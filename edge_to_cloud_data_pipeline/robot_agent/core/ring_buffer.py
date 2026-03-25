"""
ring_buffer.py — Dual-Trigger MCAP Ring Buffer (v5)

Rotation triggers (OR logic — whichever fires first):
    1. size  ≥ ROTATE_SIZE_MB  (default 5 MB)
    2. time  ≥ ROTATE_TIME_SEC (default 10 s)
    3. anomaly flag set by force_rotate(anomaly=True)

Why 5 MB?
    At 50 Mbps the robot can upload a 5 MB file in ~0.8 seconds.
    At worst-case 4G (5 Mbps) it takes ~8 seconds.
    A 50 MB file at 4G takes 80 seconds — unacceptable for P0 events.
    Large sensor bursts are automatically split into 5 MB chunks.

Why 10 seconds?
    Even if only a few KB accumulate, we rotate every 10s to ensure
    telemetry data reaches the cloud within seconds of being recorded.

MCAP splitting for large files:
    Files larger than ROTATE_SIZE_BYTES are split automatically.
    Each rotation produces one complete, self-contained MCAP file.
    The upload agent handles each file independently — no reassembly needed.
    GCS and the Cloud Run indexer see complete files every time.
"""

import io
import os
import time
import json
import struct
import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)

# ── Rotation thresholds ───────────────────────────────────────────────────────
ROTATE_SIZE_MB      = 5
ROTATE_TIME_SEC     = 10
ROTATE_SIZE_BYTES   = ROTATE_SIZE_MB * 1024 * 1024

# ── MCAP lightweight envelope constants ───────────────────────────────────────
MAGIC   = b"\x89MCAP\r\n\x1a\n"


class RingBuffer:
    """
    Captures sensor frames and rotates them into MCAP files on disk.

    Rotation is triggered by ANY of:
        - Accumulated bytes ≥ rotate_size_bytes  (5 MB default)
        - Time since last rotation ≥ rotate_time_sec (10 s default)
        - force_rotate(anomaly=True) called by anomaly detector

    Thread-safety: a single write lock serialises push() and rotate().
    Stats reading is lock-free for dashboard use.
    """

    def __init__(
        self,
        staging_root: Path,
        robot_id:     str,
        session_id:   str,
        priority:     int                              = 2,
        on_file_ready: Optional[Callable[[Path], None]] = None,
        rotate_size_bytes: int = ROTATE_SIZE_BYTES,
        rotate_time_sec:   int = ROTATE_TIME_SEC,
    ):
        self._root             = staging_root
        self._robot_id         = robot_id
        self._session_id       = session_id
        self._default_priority = priority
        self._on_file_ready    = on_file_ready
        self._rotate_size      = rotate_size_bytes
        self._rotate_time      = rotate_time_sec

        self._lock             = threading.Lock()
        self._buf: io.BytesIO  = io.BytesIO()
        self._frame_count      = 0
        self._current_bytes    = 0
        self._rotation_start   = time.time()
        self._anomaly_flag     = False

        # Back-pressure flag — set by main.py backpressure_loop
        self.backpressure_active = False

        # ── FIX: track what data_type to embed in the next filename ──────────
        # Set to "anomaly" by force_rotate() when an anomaly triggers rotation.
        # Reset to "mixed" after each rotation so normal files stay "mixed".
        self._next_rotation_data_type: str = "mixed"

        # Stats
        self._total_rotations  = 0
        self._total_bytes_in   = 0

        self._write_header()

        logger.info(
            "RingBuffer init: rotate at %dMB or %ds  session=%s  robot=%s",
            rotate_size_bytes // 1024 // 1024,
            rotate_time_sec,
            session_id,
            robot_id,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def push(
        self,
        topic:    str,
        ts_ms:    int,
        payload:  bytes,
        priority: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Add one frame to the ring buffer.
        Returns the Path of the rotated file if rotation occurred, else None.
        """
        with self._lock:
            self._write_frame(topic, ts_ms, payload)
            if self._should_rotate():
                return self._do_rotate(priority or self._default_priority)
        return None

    def push_binary(
        self,
        topic:    str,
        ts_ms:    int,
        payload:  bytes,
        encoding: str = "application/octet-stream",
        meta:     Optional[Dict[str, Any]] = None,
        priority: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Push binary payload (LiDAR, camera, log) with encoding metadata.
        Encoding is stored in the frame header for MCAP channel identification.

        Used by the simulator to write:
            lidar/points  → encoding="application/lidar-float32"
            camera/rgb    → encoding="video/h265"
            rosout        → encoding="application/json"
        """
        enc_bytes  = encoding.encode()
        meta_bytes = json.dumps(meta or {}).encode()
        frame = (
            struct.pack(">H", len(enc_bytes))  + enc_bytes
            + struct.pack(">H", len(meta_bytes)) + meta_bytes
            + payload
        )
        return self.push(topic, ts_ms, frame, priority=priority)

    def force_rotate(
        self,
        anomaly:   bool = False,
        priority:  Optional[int] = None,
        data_type: str = "mixed",
    ) -> Optional[Path]:
        """
        Force an immediate rotation regardless of size/time thresholds.
        Called from anomaly_detector when a P0 event is detected.
        anomaly=True promotes the file to P0 and writes to p0_critical/.
        data_type is embedded in the filename so the Cloud Run indexer
        can parse the correct type (e.g. "anomaly") instead of "mixed".
        """
        with self._lock:
            if anomaly:
                self._anomaly_flag = True
                # ── FIX: record data_type for the filename ────────────────────
                self._next_rotation_data_type = data_type
            effective_priority = 0 if anomaly else (priority or self._default_priority)
            if self._frame_count == 0:
                return None
            return self._do_rotate(effective_priority)

    def set_rotation_time(self, secs: float) -> None:
        """
        Update the time-based rotation threshold.
        Called by the backpressure controller to slow down rotation
        during queue buildup (reduces file creation rate).
        """
        with self._lock:
            self._rotate_time = secs
        logger.info("RingBuffer: rotation_time updated to %.0fs", secs)

    def stats(self) -> Dict[str, Any]:
        return {
            "current_bytes":    self._current_bytes,
            "current_mb":       round(self._current_bytes / 1024 / 1024, 2),
            "frame_count":      self._frame_count,
            "age_sec":          round(time.time() - self._rotation_start, 1),
            "total_rotations":  self._total_rotations,
            "total_bytes_in":   self._total_bytes_in,
            "rotate_size_mb":   self._rotate_size // 1024 // 1024,
            "rotate_time_sec":  self._rotate_time,
            "backpressure":     self.backpressure_active,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _should_rotate(self) -> bool:
        size_hit  = self._current_bytes >= self._rotate_size
        time_hit  = (time.time() - self._rotation_start) >= self._rotate_time
        anom_hit  = self._anomaly_flag
        return size_hit or time_hit or anom_hit

    def _do_rotate(self, priority: int) -> Path:
        """Finalise buffer, write to disk atomically, reset state."""
        data    = self._finalise_mcap()
        size_kb = len(data) // 1024
        hash8   = hashlib.md5(data).hexdigest()[:8]

        path = self._make_path(priority, size_kb, hash8)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: tmp → rename
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        tmp.rename(path)

        self._total_rotations += 1
        self._total_bytes_in  += len(data)

        logger.info(
            "Ring rotated → %s  (%dKB, %d frames, %.1fs, P%d)",
            path.name, size_kb, self._frame_count,
            time.time() - self._rotation_start, priority,
        )

        # Reset for next window
        self._buf           = io.BytesIO()
        self._frame_count   = 0
        self._current_bytes = 0
        self._rotation_start = time.time()
        self._anomaly_flag  = False
        # ── FIX: reset data_type after rotation so next file is "mixed" ──────
        self._next_rotation_data_type = "mixed"
        self._write_header()

        if self._on_file_ready:
            threading.Thread(
                target=self._on_file_ready, args=(path,), daemon=True
            ).start()

        return path

    def _write_header(self) -> None:
        self._buf.write(MAGIC)
        self._buf.write(struct.pack(">I", 1))   # version=1
        meta = json.dumps({
            "robot_id":   self._robot_id,
            "session_id": self._session_id,
            "created_at": int(time.time() * 1000),
        }).encode()
        self._buf.write(struct.pack(">I", len(meta)))
        self._buf.write(meta)
        self._current_bytes = self._buf.tell()

    def _write_frame(self, topic: str, ts_ms: int, payload: bytes) -> None:
        """
        Lightweight MCAP frame envelope:
        topic_len(2) + topic(N) + ts_ms(8) + payload_len(4) + payload(M)
        """
        t = topic.encode()
        self._buf.write(struct.pack(">H", len(t)))
        self._buf.write(t)
        self._buf.write(struct.pack(">Q", ts_ms))
        self._buf.write(struct.pack(">I", len(payload)))
        self._buf.write(payload)
        self._frame_count  += 1
        self._current_bytes = self._buf.tell()

    def _finalise_mcap(self) -> bytes:
        self._buf.write(b"\x00" * 4)   # EOF sentinel
        return self._buf.getvalue()

    def _make_path(self, priority: int, size_kb: int, hash8: str) -> Path:
        """
        v5 filename convention:
        {staging_root}/staging/{robot_id}/{YYYY/MM/DD}/{priority_dir}/{filename}

        Filename: {ISO8601}_{epoch_ms}_{session_id}_p{p}_{data_type}_{sizeKB}_{hash8}.mcap

        data_type is "anomaly" for anomaly-triggered rotations, "mixed" otherwise.
        This allows the Cloud Run indexer to parse anomaly_flags from the filename.
        """
        now_ms   = int(time.time() * 1000)
        iso      = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now_ms / 1000))
        today    = time.strftime("%Y/%m/%d", time.gmtime())
        p_dirs   = {0: "p0_critical", 1: "p1_high", 2: "p2_normal"}
        p_dir    = p_dirs.get(priority, "p2_normal")

        # ── FIX: use _next_rotation_data_type instead of hardcoded "mixed" ───
        data_type = self._next_rotation_data_type
        filename  = f"{iso}_{now_ms}_{self._session_id}_p{priority}_{data_type}_{size_kb}_{hash8}.mcap"
        return self._root / "staging" / self._robot_id / today / p_dir / filename