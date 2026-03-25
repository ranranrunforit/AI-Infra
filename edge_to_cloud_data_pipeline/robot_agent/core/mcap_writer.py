"""
mcap_writer.py — MCAP fastwrite strategy (v4)
Phase 1: fastwrite (no CRC, no index, no compression) → atomic rename .tmp → .mcap
Phase 2: background reindex + zstd-3 compression (lowest cgroup priority)

Uses the official Foxglove `mcap` Python library for proper MCAP format compliance.
Falls back to a lightweight binary envelope when mcap library is not available (CI/local).
"""

import os
import io
import json
import time
import struct
import hashlib
import logging
import threading
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from mcap.writer import Writer as MCAPLibWriter
    MCAP_LIB = True
except ImportError:
    MCAP_LIB = False
    logger.warning("mcap library not found — using lightweight envelope format")

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


# ── Filename convention (v4) ─────────────────────────────────────────────────
#
# {ISO8601_UTC}_{epoch_ms}_{session_id}_{priority}_{topic}_{sizeKB}_{hash8}.mcap[.zst]
# Example:
#   20250317T143022Z_1742221822000_s042_p2_lidar_512_ab12cdef.mcap
#   20250317T143022Z_1742221822000_s042_p0_anomaly_collision_128_ff01aabc.mcap
#
# Benefits:
#   ✓ ISO8601 → human-readable timeline reconstruction
#   ✓ epoch_ms → script sorting without file open
#   ✓ priority in name → upload agent sorts without manifest (cold-start safe)
#   ✓ hash8 → dedup check without full SHA-256 read
#   ✓ sizeKB → eviction scoring without stat() syscall

def make_filename(session_id: str, priority: int, topic: str,
                  size_kb: int = 0, hash8: str = "00000000",
                  ts_ms: Optional[int] = None) -> str:
    now = ts_ms or int(time.time() * 1000)
    iso = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now / 1000))
    return f"{iso}_{now}_{session_id}_p{priority}_{topic}_{size_kb}_{hash8}.mcap"


def make_staging_path(root: Path, robot_id: str, priority: int,
                      filename: str) -> Path:
    """Build hybrid directory: state-machine dirs + temporal sub-dirs (v4)."""
    today = time.strftime("%Y/%m/%d")
    priority_dir = {0: "p0_critical", 1: "p1_high", 2: "p2_normal"}.get(priority, "p2_normal")
    dest = root / "staging" / robot_id / today / priority_dir
    dest.mkdir(parents=True, exist_ok=True)
    return dest / filename


# ── Lightweight envelope (used when mcap library absent) ─────────────────────
# Header: magic(8) + version(4) + metadata_len(4) + metadata(N) + frames...
# Frame:  topic_len(2) + topic(N) + ts_ms(8) + payload_len(4) + payload(N)
MAGIC = b"\x89MCAP\r\n\x1a\n"


class LightweightMCAPWriter:
    """Minimal MCAP-envelope writer. Proper MCAP library preferred in production."""

    def __init__(self, path: Path, metadata: Dict):
        self._path = path
        self._tmp = path.with_suffix(".tmp")
        self._buf = io.BytesIO()
        # Write magic + version
        self._buf.write(MAGIC)
        self._buf.write(struct.pack(">I", 1))   # version=1
        meta_bytes = json.dumps(metadata).encode()
        self._buf.write(struct.pack(">I", len(meta_bytes)))
        self._buf.write(meta_bytes)
        self._frame_count = 0

    def write_frame(self, topic: str, ts_ms: int, payload: bytes):
        t = topic.encode()
        self._buf.write(struct.pack(">H", len(t)))
        self._buf.write(t)
        self._buf.write(struct.pack(">Q", ts_ms))
        self._buf.write(struct.pack(">I", len(payload)))
        self._buf.write(payload)
        self._frame_count += 1

    def close(self) -> Path:
        data = self._buf.getvalue()
        self._tmp.write_bytes(data)
        self._tmp.rename(self._path)   # atomic rename(2)
        logger.debug("MCAP written: %s (%d bytes, %d frames)",
                     self._path.name, len(data), self._frame_count)
        return self._path


class MCAPFastWriter:
    """
    Phase-1 fastwrite: no CRC, no index, no compression.
    Writes to .tmp then atomic rename to .mcap.
    Phase-2 background: reindex + zstd-3 compression.
    """

    CHUNK_SIZE = 4 * 1024 * 1024   # 4MB sequential chunks

    def __init__(self, staging_root: Path, robot_id: str,
                 session_id: str, priority: int, topic: str):
        self.staging_root = staging_root
        self.robot_id = robot_id
        self.session_id = session_id
        self.priority = priority
        self.topic = topic
        self._frames: List = []
        self._start_ts: Optional[int] = None
        self._writer = None
        self._metadata = {
            "robot_id": robot_id,
            "session_id": session_id,
            "priority": priority,
            "topic": topic,
            "schema_version": "1.0.0",
        }

    def add_frame(self, ts_ms: int, topic: str, payload: bytes):
        if self._start_ts is None:
            self._start_ts = ts_ms
        self._frames.append((ts_ms, topic, payload))

    def flush(self, anomaly_flags: List[str] = None) -> Optional[Path]:
        """Write buffered frames to disk using fastwrite strategy."""
        if not self._frames:
            return None

        self._metadata["anomaly_flags"] = anomaly_flags or []
        self._metadata["frame_count"] = len(self._frames)
        self._metadata["start_ts_ms"] = self._start_ts
        self._metadata["end_ts_ms"] = self._frames[-1][0]

        # Compute real hash before writing by hashing the in-memory frame payloads
        hash_ctx = hashlib.sha256()
        for f in self._frames:
            hash_ctx.update(f[2])
        real_hash8 = hash_ctx.hexdigest()[:8]
        
        size_kb = sum(len(f[2]) for f in self._frames) // 1024
        filename = make_filename(
            self.session_id, self.priority, self.topic,
            size_kb=size_kb, hash8=real_hash8,
            ts_ms=self._start_ts,
        )
        dest_path = make_staging_path(
            self.staging_root, self.robot_id, self.priority, filename
        )

        if MCAP_LIB:
            path = self._write_mcap_lib(dest_path)
        else:
            path = self._write_lightweight(dest_path)

        logger.info("MCAP flushed: %s (%d frames)", path.name, len(self._frames))
        self._frames.clear()
        return path

    def _write_lightweight(self, path: Path) -> Path:
        w = LightweightMCAPWriter(path, self._metadata)
        for ts_ms, topic, payload in self._frames:
            w.write_frame(topic, ts_ms, payload)
        return w.close()

    def _write_mcap_lib(self, path: Path) -> Path:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            writer = MCAPLibWriter(f)
            writer.start(
                library="robot-pipeline-v4",
                profile="ros2",
            )
            schema_id = writer.register_schema(
                name="sensor_data",
                encoding="jsonschema",
                data=json.dumps({
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "data": {"type": "object"},
                    }
                }).encode(),
            )
            channel_ids: Dict[str, int] = {}
            for ts_ms, topic, payload in self._frames:
                if topic not in channel_ids:
                    channel_ids[topic] = writer.register_channel(
                        schema_id=schema_id,
                        topic=topic,
                        message_encoding="json",
                    )
                writer.add_message(
                    channel_id=channel_ids[topic],
                    log_time=ts_ms * 1_000_000,   # ns
                    data=payload,
                    publish_time=ts_ms * 1_000_000,
                )
            writer.finish()
        tmp.rename(path)   # atomic rename(2)
        return path


# ── Phase 2: background reindex + compression ────────────────────────────────

class BackgroundReindexer(threading.Thread):
    """
    Lowest-priority background worker.
    Watches staging dir for uncompressed .mcap files, compresses with zstd-3.
    Paused by CpuGovernor under CPU pressure (see cpu_governor.py pause order).
    """

    def __init__(self, staging_root: Path, manifest=None, check_interval: float = 30.0):
        super().__init__(daemon=True, name="mcap-reindexer")
        self.staging_root = staging_root
        self.manifest = manifest
        self.check_interval = check_interval
        self._paused = threading.Event()
        self._paused.set()  # initially running
        self._stop = threading.Event()

    def run(self):
        logger.info("BackgroundReindexer started")
        while not self._stop.is_set():
            self._paused.wait(timeout=self.check_interval)
            if self._stop.is_set():
                break
            self._process_batch()
            time.sleep(self.check_interval)

    def _process_batch(self, batch_size: int = 5):
        uncompressed = list(self.staging_root.rglob("*.mcap"))
        for path in uncompressed[:batch_size]:
            if self._stop.is_set():
                return
            self._compress(path)

    def _compress(self, path: Path):
        out = path.with_suffix(".mcap.zst")
        if out.exists():
            return   # already compressed

        if self.manifest:
            rec = self.manifest.get_by_filename(path.name)
            if rec and rec.state in ("UPLOADING", "VERIFYING", "UPLOADED", "DELETED", "EVICTED"):
                return
        else:
            rec = None

        try:
            success = False
            if ZSTD_AVAILABLE:
                cctx = zstd.ZstdCompressor(level=3)
                with open(path, "rb") as src, open(out, "wb") as dst:
                    dst.write(cctx.compress(src.read()))
                path.unlink()
                logger.debug("Compressed: %s → %s", path.name, out.name)
                success = True
            else:
                # Fall back to system zstd
                result = subprocess.run(
                    ["zstd", "-3", "-q", "--rm", str(path)],
                    capture_output=True, timeout=60,
                )
                if result.returncode == 0:
                    logger.debug("Compressed (system zstd): %s", path.name)
                    success = True
            
            if success and self.manifest and rec:
                self.manifest.update_file_location(
                    rec.id, out.name, str(out), out.stat().st_size
                )
        except Exception as exc:
            logger.warning("Compression failed for %s: %s", path.name, exc)

    def pause(self):
        self._paused.clear()
        logger.debug("BackgroundReindexer paused")

    def resume(self):
        self._paused.set()
        logger.debug("BackgroundReindexer resumed")

    def stop(self):
        self._stop.set()
        self._paused.set()  # unblock if paused
