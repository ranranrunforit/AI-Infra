"""
codec_utils.py — Format-Aware Compression (Phase 3)

Two compression strategies for bandwidth reduction before ring buffer:

1. H.264/H.265 Hardware Encoding with mandatory short GOP
   - Jetson: nvv4l2h265enc   (NVIDIA hardware)
   - x86:   ffmpeg -hwaccel cuda / vaapi
   - CRITICAL: gop-size=30 (1 I-frame/sec at 30fps) or shorter
     Without this, anomaly detector captures 10s pre-event windows
     that are undecodable — the decoder needs the preceding I-frame
     which may be 30s back in the stream.
   - 10-16× bandwidth reduction vs zstd on raw frames

2. LiDAR Compression: bit-shuffle + zstd (lossless)
   - bitshuffle transposes float32 arrays so identical exponent bits
     group together → zstd compresses ~4:1 (vs ~1.5:1 on raw binary)
   - Lossless — preserves full SLAM mapping accuracy
   - Alternative: Draco (lossy, ~10:1, fast decode, qp=14 gives 0.6mm accuracy)
"""

import io
import logging
import struct
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


# ── H.265 Video Encoding ──────────────────────────────────────────────────────

class VideoEncoder:
    """
    Hardware-accelerated H.265 encoder.

    Wraps ffmpeg with hardware acceleration flags for Jetson (nvenc) or x86 (vaapi).
    Mandatory: GOP size = 30 (1 I-frame per second at 30fps).

    Usage:
        encoder = VideoEncoder(gop_size=30, quality=23)
        compressed = encoder.encode(raw_frame_bytes, width=640, height=480)
    """

    def __init__(self, gop_size: int = 30, quality: int = 23,
                 hw_accel: str = "auto"):
        self.gop_size = gop_size
        self.quality = quality
        self.hw_accel = hw_accel
        self._encoder_cmd = self._detect_encoder()

    def _detect_encoder(self) -> list:
        """Detect available hardware encoder."""
        # Try Jetson NVENC first
        encoders = [
            {
                "name": "nvenc",
                "cmd": [
                    "ffmpeg", "-y", "-hwaccel", "cuda",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "{width}x{height}", "-r", "30",
                    "-i", "pipe:0",
                    "-c:v", "hevc_nvenc",
                    "-preset", "fast",
                    "-g", str(self.gop_size),
                    "-qp", str(self.quality),
                    "-f", "hevc", "pipe:1",
                ],
            },
            {
                "name": "vaapi",
                "cmd": [
                    "ffmpeg", "-y",
                    "-vaapi_device", "/dev/dri/renderD128",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "{width}x{height}", "-r", "30",
                    "-i", "pipe:0",
                    "-vf", "format=nv12,hwupload",
                    "-c:v", "hevc_vaapi",
                    "-g", str(self.gop_size),
                    "-qp", str(self.quality),
                    "-f", "hevc", "pipe:1",
                ],
            },
            {
                "name": "software",
                "cmd": [
                    "ffmpeg", "-y",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "{width}x{height}", "-r", "30",
                    "-i", "pipe:0",
                    "-c:v", "libx265",
                    "-preset", "ultrafast",
                    "-x265-params", f"keyint={self.gop_size}:min-keyint={self.gop_size}",
                    "-crf", str(self.quality),
                    "-f", "hevc", "pipe:1",
                ],
            },
        ]

        if self.hw_accel != "auto":
            for e in encoders:
                if e["name"] == self.hw_accel:
                    logger.info("Using %s H.265 encoder (GOP=%d)", e["name"], self.gop_size)
                    return e["cmd"]

        # Auto-detect: try each and check if ffmpeg supports it
        for e in encoders:
            try:
                test_cmd = ["ffmpeg", "-hide_banner", "-encoders"]
                result = subprocess.run(test_cmd, capture_output=True, timeout=5)
                codec_name = "hevc_nvenc" if e["name"] == "nvenc" else (
                    "hevc_vaapi" if e["name"] == "vaapi" else "libx265"
                )
                if codec_name.encode() in result.stdout:
                    logger.info("Detected %s H.265 encoder (GOP=%d)", e["name"], self.gop_size)
                    return e["cmd"]
            except Exception:
                continue

        logger.warning("No H.265 encoder detected — video compression disabled")
        return []

    def encode(self, raw_frames: bytes, width: int, height: int) -> Optional[bytes]:
        """
        Encode raw RGB24 frames to H.265 via ffmpeg pipe.
        Returns compressed bytes or None if encoding fails.
        """
        if not self._encoder_cmd:
            return None

        cmd = [c.replace("{width}x{height}", f"{width}x{height}")
               for c in self._encoder_cmd]
        try:
            proc = subprocess.run(
                cmd, input=raw_frames,
                capture_output=True, timeout=30,
            )
            if proc.returncode == 0:
                ratio = len(raw_frames) / max(len(proc.stdout), 1)
                logger.debug("H.265 encoded: %d → %d bytes (%.1f:1)",
                             len(raw_frames), len(proc.stdout), ratio)
                return proc.stdout
            else:
                logger.warning("H.265 encoding failed: %s", proc.stderr[:200])
                return None
        except subprocess.TimeoutExpired:
            logger.warning("H.265 encoding timed out")
            return None
        except Exception as exc:
            logger.warning("H.265 encoding error: %s", exc)
            return None


# ── LiDAR Compression (bit-shuffle + zstd) ────────────────────────────────────

class LiDARCompressor:
    """
    Lossless LiDAR point cloud compression using bit-shuffle + zstd.

    Bit-shuffle transposes float32 arrays so identical exponent bits
    group together — zstd then compresses ~4:1 (vs ~1.5:1 raw).

    Alternative Draco path for lossy compression (~10:1).

    Usage:
        compressor = LiDARCompressor()
        compressed = compressor.compress(point_cloud_bytes, dtype='float32')
        original = compressor.decompress(compressed, dtype='float32')
    """

    MAGIC = b"LIDR"  # magic header for identifying compressed format

    def __init__(self, zstd_level: int = 3):
        self.zstd_level = zstd_level

    def compress(self, data: bytes, dtype: str = "float32") -> bytes:
        """
        Compress point cloud data using bit-shuffle + zstd.
        Returns: MAGIC(4) + dtype_len(2) + dtype(N) + original_len(4) + zstd_compressed_data
        """
        if not NP_AVAILABLE or not ZSTD_AVAILABLE:
            logger.debug("numpy or zstd unavailable — returning raw data")
            return data

        arr = np.frombuffer(data, dtype=dtype)

        # Bit-shuffle: transpose byte layout for better compression
        # Group all byte-0s together, then byte-1s, etc.
        element_size = arr.dtype.itemsize
        raw = arr.tobytes()
        n_elements = len(arr)

        # Manual bit-shuffle: reshape to (n_elements, element_size), transpose, flatten
        if n_elements > 0:
            byte_arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_elements, element_size)
            shuffled = byte_arr.T.tobytes()
        else:
            shuffled = raw

        # zstd compress the shuffled data
        cctx = zstd.ZstdCompressor(level=self.zstd_level)
        compressed = cctx.compress(shuffled)

        # Build output: header + compressed data
        dtype_bytes = dtype.encode()
        header = (
            self.MAGIC
            + struct.pack(">H", len(dtype_bytes))
            + dtype_bytes
            + struct.pack(">I", len(data))
        )
        result = header + compressed

        ratio = len(data) / max(len(result), 1)
        logger.debug("LiDAR compressed: %d → %d bytes (%.1f:1 ratio)",
                     len(data), len(result), ratio)
        return result

    def decompress(self, data: bytes, dtype: str = "float32") -> bytes:
        """Decompress bit-shuffle + zstd compressed point cloud."""
        if not NP_AVAILABLE or not ZSTD_AVAILABLE:
            return data

        # Parse header
        if not data.startswith(self.MAGIC):
            return data  # not our format

        offset = 4
        dtype_len = struct.unpack(">H", data[offset:offset+2])[0]
        offset += 2
        stored_dtype = data[offset:offset+dtype_len].decode()
        offset += dtype_len
        original_len = struct.unpack(">I", data[offset:offset+4])[0]
        offset += 4

        # zstd decompress
        dctx = zstd.ZstdDecompressor()
        shuffled = dctx.decompress(data[offset:])

        # Reverse bit-shuffle
        element_size = np.dtype(stored_dtype).itemsize
        n_elements = original_len // element_size

        if n_elements > 0:
            byte_arr = np.frombuffer(shuffled, dtype=np.uint8).reshape(element_size, n_elements)
            unshuffled = byte_arr.T.tobytes()
        else:
            unshuffled = shuffled

        return unshuffled[:original_len]
