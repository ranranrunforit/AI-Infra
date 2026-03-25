"""
eviction_manager.py — Dual-Watermark Eviction Cascade (v4)
Trigger at 75% disk usage, recover to 60% safe-water line.
Prevents threshold oscillation at boundary.

4-step cascade:
  Step 1: GC — delete UPLOADED files past grace period (free and safe)
  Step 2: Prune — PRAGMA incremental_vacuum
  Step 3: Compression — zstd dictionary on pending P2 (up to 90% reduction from report)
  Step 4: Priority eviction — lowest priority + oldest first. P0 NEVER evicted.
"""

import os
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

HIGH_WATER_PCT = 75.0   # trigger eviction cascade
LOW_WATER_PCT  = 60.0   # recovery target
CRITICAL_PCT   = 90.0   # P0-only territory — fire alert


class EvictionManager(threading.Thread):
    """
    Monitors disk usage every 60s.
    Runs 4-step cascade when disk > 75%.
    Thread-safe: runs as background daemon.
    """

    def __init__(self, manifest, staging_root: Path,
                 check_interval: float = 60.0,
                 quota_bytes: float = 85899345920.0,
                 dict_path: Optional[Path] = None,
                 alert_fn=None):
        super().__init__(daemon=True, name="eviction-manager")
        self.manifest = manifest
        self.staging_root = staging_root
        self.check_interval = check_interval
        self.quota_bytes = quota_bytes
        self.dict_path = dict_path or Path("/etc/robot/zstd.dict")
        self.alert_fn = alert_fn or (lambda msg: logger.critical(msg))
        self._stop = threading.Event()
        self._evictions = 0
        self._compressions = 0
        self._gc_deletes = 0

    def run(self):
        logger.info("EvictionManager started (HW=%.0f%% LW=%.0f%%)",
                    HIGH_WATER_PCT, LOW_WATER_PCT)
        while not self._stop.is_set():
            try:
                pct = self.disk_usage_pct()
                if pct >= CRITICAL_PCT:
                    self.alert_fn(
                        f"CRITICAL: disk {pct:.1f}%, only P0 files remain. "
                        f"Hardware capacity issue likely."
                    )
                if pct >= HIGH_WATER_PCT:
                    logger.warning("Disk %.1f%% ≥ %.0f%% — eviction cascade triggered",
                                   pct, HIGH_WATER_PCT)
                    self._cascade(pct)
            except Exception as exc:
                logger.error("EvictionManager error: %s", exc)

            self._stop.wait(timeout=self.check_interval)

    def _cascade(self, starting_pct: float):
        """Run 4-step cascade until disk drops below LOW_WATER_PCT."""
        logger.info("Starting eviction cascade from %.1f%%", starting_pct)

        # ── Step 1: GC — delete UPLOADED files past grace period ──────────
        expired = self.manifest.get_uploaded_grace_expired()
        for f in expired:
            try:
                Path(f.path).unlink(missing_ok=True)
                self.manifest.set_state(f.id, "DELETED")
                self._gc_deletes += 1
                logger.debug("GC deleted: %s", f.filename)
            except OSError as exc:
                logger.warning("GC delete failed %s: %s", f.path, exc)
            if self.disk_usage_pct() < LOW_WATER_PCT:
                logger.info("Recovered to <%.0f%% after GC (Step 1)", LOW_WATER_PCT)
                return

        # ── Step 2: Prune — PRAGMA incremental_vacuum ─────────────────────
        self.manifest.incremental_vacuum(pages=500)
        if self.disk_usage_pct() < LOW_WATER_PCT:
            logger.info("Recovered to <%.0f%% after vacuum (Step 2)", LOW_WATER_PCT)
            return

        # ── Step 3: Compression — zstd dictionary on pending P2 ────────────
        pending_low = self.manifest.get_evictable(min_priority=2)  # P2 only (P3 removed)
        for f in pending_low:
            p = Path(f.path)
            if not p.exists() or p.suffix == ".zst":
                continue
            try:
                new_path = self._compress_with_dict(p)
                if new_path and new_path != p:
                    self.manifest.update_file_location(
                        f.id, new_path.name, str(new_path), new_path.stat().st_size
                    )
                    self.manifest.set_state(f.id, "PENDING")
                    self._compressions += 1
            except Exception as exc:
                logger.warning("Compression failed %s: %s", f.path, exc)
            if self.disk_usage_pct() < LOW_WATER_PCT:
                logger.info("Recovered to <%.0f%% after compression (Step 3)", LOW_WATER_PCT)
                return

        # ── Step 4: Priority eviction — P1+ only, lowest priority first ───
        evictable = self.manifest.get_evictable(min_priority=1)   # never P0
        for f in evictable:
            if self.disk_usage_pct() < LOW_WATER_PCT:
                logger.info("Recovered to <%.0f%% after eviction (Step 4)", LOW_WATER_PCT)
                return
            try:
                Path(f.path).unlink(missing_ok=True)
                self.manifest.set_state(f.id, "EVICTED")
                self._evictions += 1
                logger.info("Evicted P%d: %s", f.priority, f.filename)
            except OSError as exc:
                logger.warning("Eviction failed %s: %s", f.path, exc)

        # If we're here and only P0 remains → critical alert
        final_pct = self.disk_usage_pct()
        if final_pct > CRITICAL_PCT:
            self.alert_fn(
                f"CRITICAL: disk {final_pct:.1f}% after full cascade. "
                f"Only P0 files remain or cascade failed. Immediate action required."
            )

    def _compress_with_dict(self, path: Path) -> Optional[Path]:
        """
        zstd dictionary compression.
        Report claims up to 90% reduction on structured JSON/telemetry.
        """
        out = path.with_suffix(path.suffix + ".zst")
        if out.exists():
            return out   # already compressed

        if not ZSTD_AVAILABLE:
            logger.debug("zstd not available — skipping dict compression")
            return None

        try:
            if self.dict_path.exists():
                with open(self.dict_path, "rb") as df:
                    dict_data = zstd.ZstdCompressionDict(df.read())
                cctx = zstd.ZstdCompressor(level=3, dict_data=dict_data)
            else:
                cctx = zstd.ZstdCompressor(level=3)

            with open(path, "rb") as src, open(out, "wb") as dst:
                dst.write(cctx.compress(src.read()))

            original_size = path.stat().st_size
            compressed_size = out.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100
            
            if compressed_size >= original_size * 0.95:
                logger.debug("Compression saved <5%% (%d → %d bytes), keeping original",
                             original_size, compressed_size)
                out.unlink()
                return None

            logger.debug("Compressed %s: %.1f%% reduction (%d → %d bytes)",
                         path.name, ratio, original_size, compressed_size)
            path.unlink()
            return out
        except Exception as exc:
            logger.warning("Dict compression failed: %s", exc)
            if out.exists():
                out.unlink()
            return None

    def disk_usage_pct(self) -> float:
        try:
            used = sum(f.stat().st_size for f in self.staging_root.rglob("*") if f.is_file())
            return (used / self.quota_bytes) * 100.0
        except Exception:
            return 0.0

    def stats(self) -> dict:
        return {
            "disk_usage_pct": self.disk_usage_pct(),
            "high_water_pct": HIGH_WATER_PCT,
            "low_water_pct": LOW_WATER_PCT,
            "evictions_total": self._evictions,
            "compressions_total": self._compressions,
            "gc_deletes_total": self._gc_deletes,
        }

    def stop(self):
        self._stop.set()
