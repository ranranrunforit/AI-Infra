"""
bandwidth_limiter.py — MMCF + Adaptive Token Bucket (v5.1)

Architecture:
    P0  → [Token Bucket ≤50 Mbps] → upload
    P1-P3 → [Thermal cutoff] → [MMCF cost gate] → [Token Bucket ≤50 Mbps] → upload

P0 behaviour:
    - Bypasses: thermal cutoff (P0 always uploads, even at 90°C)
    - Bypasses: MMCF cost check
    - Does NOT bypass: Token Bucket — still capped at 50 Mbps
    Rationale: 50 Mbps is a physical network constraint, not a policy.
               P0 events (collision, battery_critical) must upload immediately.
               But they cannot physically exceed the 50 Mbps network ceiling.

P1-P3 behaviour:
    Step 1 — Thermal cutoff: if cpu_temp >= throttle_temp (default 85°C),
             block immediately. This is a hard gate, not a cost calculation.
             All computation and upload resources go to SLAM/navigation.
    Step 2 — MMCF cost check: cost = W_bw*bw_usage + W_temp*temp + W_prio*priority
             If cost > limit, back off MMCF_BACKOFF_SEC and retry.
    Step 3 — Token bucket: consume bandwidth tokens at ≤50 Mbps.
"""

import os
import time
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── Hard constraints ──────────────────────────────────────────────────────────
MAX_RATE_BPS   = 6_250_000   # 50 Mbps — absolute physical ceiling, ALL priorities
MIN_RATE_BPS   = 64_000      # 512 Kbps floor (worst-case 4G)
INITIAL_RATE   = 6_250_000   # start at max ceiling — probe will throttle DOWN if link is slow

# ── MMCF thresholds (overridable via constructor) ─────────────────────────────
MMCF_COST_LIMIT    = 100.0
MMCF_BACKOFF_SEC   = 0.5
COST_W_BW          = 20.0    # bandwidth utilisation weight
COST_W_TEMP        = 1.0     # CPU temperature weight (°C above ambient)
COST_W_PRIO        = 10.0    # priority weight (1=P1, 2=P2, 3=P3)
THROTTLE_TEMP_C    = 85.0    # hard thermal cutoff for ALL P1-P3

# ── Adaptive probing ──────────────────────────────────────────────────────────
PROBE_INTERVAL_SEC  = 30
PROBE_SIZE_BYTES    = 256 * 1024
PROBE_SAFETY_MARGIN = 0.80

# ── CPU temp source ───────────────────────────────────────────────────────────
CPU_TEMP_MOCK_FILE  = "/tmp/robot_cpu_temp"
CPU_TEMP_SYSFS_FILE = "/sys/class/thermal/thermal_zone0/temp"
CPU_TEMP_FALLBACK   = 50.0


class DailyCapExceeded(Exception):
    """Retained for import compatibility — no longer raised (daily cap removed)."""


class BandwidthLimiter:
    """
    Adaptive token bucket with MMCF gate and hard thermal cutoff.

    P0 bypasses both the thermal cutoff and MMCF check.
    P0 still enters the token bucket (cannot physically exceed 50 Mbps).

    P1-P3 pass through THREE gates in order:
        1. Thermal cutoff: temp >= throttle_temp → block ALL P1-P3
        2. MMCF cost check: cost > limit → back off
        3. Token bucket: consume bandwidth at ≤ 50 Mbps
    """

    def __init__(
        self,
        initial_rate_bps: int   = INITIAL_RATE,
        max_rate_bps: int       = MAX_RATE_BPS,
        gcs_probe_bucket: Optional[str] = None,
        # MMCF overrides (read from pipeline.yaml mmcf section)
        cost_limit:   float = MMCF_COST_LIMIT,
        backoff_sec:  float = MMCF_BACKOFF_SEC,
        weight_bw:    float = COST_W_BW,
        weight_temp:  float = COST_W_TEMP,
        weight_prio:  float = COST_W_PRIO,
        throttle_temp_c: float = THROTTLE_TEMP_C,
    ):
        self._lock          = threading.Lock()
        self._max_rate      = max_rate_bps
        self._rate          = initial_rate_bps
        # Burst capacity is FIXED at 16MB — must always be >= CHUNK_SIZE (8MB).
        # Never tie _max_tokens to _rate: a slow probe rate would make the
        # bucket smaller than one chunk, causing _consume_tokens to deadlock.
        BURST_CAPACITY_BYTES = 16 * 1024 * 1024
        self._tokens        = float(BURST_CAPACITY_BYTES)
        self._max_tokens    = float(BURST_CAPACITY_BYTES)
        self._min_probe_rate = 1_250_000  # 10 Mbps floor
        self._last_refill   = time.monotonic()

        self._probe_bucket  = gcs_probe_bucket
        self._last_probe_ts = 0.0
        self._probe_lock    = threading.Lock()

        # MMCF parameters
        self._cost_limit      = cost_limit
        self._backoff_sec     = backoff_sec
        self._w_bw            = weight_bw
        self._w_temp          = weight_temp
        self._w_prio          = weight_prio
        self._throttle_temp   = throttle_temp_c

        # Stats
        self._total_p0_bytes  = 0
        self._total_p13_bytes = 0
        self._cost_backoffs   = 0
        self._thermal_blocks  = 0

        logger.info(
            "BandwidthLimiter v5.1: ceiling=%.0f Mbps  "
            "P0=token-bucket-only  P1-P3=thermal(%.0f°C)+MMCF+bucket",
            max_rate_bps / 1_000_000,
            throttle_temp_c,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def wait_for_token(self, nbytes: int, priority: int) -> None:
        """
        Block until bandwidth is available for this upload.

        P0: goes directly to token bucket. No thermal check, no MMCF.
        P1-P3: thermal cutoff → MMCF cost gate → token bucket.
        """
        if priority == 0:
            self._consume_tokens(nbytes, is_p0=True)
            return

        # P1-P3: three-gate path
        while True:
            self._maybe_probe()

            # ── Gate 1: Hard thermal cutoff ────────────────────────────────
            cpu_temp = self._read_cpu_temp()
            if cpu_temp >= self._throttle_temp:
                logger.warning(
                    "THERMAL BLOCK P%d: cpu_temp=%.1f°C >= throttle=%.1f°C "
                    "— P1-P3 uploads suspended (P0 still active)",
                    priority, cpu_temp, self._throttle_temp,
                )
                with self._lock:
                    self._thermal_blocks += 1
                time.sleep(1.0)  # poll every 1s for cooling
                continue

            # ── Gate 2: MMCF cost check ────────────────────────────────────
            with self._lock:
                self._refill()
                bw_usage = 1.0 - (self._tokens / max(self._max_tokens, 1.0))

            cost = (
                self._w_bw   * bw_usage
                + self._w_temp * cpu_temp
                + self._w_prio * priority
            )

            if cost > self._cost_limit:
                logger.debug(
                    "MMCF backoff P%d: cost=%.1f  bw=%.2f  temp=%.1f°C  limit=%.1f",
                    priority, cost, bw_usage, cpu_temp, self._cost_limit,
                )
                with self._lock:
                    self._cost_backoffs += 1
                time.sleep(self._backoff_sec)
                continue

            # ── Gate 3: Token bucket ───────────────────────────────────────
            self._consume_tokens(nbytes, is_p0=False)
            return

    def compute_cost(self, priority: int) -> float:
        """Return current MMCF cost for a given priority (for dashboard/tests)."""
        with self._lock:
            self._refill()
            bw_usage = 1.0 - (self._tokens / max(self._max_tokens, 1.0))
        cpu_temp = self._read_cpu_temp()
        return self._w_bw * bw_usage + self._w_temp * cpu_temp + self._w_prio * priority

    def is_thermally_throttled(self) -> bool:
        """True when CPU temperature is above the hard cutoff threshold."""
        return self._read_cpu_temp() >= self._throttle_temp

    def current_level(self) -> float:
        """Token bucket fill level, 0.0–1.0."""
        with self._lock:
            self._refill()
            return self._tokens / max(self._max_tokens, 1.0)

    def record_bytes(self, nbytes: int, is_p0: bool = False) -> None:
        """Compatibility shim — accounting is now internal to _consume_tokens."""
        pass

    def bytes_today(self) -> int:
        """Compatibility shim — daily cap removed."""
        return self._total_p13_bytes

    def stats(self) -> dict:
        cpu_temp = self._read_cpu_temp()
        with self._lock:
            return {
                "rate_bps":          self._rate,
                "rate_mbps":         round(self._rate / 1_000_000, 2),
                "token_level":       round(self._tokens / max(self._max_tokens, 1), 3),
                "total_p0_bytes":    self._total_p0_bytes,
                "total_p13_bytes":   self._total_p13_bytes,
                "cost_backoffs":     self._cost_backoffs,
                "thermal_blocks":    self._thermal_blocks,
                "cpu_temp_C":        cpu_temp,
                "throttle_active":   cpu_temp >= self._throttle_temp,
                "mmcf_cost_p1":      round(self._w_bw * (1.0 - self._tokens / max(self._max_tokens, 1)) + self._w_temp * cpu_temp + self._w_prio * 1, 1),
                "mmcf_cost_p3":      round(self._w_bw * (1.0 - self._tokens / max(self._max_tokens, 1)) + self._w_temp * cpu_temp + self._w_prio * 3, 1),
            }

    # ── Token bucket internals ────────────────────────────────────────────────

    def _consume_tokens(self, nbytes: int, is_p0: bool) -> None:
        """
        Block until enough tokens are available then consume them.
        The 50 Mbps ceiling applies equally to P0 and P1-P3.
        """
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= nbytes:
                    self._tokens -= nbytes
                    if is_p0:
                        self._total_p0_bytes  += nbytes
                    else:
                        self._total_p13_bytes += nbytes
                    return
                deficit = nbytes - self._tokens

            sleep_sec = deficit / max(self._rate, 1)
            time.sleep(min(sleep_sec, 1.0))

    def _refill(self) -> None:
        """Refill tokens based on elapsed wall time. Must be called under self._lock."""
        now     = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self._tokens + elapsed * self._rate, self._max_tokens)

    # ── Adaptive probing ──────────────────────────────────────────────────────

    def _maybe_probe(self) -> None:
        """Run a bandwidth probe if PROBE_INTERVAL_SEC has elapsed."""
        now = time.monotonic()
        if now - self._last_probe_ts < PROBE_INTERVAL_SEC:
            return
        if not self._probe_lock.acquire(blocking=False):
            return
        try:
            self._last_probe_ts = now
            new_rate = self._run_probe()
            if new_rate:
                with self._lock:
                    clamped = max(new_rate, self._min_probe_rate)
                    self._rate = clamped  # throttle refill speed
                    # DO NOT update _max_tokens — burst cap is fixed at 16MB
                logger.info("Bandwidth probe: %.1f Mbps", new_rate / 1_000_000)
        finally:
            self._probe_lock.release()

    def _run_probe(self) -> Optional[int]:
        """Measure actual GCS upload bandwidth. Falls back to simulation if no bucket."""
        if not self._probe_bucket:
            import random
            simulated_mbps = 20 + random.randint(0, 30)   # 20–50 Mbps range
            return min(int(simulated_mbps * 125_000), self._max_rate)
        try:
            from google.cloud import storage as gcs
            client = gcs.Client()
            blob   = client.bucket(self._probe_bucket).blob(f"_probe/{int(time.time())}.bin")
            data   = bytes(PROBE_SIZE_BYTES)
            t0     = time.perf_counter()
            blob.upload_from_string(data)
            elapsed = max(time.perf_counter() - t0, 0.001)
            blob.delete()
            measured_bps = int(len(data) / elapsed)
            safe_bps     = int(measured_bps * PROBE_SAFETY_MARGIN)
            return max(MIN_RATE_BPS, min(safe_bps, self._max_rate))
        except Exception as exc:
            logger.warning("Bandwidth probe failed: %s", exc)
            return None

    # ── CPU temperature reader ────────────────────────────────────────────────

    def _read_cpu_temp(self) -> float:
        """
        Read CPU temperature from (in order of preference):
          1. /tmp/robot_cpu_temp  — written by ThermalSimulator every tick
          2. /sys/class/thermal/  — real hardware sysfs (Linux)
          3. CPU_TEMP_FALLBACK    — conservative default (50°C)
        """
        if os.path.exists(CPU_TEMP_MOCK_FILE):
            try:
                return float(open(CPU_TEMP_MOCK_FILE).read().strip())
            except (ValueError, OSError):
                pass
        if os.path.exists(CPU_TEMP_SYSFS_FILE):
            try:
                return float(open(CPU_TEMP_SYSFS_FILE).read().strip()) / 1000.0
            except (ValueError, OSError):
                pass
        return CPU_TEMP_FALLBACK