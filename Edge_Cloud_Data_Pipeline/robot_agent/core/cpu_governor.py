"""
cpu_governor.py — Kernel Cgroup Hard Quota (v4)
cfs_period_us=100000 / cfs_quota_us=20000 → exactly 20% of 1 core.

Key insight (from report):
  nice +19 is INSUFFICIENT — a nice process still consumes 100% CPU when idle.
  On compact robot chassis this triggers thermal throttle → CPU downclocks
  → SLAM framerates drop → navigation drift / collision.
  Only cgroups provide a hard ceiling enforced by the kernel's CFS scheduler.

Child processes via ProcessPoolExecutor INHERIT parent cgroup namespace.
Aggregate CPU of all workers (GIL bypass) NEVER exceeds 20%.
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

CGROUP_ROOT = Path("/sys/fs/cgroup/robot_pipeline")

# From report: explicit kernel params (not just nice/ionice)
CFS_PERIOD_US = 100_000   # 100ms scheduling window
CFS_QUOTA_US  =  20_000   # 20ms CPU per window = exactly 20%
IO_WEIGHT     =     200   # lowest I/O priority (default=100, SLAM=1000)

# Pause order: lower index = paused first under CPU pressure
PAUSE_ORDER = [
    "log_compression",    # background reindexer — paused first
    "bulk_upload_p3",     # low-priority bulk uploads
    "bulk_upload_p2",     # normal-priority uploads
    "lidar_reindex",      # MCAP phase-2 reindex
    "anomaly_classifier", # defer ML scoring (YAML rules still run)
    # "critical_upload"   # NEVER paused — P0 always transmits
]


class CpuGovernor:
    """
    Manages cgroup lifecycle for the robot pipeline process tree.
    Call `enter()` at startup to register the main process.
    All child processes (ProcessPoolExecutor workers) automatically inherit.
    """

    def __init__(self, cgroup_root: Path = CGROUP_ROOT, dry_run: bool = False):
        self.cgroup_root = cgroup_root
        self.dry_run = dry_run
        self._registered_workers: List[str] = []

    def setup(self) -> bool:
        """Create cgroup and write kernel params. Returns True if successful."""
        if self.dry_run:
            logger.info("[DRY RUN] Would setup cgroup at %s", self.cgroup_root)
            return True

        if not os.path.exists("/sys/fs/cgroup"):
            logger.warning("cgroups not available — CPU governance disabled")
            return False

        try:
            self.cgroup_root.mkdir(parents=True, exist_ok=True)

            # Hard CPU quota: 20ms per 100ms window = 20% of 1 core
            self._write(self.cgroup_root / "cpu.cfs_period_us", str(CFS_PERIOD_US))
            self._write(self.cgroup_root / "cpu.cfs_quota_us",  str(CFS_QUOTA_US))

            # I/O weight: pipeline gets lowest priority
            io_weight_path = self.cgroup_root / "io.weight"
            if io_weight_path.exists():
                self._write(io_weight_path, f"default {IO_WEIGHT}")

            logger.info(
                "Cgroup configured: CPU quota %d/%d us (%.0f%%), I/O weight %d",
                CFS_QUOTA_US, CFS_PERIOD_US,
                CFS_QUOTA_US / CFS_PERIOD_US * 100, IO_WEIGHT,
            )
            return True
        except PermissionError:
            logger.warning(
                "No permission to configure cgroup (not root). "
                "Run as root or add CAP_SYS_ADMIN for production deployment."
            )
            return False
        except Exception as exc:
            logger.error("Cgroup setup failed: %s", exc)
            return False

    def enter(self) -> bool:
        """
        Register this process (and all future children) in the cgroup.
        Call once at main process startup.
        ProcessPoolExecutor workers inherit automatically — aggregate stays ≤20%.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would enter PID %d into cgroup", os.getpid())
            return True

        procs_file = self.cgroup_root / "cgroup.procs"
        if not procs_file.exists():
            if not self.setup():
                return False

        try:
            self._write(procs_file, str(os.getpid()))
            logger.info("PID %d entered cgroup %s", os.getpid(), self.cgroup_root)
            return True
        except PermissionError:
            logger.warning("Cannot enter cgroup (not root) — running without CPU governance")
            return False
        except Exception as exc:
            logger.error("Failed to enter cgroup: %s", exc)
            return False

    def current_usage(self) -> Optional[float]:
        """Read current CPU usage as fraction (0.0–1.0)."""
        stat_file = self.cgroup_root / "cpu.stat"
        if not stat_file.exists():
            return None
        try:
            text = stat_file.read_text()
            for line in text.splitlines():
                if line.startswith("usage_usec"):
                    current_usec = int(line.split()[1])
                    now = time.time()
                    
                    if not hasattr(self, "_last_usec"):
                        self._last_usec = current_usec
                        self._last_time = now
                        return 0.0
                        
                    delta_usec = current_usec - self._last_usec
                    delta_time = now - self._last_time
                    self._last_usec = current_usec
                    self._last_time = now
                    
                    if delta_time <= 0: return 0.0
                    
                    fraction = (delta_usec / 1_000_000) / delta_time
                    quota = CFS_QUOTA_US / CFS_PERIOD_US
                    return min(1.0, fraction / quota)
        except Exception:
            return None

    @staticmethod
    def _write(path: Path, value: str):
        with open(path, "w") as f:
            f.write(value)


# ── OS-Level SSD Wear Reduction setup script ──────────────────────────────────

OS_WEAR_SETUP = """\
#!/usr/bin/env bash
# os_wear_setup.sh — Run once at system provisioning (v4)
# Reduces eMMC P/E cycles; these 3 changes protect flash life significantly.

set -euo pipefail

# 1. journald to RAM only
echo "Configuring journald volatile mode..."
grep -q "Storage=volatile" /etc/systemd/journald.conf \
  || echo "Storage=volatile" >> /etc/systemd/journald.conf
systemctl restart systemd-journald

# 2. log2ram — intercepts all app logs, syncs to disk daily or on clean shutdown
echo "Installing log2ram..."
if ! command -v log2ram &>/dev/null; then
    echo "deb [signed-by=/usr/share/keyrings/azlux-archive-keyring.gpg] http://packages.azlux.fr/debian/ stable main" \
        > /etc/apt/sources.list.d/azlux.list
    apt-get update -qq && apt-get install -y log2ram
fi
sed -i 's/SIZE=.*/SIZE=128M/' /etc/log2ram.conf
systemctl enable --now log2ram

# 3. Disable HA micro-writes (if Proxmox/pve present)
for svc in pve-ha-lrm pve-ha-crm; do
    systemctl is-active --quiet "$svc" 2>/dev/null && systemctl disable --now "$svc" || true
done

# 4. Configure cgroup for robot pipeline
mkdir -p /sys/fs/cgroup/robot_pipeline
echo "100000" > /sys/fs/cgroup/robot_pipeline/cpu.cfs_period_us
echo "20000"  > /sys/fs/cgroup/robot_pipeline/cpu.cfs_quota_us

echo "OS wear reduction configured ✓"
"""


def write_os_setup_script(dest: Path = Path("/tmp/os_wear_setup.sh")):
    dest.write_text(OS_WEAR_SETUP)
    dest.chmod(0o755)
    logger.info("OS wear setup script written to %s", dest)
    return dest
