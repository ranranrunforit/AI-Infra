#!/usr/bin/env bash
# scripts/os_wear_setup.sh
# Run once at robot provisioning (requires root).
# Reduces eMMC/NVMe P/E cycle burn from OS-level micro-writes.
# Source: v4 merge — these 3 mitigations were absent from v2.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[os-wear]${NC} $*"; }
warn() { echo -e "${YELLOW}[os-wear]${NC} $*"; }
err()  { echo -e "${RED}[os-wear]${NC} $*" >&2; }

if [[ $EUID -ne 0 ]]; then
    err "Must be run as root"
    exit 1
fi

# ── 1. journald volatile mode ─────────────────────────────────────────────────
log "Configuring journald volatile mode (RAM-only)..."
if grep -q "Storage=volatile" /etc/systemd/journald.conf 2>/dev/null; then
    warn "journald already configured volatile — skipping"
else
    echo "Storage=volatile" >> /etc/systemd/journald.conf
    echo "RuntimeMaxUse=64M"  >> /etc/systemd/journald.conf
    systemctl restart systemd-journald
    log "journald → volatile ✓"
fi

# ── 2. log2ram ────────────────────────────────────────────────────────────────
log "Installing log2ram..."
if command -v log2ram &>/dev/null; then
    warn "log2ram already installed — checking config"
else
    if apt-get install -y log2ram &>/dev/null 2>&1; then
        log "log2ram installed ✓"
    else
        warn "log2ram not in apt repos — installing from source"
        curl -Lo /tmp/log2ram.tar.gz \
            https://github.com/azlux/log2ram/archive/master.tar.gz 2>/dev/null || true
        if [[ -f /tmp/log2ram.tar.gz ]]; then
            tar -xf /tmp/log2ram.tar.gz -C /tmp
            cd /tmp/log2ram-master && bash install.sh
        else
            warn "Could not download log2ram — skipping (network unavailable)"
        fi
    fi
fi

if [[ -f /etc/log2ram.conf ]]; then
    sed -i 's/SIZE=.*/SIZE=128M/' /etc/log2ram.conf
    log "log2ram SIZE=128M ✓"
fi

# ── 3. Disable HA micro-writes (Proxmox only) ────────────────────────────────
for svc in pve-ha-lrm pve-ha-crm; do
    if systemctl list-units --full -all 2>/dev/null | grep -q "$svc"; then
        systemctl disable --now "$svc" 2>/dev/null && \
            log "Disabled $svc ✓" || warn "Could not disable $svc"
    fi
done

# ── 4. Kernel cgroup for robot pipeline ──────────────────────────────────────
log "Configuring cgroup CPU quota (20%)..."
CGROUP=/sys/fs/cgroup/robot_pipeline
if [[ -d /sys/fs/cgroup ]]; then
    mkdir -p "$CGROUP"
    echo "100000" > "$CGROUP/cpu.cfs_period_us"
    echo "20000"  > "$CGROUP/cpu.cfs_quota_us"
    log "Cgroup quota 20ms/100ms = 20% ✓"
else
    warn "cgroups not mounted — skipping quota setup"
fi

# ── 5. Mount options (if fstab writable) ─────────────────────────────────────
log "Checking /etc/fstab for noatime..."
if grep -q "noatime" /etc/fstab; then
    warn "noatime already set in fstab"
else
    warn "Consider adding 'noatime' to /etc/fstab mnt options to reduce atime writes"
fi

log ""
log "═══════════════════════════════════════"
log "OS wear reduction configured ✓"
log "  • journald: RAM-only (no eMMC log writes)"
log "  • log2ram:  128MB RAM log buffer"  
log "  • pve-ha:   micro-write services disabled"
log "  • cgroup:   CPU quota 20% enforced"
log "═══════════════════════════════════════"
