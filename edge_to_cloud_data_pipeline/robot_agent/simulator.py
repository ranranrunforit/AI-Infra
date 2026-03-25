"""
simulator.py — Robot Sensor Data Simulator (v5 — Rich Binary Payloads)

Generates four categories of data every tick:

  1. Telemetry dict  — scalars for anomaly detector (unchanged from v4)
  2. LiDAR scan      — numpy float32 array (N×4: x,y,z,intensity)
                       Simulates a 2D spinning lidar at 360 rays, extruded to
                       8 vertical beams for a pseudo-3D scan (~23 KB/frame)
  3. Camera frame    — raw RGB bytes at 640×480 resolution (~900 KB/frame
                       before codec, ~18 KB after H.265 — demonstrates 50:1)
  4. Log records     — structured dicts with severity, node, msg, and context
                       Written to the MCAP /rosout topic as JSON payloads

Binary payloads are held separately from the telemetry dict so the anomaly
detector keeps receiving clean scalars while the MCAP writer can route each
payload type to its own channel with the correct encoding (raw / h265 / json).
"""

import os
import time
import math
import random
import logging
import threading
import struct
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger("simulator")

# ── Numpy (optional — degrades gracefully to list-based point clouds) ─────────
try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False
    logger.warning("numpy not available — LiDAR point clouds will be list-based")

# ── Pillow (optional — degrades gracefully to raw noise frames) ───────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available — camera frames will be raw noise")


# ── Thermal simulation + CPU temp sharing ────────────────────────────────────

CPU_TEMP_MOCK_FILE = "/tmp/robot_cpu_temp"


def _write_cpu_temp(temp_c: float) -> None:
    """
    Write the current simulated CPU temperature to a file
    so the BandwidthLimiter (which may run in a separate process)
    can read it via _read_cpu_temp().
    """
    try:
        with open(CPU_TEMP_MOCK_FILE, "w") as f:
            f.write(f"{temp_c:.2f}")
    except OSError:
        pass  # non-fatal: limiter will use sysfs or fallback


class ThermalSimulator:
    """
    First-order thermal model for a Jetson-class SoC.

    Physics:
        dT/dt = (P_dissipated - P_radiated) / thermal_mass

    Where:
        P_dissipated = idle_power + (max_power - idle_power) * cpu_usage
        P_radiated   = (T - T_ambient) / thermal_resistance
        thermal_mass = SoC + heatsink effective heat capacity

    Realistic parameters (Jetson Xavier NX, 15W sustained mode):
        T_ambient        = 25°C
        T_idle           ≈ 35°C (idle_power only)
        T_max_sustained  ≈ 72°C (100% CPU, no throttle)
        T_throttle       = 85°C (kernel starts reducing clocks)
        T_shutdown       = 95°C (emergency halt)

    Time constant (how fast it heats/cools):
        tau ≈ 90s (typical embedded SoC with small heatsink)

    The model is stepped at the simulator's tick rate (default 10 Hz)
    so it accumulates heat realistically during burst tests.
    """

    def __init__(
        self,
        t_ambient:          float = 25.0,   # °C
        t_idle:             float = 35.0,   # °C at 0% load
        t_max_sustained:    float = 72.0,   # °C at 100% load
        t_throttle:         float = 85.0,   # °C kernel throttle threshold
        t_shutdown:         float = 95.0,   # °C emergency shutdown threshold
        tau_sec:            float = 90.0,   # thermal time constant (seconds)
        dt_sec:             float = 0.1,    # integration step = simulator tick
        noise_std:          float = 0.3,    # °C sensor noise
    ):
        self._T_amb       = t_ambient
        self._T_idle      = t_idle
        self._T_max       = t_max_sustained
        self._T_throttle  = t_throttle
        self._T_shutdown  = t_shutdown
        self._tau         = tau_sec
        self._dt          = dt_sec
        self._noise_std   = noise_std

        # Derived: thermal resistance and capacitance from tau and power
        self._idle_power  = 3.0    # W (Jetson idle)
        self._max_power   = 15.0   # W (Jetson 15W mode)

        # thermal resistance: R = (T_idle - T_ambient) / P_idle
        self._R_thermal   = (t_idle - t_ambient) / self._idle_power

        # thermal mass: C = R * tau   →  tau = R * C
        self._C_thermal   = tau_sec / self._R_thermal

        # State
        self._temp        = t_idle   # start at idle temperature
        self._throttle_active = False

    def tick(self, cpu_usage_fraction: float) -> float:
        """
        Advance thermal model by one dt_sec step.

        Args:
            cpu_usage_fraction: 0.0 (idle) to 1.0 (100% CPU)

        Returns:
            Current CPU temperature in °C with sensor noise.

        The returned value includes realistic noise so it looks like
        a real thermal sensor reading. The internal state self._temp
        is the true (noiseless) temperature used for physics.
        """
        # Clamp input
        u = max(0.0, min(1.0, cpu_usage_fraction))

        # Apply throttle: if above throttle threshold, reduce effective load
        if self._temp >= self._T_throttle:
            u = min(u, 0.4)   # kernel forces clocks down → effective load halves
            self._throttle_active = True
        elif self._temp < self._T_throttle - 5:
            self._throttle_active = False

        # Power dissipated at this load
        p_in = self._idle_power + (self._max_power - self._idle_power) * u

        # Power radiated to ambient (Newton's law of cooling)
        p_out = (self._temp - self._T_amb) / self._R_thermal

        # dT = (p_in - p_out) / C * dt
        dT = ((p_in - p_out) / self._C_thermal) * self._dt
        self._temp = max(self._T_amb, self._temp + dT)

        # Emergency shutdown guard (simulation only)
        if self._temp >= self._T_shutdown:
            self._temp = self._T_shutdown - 0.5   # clamp, don't exceed

        # Return temperature + sensor noise
        return round(self._temp + random.gauss(0, self._noise_std), 2)

    @property
    def true_temp(self) -> float:
        """Noiseless internal temperature for testing."""
        return self._temp

    @property
    def throttle_active(self) -> bool:
        """True when kernel thermal throttle is engaged."""
        return self._throttle_active

    def reset(self, temp: float = None) -> None:
        """Reset to idle temperature (or specified value). Useful for tests."""
        self._temp = temp if temp is not None else self._T_idle
        self._throttle_active = False



# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RichSensorFrame:
    """One complete sensor snapshot at a single timestamp."""
    ts_ms: int
    # Scalar telemetry (for anomaly detector + MCAP /telemetry channel)
    telemetry: Dict[str, Any]
    # Binary payloads (routed to separate MCAP channels)
    lidar_bytes: bytes          # float32 array: (N, 4) x,y,z,intensity
    lidar_point_count: int
    camera_rgb_bytes: bytes     # raw RGB24 bytes, width×height×3
    camera_width: int
    camera_height: int
    log_records: List[Dict]     # structured log entries for /rosout


# ── LiDAR point cloud generator ───────────────────────────────────────────────

class LiDARGenerator:
    """
    Simulates a spinning 2D LiDAR extruded into pseudo-3D.

    Physical model:
      - 360 horizontal rays at 1° resolution
      - 8 vertical beams at -15° to +15°
      - Each ray returns distance, x, y, z, intensity
      - Obstacles are modelled as rectangular walls in the environment map
      - Simulated robot position updates each tick

    The resulting point cloud:
      - Normal operation: ~2880 points, ~46 KB as float32
      - Obstacle event:   fewer returns at short range
    """

    # Fixed obstacle walls: (x_min, x_max, y_min, y_max, reflectivity)
    WALLS = [
        (-5.0,  5.0, -8.0, -7.9, 0.8),   # south wall
        (-5.0,  5.0,  7.9,  8.0, 0.8),   # north wall
        (-8.0, -7.9, -5.0,  5.0, 0.7),   # west wall
        ( 7.9,  8.0, -5.0,  5.0, 0.7),   # east wall
        (-2.0, -1.0,  1.0,  3.0, 0.6),   # interior box A
        ( 1.0,  2.5, -2.0, -1.0, 0.5),   # interior box B
    ]

    V_ANGLES_DEG = [-15, -8, -3, 0, 3, 8, 12, 15]   # 8 vertical beams

    def __init__(self, max_range: float = 12.0, noise_std: float = 0.02):
        self.max_range = max_range
        self.noise_std = noise_std
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0   # radians

    def update_pose(self, displacement_m: float, dt: float = 0.1):
        """Advance robot pose for realistic scan motion."""
        self._robot_yaw += random.gauss(0, 0.01)   # slight yaw drift
        self._robot_x += displacement_m * math.cos(self._robot_yaw) * dt
        self._robot_y += displacement_m * math.sin(self._robot_yaw) * dt

    def _ray_distance(self, angle_rad: float) -> Tuple[float, float]:
        """Cast one horizontal ray, return (distance, reflectivity)."""
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        best_dist = self.max_range
        best_refl = 0.1  # background ambient

        for x_min, x_max, y_min, y_max, refl in self.WALLS:
            # Ray-AABB intersection
            t_min, t_max = -1e9, 1e9

            if abs(dx) > 1e-9:
                t1 = (x_min - self._robot_x) / dx
                t2 = (x_max - self._robot_x) / dx
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
            elif not (x_min <= self._robot_x <= x_max):
                continue

            if abs(dy) > 1e-9:
                t1 = (y_min - self._robot_y) / dy
                t2 = (y_max - self._robot_y) / dy
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
            elif not (y_min <= self._robot_y <= y_max):
                continue

            if t_min <= t_max and 0 < t_min < best_dist:
                best_dist = t_min
                best_refl = refl

        return best_dist, best_refl

    def generate(self) -> bytes:
        """
        Generate one full scan. Returns float32 bytes (N×4: x, y, z, intensity).
        Falls back to a minimal 4-point scan if numpy is unavailable.
        """
        points = []
        for h_deg in range(0, 360, 1):
            h_rad = math.radians(h_deg + self._robot_yaw)
            dist, refl = self._ray_distance(h_rad)

            for v_deg in self.V_ANGLES_DEG:
                v_rad = math.radians(v_deg)
                # Add sensor noise
                noisy_dist = dist + random.gauss(0, self.noise_std)
                noisy_dist = max(0.1, min(noisy_dist, self.max_range))

                # Project into 3D
                horiz = noisy_dist * math.cos(v_rad)
                x = horiz * math.cos(h_rad) + self._robot_x
                y = horiz * math.sin(h_rad) + self._robot_y
                z = noisy_dist * math.sin(v_rad)
                intensity = refl * random.uniform(0.85, 1.0)

                points.append((x, y, z, intensity))

        if NP_AVAILABLE:
            arr = np.array(points, dtype=np.float32)
            return arr.tobytes()
        else:
            # Fallback: pack manually as float32 big-endian
            buf = b""
            for x, y, z, i in points:
                buf += struct.pack(">ffff", x, y, z, i)
            return buf

    def point_count(self) -> int:
        return 360 * len(self.V_ANGLES_DEG)


# ── Camera frame generator ────────────────────────────────────────────────────

class CameraGenerator:
    """
    Generates synthetic RGB frames that look plausible from a robot perspective.

    Uses PIL when available for a richer scene (gradient floor, walls, overlaid
    status text). Falls back to structured random noise that still benefits from
    H.265 compression (temporal coherence across ticks).

    Resolution: 640×480 — realistic for a small service robot (not 4K).
    640×480×3 = 921,600 bytes raw → ~18KB after H.265 (50:1 ratio).
    """

    WIDTH  = 640
    HEIGHT = 480

    def __init__(self):
        self._frame_num = 0
        self._last_frame: Optional[bytes] = None   # for temporal coherence

    def generate(self, telemetry: Dict[str, Any]) -> bytes:
        """Returns raw RGB bytes, WIDTH×HEIGHT×3."""
        self._frame_num += 1

        if PIL_AVAILABLE:
            return self._pil_frame(telemetry)
        else:
            return self._noise_frame()

    def _pil_frame(self, tel: Dict) -> bytes:
        """Pillow-rendered scene: floor gradient + walls + HUD overlay."""
        img = Image.new("RGB", (self.WIDTH, self.HEIGHT))
        draw = ImageDraw.Draw(img)

        battery = tel.get("battery_pct", 100)
        obstacle = tel.get("obstacle_dist_m", 3.0)
        cpu_temp = tel.get("cpu_temp_C", 50)

        # Sky gradient (top third)
        sky_r = int(100 + battery * 0.5)
        for y in range(self.HEIGHT // 3):
            shade = int(180 - y * 0.5)
            draw.line([(0, y), (self.WIDTH, y)],
                      fill=(shade, shade + 10, sky_r))

        # Floor (bottom two thirds) — perspective lines
        floor_g = int(60 + obstacle * 5)
        for y in range(self.HEIGHT // 3, self.HEIGHT):
            brightness = int(40 + (y - self.HEIGHT // 3) * 0.4)
            draw.line([(0, y), (self.WIDTH, y)],
                      fill=(brightness, min(255, brightness + floor_g), brightness))

        # Simulated obstacle box when close
        if obstacle < 2.0:
            box_w = int(200 * (2.0 - obstacle) / 2.0)
            box_h = int(150 * (2.0 - obstacle) / 2.0)
            cx, cy = self.WIDTH // 2, self.HEIGHT // 2
            draw.rectangle(
                [cx - box_w, cy - box_h, cx + box_w, cy + box_h],
                fill=(200, 80, 60), outline=(255, 120, 80), width=3
            )

        # HUD text overlay
        ts = time.strftime("%H:%M:%S")
        hud_lines = [
            f"BAT: {battery:.0f}%   TEMP: {cpu_temp:.0f}C",
            f"DIST: {obstacle:.2f}m   t={ts}",
            f"FRAME #{self._frame_num}",
        ]
        y_pos = 8
        for line in hud_lines:
            draw.text((8, y_pos), line, fill=(220, 220, 60))
            y_pos += 16

        # Add per-frame noise patch (prevents encoding trivially from removing motion)
        noise_x = random.randint(0, self.WIDTH - 32)
        noise_y = random.randint(0, self.HEIGHT - 32)
        for dy in range(32):
            for dx in range(32):
                r = random.randint(0, 10)
                px = img.getpixel((noise_x + dx, noise_y + dy))
                img.putpixel((noise_x + dx, noise_y + dy),
                             (min(255, px[0] + r), min(255, px[1] + r), min(255, px[2] + r)))

        return img.tobytes()  # raw RGB24

    def _noise_frame(self) -> bytes:
        """
        Structured noise with temporal coherence — H.265 still compresses well
        because consecutive frames share most pixel values.
        """
        if self._last_frame is None:
            # First frame: pure structured noise
            frame = bytearray(self.WIDTH * self.HEIGHT * 3)
            for i in range(0, len(frame), 3):
                v = random.randint(60, 180)
                frame[i]     = v                       # R
                frame[i + 1] = min(255, v + 20)       # G (floor tint)
                frame[i + 2] = max(0, v - 20)         # B
            self._last_frame = bytes(frame)

        # Evolve 5% of pixels each frame (temporal coherence for codec)
        frame = bytearray(self._last_frame)
        n_changes = (self.WIDTH * self.HEIGHT * 3) // 20
        for _ in range(n_changes):
            i = random.randrange(0, len(frame) - 2)
            delta = random.randint(-8, 8)
            frame[i] = max(0, min(255, frame[i] + delta))

        self._last_frame = bytes(frame)
        return self._last_frame


# ── Structured log generator ──────────────────────────────────────────────────

class LogGenerator:
    """
    Generates realistic ROS2-style log records as structured dicts.

    Replaces the single `has_fatal_log: bool` with full log records that
    can be written to the MCAP /rosout topic as JSON payloads.

    Each record matches the ROS2 Log message format:
      { level, name, msg, file, function, line, stamp_ns }
    """

    LEVEL_DEBUG = 10
    LEVEL_INFO  = 20
    LEVEL_WARN  = 30
    LEVEL_ERROR = 40
    LEVEL_FATAL = 50

    # Normal-operation log templates
    NORMAL_LOGS = [
        (LEVEL_INFO,  "navigation",    "Waypoint {wp} reached in {t:.1f}s"),
        (LEVEL_INFO,  "slam",          "Map updated: {pts} keypoints, confidence={conf:.2f}"),
        (LEVEL_INFO,  "battery_mgr",   "Battery at {pct:.0f}%, drain rate {rate:.3f}%/s"),
        (LEVEL_DEBUG, "lidar_driver",  "Scan complete: {n} points in {ms:.1f}ms"),
        (LEVEL_DEBUG, "camera_node",   "Frame {f} encoded: {kb:.0f}KB ({ratio:.0f}:1 ratio)"),
        (LEVEL_INFO,  "upload_agent",  "Uploaded {fname}, {kb:.0f}KB, P{pri}"),
        (LEVEL_DEBUG, "ring_buffer",   "Rotated: {n} frames → {path}"),
        (LEVEL_INFO,  "health_check",  "All systems nominal. CPU={cpu:.0f}% MEM={mem:.0f}MB"),
    ]

    # Anomaly-correlated log templates
    WARN_LOGS = [
        (LEVEL_WARN,  "motor_ctrl",    "Motor current elevated: {amps:.1f}A (threshold {thresh:.1f}A)"),
        (LEVEL_WARN,  "battery_mgr",   "Battery draining faster than expected: {rate:.3f}%/s"),
        (LEVEL_WARN,  "navigation",    "Obstacle at {dist:.2f}m — replanning route"),
        (LEVEL_WARN,  "thermal",       "CPU temp {temp:.0f}°C — throttling to {pct:.0f}% capacity"),
    ]

    ERROR_LOGS = [
        (LEVEL_ERROR, "slam",          "Localization lost — reinitializing from last known pose"),
        (LEVEL_ERROR, "upload_agent",  "GCS upload failed after {retries} retries: {reason}"),
        (LEVEL_ERROR, "motor_ctrl",    "Motor overcurrent detected: {amps:.1f}A — emergency stop"),
        (LEVEL_FATAL, "watchdog",      "Process {proc} unresponsive for {t:.0f}s — restarting"),
    ]

    def __init__(self, robot_id: str = "robot-demo"):
        self.robot_id = robot_id
        self._seq     = 0
        self._wp      = 1
        self._frame   = 0

    def generate(self, telemetry: Dict[str, Any], anomaly_active: bool = False) -> List[Dict]:
        """Generate 1-4 log records appropriate to the current state."""
        records = []
        ts_ns = int(time.time() * 1e9)

        # Always emit 1-2 normal/debug logs
        n_normal = random.randint(1, 2)
        for _ in range(n_normal):
            level, node, tmpl = random.choice(self.NORMAL_LOGS)
            self._frame += 1
            self._seq   += 1
            msg = self._render(tmpl, telemetry)
            records.append(self._record(level, node, msg, ts_ns + self._seq))

        # Conditionally emit warnings based on telemetry thresholds
        if telemetry.get("motor_current_A", 0) > 8.0:
            level, node, tmpl = random.choice(self.WARN_LOGS)
            records.append(self._record(level, node,
                self._render(tmpl, telemetry), ts_ns + 1000))

        if telemetry.get("battery_pct", 100) < 25:
            records.append(self._record(
                self.LEVEL_WARN, "battery_mgr",
                f"Battery low: {telemetry['battery_pct']:.0f}% — returning to dock",
                ts_ns + 2000))

        # Emit error/fatal if anomaly is active
        if anomaly_active:
            level, node, tmpl = random.choice(self.ERROR_LOGS)
            msg = self._render(tmpl, telemetry)
            records.append(self._record(level, node, msg, ts_ns + 3000))

        return records

    def _render(self, template: str, tel: Dict) -> str:
        """Fill template with realistic values."""
        try:
            return template.format(
                wp=self._wp,
                t=random.uniform(1.5, 8.0),
                pts=random.randint(800, 2400),
                conf=random.uniform(0.85, 0.99),
                pct=tel.get("battery_pct", 80),
                rate=tel.get("battery_drain_rate", 0.01),
                n=2880,
                ms=random.uniform(8, 25),
                f=self._frame,
                kb=random.uniform(15, 22),
                ratio=random.uniform(40, 55),
                fname=f"mcap_{self._seq:05d}.mcap",
                pri=random.randint(1, 3),
                path=f"/tmp/robot_data/staging/robot-demo/p2_normal/",
                cpu=tel.get("cpu_usage_pct", 20),
                mem=random.uniform(150, 280),
                amps=tel.get("motor_current_A", 3.0),
                thresh=10.0,
                dist=tel.get("obstacle_dist_m", 2.0),
                temp=tel.get("cpu_temp_C", 50),
                retries=random.randint(2, 5),
                reason=random.choice(["503 Service Unavailable", "408 Timeout", "ConnectionReset"]),
                proc=random.choice(["slam_node", "motor_ctrl", "upload_agent"]),
            )
        except KeyError:
            return template  # partial render is fine

    def _record(self, level: int, node: str, msg: str, ts_ns: int) -> Dict:
        return {
            "stamp_ns":  ts_ns,
            "level":     level,
            "name":      f"/robot/{self.robot_id}/{node}",
            "msg":       msg,
            "file":      f"{node}.py",
            "function":  "run",
            "line":      random.randint(100, 800),
            "robot_id":  self.robot_id,
        }


# ── Main SensorSimulator class ────────────────────────────────────────────────

class SensorSimulator:
    """
    Full sensor simulator producing both scalar telemetry and binary payloads.

    Call `read()` each tick to get a RichSensorFrame containing:
      - telemetry dict  (scalars, same shape as v4 — anomaly detector unchanged)
      - lidar_bytes     (float32 point cloud, ~46 KB per frame)
      - camera_rgb_bytes (raw RGB24, 640×480, ~900 KB before compression)
      - log_records     (1-4 structured log dicts)
    """

    def __init__(self, robot_id: str = "robot-demo", hz: float = 10.0):
        self.robot_id   = robot_id
        self.hz         = hz
        self._interval  = 1.0 / hz
        self._t         = 0.0
        self._seq       = 0

        # Scalar state (same as v4)
        self._battery_pct    = 85.0
        self._motor_temp_C   = 30.0
        self._cpu_temp_C     = 45.0
        self._restart_count  = 0
        self._slam_localized = True
        self._wifi_rssi      = -45.0
        self._at_dock        = False

        # Anomaly injection flags
        self._overcurrent_active = False
        self._collision_event    = False
        self._fast_drain         = False

        # Scheduled anomaly queue: [(fire_at_time, anomaly_type), ...]
        self._anomaly_queue: List[Tuple[float, str]] = []

        # Rich payload generators
        self._lidar   = LiDARGenerator()
        self._camera  = CameraGenerator()
        self._logger  = LogGenerator(robot_id)

        self._thermal = ThermalSimulator()

    # ── Anomaly scheduling ────────────────────────────────────────────────────

    def inject_anomaly(self, anomaly_type: str, delay_secs: float = 0.0):
        fire_at = time.time() + delay_secs
        self._anomaly_queue.append((fire_at, anomaly_type))
        self._anomaly_queue.sort(key=lambda x: x[0])
        logger.info("[SIM] Anomaly scheduled: %s in %.0fs", anomaly_type, delay_secs)

    def _apply_scheduled_anomalies(self):
        now = time.time()
        while self._anomaly_queue and self._anomaly_queue[0][0] <= now:
            _, atype = self._anomaly_queue.pop(0)
            self._trigger_anomaly(atype)

    def _trigger_anomaly(self, atype: str):
        logger.warning("[SIM] INJECTING ANOMALY: %s", atype)
        if atype == "battery_critical":
            self._battery_pct = 10.0
        elif atype == "motor_overcurrent":
            self._overcurrent_active = True
        elif atype == "collision":
            self._collision_event = True
        elif atype == "high_temp":
            self._motor_temp_C = 82.0
        elif atype == "nav_stuck":
            self._displacement_override = 0.0
        elif atype == "fast_battery_drain":
            self._fast_drain = True
        elif atype == "slam_lost":
            self._slam_localized = False

    # ── Scalar telemetry (identical structure to v4) ──────────────────────────

    def _scalar_telemetry(self) -> Dict[str, Any]:
        """Compute scalar telemetry dict — anomaly detector reads this."""
        self._t       += self._interval
        self._seq     += 1
        t              = self._t

        self._apply_scheduled_anomalies()

        # Battery: dock recharge when critically low
        if self._at_dock:
            self._battery_pct = min(80.0, self._battery_pct + 0.3)
            if self._battery_pct >= 80.0:
                self._at_dock = False
                self._fast_drain = False
        else:
            drain_rate = 0.15 if self._fast_drain else 0.01
            self._battery_pct = max(0.0, self._battery_pct - drain_rate)
            if self._battery_pct < 5.0:
                self._at_dock = True
                logger.info("[SIM] Battery critical — returning to dock")

        # Motor temperature
        load = 0.5 + 0.3 * math.sin(t / 30)
        self._motor_temp_C = min(90, self._motor_temp_C + load * 0.02 - 0.01)

        # CPU temperature — first-order thermal model, result written to
        # /tmp/robot_cpu_temp so BandwidthLimiter can read it for MMCF cost.
        cpu_usage_pct_val = round(20 + load * 30 + random.gauss(0, 2), 1)
        self._cpu_temp_C = self._thermal.tick(cpu_usage_pct_val / 100.0)
        _write_cpu_temp(self._cpu_temp_C)

        # Motor current
        base_current = 3.0 + load * 4.0
        motor_current = base_current + random.gauss(0, 0.3)
        motor_torque  = 2.0 + load * 3.0 + random.gauss(0, 0.2)
        if self._overcurrent_active:
            motor_current = 16.0 + random.gauss(0, 0.5)
            if random.random() < 0.1:
                self._overcurrent_active = False

        # IMU
        accel_x = random.gauss(0, 0.5)
        accel_y = random.gauss(0, 0.5)
        accel_z = 9.81 + random.gauss(0, 0.1)
        linear_accel = math.sqrt(accel_x ** 2 + accel_y ** 2)
        if self._collision_event:
            linear_accel = 12.0
            accel_x = 9.0
            self._collision_event = False

        # Navigation
        displacement = max(0, 0.3 + 0.3 * math.sin(t / 5) + random.gauss(0, 0.05))
        if hasattr(self, "_displacement_override"):
            displacement = self._displacement_override

        obstacle_dist = max(0.01, 2.0 + random.gauss(0, 0.5))
        if random.random() < 0.001:
            obstacle_dist = 0.05

        # WiFi
        self._wifi_rssi = -45 + random.gauss(0, 5) + 20 * math.sin(t / 300)

        # SLAM recovery
        if not self._slam_localized and random.random() < 0.05:
            self._slam_localized = True

        battery_drain_rate = (0.15 if self._fast_drain else 0.01) * self.hz

        return {
            "battery_pct":         round(self._battery_pct, 2),
            "battery_voltage_V":   round(3.7 + self._battery_pct / 100 * 0.5, 3),
            "battery_drain_rate":  round(battery_drain_rate, 4),
            "motor_current_A":     round(motor_current, 2),
            "motor_torque_Nm":     round(motor_torque, 2),
            "motor_temp_C":        round(self._motor_temp_C, 1),
            "motor_threshold_A":   10.0,
            "linear_accel_mps2":   round(linear_accel, 3),
            "accel_x":             round(accel_x, 3),
            "accel_y":             round(accel_y, 3),
            "accel_z":             round(accel_z, 3),
            "displacement_m":      round(displacement, 3),
            "obstacle_dist_m":     round(obstacle_dist, 3),
            "cmd_vel_nonzero":     displacement > 0.01,
            "nav_stuck":           displacement < 0.05 and displacement > 0.001,
            "slam_localized":      self._slam_localized,
            "cpu_temp_C":          round(self._cpu_temp_C, 1),
            "cpu_usage_pct":       cpu_usage_pct_val,
            "wifi_rssi":           round(self._wifi_rssi, 1),
            "restart_count":       self._restart_count,
            "has_fatal_log":       False,   # kept for YAML rule compat
            "ts_ms":               int(time.time() * 1000),
            "at_dock":             self._at_dock,
        }

    # ── Main read() method ────────────────────────────────────────────────────

    def read(self) -> RichSensorFrame:
        """
        Generate one complete sensor snapshot.

        Returns a RichSensorFrame with all four data types. The telemetry dict
        is identical in structure to v4 so the anomaly detector requires no changes.

        Binary payloads are large but generated efficiently:
          LiDAR:  ~46 KB  (2880 float32 points × 4 channels × 4 bytes)
          Camera: ~900 KB (640×480×3 bytes — compresses to ~18 KB with H.265)
        """
        tel = self._scalar_telemetry()

        # Update LiDAR pose to match robot movement
        self._lidar.update_pose(tel["displacement_m"], self._interval)

        # Determine if an anomaly is active (for log severity selection)
        anomaly_active = (
            tel["battery_pct"] < 15
            or tel["motor_current_A"] > 12
            or tel["linear_accel_mps2"] > 8
            or not tel["slam_localized"]
        )

        # Generate binary payloads
        lidar_bytes  = self._lidar.generate()
        camera_bytes = self._camera.generate(tel)
        log_records  = self._logger.generate(tel, anomaly_active)

        return RichSensorFrame(
            ts_ms             = tel["ts_ms"],
            telemetry         = tel,
            lidar_bytes       = lidar_bytes,
            lidar_point_count = self._lidar.point_count(),
            camera_rgb_bytes  = camera_bytes,
            camera_width      = CameraGenerator.WIDTH,
            camera_height     = CameraGenerator.HEIGHT,
            log_records       = log_records,
        )


# ── Demo runner ───────────────────────────────────────────────────────────────

def run_demo_simulation(pipeline, cfg: dict, duration_secs: float = 0.0):
    """
    Run the full simulation, pushing all four data types into the pipeline.

    The pipeline's ingest() receives the telemetry dict (for anomaly detection
    + scalar MCAP channel) plus the binary payloads via separate ingest_binary()
    calls for the lidar, camera, and rosout channels.
    """
    hz       = cfg.get("simulation", {}).get("sensor_hz", 10)
    robot_id = cfg["robot"]["id"]
    interval = 1.0 / hz

    sim = SensorSimulator(robot_id=robot_id, hz=hz)

    # Schedule demo anomaly sequence
    for delay, atype in [
        (30,  "fast_battery_drain"),
        (60,  "high_temp"),
        (90,  "motor_overcurrent"),
        (120, "collision"),
        (150, "nav_stuck"),
        (180, "slam_lost"),
        (210, "battery_critical"),
    ]:
        sim.inject_anomaly(atype, delay_secs=delay)

    start       = time.time()
    frame_count = 0

    logger.info("[SIM] Simulation started @ %.0fHz (LiDAR+Camera+Logs enabled)", hz)
    logger.info("[SIM] LiDAR: ~%d pts/frame (~%d KB raw)",
                sim._lidar.point_count(),
                sim._lidar.point_count() * 16 // 1024)
    logger.info("[SIM] Camera: %dx%d RGB (~%d KB raw, ~18KB after H.265)",
                CameraGenerator.WIDTH, CameraGenerator.HEIGHT,
                CameraGenerator.WIDTH * CameraGenerator.HEIGHT * 3 // 1024)

    while True:
        loop_start = time.time()
        frame      = sim.read()
        frame_count += 1

        # Push scalar telemetry to pipeline (anomaly detector + /telemetry channel)
        pipeline.ingest("telemetry", frame.telemetry)

        # Push binary payloads to separate MCAP channels
        if hasattr(pipeline, "ingest_binary"):
            pipeline.ingest_binary("lidar/points", frame.ts_ms, frame.lidar_bytes,
                                   encoding="application/lidar-float32",
                                   meta={"point_count": frame.lidar_point_count})
            pipeline.ingest_binary("camera/rgb", frame.ts_ms, frame.camera_rgb_bytes,
                                   encoding="video/h265",
                                   meta={"width": frame.camera_width,
                                         "height": frame.camera_height})
            for log in frame.log_records:
                import json
                pipeline.ingest_binary("rosout", frame.ts_ms,
                                       json.dumps(log).encode(),
                                       encoding="application/json")

        # Progress logging every 10 seconds
        if frame_count % (hz * 10) == 0:
            elapsed = time.time() - start
            stats   = pipeline.manifest.stats() if pipeline.manifest else {}
            bw_s    = pipeline.bw.stats()       if pipeline.bw       else {}
            ev_s    = pipeline.evictor.stats()  if pipeline.evictor  else {}

            lidar_kb  = frame.lidar_point_count * 16 / 1024
            cam_kb    = len(frame.camera_rgb_bytes) / 1024
            logger.info(
                "[SIM] t=%.0fs | frames=%d | lidar=%.0fKB | cam=%.0fKB "
                "| db=%s | disk=%.1f%% | bw=%.2f",
                elapsed, frame_count, lidar_kb, cam_kb,
                stats.get("counts_by_state", {}),
                ev_s.get("disk_usage_pct", 0),
                bw_s.get("token_level", 1),
            )

        if duration_secs and (time.time() - start) > duration_secs:
            logger.info("[SIM] Complete after %.0fs (%d frames)", time.time() - start, frame_count)
            break

        elapsed_loop = time.time() - loop_start
        sleep_time   = max(0, interval - elapsed_loop)
        time.sleep(sleep_time)