"""
ThermalSimulator — addition to simulator.py

Paste this class into your existing simulator.py, then:

  1. In SensorSimulator.__init__:
        self._thermal = ThermalSimulator()

  2. In SensorSimulator._scalar_telemetry, replace:
        self._cpu_temp_C = 45 + 15 * load + random.gauss(0, 1)
     with:
        self._cpu_temp_C = self._thermal.tick(cpu_usage_fraction)
        _write_cpu_temp(self._cpu_temp_C)   # shared with bandwidth_limiter

The thermal model gates on cpu_usage_fraction (0.0–1.0).
At sustained 80% load the CPU approaches ~72°C after ~2 minutes,
triggering MMCF backoff for P3 (cost ≈ 72 + 10*3 = 102 > 100).
"""

import os
import math
import random

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


# ── Integration snippet for SensorSimulator ───────────────────────────────────
#
# In SensorSimulator.__init__ add:
#
#     self._thermal = ThermalSimulator()
#
# In SensorSimulator._scalar_telemetry replace the temperature line with:
#
#     load_fraction = (cpu_usage_pct / 100.0)   # or however you compute load
#     self._cpu_temp_C = self._thermal.tick(load_fraction)
#     _write_cpu_temp(self._cpu_temp_C)          # make visible to BandwidthLimiter
#
# The bandwidth limiter will read /tmp/robot_cpu_temp before each MMCF decision,
# so high CPU load during a burst test will naturally raise upload costs for P3.
