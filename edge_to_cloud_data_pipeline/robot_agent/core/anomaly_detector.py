"""
anomaly_detector.py — 3-Layer Detection (v4)
Layer 1: Deterministic YAML rules (zero-latency, per-rule capture windows)
Layer 2: Z-Score rolling window (univariate — battery slope, motor temp)
Layer 3: Isolation Forest (multivariate — torque + current + IMU)

Score formula (v2, kept):
  score = type_weight × anomaly_bonus × recency_decay × size_penalty
  → 0–100 continuous, maps to P-bucket via thresholds
"""

import time
import math
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple

import yaml

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("sklearn/numpy not available — ML layer disabled")

try:
    import statistics as stats_mod
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

try:
    from simpleeval import EvalWithCompoundTypes
    SIMPLEEVAL_AVAILABLE = True
except ImportError:
    SIMPLEEVAL_AVAILABLE = False
    logger.warning("simpleeval not installed — falling back to lambda-only rule compilation")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    rule_id: str
    layer: str            # "yaml_rule" | "z_score" | "isolation_forest"
    score: float          # 0–100
    priority: int         # promoted priority (0–3)
    ts_ms: int
    flags: List[str]
    capture_pre_sec: float = 10.0
    capture_post_sec: float = 5.0
    data: Dict = field(default_factory=dict)


@dataclass
class TriggerRule:
    id: str
    promote_to: int
    capture_pre_sec: float
    capture_post_sec: float
    eval_fn: Callable       # compiled from YAML trigger expression


# ── Layer 1: YAML Rule Engine ─────────────────────────────────────────────────

class YAMLRuleEngine:
    """
    Loads /etc/robot/priority_rules.yaml.
    Each rule has: trigger condition, promote_to priority, pre/post capture windows.
    Evaluation is deterministic — no model loading, zero warm-up latency.
    """

    def __init__(self, rules_path: Path):
        self.rules: List[TriggerRule] = []
        self._load(rules_path)

    def _load(self, path: Path):
        if not path.exists():
            logger.warning("Rules file not found: %s — using defaults", path)
            self._load_defaults()
            return

        with open(path) as f:
            doc = yaml.safe_load(f)

        for r in doc.get("rules", []):
            try:
                trigger_expr = r["trigger"]
                # Compile trigger as a safe lambda over telemetry dict
                # e.g. "telemetry.battery_pct < 15" → lambda t: t.get('battery_pct',100) < 15
                eval_fn = self._compile_trigger(trigger_expr)
                self.rules.append(TriggerRule(
                    id=r["id"],
                    promote_to=int(r.get("promote_to", 1).replace("p", "")
                                   if isinstance(r.get("promote_to"), str) else r.get("promote_to", 1)),
                    capture_pre_sec=float(r.get("capture_pre_sec", 10)),
                    capture_post_sec=float(r.get("capture_post_sec", 5)),
                    eval_fn=eval_fn,
                ))
                logger.debug("Loaded rule: %s", r["id"])
            except Exception as exc:
                logger.error("Failed to load rule %s: %s", r.get("id"), exc)

        logger.info("Loaded %d YAML trigger rules", len(self.rules))

    def _load_defaults(self):
        """Built-in fallback rules matching the v4 spec."""
        defaults = [
            ("battery_critical",  lambda t: t.get("battery_pct", 100) < 15,      0, 30, 0),
            ("motor_overcurrent",  lambda t: t.get("motor_current_A", 0) > t.get("motor_threshold_A", 10) * 1.5, 0, 10, 5),
            ("collision_imu",      lambda t: t.get("linear_accel_mps2", 0) > 8.0 or t.get("obstacle_dist_m", 99) < 0.1, 0, 5, 10),
            ("nav_stuck",          lambda t: t.get("displacement_m", 1) < 0.05 and t.get("cmd_vel_nonzero", False), 1, 5, 0),
            ("fatal_log",          lambda t: t.get("log_severity", "") in ("FATAL", "ERROR"), 0, 0, 0),
            ("high_temp",          lambda t: t.get("motor_temp_C", 0) > 75, 1, 15, 5),
        ]
        for rule_id, fn, priority, pre, post in defaults:
            self.rules.append(TriggerRule(
                id=rule_id, promote_to=priority,
                capture_pre_sec=pre, capture_post_sec=post,
                eval_fn=fn,
            ))

    @staticmethod
    def _compile_trigger(expr: str) -> Callable:
        """
        Safely compile a trigger expression string using simpleeval.
        Maps 'telemetry.X' to direct dict key lookups.
        Handles and/or/==/bare booleans; raises on unsafe builtins.
        """
        import re
        # Normalise dotted access: telemetry.battery_pct → battery_pct
        clean = re.sub(r'(telemetry|imu|lidar|odometry|log)\.', '', expr)

        if SIMPLEEVAL_AVAILABLE:
            evaluator = EvalWithCompoundTypes()
            def _eval(t: dict) -> bool:
                evaluator.names = t
                try:
                    return bool(evaluator.eval(clean))
                except Exception:
                    return False
            return _eval
        else:
            # Fallback: restricted eval (no builtins)
            import ast
            safe_expr = re.sub(r'(\w+)', r"__t__.get('\1', __t__.get('\1'))", clean)
            try:
                tree = ast.parse(safe_expr, mode="eval")
                code = compile(tree, "<rule>", "eval")
                return lambda t: eval(code, {"__builtins__": {}}, {"__t__": t})
            except Exception as exc:
                logger.warning("Could not compile trigger: %s (error: %s)", expr, exc)
                return lambda t: False

    def evaluate(self, telemetry: Dict) -> List[AnomalyEvent]:
        events = []
        ts_ms = int(time.time() * 1000)
        for rule in self.rules:
            try:
                if rule.eval_fn(telemetry):
                    score = _score_anomaly(
                        type_weight=1.0 if rule.promote_to == 0 else 0.7,
                        anomaly_bonus=2.0,
                        recency_decay=1.0,
                        size_penalty=1.0,
                    )
                    events.append(AnomalyEvent(
                        rule_id=rule.id,
                        layer="yaml_rule",
                        score=score,
                        priority=rule.promote_to,
                        ts_ms=ts_ms,
                        flags=[rule.id],
                        capture_pre_sec=rule.capture_pre_sec,
                        capture_post_sec=rule.capture_post_sec,
                        data=telemetry,
                    ))
            except Exception as exc:
                logger.debug("Rule %s eval error: %s", rule.id, exc)
        return events


# ── Layer 2: Z-Score Detector (univariate rolling window) ────────────────────

class ZScoreDetector:
    """
    Rolling Z-Score detector.
    Applied to: battery discharge slope, motor temperature, CPU temperature.

    Physical insight (from report):
      Static threshold (e.g. 80°C) misses a motor at 60°C trending upward
      (bearing wear, partial blockage). Z-Score detects the rate of change
      deviation — early warning before threshold breach.
    """

    def __init__(self, window: int = 300, sigma_threshold: float = 3.0,
                 warmup: int = 30):
        self.window = deque(maxlen=window)   # rolling 5-min at 1Hz
        self.sigma = sigma_threshold
        self.warmup = warmup

    def feed(self, value: float) -> Tuple[bool, float]:
        """
        Returns (is_anomaly, z_score).
        Needs warmup samples before detecting.
        """
        self.window.append(value)
        if len(self.window) < self.warmup:
            return False, 0.0

        try:
            if STATS_AVAILABLE:
                mu = stats_mod.mean(self.window)
                sigma = stats_mod.stdev(self.window)
            else:
                vals = list(self.window)
                mu = sum(vals) / len(vals)
                variance = sum((x - mu) ** 2 for x in vals) / len(vals)
                sigma = math.sqrt(variance)

            z = abs(value - mu) / (sigma + 1e-9)
            return z > self.sigma, z
        except Exception:
            return False, 0.0


class MultiChannelZScore:
    """One ZScoreDetector per channel (topic + field)."""

    def __init__(self, sigma: float = 3.0):
        self.sigma = sigma
        self._detectors: Dict[str, ZScoreDetector] = {}

    def feed(self, channel: str, value: float) -> Tuple[bool, float]:
        if channel not in self._detectors:
            self._detectors[channel] = ZScoreDetector(sigma_threshold=self.sigma)
        return self._detectors[channel].feed(value)

    def evaluate(self, telemetry: Dict) -> List[AnomalyEvent]:
        events = []
        ts_ms = int(time.time() * 1000)
        checks = [
            ("battery_slope",   telemetry.get("battery_drain_rate", 0)),
            ("motor_temp_slope", telemetry.get("motor_temp_C", 0)),
            ("cpu_temp",         telemetry.get("cpu_temp_C", 0)),
        ]
        for channel, value in checks:
            if value is None:
                continue
            is_anom, z = self.feed(channel, float(value))
            if is_anom:
                score = min(100.0, 50.0 + z * 10)
                events.append(AnomalyEvent(
                    rule_id=f"z_score_{channel}",
                    layer="z_score",
                    score=score,
                    priority=1 if score < 80 else 0,
                    ts_ms=ts_ms,
                    flags=[f"z_score_{channel}"],
                    capture_pre_sec=15.0,
                    capture_post_sec=5.0,
                    data={"channel": channel, "z_score": z, "value": value},
                ))
        return events


# ── Layer 3: Isolation Forest (multivariate) ──────────────────────────────────

class IsolationForestDetector:
    """
    Multivariate anomaly detection on (motor_torque, current_draw, imu_accel).

    Physical insight (from report):
      Joint distribution catches correlated failures invisible to 1D thresholds.
      e.g. normal torque + high current = winding fault; seen as isolated point
      in 5D space at shallow tree depth.

    Edge deployment: n_estimators=50, max_samples=256 → ~0.2ms per sample.
    Model pre-trained offline, loaded as pickle.
    """

    MODEL_PATH = Path("/etc/robot/isoforest_model.pkl")

    def __init__(self, model_path: Optional[Path] = None,
                 contamination: float = 0.01):
        self.contamination = contamination
        self._model = None
        self._lock = threading.Lock()
        path = model_path or self.MODEL_PATH

        if ML_AVAILABLE:
            logger.info("Online-only ML training mode enabled for IsolationForest.")
            self._init_online_model()
        else:
            logger.warning("sklearn unavailable — IsolationForest disabled")

        self._online_buffer = []
        self._online_trained = False

    def _init_online_model(self):
        if not ML_AVAILABLE:
            return
        self._model = IsolationForest(
            n_estimators=50,
            max_samples=256,
            contamination=self.contamination,
            random_state=42,
        )

    def feed_and_score(self, torque: float, current: float,
                       accel_x: float, accel_y: float,
                       accel_z: float) -> Tuple[bool, float]:
        """
        Returns (is_anomaly, anomaly_score).
        Online: buffers 512 samples before first fit.
        """
        if not ML_AVAILABLE or self._model is None:
            return False, 0.0

        sample = [torque, current, accel_x, accel_y, accel_z]
        with self._lock:
            if not self._online_trained:
                self._online_buffer.append(sample)
                if len(self._online_buffer) >= 512:
                    import numpy as _np
                    self._model.fit(_np.array(self._online_buffer))
                    self._online_trained = True
                    logger.info("IsolationForest online fit complete (%d samples)",
                                len(self._online_buffer))
                return False, 0.0

            import numpy as _np
            X = _np.array([sample])
            score = float(self._model.decision_function(X)[0])
            # Negative decision function score = more anomalous
            is_anom = score < -0.1
            # Normalise to 0–100 (more negative = higher score)
            normalised = min(100.0, max(0.0, (-score + 0.1) * 200))
            return is_anom, normalised

    def evaluate(self, telemetry: Dict) -> List[AnomalyEvent]:
        torque  = telemetry.get("motor_torque_Nm", 0.0)
        current = telemetry.get("motor_current_A", 0.0)
        ax      = telemetry.get("accel_x", 0.0)
        ay      = telemetry.get("accel_y", 0.0)
        az      = telemetry.get("accel_z", 9.8)

        try:
            is_anom, score = self.feed_and_score(torque, current, ax, ay, az)
        except Exception as exc:
            logger.debug("IsoForest eval error: %s", exc)
            return []

        if not is_anom:
            return []

        ts_ms = int(time.time() * 1000)
        return [AnomalyEvent(
            rule_id="isolation_forest_multivariate",
            layer="isolation_forest",
            score=score,
            priority=0 if score > 80 else 1,
            ts_ms=ts_ms,
            flags=["isolation_forest"],
            capture_pre_sec=10.0,
            capture_post_sec=5.0,
            data={"score": score, "torque": torque,
                  "current": current, "accel": [ax, ay, az]},
        )]


# ── v2 Priority Scoring Formula ───────────────────────────────────────────────

def _score_anomaly(type_weight: float, anomaly_bonus: float,
                   recency_decay: float, size_penalty: float) -> float:
    """
    v2 numerical score formula: type_weight × anomaly_bonus × recency_decay × size_penalty
    Clamped to 0–100.
    """
    raw = type_weight * anomaly_bonus * recency_decay * size_penalty * 25
    return min(100.0, max(0.0, raw))


def priority_from_score(score: float) -> int:
    """Map continuous 0–100 score to P-bucket."""
    if score >= 90: return 0
    if score >= 60: return 1
    if score >= 30: return 2
    return 3


# ── Unified Detector ──────────────────────────────────────────────────────────

class AnomalyDetector:
    """Orchestrates all 3 layers. Returns merged list of AnomalyEvents."""

    def __init__(self, rules_path: Path):
        self.yaml_engine  = YAMLRuleEngine(rules_path)
        self.zscore       = MultiChannelZScore(sigma=3.0)
        self.isoforest    = IsolationForestDetector()
        self._callbacks: List[Callable[[List[AnomalyEvent]], None]] = []
        self._last_trigger_times: Dict[str, float] = {}
        # Build per-rule cooldown lookup from YAML rules
        self._cooldowns: Dict[str, float] = {}
        for rule in self.yaml_engine.rules:
            self._cooldowns[rule.id] = rule.capture_pre_sec + rule.capture_post_sec

    def on_anomaly(self, callback: Callable[[List[AnomalyEvent]], None]):
        self._callbacks.append(callback)

    def evaluate(self, telemetry: Dict) -> List[AnomalyEvent]:
        raw_events: List[AnomalyEvent] = []
        raw_events.extend(self.yaml_engine.evaluate(telemetry))
        raw_events.extend(self.zscore.evaluate(telemetry))
        raw_events.extend(self.isoforest.evaluate(telemetry))

        now = time.time()
        events = []
        for e in raw_events:
            # Per-rule cooldown: suppress for (capture_pre + capture_post) seconds
            # This is the exact window already defined per rule, preventing anomaly storms
            cooldown = self._cooldowns.get(e.rule_id, e.capture_pre_sec + e.capture_post_sec)
            cooldown = max(cooldown, 5.0)  # minimum 5s cooldown
            last_fired = self._last_trigger_times.get(e.rule_id, 0)
            if (now - last_fired) >= cooldown:
                self._last_trigger_times[e.rule_id] = now
                events.append(e)

        # Fire callbacks for the batch of detected anomalies to deduplicate captures
        if events:
            for cb in self._callbacks:
                try:
                    cb(events)
                except Exception as exc:
                    logger.error("Anomaly callback error: %s", exc)

        return events
