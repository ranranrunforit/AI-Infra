"""
main.py — Robot Data Pipeline v5 Entry Point

Wires:
  CpuGovernor → RingBuffer → AnomalyDetector →
  ManifestDB → BandwidthLimiter (MMCF + 50Mbps ceiling) →
  UploadAgent → EvictionManager → Metrics (Prometheus)

Key v5 changes vs v4:
  - RingBuffer v5 API (staging_root, robot_id, session_id)
  - BandwidthLimiter: MMCF weights loaded from config, no manifest_db arg
  - Thermal cutoff: P1-P3 blocked when cpu_temp >= throttle_temp_c
  - P0 bypasses MMCF, still subject to 50 Mbps token bucket
  - Daily cap: REMOVED
  - ingest_binary() added for simulator LiDAR/Camera/Log data
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline.main")

sys.path.insert(0, str(Path(__file__).parent.parent))

from robot_agent.core.manifest_db       import ManifestDB
from robot_agent.core.ring_buffer       import RingBuffer
from robot_agent.core.mcap_writer       import MCAPFastWriter, BackgroundReindexer
from robot_agent.core.anomaly_detector  import AnomalyDetector, AnomalyEvent
from robot_agent.core.bandwidth_limiter import BandwidthLimiter
from robot_agent.core.upload_agent      import UploadAgent
from robot_agent.core.eviction_manager  import EvictionManager
from robot_agent.core.cpu_governor      import CpuGovernor
from robot_agent.core.metrics_exporter  import RobotMetrics

try:
    from google.cloud import pubsub_v1
    PUBSUB_AVAILABLE = True
except ImportError:
    PUBSUB_AVAILABLE = False

try:
    from systemd.daemon import notify as sd_notify
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Environment variable overrides
    cfg["robot"]["id"]       = os.getenv("ROBOT_ID",       cfg["robot"]["id"])
    cfg["cloud"]["bucket"]   = os.getenv("GCS_BUCKET",     cfg["cloud"]["bucket"])
    if "database" in cfg:
        cfg["database"]["host"]     = os.getenv("POSTGRES_HOST",     cfg["database"].get("host", ""))
        cfg["database"]["password"] = os.getenv("POSTGRES_PASSWORD", cfg["database"].get("password", ""))

    # Demo mode: redirect storage to /tmp, disable cgroup enforcement
    if os.getenv("DEMO_MODE", "0") == "1":
        cfg["storage"]["root"] = "/tmp/robot_data"
        cfg["cpu"]["dry_run"]  = True

    return cfg


class RobotPipeline:
    """
    Full v5 pipeline. Call start() to initialise all subsystems.
    Simulator calls ingest() and ingest_binary() to push sensor data.
    """

    def __init__(self, cfg: dict):
        self.cfg          = cfg
        self.robot_id     = cfg["robot"]["id"]
        self.staging_root = Path(cfg["storage"]["root"])
        self._shutdown    = threading.Event()

        # Subsystems — initialised in start()
        self.governor:   CpuGovernor        = None
        self.manifest:   ManifestDB         = None
        self.ring:       RingBuffer         = None
        self.detector:   AnomalyDetector    = None
        self.bw:         BandwidthLimiter   = None
        self.uploader:   UploadAgent        = None
        self.evictor:    EvictionManager    = None
        self.reindexer:  BackgroundReindexer = None
        self.metrics:    RobotMetrics       = None

        self._writer_lock    = threading.Lock()
        self._active_writer: MCAPFastWriter = None
        self._session_id     = self._get_or_create_session_id()
        self._frame_seq      = 0

        # Back-pressure state
        self._default_rotation_secs = cfg["ring_buffer"]["rotate_time_sec"]
        self._backpressure_level    = 0

        # ── FIX: store latest anomaly context so _on_file_ready can use it ───
        # _on_anomaly fires BEFORE force_rotate, so by the time _on_file_ready
        # is called these are already set with the correct values.
        self._last_anomaly_flags:     list = []
        self._last_anomaly_data_type: str  = "mixed"

        # Pub/Sub for P0 real-time alerts
        self._pubsub_publisher = None
        self._pubsub_topic     = None
        if PUBSUB_AVAILABLE:
            try:
                project_id = cfg["cloud"].get("project_id", "")
                topic_name = cfg["cloud"].get("pubsub_topic", "robot-p0-alerts")
                if project_id:
                    self._pubsub_publisher = pubsub_v1.PublisherClient()
                    self._pubsub_topic = f"projects/{project_id}/topics/{topic_name}"
                    logger.info("Pub/Sub P0 alerts: %s", self._pubsub_topic)
            except Exception as exc:
                logger.warning("Pub/Sub init failed (P0 alerts disabled): %s", exc)

    # ── start() — wires all subsystems ───────────────────────────────────────

    def start(self):
        logger.info("═══ Robot Pipeline v5 starting (robot_id=%s) ═══", self.robot_id)
        self.staging_root.mkdir(parents=True, exist_ok=True)

        # CPU governance
        self.governor = CpuGovernor(dry_run=self.cfg["cpu"].get("dry_run", True))
        self.governor.enter()

        # Manifest DB
        db_path       = self.staging_root / self.cfg["storage"]["db_path"]
        self.manifest = ManifestDB(db_path)

        result = self.manifest.orphan_reconciliation(self.staging_root)
        if result["orphans_recovered"] > 0:
            logger.warning("Recovered %d orphaned files", result["orphans_recovered"])

        last_session = self.manifest.get_last_session_id()
        if last_session:
            self._session_id = last_session
            try:
                (self.staging_root / ".session_id").write_text(last_session)
            except Exception:
                pass

        # ── Ring buffer (v5 API) ───────────────────────────────────────────
        rb_cfg = self.cfg["ring_buffer"]
        self.ring = RingBuffer(
            staging_root      = self.staging_root,
            robot_id          = self.robot_id,
            session_id        = self._session_id,
            priority          = 2,               # default P2 for telemetry
            on_file_ready     = self._on_file_ready,
            rotate_size_bytes = int(rb_cfg["rotate_size_mb"] * 1024 * 1024),
            rotate_time_sec   = int(rb_cfg["rotate_time_sec"]),
        )

        # ── Anomaly detector ───────────────────────────────────────────────
        rules_path   = Path(__file__).parent.parent / "config" / "priority_rules.yaml"
        self.detector = AnomalyDetector(rules_path)
        self.detector.on_anomaly(self._on_anomaly)

        # ── Bandwidth limiter (MMCF + 50 Mbps token bucket) ───────────────
        bw_cfg   = self.cfg["bandwidth"]
        mmcf_cfg = bw_cfg.get("mmcf", {})
        self.bw  = BandwidthLimiter(
            initial_rate_bps = bw_cfg["rate_bps"],
            max_rate_bps     = bw_cfg["rate_bps"],
            gcs_probe_bucket = self.cfg.get("cloud", {}).get("bucket"),
            # MMCF parameters from config
            cost_limit       = mmcf_cfg.get("cost_limit",   100.0),
            backoff_sec      = mmcf_cfg.get("backoff_sec",  0.5),
            weight_bw        = mmcf_cfg.get("weight_bw",    20.0),
            weight_temp      = mmcf_cfg.get("weight_temp",  1.0),
            weight_prio      = mmcf_cfg.get("weight_prio",  10.0),
            throttle_temp_c  = mmcf_cfg.get("throttle_temp_c", 85.0),
        )

        # ── Upload agent ───────────────────────────────────────────────────
        up_cfg = self.cfg.get("upload", {})
        self.uploader = UploadAgent(
            manifest        = self.manifest,
            bw              = self.bw,
            gcs_bucket      = self.cfg["cloud"]["bucket"],
            staging_root    = self.staging_root,
            poll_interval   = up_cfg.get("poll_interval_secs", 1),
            grace_period_secs = up_cfg.get("grace_period_secs", 3600),
        )
        self.uploader.start()

        # ── Eviction manager ───────────────────────────────────────────────
        ev_cfg = self.cfg["eviction"]
        self.evictor = EvictionManager(
            manifest        = self.manifest,
            staging_root    = self.staging_root,
            check_interval  = ev_cfg["check_interval_secs"],
            quota_bytes     = ev_cfg.get("quota_bytes", 85899345920),
            dict_path       = Path(ev_cfg["zstd_dict_path"]),
            alert_fn        = self._alert,
        )
        self.evictor.start()

        # ── Background reindexer ───────────────────────────────────────────
        self.reindexer = BackgroundReindexer(self.staging_root, manifest=self.manifest)
        self.reindexer.start()

        # ── Prometheus metrics ─────────────────────────────────────────────
        self.metrics = RobotMetrics(robot_id=self.robot_id)
        self.uploader.metrics = self.metrics   # FIX: inject metrics into uploader
        self.metrics.start_http_server(
            port=self.cfg["monitoring"]["prometheus_port"]
        )

        # ── Background threads ─────────────────────────────────────────────
        for name, target in [
            ("metrics-loop",    self._metrics_loop),
            ("cpu-monitor",     self._cpu_monitor_loop),
            ("backpressure",    self._backpressure_loop),
        ]:
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()

        logger.info("All subsystems started ✓")

    # ── Sensor data ingestion (called by simulator) ───────────────────────────

    def ingest(self, topic: str, data: dict) -> None:
        """
        Ingest a telemetry dict. Runs anomaly detection and pushes to ring buffer.
        Called by the simulator 10x/sec for scalar sensor readings.
        """
        try:
            # Run anomaly detection on scalar telemetry
            if self.detector:
                self.detector.evaluate(data)

            # Push JSON-encoded telemetry to ring buffer
            import json
            payload  = json.dumps(data).encode()
            ts_ms    = data.get("ts_ms", int(time.time() * 1000))
            self.ring.push(topic, ts_ms, payload)

            # Update Prometheus battery slope metric from telemetry
            if self.metrics:
                self.metrics.update_battery_slope(
                    data.get("battery_drain_rate", 0.0)
                )
        except Exception as exc:
            logger.debug("ingest error: %s", exc)

    def ingest_binary(
        self,
        topic:    str,
        ts_ms:    int,
        payload:  bytes,
        encoding: str = "application/octet-stream",
        meta:     dict = None,
    ) -> None:
        """
        Ingest binary sensor data (LiDAR, camera frames, log records).
        Called by simulator for:
            lidar/points  → float32 point cloud array
            camera/rgb    → raw RGB bytes (900 KB/frame, simulated H.265)
            rosout        → JSON log records
        """
        try:
            if self.ring:
                self.ring.push_binary(
                    topic    = topic,
                    ts_ms    = ts_ms,
                    payload  = payload,
                    encoding = encoding,
                    meta     = meta,
                )
        except Exception as exc:
            logger.debug("ingest_binary error: %s", exc)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_file_ready(self, path: Path) -> None:
        """Called by RingBuffer when a file is rotated to disk."""
        try:
            size    = path.stat().st_size if path.exists() else 0
            p_dirs  = {"p0_critical": 0, "p1_high": 1, "p2_normal": 2}
            # Infer priority from path segment (P3 removed — defaults to P2)
            priority = 2
            for part in path.parts:
                if part in p_dirs:
                    priority = p_dirs[part]
                    break

            # ── FIX: use real anomaly context for P0 files ────────────────────
            # When _on_anomaly fires it stores flags + data_type on self BEFORE
            # calling force_rotate. By the time this callback runs those values
            # are already set, so we pick them up here and immediately reset
            # them so the next non-anomaly file gets clean defaults.
            if priority == 0:
                data_type     = self._last_anomaly_data_type
                anomaly_flags = list(self._last_anomaly_flags)
                # Reset so subsequent non-anomaly files don't inherit these
                self._last_anomaly_flags     = []
                self._last_anomaly_data_type = "mixed"
            else:
                data_type     = "mixed"
                anomaly_flags = []

            from robot_agent.core.manifest_db import FileRecord
            rec = FileRecord(
                filename       = path.name,
                path           = str(path),
                session_id     = self._session_id,
                data_type      = data_type,
                priority       = priority,
                score          = 100.0 if priority == 0 else float(50 * (3 - priority)),
                size_bytes     = size,
                anomaly_flags  = anomaly_flags,
                schema_version = self.cfg["robot"].get("schema_version", "v5"),
            )
            self.manifest.insert(rec)
            if self.metrics:
                self.metrics.record_write(size)
        except Exception as exc:
            logger.warning("_on_file_ready error for %s: %s", path.name, exc)

    def _on_anomaly(self, events: list) -> None:
        """Called by AnomalyDetector when anomaly rules fire."""
        priority = min((e.priority for e in events), default=3)
        flags = [e.rule_id for e in events]
        logger.warning(
            "ANOMALY BATCH [%d events] max_score=%d P%d flags=%s",
            len(events), max((e.score for e in events), default=0),
            priority, flags,
        )

        import time as _time
        ts_now = int(_time.time() * 1000)
        for ev in events:
            pre_ms  = int(getattr(ev, 'capture_pre_sec',  10) * 1000)
            post_ms = int(getattr(ev, 'capture_post_sec', 10) * 1000)
            if self.manifest:
                try:
                    self.manifest.promote_priority(
                        session_id=self._session_id,
                        new_priority=ev.priority,
                        start_ts=ts_now - pre_ms,
                        end_ts=ts_now + post_ms,
                    )
                except Exception as exc:
                    logger.debug("promote_priority error: %s", exc)

        # ── FIX: store anomaly context BEFORE force_rotate ────────────────────
        # _on_file_ready will pick these up when the rotated file lands.
        if priority == 0:
            self._last_anomaly_flags     = flags
            self._last_anomaly_data_type = "anomaly"

        if priority == 0 and self.ring:
            self.ring.force_rotate(anomaly=True, data_type="anomaly")

        if self.metrics:
            for ev in events:
                self.metrics.record_anomaly(ev.layer, ev.rule_id)

        if priority == 0 and self._pubsub_publisher:
            self._publish_p0_alert(events, flags)

    # ── Background loops ──────────────────────────────────────────────────────

    def _metrics_loop(self):
        """Update Prometheus gauges every 15 seconds."""
        while not self._shutdown.is_set():
            try:
                ev_stats = self.evictor.stats() if self.evictor else {}
                bw_stats = self.bw.stats()       if self.bw       else {}
                db_stats = self.manifest.stats() if self.manifest else {}

                counts = db_stats.get("counts_by_state", {})
                # Build per-priority pending count
                pending_by_priority = {i: 0 for i in range(4)}
                # Try per-priority query if available
                try:
                    for p in range(4):
                        pending_by_priority[p] = self.manifest.count_pending_by_priority(p)
                except Exception:
                    pass

                if self.metrics:
                    # CPU: try psutil (works in Docker), fallback to 0
                    cpu_pct = 0.0
                    try:
                        import psutil
                        proc = psutil.Process()
                        total_pct = proc.cpu_percent(interval=None)
                        for child in proc.children(recursive=True):
                            try:
                                total_pct += child.cpu_percent(interval=None)
                            except Exception:
                                pass
                        quota = self.cfg.get("cpu", {}).get("quota_pct", 20.0)
                        cpu_pct = min(1.0, total_pct / max(quota, 1.0))
                    except Exception:
                        pass

                    self.metrics.update(
                        cpu_pct           = cpu_pct,
                        bucket_level      = bw_stats.get("token_level", 1.0),
                        disk_pct          = ev_stats.get("disk_usage_pct", 0.0),
                        queue_by_priority = pending_by_priority,
                        battery_slope     = 0.0,
                    )

                    # Push per-state per-priority breakdown for Grafana
                    try:
                        state_prio = self.manifest.count_by_state_and_priority()
                        for key, val in state_prio.items():
                            parts = key.rsplit("_p", 1)
                            if len(parts) == 2:
                                state_label, prio_str = parts[0].lower(), parts[1]
                                self.metrics.queue_depth.labels(
                                    robot_id=self.robot_id,
                                    priority=prio_str,
                                    state=state_label,
                                ).set(val)
                    except Exception as exc:
                        logger.debug("State breakdown metrics error: %s", exc)
                    # Add MMCF metrics to Prometheus
                    if hasattr(self.metrics, "update_mmcf"):
                        self.metrics.update_mmcf(
                            cpu_temp_c      = bw_stats.get("cpu_temp_C", 50.0),
                            throttle_active = bw_stats.get("throttle_active", False),
                            cost_p1         = bw_stats.get("mmcf_cost_p1", 0.0),
                            cost_p3         = bw_stats.get("mmcf_cost_p3", 0.0),
                            thermal_blocks  = bw_stats.get("thermal_blocks", 0),
                        )
            except Exception as exc:
                logger.debug("Metrics update error: %s", exc)

            if WATCHDOG_AVAILABLE:
                try:
                    sd_notify("WATCHDOG=1")
                except Exception:
                    pass

            self._shutdown.wait(timeout=15)

    def _cpu_monitor_loop(self):
        """Pause background tasks if CPU > 80% quota."""
        while not self._shutdown.is_set():
            try:
                usage = self.governor.current_usage()
                if usage is not None:
                    if usage > 0.80:
                        if self.reindexer: self.reindexer.pause()
                        if self.uploader:  self.uploader.pause()
                        logger.debug("CPU %.0f%% — background tasks paused", usage * 100)
                    elif usage < 0.60:
                        if self.reindexer: self.reindexer.resume()
                        if self.uploader:  self.uploader.resume()
            except Exception as exc:
                logger.debug("CPU monitor error: %s", exc)
            self._shutdown.wait(timeout=5)

    def _backpressure_loop(self):
        """
        Adaptive back-pressure controller based on queue depth and disk usage.

        Level 0 (normal):   queue < 200MB
        Level 1 (adaptive): queue > 500MB  → slow rotation to 120s
        Level 2 (shedding): queue > 800MB AND disk > 70%  → flag ring buffer
        Recovery:           queue < 200MB  → restore defaults
        """
        while not self._shutdown.is_set():
            try:
                ev_stats  = self.evictor.stats() if self.evictor else {}
                disk_pct  = ev_stats.get("disk_usage_pct", 0)
                db_stats  = self.manifest.stats() if self.manifest else {}
                counts    = db_stats.get("counts_by_state", {})
                pending   = counts.get("PENDING", 0) + counts.get("UPLOADING", 0)
                queue_mb  = pending * 10   # rough estimate: ~10MB per file

                if queue_mb < 200 and self._backpressure_level > 0:
                    self._backpressure_level     = 0
                    self.ring.set_rotation_time(self._default_rotation_secs)
                    self.ring.backpressure_active = False
                    logger.info("Back-pressure: RECOVERED (queue ~%dMB)", queue_mb)

                elif queue_mb > 800 and disk_pct > 70:
                    if self._backpressure_level < 2:
                        self._backpressure_level     = 2
                        self.ring.set_rotation_time(120.0)
                        self.ring.backpressure_active = True
                        logger.warning(
                            "Back-pressure LEVEL 2: SHEDDING low-prio (queue ~%dMB, disk %.1f%%)",
                            queue_mb, disk_pct,
                        )

                elif queue_mb > 500:
                    if self._backpressure_level < 1:
                        self._backpressure_level     = 1
                        self.ring.set_rotation_time(120.0)
                        self.ring.backpressure_active = False
                        logger.warning(
                            "Back-pressure LEVEL 1: ADAPTIVE rotation 120s (queue ~%dMB)",
                            queue_mb,
                        )

            except Exception as exc:
                logger.debug("Back-pressure loop error: %s", exc)
            self._shutdown.wait(timeout=30)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_p0_alert(self, events: list, flags: list):
        import json
        try:
            data = json.dumps({
                "robot_id":    self.robot_id,
                "session_id":  self._session_id,
                "ts_ms":       events[0].ts_ms,
                "flags":       flags,
                "event_count": len(events),
                "max_score":   max(e.score for e in events),
            }).encode()
            future = self._pubsub_publisher.publish(
                self._pubsub_topic, data=data,
                robot_id=self.robot_id, priority="P0",
            )
            logger.info("P0 alert published (msg_id=%s)", future.result(timeout=5))
        except Exception as exc:
            logger.warning("Pub/Sub P0 alert failed: %s", exc)

    def _alert(self, msg: str):
        logger.critical("ALERT: %s", msg)

    def shutdown(self):
        logger.info("Shutting down pipeline...")
        self._shutdown.set()
        if self.uploader:  self.uploader.stop()
        if self.evictor:   self.evictor.stop()
        if self.reindexer: self.reindexer.stop()
        logger.info("Pipeline shutdown complete")

    def _get_or_create_session_id(self) -> str:
        import uuid
        session_file = self.staging_root / ".session_id"
        if session_file.exists():
            try:
                return session_file.read_text().strip()
            except Exception:
                pass
        new_id = f"s{uuid.uuid4().hex[:8]}"
        try:
            self.staging_root.mkdir(parents=True, exist_ok=True)
            session_file.write_text(new_id)
        except Exception:
            pass
        return new_id


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Robot Data Pipeline v5")
    parser.add_argument("--config",    default="config/pipeline.yaml")
    parser.add_argument("--demo",      action="store_true",
                        help="Run with sensor simulator (sets DEMO_MODE=1)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.demo:
        os.environ["DEMO_MODE"] = "1"

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path(__file__).parent.parent / "config" / "pipeline.yaml"

    cfg      = load_config(cfg_path)
    pipeline = RobotPipeline(cfg)

    def handle_signal(sig, frame):
        logger.info("Signal %s — shutting down", sig)
        pipeline.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT,  handle_signal)

    pipeline.start()

    if args.demo:
        logger.info("Demo mode: starting simulator")
        from robot_agent.simulator import run_demo_simulation
        run_demo_simulation(pipeline, cfg)
    else:
        logger.info("Pipeline running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(10)
                if pipeline.manifest:
                    logger.info("Status: %s", pipeline.manifest.stats())
        except KeyboardInterrupt:
            pipeline.shutdown()
            sys.exit(0)


if __name__ == "__main__":
    main()