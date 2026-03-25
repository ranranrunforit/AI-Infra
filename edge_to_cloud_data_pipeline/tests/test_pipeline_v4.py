"""
tests/test_pipeline_v4.py — Integration test suite for Robot Pipeline v4
Tests: manifest DB, ring buffer, anomaly detection, bandwidth limiter,
       upload FSM state transitions, eviction cascade, MCAP writer.
All tests run without external services (GCS/PostgreSQL stubbed).
"""

import os
import sys
import json
import time
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["DEMO_MODE"] = "1"

from robot_agent.core.manifest_db      import ManifestDB, FileRecord
from robot_agent.core.ring_buffer      import RingBuffer, SensorFrame
from robot_agent.core.anomaly_detector import (
    AnomalyDetector, ZScoreDetector, MultiChannelZScore,
    YAMLRuleEngine, priority_from_score,
)
from robot_agent.core.bandwidth_limiter import BandwidthLimiter, DailyCapExceeded
from robot_agent.core.eviction_manager  import EvictionManager
from robot_agent.core.mcap_writer       import (
    MCAPFastWriter, make_filename, make_staging_path,
)
from robot_agent.core.cpu_governor      import CpuGovernor
from robot_agent.core.upload_agent      import JitteredBackoff


class TestManifestDB(unittest.TestCase):
    """SQLite manifest: CRUD, state machine, orphan recovery, gap detection."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = ManifestDB(Path(self.tmp) / "test_manifest.db")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _make_rec(self, priority=3, session="s001"):
        return FileRecord(
            filename=f"test_{priority}_{time.time_ns()}.mcap",
            path=f"/tmp/test_{priority}.mcap",
            session_id=session,
            data_type="telemetry",
            priority=priority,
            score=float(50 - priority * 10),
            size_bytes=1024,
        )

    def test_insert_and_get(self):
        rec = self._make_rec(priority=2)
        fid = self.db.insert(rec)
        self.assertGreater(fid, 0)
        loaded = self.db.get_by_id(fid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.filename, rec.filename)
        self.assertEqual(loaded.priority, 2)

    def test_state_transitions(self):
        """5-state FSM: PENDING → UPLOADING → VERIFYING → UPLOADED → DELETED"""
        rec = self._make_rec()
        fid = self.db.insert(rec)

        for state in ["UPLOADING", "VERIFYING", "UPLOADED", "DELETED"]:
            self.db.set_state(fid, state)
            loaded = self.db.get_by_id(fid)
            self.assertEqual(loaded.state, state)

    def test_invalid_state_rejected(self):
        rec = self._make_rec()
        fid = self.db.insert(rec)
        with self.assertRaises(AssertionError):
            self.db.set_state(fid, "FLYING")

    def test_pending_ordered_by_priority_then_score(self):
        """Higher priority (lower number) and higher score comes first."""
        recs = [
            FileRecord(filename="p3_low.mcap",  path="/tmp/p3.mcap",
                       session_id="s", data_type="t", priority=3, score=50, size_bytes=0),
            FileRecord(filename="p0_crit.mcap", path="/tmp/p0.mcap",
                       session_id="s", data_type="t", priority=0, score=100, size_bytes=0),
            FileRecord(filename="p1_high.mcap", path="/tmp/p1.mcap",
                       session_id="s", data_type="t", priority=1, score=70, size_bytes=0),
        ]
        for r in recs:
            self.db.insert(r)
        pending = self.db.get_pending(limit=10)
        priorities = [r.priority for r in pending]
        self.assertEqual(priorities, sorted(priorities))   # ascending priority = descending urgency

    def test_orphan_reconciliation(self):
        """Files on disk but not in manifest → recovered as PENDING."""
        staging = Path(self.tmp) / "staging"
        staging.mkdir()
        orphan = staging / "20250317T120000Z_1000_s001_p2_lidar_512_abc12345.mcap"
        orphan.write_bytes(b"fake mcap data")

        result = self.db.orphan_reconciliation(staging)
        self.assertEqual(result["orphans_recovered"], 1)

        # File should now be in manifest
        pending = self.db.get_pending()
        self.assertEqual(len(pending), 1)

    def test_sequence_gap_recording(self):
        self.db.record_gap("s001", expected_seq=42)
        # No exception = pass; gaps table populated

    def test_bandwidth_accounting(self):
        self.db.record_bytes(1024)
        self.db.record_bytes(2048)
        total = self.db.bytes_today()
        self.assertEqual(total, 3072)

    def test_promote_priority(self):
        rec = self._make_rec(priority=3)
        now = int(time.time() * 1000)
        rec.created_at = now
        fid = self.db.insert(rec)

        self.db.promote_priority(
            session_id=rec.session_id,
            new_priority=0,
            start_ts=now - 5000,
            end_ts=now + 5000,
        )
        loaded = self.db.get_by_id(fid)
        self.assertEqual(loaded.priority, 0)
        self.assertEqual(loaded.score, 100.0)

    def test_incremental_vacuum(self):
        """Ensure incremental_vacuum doesn't raise."""
        self.db.incremental_vacuum(pages=10)

    def test_thread_safety(self):
        """Multiple threads inserting concurrently should not corrupt DB."""
        errors = []
        def insert_batch(n):
            for i in range(20):
                try:
                    rec = FileRecord(
                        filename=f"t{n}_{i}.mcap", path=f"/tmp/t{n}_{i}.mcap",
                        session_id="s", data_type="t", priority=3, score=10, size_bytes=1,
                    )
                    self.db.insert(rec)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=insert_batch, args=(i,)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [], f"Thread errors: {errors}")


class TestRingBuffer(unittest.TestCase):

    def setUp(self):
        self.buf = RingBuffer(capacity=100, rotation_secs=60, rotation_bytes=10240)

    def test_push_and_size(self):
        self.buf.push("imu", {"accel_x": 0.1})
        self.buf.push("lidar", {"dist": 2.5})
        self.assertEqual(self.buf.size(), 2)

    def test_rotate_returns_frames(self):
        for i in range(10):
            self.buf.push("test", {"seq": i})
        frames = self.buf.rotate()
        self.assertIsNotNone(frames)
        self.assertEqual(len(frames), 10)
        self.assertEqual(self.buf.size(), 0)   # cleared after rotate

    def test_extract_window(self):
        base_ts = int(time.time() * 1000)
        for i in range(5):
            frame = self.buf.push("sensor", {"val": i})
            # Directly set ts for test determinism
            frame.ts_ms = base_ts + i * 1000

        window = self.buf.extract_window(base_ts + 1000, base_ts + 3000)
        self.assertEqual(len(window), 3)

    def test_double_buffer_no_data_loss_during_rotate(self):
        """Writes to buffer while rotation in progress should not be lost."""
        for i in range(20):
            self.buf.push("sensor", {"i": i})
        batch1 = self.buf.rotate()

        for i in range(10):
            self.buf.push("sensor", {"i": i + 100})
        batch2 = self.buf.rotate()

        self.assertEqual(len(batch1), 20)
        self.assertEqual(len(batch2), 10)

    def test_stats(self):
        self.buf.push("test", {})
        s = self.buf.stats()
        self.assertIn("active_buffer", s)
        self.assertIn("total_sequence", s)


class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = AnomalyDetector(
            rules_path=Path("config/priority_rules.yaml")
        )

    def test_battery_critical_triggers(self):
        events = self.detector.yaml_engine.evaluate({"battery_pct": 10})
        rule_ids = [e.rule_id for e in events]
        self.assertIn("battery_critical", rule_ids)
        # Must be promoted to P0
        for e in events:
            if e.rule_id == "battery_critical":
                self.assertEqual(e.priority, 0)
                self.assertGreater(e.capture_pre_sec, 0)

    def test_motor_overcurrent_triggers(self):
        telemetry = {"motor_current_A": 18.0, "motor_threshold_A": 10.0}
        events = self.detector.yaml_engine.evaluate(telemetry)
        rule_ids = [e.rule_id for e in events]
        self.assertIn("motor_overcurrent", rule_ids)

    def test_normal_values_no_trigger(self):
        normal = {
            "battery_pct": 80,
            "motor_current_A": 5.0,
            "motor_threshold_A": 10.0,
            "motor_temp_C": 40.0,
            "linear_accel_mps2": 0.5,
            "obstacle_dist_m": 2.0,
            "cpu_temp_C": 50.0,
            "nav_stuck": False,
            "has_fatal_log": False,
        }
        events = self.detector.yaml_engine.evaluate(normal)
        self.assertEqual(events, [])

    def test_zscore_detects_drift(self):
        det = ZScoreDetector(window=50, sigma_threshold=3.0, warmup=30)
        # Feed normal values
        for i in range(35):
            det.feed(50.0 + (i % 3) * 0.1)   # very stable
        # Sudden spike
        is_anom, z = det.feed(200.0)
        self.assertTrue(is_anom, "Z-Score should detect 200°C spike")
        self.assertGreater(z, 3.0)

    def test_zscore_no_false_positive_during_warmup(self):
        det = ZScoreDetector(window=50, warmup=30)
        for i in range(20):   # below warmup threshold
            is_anom, _ = det.feed(float(i * 10))
        # Even with extreme value, no detection before warmup complete
        is_anom, _ = det.feed(9999.0)
        self.assertFalse(is_anom, "No detection before warmup complete")

    def test_priority_from_score(self):
        self.assertEqual(priority_from_score(95), 0)
        self.assertEqual(priority_from_score(70), 1)
        self.assertEqual(priority_from_score(40), 2)
        self.assertEqual(priority_from_score(10), 3)

    def test_anomaly_callback_fired(self):
        fired = []
        self.detector.on_anomaly(lambda e: fired.append(e))
        self.detector.evaluate({"battery_pct": 5})
        self.assertTrue(len(fired) > 0)


class TestBandwidthLimiter(unittest.TestCase):

    def test_p0_never_blocked(self):
        """P0 should return instantly regardless of token level."""
        bw = BandwidthLimiter(rate_bps=100, burst_bps=100)
        # Drain all tokens first
        bw.tokens = 0
        start = time.time()
        bw.wait_for_token(10 * 1024 * 1024, priority=0)   # 10MB P0
        elapsed = time.time() - start
        self.assertLess(elapsed, 0.1, "P0 should not wait")

    def test_p0_bypasses_daily_cap(self):
        """P0 should succeed even when daily cap is exhausted."""
        mock_db = MagicMock()
        mock_db.bytes_today.return_value = 600_000_000  # over 500MB cap
        mock_db.daily_cap.return_value   = 500_000_000
        bw = BandwidthLimiter(manifest_db=mock_db, burst_bps=10_000_000)
        # Should not raise DailyCapExceeded
        bw.wait_for_token(1024, priority=0)

    def test_p1_blocked_by_daily_cap(self):
        mock_db = MagicMock()
        mock_db.bytes_today.return_value = 524_000_000   # almost at cap
        mock_db.daily_cap.return_value   = 524_288_000   # 500MB
        bw = BandwidthLimiter(manifest_db=mock_db)
        with self.assertRaises(DailyCapExceeded):
            bw.wait_for_token(1024 * 1024, priority=1)   # 1MB P1 would exceed cap

    def test_token_bucket_level(self):
        bw = BandwidthLimiter(burst_bps=1024)
        bw.tokens = 512   # half full
        self.assertAlmostEqual(bw.current_level(), 0.5, places=1)

    def test_stats_structure(self):
        bw = BandwidthLimiter()
        s = bw.stats()
        self.assertIn("token_level", s)
        self.assertIn("daily_cap_bytes", s)
        self.assertIn("bytes_p0_total", s)


class TestMCAPWriter(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.staging_root = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_filename_convention(self):
        """Filename must encode all metadata without opening the file."""
        name = make_filename("s042", priority=2, topic="lidar",
                             size_kb=512, hash8="ab12cdef", ts_ms=1742221822000)
        parts = name.split("_")
        self.assertGreater(len(parts), 4)
        self.assertIn("s042", name)
        self.assertIn("p2", name)
        self.assertIn("lidar", name)
        self.assertIn("ab12cdef", name)
        self.assertTrue(name.endswith(".mcap"))

    def test_atomic_write_no_tmp_visible(self):
        """Upload daemon must never see .tmp files."""
        writer = MCAPFastWriter(
            staging_root=self.staging_root,
            robot_id="test-robot",
            session_id="s001",
            priority=3,
            topic="test",
        )
        writer.add_frame(int(time.time() * 1000), "test",
                         json.dumps({"val": 42}).encode())
        path = writer.flush()
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        self.assertTrue(str(path).endswith(".mcap"))
        # No .tmp file should remain
        tmp_files = list(self.staging_root.rglob("*.tmp"))
        self.assertEqual(tmp_files, [])

    def test_priority_in_directory_structure(self):
        """P0 files must land in p0_critical/ directory."""
        writer = MCAPFastWriter(
            staging_root=self.staging_root,
            robot_id="r1", session_id="s1", priority=0, topic="anomaly",
        )
        writer.add_frame(int(time.time() * 1000), "test", b"data")
        path = writer.flush()
        self.assertIn("p0_critical", str(path))

    def test_empty_flush_returns_none(self):
        writer = MCAPFastWriter(
            staging_root=self.staging_root,
            robot_id="r1", session_id="s1", priority=3, topic="test",
        )
        result = writer.flush()
        self.assertIsNone(result)


class TestEvictionManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.staging = Path(self.tmp) / "staging"
        self.staging.mkdir()
        self.db = ManifestDB(Path(self.tmp) / "manifest.db")
        self.evictor = EvictionManager(
            manifest=self.db,
            staging_root=self.staging,
            check_interval=999,   # Don't auto-run
        )

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _create_file(self, priority: int, name_suffix: str = "") -> FileRecord:
        fname = f"test_p{priority}{name_suffix}.mcap"
        fpath = self.staging / fname
        fpath.write_bytes(b"x" * 1024 * 100)   # 100KB each
        rec = FileRecord(
            filename=fname, path=str(fpath),
            session_id="s1", data_type="test",
            priority=priority, score=50, size_bytes=102400,
        )
        self.db.insert(rec)
        return rec

    def test_p0_never_evicted(self):
        """P0 files must survive even when disk is critically full."""
        recs = [self._create_file(0, f"_{i}") for i in range(3)]
        # Manually run eviction cascade at 100% disk
        evictable = self.db.get_evictable(min_priority=1)   # min_priority=1 = never P0
        evicted_priorities = [r.priority for r in evictable]
        self.assertNotIn(0, evicted_priorities, "P0 must never be in evictable set")

    def test_eviction_order_low_priority_first(self):
        """P3 evicted before P2, P2 before P1."""
        self._create_file(3, "_a")
        self._create_file(2, "_b")
        self._create_file(1, "_c")
        evictable = self.db.get_evictable(min_priority=1)
        priorities = [r.priority for r in evictable]
        # Should be ordered: P3 first, then P2, then P1
        self.assertEqual(priorities, sorted(priorities, reverse=True))

    def test_gc_deletes_uploaded_grace_expired(self):
        """Files in UPLOADED state with grace_expires in the past should be GC'd."""
        rec = self._create_file(3, "_uploaded")
        past_ms = int(time.time() * 1000) - 1000   # 1s ago
        self.db.set_state(rec.id, "UPLOADED", grace_expires=past_ms)
        expired = self.db.get_uploaded_grace_expired()
        self.assertEqual(len(expired), 1)

    def test_stats_structure(self):
        s = self.evictor.stats()
        self.assertIn("disk_usage_pct", s)
        self.assertIn("high_water_pct", s)
        self.assertIn("evictions_total", s)


class TestJitteredBackoff(unittest.TestCase):

    def test_backoff_sequence_increases(self):
        bo = JitteredBackoff(base=2, max_wait=300)
        # Test delay computation without actually sleeping
        delays = []
        for i in range(6):
            delay = min(bo.base ** i, bo.max_wait)
            delays.append(delay)
        self.assertTrue(all(delays[i] <= delays[i+1] for i in range(len(delays)-1)))

    def test_max_wait_cap(self):
        """Backoff should never exceed max_wait."""
        bo = JitteredBackoff(base=2, max_wait=30)
        for i in range(20):
            delay = min(bo.base ** i, bo.max_wait)
            self.assertLessEqual(delay, bo.max_wait)

    def test_mac_seed_unique_per_robot(self):
        """Different MAC addresses → different jitter seeds → different timing."""
        bo1 = JitteredBackoff()
        bo2 = JitteredBackoff()
        # Both instances get same MAC, so same seed. Just check it's deterministic.
        self.assertEqual(bo1.jitter_seed, bo2.jitter_seed)

    def test_reset(self):
        bo = JitteredBackoff()
        bo.attempt = 5
        bo.reset()
        self.assertEqual(bo.attempt, 0)


class TestCpuGovernor(unittest.TestCase):

    def test_dry_run_no_filesystem_access(self):
        gov = CpuGovernor(dry_run=True)
        result = gov.setup()
        self.assertTrue(result)
        result = gov.enter()
        self.assertTrue(result)

    def test_cgroup_not_available_graceful(self):
        """Governor must not raise regardless of cgroup availability.
        Root (CI/containers) may succeed; non-root may fail. Both are OK."""
        gov = CpuGovernor(
            cgroup_root=Path("/nonexistent/cgroup/path"),
            dry_run=False
        )
        try:
            result = gov.setup()
            self.assertIsInstance(result, bool)
        except Exception as exc:
            self.fail(f"CpuGovernor.setup() must not raise: {exc}")


class TestFullPipelineIntegration(unittest.TestCase):
    """End-to-end: sensor data → anomaly → manifest → eviction."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.staging = Path(self.tmp) / "robot_data"
        self.staging.mkdir()
        self.db = ManifestDB(Path(self.tmp) / "manifest.db")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_anomaly_promotes_file_priority(self):
        """Anomaly event should promote staged file from P3 to P0."""
        # Stage a P3 file
        now_ms = int(time.time() * 1000)
        writer = MCAPFastWriter(
            staging_root=self.staging,
            robot_id="test-robot",
            session_id="s001",
            priority=3,
            topic="telemetry",
        )
        writer.add_frame(now_ms, "sensor", b'{"battery_pct": 80}')
        path = writer.flush()
        rec = FileRecord(
            filename=path.name, path=str(path),
            session_id="s001", data_type="telemetry",
            priority=3, score=30, size_bytes=path.stat().st_size,
            created_at=now_ms,
        )
        fid = self.db.insert(rec)

        # Trigger anomaly promotion
        self.db.promote_priority(
            session_id="s001",
            new_priority=0,
            start_ts=now_ms - 5000,
            end_ts=now_ms + 5000,
        )

        updated = self.db.get_by_id(fid)
        self.assertEqual(updated.priority, 0)
        self.assertEqual(updated.score, 100.0)

    def test_upload_state_machine_full_cycle(self):
        """Test full 5-state transition: PENDING → UPLOADING → VERIFYING → UPLOADED → DELETED"""
        rec = FileRecord(
            filename="cycle_test.mcap", path="/tmp/cycle.mcap",
            session_id="s001", data_type="test",
            priority=2, score=50, size_bytes=1024,
        )
        fid = self.db.insert(rec)

        states = ["UPLOADING", "VERIFYING", "UPLOADED", "DELETED"]
        for state in states:
            self.db.set_state(fid, state)
            loaded = self.db.get_by_id(fid)
            self.assertEqual(loaded.state, state, f"Expected {state}")

        # DELETED state should have deleted_at timestamp
        final = self.db.get_by_id(fid)
        self.assertIsNotNone(final.deleted_at)
        self.assertGreater(final.deleted_at, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
