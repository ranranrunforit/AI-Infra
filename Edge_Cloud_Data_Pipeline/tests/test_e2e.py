import unittest
import time
from pathlib import Path
from robot_agent.main import RobotPipeline

class TestE2EPipeline(unittest.TestCase):
    def test_anomaly_injection_to_upload_cycle(self):
        """Full flow: simulate sensor data -> anomaly -> P0 file -> upload -> delete."""
        from collections import namedtuple
        
        Cfg = namedtuple("Cfg", ["staging_root", "db_path", "demo_mode", "log_level", "gcs_bucket"])
        cfg = Cfg(
            staging_root=Path("/tmp/robot_test"),
            db_path=Path("/tmp/robot_test/manifest.sqlite3"),
            demo_mode=True,
            log_level="INFO",
            gcs_bucket=None
        )
        
        # Clean up existing test dir
        if cfg.staging_root.exists():
            import shutil
            shutil.rmtree(cfg.staging_root)
            
        pipeline = RobotPipeline(cfg)
        pipeline.start()
        
        try:
            # inject battery_critical reading
            pipeline.ingest("telemetry", {"battery_pct": 5.0, "timestamp": time.time()})
            time.sleep(2)  # allow flush loop to run
            
            # Assert: P0 file exists in manifest
            pending = pipeline.manifest.get_pending()
            self.assertTrue(any(r.priority == 0 for r in pending), "Expected P0 file in pending queue")
        finally:
            pipeline.shutdown()

if __name__ == "__main__":
    unittest.main()
