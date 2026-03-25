"""
manifest_db.py — SQLite Manifest (v4)
WAL + INCREMENTAL vacuum + page_size=4096
All file lifecycle states: PENDING → UPLOADING → VERIFYING → UPLOADED → DELETED
Plus: seq_gaps, bandwidth_log, metrics_buffer tables
"""

import sqlite3
import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DB_PATH = Path("/mnt/robot_data/meta/manifest.db")


@dataclass
class FileRecord:
    id: int = 0
    filename: str = ""
    path: str = ""
    session_id: str = ""
    data_type: str = ""
    priority: int = 2          # 0=critical, 1=high, 2=normal (P3 removed)
    score: float = 0.0         # 0–100 continuous score (v2 formula)
    size_bytes: int = 0
    checksum_sha256: str = ""
    state: str = "PENDING"     # PENDING|UPLOADING|VERIFYING|UPLOADED|DELETED|EVICTED|STUCK
    cloud_uri: str = ""
    cloud_etag: str = ""
    upload_session: str = ""   # resumable GCS session URI
    last_part: int = 0         # last completed chunk index
    retry_count: int = 0
    anomaly_flags: List[str] = field(default_factory=list)
    schema_version: str = "1.0.0"
    created_at: int = 0
    uploaded_at: int = 0
    deleted_at: int = 0
    grace_expires: int = 0     # unix_ms; DELETED blocked until this passes
    robot_id: str = "unknown"


class ManifestDB:
    """Thread-safe SQLite manifest with v4 pragmas."""

    VALID_STATES = frozenset(
        ["PENDING", "UPLOADING", "VERIFYING", "UPLOADED", "DELETED", "EVICTED", "STUCK", "ERROR"]
    )

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        self._conn().execute("PRAGMA wal_checkpoint(PASSIVE)")

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            db_exists = self.db_path.exists()
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            conn.row_factory = sqlite3.Row
            # ── v4 pragmas ──────────────────────────────────────────────────
            if not db_exists:
                conn.execute("PRAGMA page_size = 4096")            # match Linux page size → fewer fsync calls
            conn.execute("PRAGMA journal_mode = WAL")          # zero lock contention
            conn.execute("PRAGMA synchronous = NORMAL")        # WAL safe at NORMAL
            conn.execute("PRAGMA cache_size = -16000")         # 16MB page cache
            conn.execute("PRAGMA temp_store = MEMORY")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT UNIQUE NOT NULL,
                path            TEXT,
                session_id      TEXT,
                data_type       TEXT,
                priority        INTEGER DEFAULT 2,
                score           REAL DEFAULT 0.0,
                size_bytes      INTEGER DEFAULT 0,
                checksum_sha256 TEXT,
                state           TEXT DEFAULT 'PENDING',
                cloud_uri       TEXT,
                cloud_etag      TEXT,
                upload_session  TEXT,
                last_part       INTEGER DEFAULT 0,
                retry_count     INTEGER DEFAULT 0,
                anomaly_flags   TEXT DEFAULT '[]',
                schema_version  TEXT DEFAULT '1.0.0',
                firmware_ver    TEXT DEFAULT 'unknown',
                robot_id        TEXT,
                created_at      INTEGER,
                uploaded_at     INTEGER,
                deleted_at      INTEGER,
                grace_expires   INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_state      ON files(state);
            CREATE INDEX IF NOT EXISTS idx_priority   ON files(priority, state);
            CREATE INDEX IF NOT EXISTS idx_session    ON files(session_id);
            CREATE INDEX IF NOT EXISTS idx_created_at ON files(created_at);

            -- v2: gap detector (research report lacked this)
            CREATE TABLE IF NOT EXISTS seq_gaps (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                expected_seq INTEGER NOT NULL,
                detected_at INTEGER NOT NULL
            );

            -- daily bandwidth accounting
            CREATE TABLE IF NOT EXISTS bandwidth_log (
                day        TEXT PRIMARY KEY,
                bytes_sent INTEGER DEFAULT 0,
                cap_bytes  INTEGER DEFAULT 524288000,
                p0_bytes   INTEGER DEFAULT 0
            );

            -- offline Prometheus WAL (from report)
            CREATE TABLE IF NOT EXISTS metrics_buffer (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      INTEGER NOT NULL,
                metric  TEXT NOT NULL,
                labels  TEXT DEFAULT '{}',
                value   REAL NOT NULL,
                flushed INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_unflushed
                ON metrics_buffer(flushed, ts);
        """)
        conn.commit()
        try:
            conn.execute("ALTER TABLE bandwidth_log ADD COLUMN p0_bytes INTEGER DEFAULT 0")
            conn.commit()
        except sqlite3.OperationalError:
            pass
        logger.info("ManifestDB initialised at %s", self.db_path)

    # ── CRUD ────────────────────────────────────────────────────────────────

    def insert(self, rec: FileRecord) -> int:
        conn = self._conn()
        cur = conn.execute("""
            INSERT OR IGNORE INTO files
              (filename, path, session_id, data_type, priority, score,
               size_bytes, checksum_sha256, state, anomaly_flags,
               schema_version, firmware_ver, robot_id, created_at, grace_expires)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            rec.filename, rec.path, rec.session_id, rec.data_type,
            rec.priority, rec.score, rec.size_bytes, rec.checksum_sha256,
            rec.state, json.dumps(rec.anomaly_flags),
            rec.schema_version, "v4.0.0", rec.robot_id,
            rec.created_at or self._now_ms(),
            rec.grace_expires,
        ))
        conn.commit()
        rec.id = cur.lastrowid or 0
        return rec.id

    def get_by_id(self, file_id: int) -> Optional[FileRecord]:
        row = self._conn().execute(
            "SELECT * FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_filename(self, filename: str) -> Optional[FileRecord]:
        row = self._conn().execute(
            "SELECT * FROM files WHERE filename = ?", (filename,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def update_file_location(self, file_id: int, new_filename: str, new_path: str, new_size: int):
        conn = self._conn()
        conn.execute(
            "UPDATE files SET filename=?, path=?, size_bytes=? WHERE id=?",
            (new_filename, new_path, new_size, file_id)
        )
        conn.commit()

    def get_pending(self, limit: int = 50, min_priority: int = 0) -> List[FileRecord]:
        """Return pending files ordered by priority then score (highest first)."""
        rows = self._conn().execute("""
            SELECT * FROM files
            WHERE state = 'PENDING' AND priority >= ?
            ORDER BY priority ASC, score DESC
            LIMIT ?
        """, (min_priority, limit)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_pending_by_priority(self, limit: int = 50, priority: int = 0) -> List[FileRecord]:
        rows = self._conn().execute("""
            SELECT * FROM files
            WHERE state = 'PENDING' AND priority = ?
            ORDER BY score DESC
            LIMIT ?
        """, (priority, limit)).fetchall()
        return [self._row_to_record(r) for r in rows]


    def count_pending_by_priority(self, priority: int) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) FROM files WHERE state='PENDING' AND priority=?",
            (priority,)
        ).fetchone()
        return row[0] if row else 0

    def count_by_state_and_priority(self) -> dict:
        conn = self._conn()
        result = {}
        for state in ("PENDING", "UPLOADING", "UPLOADED", "EVICTED"):
            rows = conn.execute(
                "SELECT priority, COUNT(*) FROM files WHERE state=? GROUP BY priority",
                (state,)
            ).fetchall()
            for p, c in rows:
                result[f"{state}_p{p}"] = c
        return result

    def set_state_if_pending(self, file_id: int, new_state: str) -> bool:
        """Atomically transition a file from PENDING → new_state.
        Returns True if the update succeeded (file was PENDING),
        False if another worker already claimed it.
        """
        rows = self._conn().execute(
            "UPDATE files SET state=? WHERE id=? AND state='PENDING'",
            (new_state, file_id)
        ).rowcount
        self._conn().commit()
        return rows > 0

    def get_stuck(self) -> List[FileRecord]:
        rows = self._conn().execute("""
            SELECT * FROM files WHERE state = 'STUCK'
        """).fetchall()
        return [self._row_to_record(r) for r in rows]

    def set_state(self, file_id: int, state: str, **kwargs):
        assert state in self.VALID_STATES, f"Unknown state: {state}"
        conn = self._conn()
        fields = {"state": state}
        if state == "UPLOADED":
            fields["uploaded_at"] = self._now_ms()
        if state == "DELETED":
            fields["deleted_at"] = self._now_ms()
        fields.update(kwargs)

        set_clause = ", ".join(f"{k}=?" for k in fields)
        conn.execute(
            f"UPDATE files SET {set_clause} WHERE id=?",
            (*fields.values(), file_id),
        )
        conn.commit()

    def save_session(self, file_id: int, session_uri: str):
        self._conn().execute(
            "UPDATE files SET upload_session=? WHERE id=?",
            (session_uri, file_id),
        )
        self._conn().commit()

    def update_last_part(self, file_id: int, part: int):
        self._conn().execute(
            "UPDATE files SET last_part=? WHERE id=?", (part, file_id)
        )
        self._conn().commit()

    def increment_retry(self, file_id: int) -> int:
        conn = self._conn()
        conn.execute(
            "UPDATE files SET retry_count = retry_count + 1 WHERE id=?",
            (file_id,),
        )
        conn.commit()
        row = conn.execute(
            "SELECT retry_count FROM files WHERE id=?", (file_id,)
        ).fetchone()
        return row["retry_count"] if row else 0

    def promote_priority(self, session_id: str, new_priority: int,
                         start_ts: int, end_ts: int):
        """Promote files in a time window to higher priority (anomaly context)."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT id, path FROM files
            WHERE session_id=? AND created_at BETWEEN ? AND ?
              AND state='PENDING' AND priority > ?
        """, (session_id, start_ts, end_ts, new_priority)).fetchall()
        
        for r in rows:
            old_path = Path(r["path"])
            if old_path.exists():
                priority_dir = {0: "p0_critical", 1: "p1_high", 2: "p2_normal"}.get(new_priority, "p2_normal")
                new_path = old_path.parent.parent / priority_dir / old_path.name
                new_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.move(str(old_path), str(new_path))
                conn.execute(
                    "UPDATE files SET priority=?, score=100.0, path=? WHERE id=?",
                    (new_priority, str(new_path), r["id"])
                )
        conn.commit()

    def get_last_session_id(self) -> Optional[str]:
        """Fetch the most recent session ID from files to resume post-crash"""
        row = self._conn().execute(
            "SELECT session_id FROM files ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return row["session_id"] if row else None

    # ── Orphan recovery (from report — critical for kernel panic) ────────────

    def orphan_reconciliation(self, staging_root: Path) -> Dict[str, int]:
        """
        On every boot: scan disk vs manifest.
        - Files on disk but not in manifest → insert as PENDING (kernel panic orphans)
        - Rows in manifest but file missing → mark EVICTED (inconsistent ghost rows)
        """
        on_disk = {str(p) for p in staging_root.rglob("*.mcap")}
        on_disk |= {str(p) for p in staging_root.rglob("*.mcap.zst")}

        conn = self._conn()
        in_manifest = {
            row["path"]
            for row in conn.execute(
                "SELECT path FROM files WHERE state NOT IN ('DELETED','EVICTED')"
            ).fetchall()
        }

        orphans = on_disk - in_manifest
        ghosts = in_manifest - on_disk
        recovered = 0
        cleaned = 0

        for path in orphans:
            try:
                meta = _extract_meta_from_path(path)
                conn.execute("""
                    INSERT OR IGNORE INTO files
                      (filename, path, session_id, data_type, priority,
                       state, created_at, robot_id)
                    VALUES (?,?,?,?,?,'PENDING',?,?)
                """, (
                    meta["filename"], path, meta["session_id"],
                    meta["data_type"], meta["priority"],
                    meta["created_at"], "robot-demo",
                ))
                recovered += 1
                logger.warning("Orphan recovered: %s", path)
            except Exception as exc:
                logger.error("Failed to recover orphan %s: %s", path, exc)

        for path in ghosts:
            conn.execute(
                "UPDATE files SET state='EVICTED' WHERE path=?", (path,)
            )
            cleaned += 1

        conn.commit()
        logger.info(
            "Reconciliation complete: %d orphans recovered, %d ghosts cleaned",
            recovered, cleaned,
        )
        return {"orphans_recovered": recovered, "ghosts_cleaned": cleaned}

    # ── Bandwidth accounting ─────────────────────────────────────────────────

    def record_bytes(self, nbytes: int, is_p0: bool = False):
        today = time.strftime("%Y-%m-%d", time.gmtime())
        col = "p0_bytes" if is_p0 else "bytes_sent"
        conn = self._conn()
        conn.execute(f"""
            INSERT INTO bandwidth_log (day, {col}) VALUES (?, ?)
            ON CONFLICT(day) DO UPDATE SET {col} = {col} + excluded.{col}
        """, (today, nbytes))
        conn.commit()

    def bytes_today(self) -> int:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        row = self._conn().execute(
            "SELECT bytes_sent FROM bandwidth_log WHERE day=?", (today,)
        ).fetchone()
        return row["bytes_sent"] if row else 0

    def daily_cap(self) -> int:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        row = self._conn().execute(
            "SELECT cap_bytes FROM bandwidth_log WHERE day=?", (today,)
        ).fetchone()
        return row["cap_bytes"] if row else 524_288_000  # 500MB default

    # ── Sequence gap detection (v2 — research report lacked this) ────────────

    def record_gap(self, session_id: str, expected_seq: int):
        self._conn().execute(
            "INSERT INTO seq_gaps (session_id, expected_seq, detected_at) VALUES (?,?,?)",
            (session_id, expected_seq, self._now_ms()),
        )
        self._conn().commit()

    # ── Offline metrics buffer ────────────────────────────────────────────────

    def buffer_metric(self, metric: str, value: float, labels: Dict[str, str] = None):
        self._conn().execute(
            "INSERT INTO metrics_buffer (ts, metric, labels, value) VALUES (?,?,?,?)",
            (self._now_ms(), metric, json.dumps(labels or {}), value),
        )
        self._conn().commit()

    def drain_metrics(self, limit: int = 1000) -> List[Dict]:
        conn = self._conn()
        rows = conn.execute("""
            SELECT id, ts, metric, labels, value
            FROM metrics_buffer WHERE flushed=0
            ORDER BY ts ASC LIMIT ?
        """, (limit,)).fetchall()
        if rows:
            ids = [r["id"] for r in rows]
            conn.execute(
                f"UPDATE metrics_buffer SET flushed=1 WHERE id IN ({','.join('?'*len(ids))})",
                ids,
            )
            conn.commit()
        return [dict(r) for r in rows]

    # ── Eviction helpers ─────────────────────────────────────────────────────

    def get_evictable(self, min_priority: int = 1) -> List[FileRecord]:
        """Return PENDING files eligible for eviction (never P0)."""
        rows = self._conn().execute("""
            SELECT * FROM files
            WHERE state='PENDING' AND priority > ?
            ORDER BY priority DESC, created_at ASC
        """, (min_priority - 1,)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_uploaded_grace_expired(self) -> List[FileRecord]:
        rows = self._conn().execute("""
            SELECT * FROM files
            WHERE state='UPLOADED' AND grace_expires < ?
            ORDER BY priority DESC, created_at ASC
        """, (self._now_ms(),)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def incremental_vacuum(self, pages: int = 100):
        self._conn().execute(f"PRAGMA incremental_vacuum({pages})")
        self._conn().commit()

    def prune_old_metrics(self, days: int = 30):
        cutoff_ms = self._now_ms() - days * 86_400_000
        self._conn().execute(
            "DELETE FROM metrics_buffer WHERE ts < ? AND flushed = 1", (cutoff_ms,)
        )
        self._conn().execute(
            "DELETE FROM bandwidth_log WHERE day < ?",
            (time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400)),),
        )
        self._conn().commit()

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        conn = self._conn()
        counts = dict(conn.execute("""
            SELECT state, COUNT(*) FROM files GROUP BY state
        """).fetchall())
        return {
            "counts_by_state": counts,
            "bytes_today": self.bytes_today(),
            "daily_cap": self.daily_cap(),
        }

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FileRecord:
        d = dict(row)
        d["anomaly_flags"] = json.loads(d.get("anomaly_flags") or "[]")
        return FileRecord(**{k: v for k, v in d.items() if k in FileRecord.__dataclass_fields__})


def _extract_meta_from_path(path: str) -> Dict:
    """
    Parse filename convention:
    {ISO8601}_{epoch_ms}_{session}_{priority}_{topic}_{sizeKB}_{hash8}.mcap[.zst]
    Falls back gracefully on parse failure.
    """
    import re
    name = Path(path).name.replace(".mcap.zst", "").replace(".mcap", "")
    parts = name.split("_")
    try:
        return {
            "filename": Path(path).name,
            "created_at": int(parts[1]) if len(parts) > 1 else int(time.time() * 1000),
            "session_id": parts[2] if len(parts) > 2 else "unknown",
            "priority": min(int(parts[3].replace("p", "")), 2) if len(parts) > 3 else 2,
            "data_type": parts[4] if len(parts) > 4 else "unknown",
        }
    except (IndexError, ValueError):
        return {
            "filename": Path(path).name,
            "created_at": int(time.time() * 1000),
            "session_id": "recovered",
            "priority": 2,
            "data_type": "unknown",
        }