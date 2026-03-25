"""
main.py — Cloud Run MCAP Indexer (v5 + VLM tagging)
Triggered by GCS ObjectFinalize via Eventarc.

Database backend support:
  DB_BACKEND=cloudsql     → psycopg2 over private VPC (Cloud SQL)  ← DEFAULT

Required Cloud Run env vars for Cloud SQL backend:
  DB_BACKEND        = cloudsql   (or omit — cloudsql is the default)
  POSTGRES_HOST     = /cloudsql/<project>:<region>:<instance>  (Unix socket) or private IP
  POSTGRES_DB       = robot_pipeline
  POSTGRES_USER     = pipeline
  POSTGRES_PASSWORD = <from Secret Manager>
  GCS_BUCKET        = robot-data-pipeline-cz78-demo  (must match robot agent config)

VLM tagging (optional — indexing works without it):
  GEMINI_API_KEY    = <from Secret Manager>  ← NEW
  If unset, VLM tagging is silently skipped. Indexing is never blocked by VLM errors.

Partial transfer in Cloud Run context:
  This function only reads the last 64KB (MCAP trailer) from GCS — it never
  transfers the full file.  So partial transfer here means: the 64KB range
  request itself fails.  We handle this via:
    1. Retry with exponential backoff on transient GCS errors
    2. Cloud Functions automatic retry on unhandled exception (via raise)
    3. Dead-letter Pub/Sub topic for permanent failures (configured in Terraform)
"""

import os, json, struct, logging, base64, time, contextlib, re
from datetime import datetime, timezone
from typing import Optional

import functions_framework
from google.cloud import storage as gcs


# ── VLM tagger (imported lazily so missing dep doesn't break indexing) ────────
try:
    import vlm_tagger
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── DB Backend Abstraction ────────────────────────────────────────────────────

DB_BACKEND = os.environ.get("DB_BACKEND", "cloudsql").lower()

try:
    import psycopg2
    from psycopg2 import pool as pg_pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.error("psycopg2 not installed — database writes disabled")

_pool: Optional[pg_pool.ThreadedConnectionPool] = None
GCS_CLIENT = None


def _gcs():
    global GCS_CLIENT
    if GCS_CLIENT is None:
        GCS_CLIENT = gcs.Client()
    return GCS_CLIENT


def _build_pool() -> pg_pool.ThreadedConnectionPool:
    """
    Build a connection pool appropriate for the configured backend.

    Cloud SQL: uses individual POSTGRES_* env vars (VPC private IP or Unix socket)
    """
    pg_host  = os.environ["POSTGRES_HOST"]
    is_socket = pg_host.startswith("/")
    kwargs = dict(
        host=pg_host,
        dbname=os.environ.get("POSTGRES_DB", "robot_pipeline"),
        user=os.environ.get("POSTGRES_USER", "pipeline"),
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=10,
        sslmode="disable" if is_socket else os.environ.get("POSTGRES_SSLMODE", "require"),
    )
    if not is_socket:
        kwargs["port"] = int(os.environ.get("POSTGRES_PORT", 5432))
    return pg_pool.ThreadedConnectionPool(1, 5, **kwargs)


@contextlib.contextmanager
def _db_conn():
    """Thread-safe connection acquisition with automatic return to pool."""
    global _pool
    if _pool is None:
        _pool = _build_pool()
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def _execute_with_retry(conn, sql: str, params: dict, max_retries: int = 3):
    """
    Execute SQL with retry logic for serialization errors (SQLSTATE 40001).
    Cloud SQL does not produce 40001 under normal conditions — the retry is
    a no-op for that backend.
    """
    for attempt in range(max_retries):
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
            return
        except psycopg2.errors.SerializationFailure:
            if attempt == max_retries - 1:
                raise
            conn.rollback()
            backoff = (2 ** attempt) * 0.1
            logger.warning("Serialization retry %d (backoff %.1fs)", attempt + 1, backoff)
            time.sleep(backoff)
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            logger.debug("Duplicate gcs_uri — already indexed")
            return


# ── MCAP Trailer Parser ────────────────────────────────────────────────────────

MCAP_MAGIC         = b"\x89MCAP\r\n\x1a\n"
TRAILER_READ_BYTES = 65_536


def read_mcap_trailer(bucket_name: str, object_name: str, file_size: int) -> dict:
    """
    Read only the last 64KB of the MCAP file.
    Retries up to 3 times on transient GCS errors before re-raising.
    """
    client = _gcs()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(object_name)

    read_start = max(0, file_size - TRAILER_READ_BYTES)

    last_exc = None
    for attempt in range(3):
        try:
            trailer = blob.download_as_bytes(start=read_start, end=file_size - 1)
            return parse_mcap_trailer(trailer, object_name, blob.metadata or {})
        except Exception as exc:
            last_exc = exc
            wait = (2 ** attempt) * 0.5
            logger.warning("GCS trailer read attempt %d failed: %s (retry in %.1fs)",
                           attempt + 1, exc, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed to read GCS trailer after 3 attempts: {last_exc}") from last_exc


def parse_mcap_trailer(trailer: bytes, object_name: str, blob_metadata: dict) -> dict:
    meta = {}
    try:
        search_area = trailer[:512]
        for start in range(0, min(len(search_area), 256), 4):
            chunk     = search_area[start:]
            brace_pos = chunk.find(b"{")
            if brace_pos >= 0:
                candidate = chunk[brace_pos:]
                end = candidate.find(b"\x00") or len(candidate)
                try:
                    parsed = json.loads(candidate[:end].decode("utf-8", errors="ignore").rstrip("\x00"))
                    if "session_id" in parsed or "robot_id" in parsed:
                        meta = parsed
                        break
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
    except Exception as exc:
        logger.warning("Trailer parse error: %s", exc)

    filename   = object_name.split("/")[-1]
    file_meta  = _parse_filename(object_name)

    # Layer 1: trailer JSON values override everything else if they exist
    merged             = {**file_meta, **blob_metadata, **meta}
    merged["gcs_uri"]  = f"gs://{os.environ.get('GCS_BUCKET', 'unknown')}/{object_name}"
    merged["filename"] = filename
    return merged


def _parse_filename(object_name: str) -> dict:
    path_parts = object_name.split("/")
    filename   = path_parts[-1]
    name  = filename.replace(".mcap.zst", "").replace(".mcap", "")
    parts = name.split("_")
    
    result = {
        "robot_id":       None,
        "session_id":     None,
        "priority":       3,
        "data_type":      "unknown",
        "schema_version": "unknown",
        "anomaly_flags":  [],
        "topics":         [],
    }
    
    # Layer 2: GCS path
    if len(path_parts) == 5:
        result["robot_id"] = path_parts[2]
        result["session_id"] = path_parts[3]

    try:
        # Layer 3 for session_id: parts[2]
        if len(parts) >= 3 and not result["session_id"]:
            result["session_id"] = parts[2]
            
        # Layer 3 for robot_id: filename prefix validation
        if not result["robot_id"] and len(parts) > 0:
            candidate = parts[0]
            if re.match(r'^[a-z0-9][a-z0-9\-]{2,}$', candidate):
                result["robot_id"] = candidate
                
        if len(parts) >= 4:
            result["priority"] = int(parts[3].replace("p", ""))
        if len(parts) >= 5:
            topic              = parts[4]
            result["data_type"] = topic
            result["topics"]   = [topic]
            if any(kw in topic for kw in ("anomaly", "collision", "overcurrent", "critical")):
                result["anomaly_flags"] = [topic]
        if len(parts) > 1:
            epoch_ms          = int(parts[1])
            result["start_ts"] = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()
    except (IndexError, ValueError) as exc:
        logger.debug("Filename parse partial failure: %s — %s", filename, exc)
        
    # Layer 4: Fallback
    if not result["robot_id"]:
        result["robot_id"] = "unknown"
    if not result["session_id"]:
        result["session_id"] = "unknown"
        
    return result


# ── PostgreSQL Insert ─────────────────────────────────────────────────────────

INSERT_SQL = """
    INSERT INTO robot_files
      (robot_id, session_id, gcs_uri, topics, time_start, time_end,
       anomaly_flags, schema_version, priority, size_bytes, firmware_ver,
       filename, data_type)
    VALUES
      (%(robot_id)s, %(session_id)s, %(gcs_uri)s, %(topics)s,
       %(time_start)s, %(time_end)s, %(anomaly_flags)s, %(schema_version)s,
       %(priority)s, %(size_bytes)s, %(firmware_ver)s,
       %(filename)s, %(data_type)s)
    ON CONFLICT (gcs_uri) DO UPDATE SET
      anomaly_flags  = EXCLUDED.anomaly_flags,
      time_end       = EXCLUDED.time_end,
      indexed_at     = NOW()
"""

# SQL to write VLM results back to the same row
UPDATE_EMBEDDING_SQL = """
    UPDATE robot_files
    SET    vlm_description = %(description)s,
           embedding       = %(embedding)s::vector
    WHERE  gcs_uri = %(gcs_uri)s
"""


def index_in_database(meta: dict, file_size: int):
    params = {
        "robot_id":       meta.get("robot_id", "unknown"),
        "session_id":     meta.get("session_id", "unknown"),
        "gcs_uri":        meta["gcs_uri"],
        "topics":         meta.get("topics", []),
        "time_start":     meta.get("start_ts") or meta.get("time_start"),
        "time_end":       meta.get("end_ts")   or meta.get("time_end"),
        "anomaly_flags":  meta.get("anomaly_flags", []),
        "schema_version": meta.get("schema_version", "unknown"),
        "priority":       int(meta.get("priority", 3)),
        "size_bytes":     file_size,
        "firmware_ver":   meta.get("firmware_ver", "unknown"),
        "filename":       meta.get("filename", ""),
        "data_type":      meta.get("data_type", "unknown"),
    }
    with _db_conn() as conn:
        _execute_with_retry(conn, INSERT_SQL, params)

    logger.info("Indexed: %s (robot=%s, session=%s, P%s, backend=%s)",
                meta.get("filename"), meta.get("robot_id"),
                meta.get("session_id"), meta.get("priority"), DB_BACKEND)


# ── Write VLM description + embedding to the already-inserted row ─────────────

def _store_embedding(gcs_uri: str, description: str, embedding: list):
    """
    UPDATE the row we just inserted with the VLM description and embedding vector.

    Called only when both description and embedding are non-None.
    Failure here is caught by the caller and logged — never raises.

    FIX #2: Removed the redundant conn.commit() that was called after the
    cursor.execute(). The _db_conn() context manager already commits on clean
    exit (__exit__). Calling commit() a second time was harmless on Cloud SQL
    but would raise an InterfaceError if the connection was in a broken state.
    """
    # pgvector expects a string like '[0.1,0.2,...]'
    embedding_str = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"

    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(UPDATE_EMBEDDING_SQL, {
                "description": description,
                "embedding":   embedding_str,
                "gcs_uri":     gcs_uri,
            })
        # No explicit conn.commit() here — _db_conn().__exit__ handles it.

    logger.info("VLM tagged: %s (%d-dim embedding)", gcs_uri.split("/")[-1], len(embedding))


# ── Cloud Function Entry Point ─────────────────────────────────────────────────

@functions_framework.cloud_event
def index_mcap_file(cloud_event):
    """
    Triggered by GCS ObjectFinalize via Eventarc.

    Partial-transfer safety:
      This function handles its own partial-read failures by retrying the GCS
      range request up to 3 times.  If all retries fail, we raise — Eventarc
      will retry with exponential backoff for up to 24 hours.  After that,
      the event goes to the dead-letter Pub/Sub topic defined in Terraform.

      There is NO partial-write risk here: the 64KB read is atomic from GCS's
      perspective (the object is finalized before Eventarc fires).

    VLM tagging:
      Runs AFTER index_in_database() so a VLM failure never blocks indexing.
      If GEMINI_API_KEY is absent or any Gemini call fails, we log a warning
      and continue — the row is indexed without an embedding.
    """
    data        = cloud_event.data
    bucket_name = data["bucket"]
    object_name = data["name"]
    file_size   = int(data.get("size", 0))

    if not (object_name.endswith(".mcap") or object_name.endswith(".mcap.zst")):
        return "skipped", 200

    logger.info("Indexing: gs://%s/%s (%d bytes, backend=%s)",
                bucket_name, object_name, file_size, DB_BACKEND)

    try:
        meta               = read_mcap_trailer(bucket_name, object_name, file_size)
        meta["size_bytes"] = file_size  # pass size into meta for VLM description
        index_in_database(meta, file_size)
    except Exception as exc:
        logger.error("Indexing failed for %s: %s", object_name, exc)
        raise  # triggers Eventarc retry → DLQ after max retries

    # ── VLM tagging (never raises, never blocks the response) ─────────────────
    if VLM_AVAILABLE:
        try:
            description, embedding = vlm_tagger.tag_file(meta)
            if description and embedding:
                _store_embedding(meta["gcs_uri"], description, embedding)
            elif description and not embedding:
                # Partial result: description succeeded but embedding failed.
                # Store the description alone so it's at least text-searchable.
                logger.warning("Storing description only (no embedding) for %s", object_name)
                with _db_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE robot_files SET vlm_description = %s WHERE gcs_uri = %s",
                            (description, meta["gcs_uri"]),
                        )
        except Exception as exc:
            # Log but do NOT re-raise — indexing already succeeded above
            logger.warning("VLM tagging failed for %s (non-fatal): %s", object_name, exc)
    # ──────────────────────────────────────────────────────────────────────────

    return "indexed", 200
