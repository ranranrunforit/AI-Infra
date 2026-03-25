"""
upload_agent.py — 5-State Upload FSM (v6 — Correct Resumable + Streaming CRC)

States: PENDING → UPLOADING → VERIFYING → UPLOADED → DELETED

v6 fixes over v5:
  1. create_session() uses ResumableUpload.initiate() with the REAL file stream
     seeked to start_byte — this returns a genuine HTTPS session URI.
  2. _stream_chunks() wraps every transmit_next_chunk() in urllib3.Retry for
     transparent retry on transient 408/429/503 errors.
  3. CRC32C is computed DURING the chunk upload loop (streaming), not by
     re-reading the 50MB file post-upload.  Halves disk I/O.
  4. verify_etag() compares the streaming CRC against GCS blob.crc32c.
  5. STUCK recovery: hourly retries for 24h, then CRITICAL metric.
"""

import os, io, time, random, hashlib, logging, threading, base64, struct
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from google.cloud import storage as gcs_storage
    from google.resumable_media.requests import ResumableUpload
    from google.auth.transport.requests import AuthorizedSession
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("google-cloud-storage not installed — upload disabled in production mode")

try:
    import google_crc32c
    CRC32C_AVAILABLE = True
except ImportError:
    CRC32C_AVAILABLE = False

try:
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False

from .manifest_db import ManifestDB, FileRecord
from .bandwidth_limiter import BandwidthLimiter, DailyCapExceeded

CHUNK_SIZE      = 8 * 1024 * 1024   # 8 MB — GCS minimum resumable chunk
MAX_RETRIES     = 10
GRACE_PERIOD_MS = 60 * 60 * 1000    # 1 h default (overridden from config)
SESSION_MAX_AGE = 23 * 3600         # GCS sessions expire after 24h; use 23h to be safe
STUCK_RETRY_INTERVAL = 3600         # 1 hour between STUCK retries
STUCK_MAX_AGE   = 24 * 3600         # After 24h mark permanently failed

# Retry policy for transient GCS errors during chunk upload
_RETRY_POLICY = None
if RETRY_AVAILABLE:
    _RETRY_POLICY = Retry(
        total=3,
        status_forcelist=[408, 429, 500, 502, 503],
        backoff_factor=0.5,
        allowed_methods=["PUT"],
    )


# ── Jittered Backoff ──────────────────────────────────────────────────────────

class JitteredBackoff:
    def __init__(self, base: float = 2.0, max_wait: float = 300.0):
        self.base = base
        self.max_wait = max_wait
        self.attempt = 0
        mac = _get_mac_int()
        self.jitter_seed = mac % 1000

    def wait(self):
        delay = min(self.base ** self.attempt, self.max_wait)
        jitter = random.uniform(0, delay * 0.3)
        total = delay + jitter + self.jitter_seed * 0.001
        logger.debug("Backoff wait %.2fs (attempt %d)", total, self.attempt)
        time.sleep(total)
        self.attempt += 1

    def reset(self):
        self.attempt = 0


def _get_mac_int() -> int:
    try:
        import uuid
        return uuid.getnode()
    except Exception:
        return random.randint(0, 999)


# ── GCS Resumable Upload ──────────────────────────────────────────────────────

class GCSUploader:
    """
    Manages GCS resumable uploads with real partial-transfer recovery
    and streaming CRC32C verification.

    Session lifecycle:
      create_session()  — initiates a real HTTPS resumable session on GCS;
                          returns the session URI (https://storage.googleapis.com/...)
      upload_file()     — streams file in CHUNK_SIZE pieces, honouring bandwidth
                          limiter; resumes from the last committed byte if the
                          session URI is already in the manifest.
                          Computes CRC32C inline during the chunk loop.
      verify_etag()     — compares the inline CRC against GCS blob metadata
    """

    UPLOAD_URL_TEMPLATE = (
        "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
        "?uploadType=resumable"
    )

    def __init__(self, bucket_name: str, prefix: str = "robot-data"):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self._client: Optional[gcs_storage.Client] = None
        self._authed_session: Optional[AuthorizedSession] = None

        if GCS_AVAILABLE:
            self._client = gcs_storage.Client()
            import google.auth
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
            )
            self._authed_session = AuthorizedSession(credentials)
            # Mount retry adapter for transient errors
            if _RETRY_POLICY and RETRY_AVAILABLE:
                adapter = HTTPAdapter(max_retries=_RETRY_POLICY)
                self._authed_session.mount("https://", adapter)

    def _blob_name(self, rec: FileRecord) -> str:
        pdir = {0: "p0_critical", 1: "p1_high", 2: "p2_normal"}.get(
            rec.priority, "p2_normal"
        )
        return f"{self.prefix}/{pdir}/{rec.robot_id}/{rec.session_id}/{rec.filename}"

    # ── Session management ─────────────────────────────────────────────────

    def create_session(self, rec: FileRecord, path: Path) -> str:
        """
        Initiate a GCS resumable upload session.
        Returns the real HTTPS session URI — this is what gets persisted to the
        manifest before byte 1 and used to resume after a crash.
        """
        if not GCS_AVAILABLE:
            return f"mock://session/{rec.filename}"

        blob_name = self._blob_name(rec)
        upload_url = self.UPLOAD_URL_TEMPLATE.format(bucket=self.bucket_name)
        metadata = {
            "name": blob_name,
            "metadata": {
                "robot_id":       rec.robot_id,
                "session_id":     rec.session_id,
                "priority":       str(rec.priority),
                "anomaly_flags":  str(rec.anomaly_flags),
                "schema_version": rec.schema_version,
            },
        }
        file_size = path.stat().st_size

        upload = ResumableUpload(upload_url, chunk_size=CHUNK_SIZE)
        # initiate() requires a seekable stream at byte 0 and total_bytes
        with open(path, "rb") as f:
            upload.initiate(
                self._authed_session,
                f,
                metadata,
                content_type="application/octet-stream",
                total_bytes=file_size,
                stream_final=False,
            )
        return upload.resumable_url  # real HTTPS URI

    # ── Upload with streaming CRC32C ───────────────────────────────────────

    def upload_file(self, rec: FileRecord, path: Path,
                    bw: BandwidthLimiter,
                    on_progress=None) -> str:
        """
        Upload file to GCS, resuming from the last committed byte if a session
        URI is already in the manifest (crash recovery path).

        Returns the base64-encoded CRC32C of the uploaded file (computed inline).
        """
        if not GCS_AVAILABLE:
            logger.info("[MOCK] Would upload %s to GCS", path.name)
            time.sleep(0.1)
            return hashlib.md5(path.read_bytes()).hexdigest()[:16] if path.exists() else "mock_etag"

        blob_name = self._blob_name(rec)
        file_size = path.stat().st_size

        # ── Determine resume offset ────────────────────────────────────────
        start_byte = 0
        session_uri = rec.upload_session

        if session_uri and session_uri.startswith("https://"):
            resume_offset = self._query_resume_offset(session_uri, file_size)
            if resume_offset is None:
                # Session expired or invalid — start fresh
                logger.warning("GCS session expired for %s — restarting upload", rec.filename)
                session_uri = self.create_session(rec, path)
            elif resume_offset == file_size:
                # Already fully uploaded (duplicate trigger)
                logger.info("File already fully uploaded: %s", rec.filename)
                blob = self._client.bucket(self.bucket_name).blob(blob_name)
                blob.reload()
                return blob.crc32c or blob.etag or ""
            else:
                start_byte = resume_offset
                logger.info("Resuming upload of %s from byte %d / %d (%.1f%%)",
                            rec.filename, start_byte, file_size,
                            100 * start_byte / max(file_size, 1))
        else:
            # No existing session — create one
            session_uri = self.create_session(rec, path)

        # ── Stream upload with inline CRC32C ───────────────────────────────
        crc32c_hash = self._stream_chunks(
            blob_name, path, file_size, start_byte, session_uri, bw,
            priority=rec.priority, on_progress=on_progress,
        )

        return crc32c_hash

    def _stream_chunks(self, blob_name: str, path: Path, file_size: int,
                       start_byte: int, session_uri: str, bw: BandwidthLimiter,
                       priority: int = 3, on_progress=None) -> str:
        """
        Low-level chunked upload using google.resumable_media.
        Computes CRC32C inline during the upload loop.

        urllib3.Retry is mounted on the AuthorizedSession adapter — each
        transmit_next_chunk() automatically retries on 408/429/500/502/503.
        """
        upload_url = self.UPLOAD_URL_TEMPLATE.format(bucket=self.bucket_name)
        upload = ResumableUpload(upload_url, chunk_size=CHUNK_SIZE)

        # Streaming CRC32C — computed over every byte as we upload
        crc = google_crc32c.Checksum() if CRC32C_AVAILABLE else None

        with open(path, "rb") as f:
            # If resuming, we need to feed the CRC the bytes we're skipping
            if crc and start_byte > 0:
                bytes_read = 0
                while bytes_read < start_byte:
                    to_read = min(65536, start_byte - bytes_read)
                    chunk = f.read(to_read)
                    if not chunk:
                        break
                    crc.update(chunk)
                    bytes_read += len(chunk)
            else:
                f.seek(start_byte)

            # Wire up the ResumableUpload to the existing session
            upload._resumable_url = session_uri
            upload._bytes_uploaded = start_byte
            upload._total_bytes = file_size
            upload._stream = f

            chunk_num = start_byte // CHUNK_SIZE
            while not upload.finished:
                # Bandwidth limiter — P0 bypasses
                remaining = file_size - upload.bytes_uploaded
                bw.wait_for_token(min(CHUNK_SIZE, remaining), priority=priority)

                # Read the next chunk and feed CRC before transmitting
                pos_before = f.tell()
                next_chunk_data = f.read(min(CHUNK_SIZE, remaining))
                if crc and next_chunk_data:
                    crc.update(next_chunk_data)
                f.seek(pos_before)  # seek back — transmit_next_chunk reads from stream

                # Transmit — urllib3.Retry handles transient retries transparently
                upload.transmit_next_chunk(self._authed_session)
                chunk_num += 1
                if on_progress:
                    on_progress(chunk_num, upload.bytes_uploaded)

        # Return base64-encoded CRC32C for verification
        if crc:
            crc_bytes = crc.digest()
            return base64.b64encode(crc_bytes).decode("ascii")
        return ""

    # ── Resume offset query ────────────────────────────────────────────────

    def _query_resume_offset(self, session_uri: str, file_size: int) -> Optional[int]:
        """
        Query GCS for how many bytes have been committed to a resumable session.

        Protocol (RFC 7233 + GCS docs):
          PUT {session_uri}
          Content-Range: bytes */{file_size}
          Content-Length: 0

        Responses:
          308 Resume Incomplete  → Range: 0-{N} means N+1 bytes committed
          200/201 Complete       → all bytes committed
          404/410 Not Found      → session expired; must restart
        """
        if not GCS_AVAILABLE:
            return 0

        try:
            resp = self._authed_session.put(
                session_uri,
                headers={
                    "Content-Range": f"bytes */{file_size}",
                    "Content-Length": "0",
                },
                timeout=15,
            )
            if resp.status_code in (200, 201):
                return file_size  # fully uploaded
            elif resp.status_code == 308:
                range_header = resp.headers.get("Range", "")
                if range_header:
                    last_byte = int(range_header.split("-")[-1])
                    return last_byte + 1
                return 0  # no bytes committed yet
            elif resp.status_code in (404, 410):
                return None  # session expired
            else:
                logger.warning("Unexpected GCS resume query response: %d", resp.status_code)
                return None
        except Exception as exc:
            logger.warning("Resume offset query failed: %s", exc)
            return None

    # ── CRC32C verification ────────────────────────────────────────────────

    def verify_crc32c(self, local_crc_b64: str, rec: FileRecord) -> bool:
        """
        Verify upload integrity by comparing inline CRC32C against GCS blob metadata.
        No re-read of the file — CRC was computed during upload.
        """
        if not GCS_AVAILABLE or not local_crc_b64:
            return True

        try:
            blob_name = self._blob_name(rec)
            blob = self._client.bucket(self.bucket_name).blob(blob_name)
            blob.reload()
            cloud_crc_b64 = blob.crc32c

            if not cloud_crc_b64:
                logger.warning("GCS blob has no CRC32C metadata — skipping verification")
                return True

            match = (local_crc_b64 == cloud_crc_b64)
            if not match:
                logger.error("CRC32C mismatch for %s: local=%s cloud=%s",
                             rec.filename, local_crc_b64, cloud_crc_b64)
            return match
        except Exception as exc:
            logger.error("CRC32C verification error: %s", exc)
            return False


# ── Upload FSM ────────────────────────────────────────────────────────────────

class UploadAgent(threading.Thread):
    """
    Continuously dequeues PENDING files from manifest, uploads to GCS,
    verifies CRC32C, transitions through 5-state FSM.

    P0 uploads bypass the pause gate — they always run even under CPU pressure.
    """

    def __init__(self, manifest: ManifestDB, bw: BandwidthLimiter,
                 gcs_bucket: str, staging_root: Path,
                 poll_interval: float = 5.0,
                 grace_period_secs: int = 3600,
                 metrics=None):
        super().__init__(daemon=True, name="upload-agent")
        self.manifest        = manifest
        self.bw              = bw
        self.staging_root    = staging_root
        self.poll_interval   = poll_interval
        self.grace_period_secs = grace_period_secs
        self.metrics         = metrics
        self.uploader        = GCSUploader(gcs_bucket)
        self.backoff         = JitteredBackoff()
        self._stop           = threading.Event()
        self._paused         = threading.Event()
        self._paused.set()   # initially running

    def run(self):
        logger.info("UploadAgent started")
        while not self._stop.is_set():
            try:
                # P0 always drains regardless of pause state
                self._process_p0()
                # P1-P3 only when not paused
                if self._paused.is_set():
                    self._process_batch(min_priority=1)
                # Hourly STUCK recovery
                self._retry_stuck()
            except Exception as exc:
                logger.error("UploadAgent error: %s", exc)
            self._stop.wait(timeout=self.poll_interval)

    def _process_p0(self):
        p0_files = self.manifest.get_pending_by_priority(limit=10, priority=0)
        for rec in p0_files:
            if self._stop.is_set():
                return
            try:
                self._upload_one(rec)
                self.backoff.reset()
            except DailyCapExceeded:
                pass  # P0 bypasses daily cap — this should never fire for P0
            except Exception as exc:
                self._handle_failure(rec, exc)

    def _process_batch(self, min_priority: int = 1, batch_size: int = 8):
        """Upload P1-P2 files. Worker count adapts to measured bandwidth:
        - slow links (< 1 Mbps): 1 worker — parallel streams hurt on narrow pipes
        - medium (1-2 Mbps):     2 workers
        - fast (> 2 Mbps):       4 workers
        This keeps UPLOADING count sane and prevents 60-80s latency ballooning.
        """
        pending = self.manifest.get_pending(limit=batch_size, min_priority=min_priority)
        if not pending:
            return

        # Claim files atomically before spawning workers
        claimed = []
        for rec in pending:
            if self._stop.is_set():
                break
            try:
                if self.manifest.set_state_if_pending(rec.id, "UPLOADING"):
                    claimed.append(rec)
            except Exception:
                claimed.append(rec)  # fallback: let _upload_one handle it
        if not claimed:
            return

        # Adaptive worker count
        bw_mbps = self.bw.stats().get("rate_mbps", 1.0) if self.bw else 1.0
        if bw_mbps < 1.0:
            num_workers = 1
        elif bw_mbps < 2.0:
            num_workers = 2
        else:
            num_workers = 4
        num_workers = min(num_workers, len(claimed))

        if num_workers == 1:
            # Single-threaded path — simpler, no thread overhead
            for rec in claimed:
                if self._stop.is_set():
                    return
                try:
                    self._upload_one(rec)
                    self.backoff.reset()
                except DailyCapExceeded as exc:
                    logger.info("Daily cap reached: %s", exc)
                    now = time.time()
                    self._stop.wait(timeout=((now // 86400 + 1) * 86400) - now)
                    return
                except Exception as exc:
                    self._handle_failure(rec, exc)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def _worker(rec):
                try:
                    self._upload_one(rec)
                    self.backoff.reset()
                    return None
                except Exception as exc:
                    return (rec, exc)
            with ThreadPoolExecutor(max_workers=num_workers,
                                    thread_name_prefix="upload-worker") as pool:
                futures = {pool.submit(_worker, rec): rec for rec in claimed}
                for fut in as_completed(futures):
                    result = fut.result()
                    if result is not None:
                        self._handle_failure(result[0], result[1])

    def _handle_failure(self, rec: FileRecord, exc: Exception):
        retries = self.manifest.increment_retry(rec.id)
        logger.warning("Upload failed (attempt %d): %s — %s", retries, rec.filename, exc)
        if retries >= MAX_RETRIES:
            self.manifest.set_state(rec.id, "STUCK")
            logger.error("File STUCK after %d retries: %s", MAX_RETRIES, rec.filename)
            if self.metrics:
                self.metrics.record_stuck(rec.priority)
        else:
            self.manifest.set_state(rec.id, "PENDING")
            self.backoff.wait()

    def _retry_stuck(self):
        """
        STUCK recovery: retry files hourly for 24h.
        After 24h unresolved → emit CRITICAL metric.
        """
        stuck = self.manifest.get_stuck()
        now_ms = int(time.time() * 1000)
        for rec in stuck:
            age_secs = (now_ms - rec.created_at) / 1000 if rec.created_at else 0
            if age_secs > STUCK_MAX_AGE:
                logger.critical("File permanently STUCK (>24h): %s", rec.filename)
                if self.metrics:
                    self.metrics.record_stuck_critical(rec.priority)
                continue
            # Only retry once per STUCK_RETRY_INTERVAL
            if rec.uploaded_at and (now_ms - rec.uploaded_at) < STUCK_RETRY_INTERVAL * 1000:
                continue
            logger.info("Retrying STUCK file: %s (age=%.1fh)", rec.filename, age_secs / 3600)
            self.manifest.set_state(rec.id, "PENDING")

    def _upload_one(self, rec: FileRecord):
        t_start = time.time()
        path = Path(rec.path)

        if not path.exists():
            logger.warning("File missing: %s → EVICTED", rec.path)
            self.manifest.set_state(rec.id, "EVICTED")
            return

        # ── Pre-upload integrity check (silent eMMC bit-rot guard) ─────────
        try:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if rec.checksum_sha256 and actual_hash != rec.checksum_sha256:
                self.manifest.set_state(rec.id, "CORRUPTED")
                logger.error("eMMC corruption detected before upload: %s", rec.filename)
                return
        except OSError as exc:
            logger.error("Cannot read file for checksum: %s — %s", rec.filename, exc)
            return

        # ── UPLOADING ──────────────────────────────────────────────────────
        self.manifest.set_state(rec.id, "UPLOADING")

        # Persist session URI BEFORE byte 1 — crash recovers from here.
        if not (rec.upload_session and rec.upload_session.startswith("https://")):
            session_uri = self.uploader.create_session(rec, path)
            self.manifest.save_session(rec.id, session_uri)
            rec.upload_session = session_uri

        logger.info("Uploading: %s (P%d, %.1f KB, resume=%s)",
                    rec.filename, rec.priority, rec.size_bytes / 1024,
                    bool(rec.upload_session))

        cloud_crc = self.uploader.upload_file(
            rec, path, self.bw,
            on_progress=lambda part, _: self.manifest.update_last_part(rec.id, part),
        )

        # ── VERIFYING ─────────────────────────────────────────────────────
        self.manifest.set_state(rec.id, "VERIFYING", cloud_etag=cloud_crc)
        if not self.uploader.verify_crc32c(cloud_crc, rec):
            self.manifest.set_state(rec.id, "ERROR")
            logger.error("CRC32C mismatch after upload — integrity failure: %s", rec.filename)
            return

        # ── UPLOADED ───────────────────────────────────────────────────────
        grace_ms = int(time.time() * 1000) + self.grace_period_secs * 1000
        self.manifest.set_state(rec.id, "UPLOADED", grace_expires=grace_ms)

        latency = time.time() - t_start
        logger.info("✓ Uploaded & verified: %s (%.1fs)", rec.filename, latency)
        if self.metrics:
            self.metrics.record_upload(rec, latency)

    def pause(self):
        """Pause P1-P3 uploads under CPU pressure. P0 always continues."""
        self._paused.clear()
        logger.info("UploadAgent P1-P3 uploads paused (CPU pressure)")

    def resume(self):
        self._paused.set()
        logger.info("UploadAgent resumed")

    def stop(self):
        self._stop.set()
        self._paused.set()