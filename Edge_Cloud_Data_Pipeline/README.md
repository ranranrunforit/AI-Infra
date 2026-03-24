# Edge-to-Cloud Data Pipeline
### Production-Grade Telemetry Infrastructure + Semantic Search

> A mobile robot generates 60Hz sensor data on a Jetson-class SoC with a weak 4G connection, 80GB of flash storage, and a CPU budget of 20%. It just crashed. You need the sensor data from the 10 seconds before impact — reliably, automatically, every time. This system solves that problem end-to-end, from byte capture on the edge to natural-language search in the cloud.

---

## Table of Contents

1. [The Problem This Solves](#1-the-problem-this-solves)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Edge Agent: Data Capture & Processing](#3-edge-agent-data-capture--processing)
4. [Edge Agent: Upload & Bandwidth Control](#4-edge-agent-upload--bandwidth-control)
5. [Edge Agent: State Management & Storage](#5-edge-agent-state-management--storage)
6. [Cloud Infrastructure](#6-cloud-infrastructure)
7. [Vector DB + VLM Tagging (v5 New Feature)](#7-vector-db--vlm-tagging-v5-new-feature)
8. [Observability: Prometheus + Grafana](#8-observability-prometheus--grafana)
9. [Demo Walkthrough](#9-demo-walkthrough)
10. [Architecture Decisions & Tradeoffs](#10-architecture-decisions--tradeoffs)
11. [Project Structure](#11-project-structure)
12. [Quick Start](#12-quick-start)

---

## 1. The Problem This Solves

Most robotics data pipelines are designed for ideal conditions — reliable WiFi, idle CPUs, and disposable flash storage. Real warehouse and field robots operate under four hard constraints simultaneously:

| Constraint | Requirement | Why It Matters |
|---|---|---|
| **CPU** | ≤ 20% of 1 core | 80% must remain for SLAM / navigation |
| **Bandwidth** | ≤ 50 Mbps total | Physical 4G NIC ceiling |
| **Storage** | ≤ 80GB quota | eMMC flash wears out; random writes kill it |
| **Thermal** | Halt uploads at ≥ 85°C | Network I/O adds heat; SLAM drift kills the robot |

Beyond the constraints, there is a data *retrieval* problem: once 6,000+ recordings are in cloud storage, finding the right one requires either manual bucket scanning (minutes to hours) or writing exact SQL filters. Neither works for open-ended investigation like "find all sessions where the robot seemed stuck near an obstacle."

**v5 solves both problems:**
- The edge pipeline captures, compresses, prioritizes, and uploads data under all four constraints with zero data loss
- The cloud layer automatically tags every file with a semantic description and 1536-dimensional embedding, making the entire archive searchable in plain English

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EDGE (Robot)                                │
│                                                                     │
│  Sensors (60Hz)                                                     │
│  LiDAR / Camera / IMU / Logs                                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐    ┌──────────────────┐                           │
│  │ CPU Governor │    │ Anomaly Detector │                           │
│  │ cgroups 20% │    │ 3-layer engine   │──── P0 Alert ──► Pub/Sub  │
│  └──────┬──────┘    └────────┬─────────┘                           │
│         │                   │ priority boost                       │
│         ▼                   ▼                                       │
│  ┌─────────────────────────────────┐                               │
│  │  Ring Buffer (256MB RAM)        │                               │
│  │  Dual trigger: 25MB OR 15s      │                               │
│  │  3-stage backpressure           │                               │
│  └────────────────┬────────────────┘                               │
│                   ▼                                                 │
│  ┌─────────────────────────────────┐                               │
│  │  MCAP FastWriter                │                               │
│  │  Phase 1: hash-free fast write  │                               │
│  │  Phase 2: Zstd compress + index │                               │
│  └────────────────┬────────────────┘                               │
│                   ▼                                                 │
│  ┌─────────────────────────────────┐                               │
│  │  Manifest DB (SQLite WAL)       │                               │
│  │  Single source of truth         │                               │
│  │  PENDING→UPLOADING→UPLOADED     │                               │
│  └────────────────┬────────────────┘                               │
│                   ▼                                                 │
│  ┌─────────────────────────────────┐                               │
│  │  Bandwidth Limiter              │                               │
│  │  Gate 1: Thermal cutoff 85°C    │                               │
│  │  Gate 2: MMCF cost function     │                               │
│  │  Gate 3: 50Mbps token bucket    │                               │
│  └────────────────┬────────────────┘                               │
│                   │                                                 │
└───────────────────┼─────────────────────────────────────────────────┘
                    │ HTTPS Resumable Upload + Streaming CRC32C
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         GOOGLE CLOUD                                │
│                                                                     │
│  Google Cloud Storage (GCS)                                         │
│  p0_critical/ p1_high/ p2_normal/                                  │
│       │                                                             │
│       │ Eventarc (ObjectFinalize)                                   │
│       ▼                                                             │
│  Cloud Run: mcap-indexer                                            │
│  ├── Reads last 64KB trailer only                                   │
│  ├── Parses robot_id, session, anomaly_flags, timestamps            │
│  ├── INSERT into robot_files (PostgreSQL)                           │
│  └── VLM tagging (NEW v5)                                           │
│      ├── Gemini 3 Flash → description text                          │
│      └── gemini-embedding-001 → 1536-dim vector                    │
│           │                                                         │
│           ▼                                                         │
│  Cloud SQL PostgreSQL                                               │
│  ├── robot_files (+ embedding VECTOR(1536))                        │
│  ├── IVFFlat cosine index                                           │
│  └── Semantic search via pgvector <=> operator                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Edge Agent: Data Capture & Processing

### 3.1 CPU Governor (`cpu_governor.py`)

The pipeline uses Linux kernel CFS (Completely Fair Scheduler) hard enforcement, not `nice` values. `nice` still burns CPU when the system is idle, which causes thermal throttling and SLAM drift.

```
cfs_period_us = 100000   # 100ms window
cfs_quota_us  = 20000    # 20ms allowed = 20% of 1 core
```

This is a hard ceiling. No matter how many threads the pipeline spawns, the kernel enforces the total. When CPU pressure is extreme, the Governor suspends lower-priority threads in order: background compression first, bulk P2 uploads second. P0 emergency data is never suspended.

Additional OS-level protections:
- `log2ram` deployed via init script — system logs written to RAM, flushed periodically, never directly to eMMC
- `journald` configured RAM-only mode
- `systemd` unit with `WatchdogSec=15` — dead-lock auto-restart in 15 seconds

### 3.2 Codec Pipeline (`codec_utils.py`)

Before data enters the ring buffer, large channels are pre-compressed:

**Camera (H.265 Short-GOP Encoding)**
Hardware-accelerated via ffmpeg NVENC or VAAPI. The critical constraint: `GOP=30` (one I-frame per second). Why? The system must be able to extract 10 seconds of footage before a collision. A long GOP means the first several seconds have no I-frame and cannot be decoded. Short GOP guarantees every second is independently decodable. Compression ratio vs raw frames: ~50:1.

**LiDAR (Bit-Shuffle + Zstd)**
Standard Zstd on raw float32 point cloud data achieves ~1.5:1. This is because spatial coordinates have no predictable byte patterns. The fix: bit-shuffle the data first — rearrange all sign bits together, all exponent bits together, all mantissa bits together. This creates long runs of similar bytes that Zstd compresses efficiently. Result: ~4:1 lossless compression on LiDAR data.

### 3.3 Ring Buffer & Backpressure (`ring_buffer.py`)

The ring buffer holds up to 256MB of active sensor data in RAM. It triggers a flush to disk when either threshold is hit first:
- **Volume:** 25MB accumulated
- **Time:** 30 seconds elapsed (ensures maximum data staleness)

**Three-Stage Adaptive Backpressure:**

| State | Condition | Action |
|---|---|---|
| Normal | Queue < 200MB | 30s rotation threshold |
| Adaptive | Queue > 500MB | Rotation extended to 120s, reduces SQLite fragmentation from many small files |
| Shed | Queue > 800MB AND disk > 70% | Silently discard incoming low-priority data — it would be evicted before upload anyway |

The shedding logic is intentional: low-priority telemetry data that cannot be stored or uploaded has negative value. Writing it wastes I/O cycles and accelerates flash wear.

### 3.4 Three-Layer Anomaly Detection (`anomaly_detector.py`)

Three complementary detection methods run in parallel on every incoming data frame:

**Layer 1: YAML Rule Engine (Zero-latency)**
Rules are defined in `config/anomaly_rules.yaml` as human-readable expressions. The engine uses `simpleeval` — a sandboxed AST parser — instead of `eval()` or `pickle`. No injection surface. Rules carry individual debounce timers to prevent alert storms (e.g., "battery low" firing 600 times per minute). Each rule also specifies `pre_sec` and `post_sec` — how many seconds before and after the event to retain at elevated priority.

```yaml
# Example rule
- id: battery_critical
  expr: "battery_level < 10"
  priority: 0
  pre_sec: 30
  post_sec: 0
  debounce_sec: 60
```

**Layer 2: Z-Score Sliding Window (Rate-of-change detection)**
Evaluates deviation from a 30-second rolling baseline. This detects *trends*, not just thresholds. A motor temperature rising from 60°C to 78°C over 5 seconds triggers the detector even if it never crossed the absolute 85°C threshold. Static threshold rules miss this entirely.

**Layer 3: Multivariate Online Isolation Forest**
Runs on multi-dimensional feature vectors (e.g., torque × current × temperature simultaneously). Completely online — it trains on the first 512 samples after startup and continuously updates. No offline model, no serialization, no pickle vulnerabilities. Detects complex cross-sensor anomalies that single-variable rules cannot express.

When any layer fires, the detector dynamically re-prioritizes the ring buffer frames around the event window, ensuring pre- and post-event data is written to disk and uploaded at the appropriate priority level.

---

## 4. Edge Agent: Upload & Bandwidth Control

### 4.1 The Three-Gate Bandwidth Controller (`bandwidth_limiter.py`)

Every upload attempt must pass three gates in sequence. P0 data bypasses gates 1 and 2 entirely.

```
File queued for upload
        │
        ▼
   Is P0 (critical)?
   ├── YES → skip to Token Bucket
   └── NO  ▼
        Gate 1: Thermal Cutoff
        CPU temp ≥ 85°C?
        ├── YES → BLOCK (suspend all P1-P2 uploads)
        └── NO  ▼
        Gate 2: MMCF Cost Function
        Cost = (20 × bandwidth_usage) + (1 × cpu_temp_C) + (10 × priority_level)
        Cost > 100?
        ├── YES → BACKOFF (wait 0.5s, recalculate)
        └── NO  ▼
        Gate 3: Token Bucket
        50 Mbps physical ceiling
        └── Upload chunk (8MB) via GCS Resumable Upload
```

**MMCF worked examples:**
- P2 at idle (50°C, 0% BW): `20×0 + 1×50 + 10×2 = 70` → allowed
- P2 at 50% BW, 70°C: `20×0.5 + 1×70 + 10×2 = 100` → borderline
- P1 at 80°C, 0% BW: `20×0 + 1×80 + 10×1 = 90` → allowed
- Any P1-P2 at ≥ 85°C: thermal cutoff fires first regardless of MMCF score

### 4.2 Upload Agent & Resumable Upload (`upload_agent.py`)

The upload agent implements true GCS Resumable Upload — not a wrapper around a single HTTP PUT.

**Standard (fragile) approach:**
```
PUT entire_file → success or retry from zero
```

**This system:**
```
POST /upload/resumable → receive session_uri (persisted to SQLite)
loop:
    GET session_uri → "308 Resume Incomplete, Range: 0-{n}"
    f.seek(n+1)
    PUT chunk (8MB) with Content-Range header
    streaming CRC32C.update(chunk) as bytes flow
until 200 OK
```

The session URI is persisted in the Manifest DB before the first byte is sent. If the robot loses power mid-upload at byte 56MB, on restart it queries the session URI, GCS responds with the committed byte offset, and upload resumes from exactly that byte. This is hardware-level power loss protection, not just process crash recovery.

**Streaming CRC32C:** The checksum is computed *as bytes flow*, using `google_crc32c.Checksum.update()`. This eliminates the "upload then re-read the file to verify ETag" pattern that doubles I/O on large files.

### 4.3 5-Stage State Machine

Every file's lifecycle is tracked in SQLite:

```
PENDING → UPLOADING → VERIFYING → UPLOADED → DELETED
                                              ↑
                                           EVICTED (if disk pressure)
                                              ↑
                                           STUCK (after 10 failed retries)
```

STUCK files are re-activated every 1 hour automatically. After 24 hours of STUCK state, a `CRITICAL` alert is pushed to the monitoring dashboard for human intervention.

---

## 5. Edge Agent: State Management & Storage

### 5.1 SQLite Manifest DB (`manifest_db.py`)

The manifest database is the single source of truth for all file state. It uses WAL (Write-Ahead Log) mode with page size forced to 4096 bytes (matching the OS page size) for minimal I/O amplification.

**Orphan reconciliation on startup:** When the robot hard-reboots, the engine reconciles disk state with DB state:
- Files on disk but not in DB → re-imported as PENDING
- Files in DB but missing from disk → marked EVICTED

This prevents phantom uploads and ensures no file is silently lost across power cycles.

**Offline metrics buffer:** The SQLite DB also stores Prometheus metrics locally when the network is unavailable. Metrics are pushed to the remote endpoint when connectivity resumes, preserving observability continuity through outages.

### 5.2 Eviction Manager — Dual Watermark (`eviction_manager.py`)

When local storage exceeds 75% of the 80GB quota, a cascade begins. It stops only when storage drops below 60%.

The cascade executes in order, stopping as soon as the 60% target is reached:

**Step 1:** Delete files in UPLOADED state that are older than the 1-hour grace period. These are safe to remove — they exist in GCS.

**Step 2:** Run `incremental_vacuum` on the SQLite database. Deleted rows leave gaps; this reclaims that space.

**Step 3:** Apply Zstd dictionary compression to all PENDING P2 files. A pre-trained dictionary on structured telemetry data achieves 60–70% additional size reduction on data that is already Zstd-compressed without a dictionary.

**Step 4:** Delete PENDING files by priority, lowest first. P2 first, then P1. **P0 data is never deleted by the eviction manager under any circumstances.** If only P0 data remains and storage is still over the limit, a CRITICAL human alert is sent.

---

## 6. Cloud Infrastructure

### 6.1 Terraform IaC (`cloud/terraform/`)

The entire cloud infrastructure is defined as code and deployable with `terraform apply`:

- **GCS bucket** with lifecycle rules: non-critical data archived to Nearline at 30 days, Coldline at 90 days
- **Cloud SQL PostgreSQL 15** on private VPC with no public IP
- **Cloud Run service** (`mcap-indexer`) with VPC connector for private Cloud SQL access
- **Eventarc trigger** on GCS `ObjectFinalize` events
- **Pub/Sub topic** for P0 real-time alerts from edge nodes
- **Secret Manager** secrets for DB password and Gemini API key
- **Least-privilege service accounts** — the indexer can read GCS and write to Cloud SQL, nothing else

### 6.2 MCAP Indexer Cloud Run Function (`cloud/functions/mcap_indexer/main.py`)

The indexer is triggered by Eventarc on every new GCS object. Its key design constraint: **it never reads more than 64KB from any file**.

The MCAP format stores a trailer at the end of the file containing robot UUID, session ID, anomaly flags, firmware version, schema version, and message timestamps. Reading only the trailer means a 500MB recording costs the same as a 1MB recording to index — one 64KB GCS range request.

The indexer:
1. Downloads the last 64KB via `blob.download_as_bytes(start=file_size-65536, end=file_size-1)`
2. Parses the JSON metadata from the trailer
3. Falls back to filename parsing if trailer is malformed
4. Inserts a row into `robot_files` with full metadata
5. Calls the VLM tagger (v5 new) — described in section 7

Retry logic: 3 GCS retries with exponential backoff. If all fail, the exception propagates and Eventarc retries with backoff for up to 24 hours. After that, the event goes to a dead-letter Pub/Sub topic.

### 6.3 Database Schema (`schema/init.sql`)

**`robot_files`** — primary index of all recordings:
```sql
robot_id        TEXT NOT NULL
session_id      TEXT NOT NULL
gcs_uri         TEXT UNIQUE NOT NULL
topics          TEXT[]           -- GIN indexed
anomaly_flags   TEXT[]           -- GIN indexed
priority        SMALLINT         -- 0=critical, 1=high, 2=normal
size_bytes      BIGINT
time_start      TIMESTAMPTZ
time_end        TIMESTAMPTZ
schema_version  TEXT
firmware_ver    TEXT
vlm_description TEXT             -- NEW v5: Gemini-generated tag
embedding       VECTOR(1536)     -- NEW v5: semantic search vector
```

**Key indexes:**
- `idx_robot_time` — `(robot_id, time_start DESC)` for time-range queries per robot
- `idx_anomaly_flags` — GIN on `anomaly_flags` array for fleet-wide anomaly queries
- `idx_topics` — GIN on `topics` for sensor type filtering
- `idx_embedding_cosine` — IVFFlat cosine index on `embedding` for ANN search

**Built-in views:**
- `v_fleet_anomalies_24h` — aggregated anomaly counts across all robots, last 24h
- `v_schema_versions` — detects firmware/schema drift across the fleet
- `v_storage_by_priority` — cost attribution by priority tier
- `v_vlm_coverage` — percentage of files with embeddings (v5 new)

---

## 7. Vector DB + VLM Tagging (v5 New Feature)

### 7.1 The Problem It Solves

After indexing, a data scientist can query:
```sql
SELECT * FROM robot_files WHERE 'battery_critical' = ANY(anomaly_flags);
```

But they cannot query:
```
"find sessions where the robot seemed stuck near an obstacle while the battery was degrading"
```

There is no column for "seemed stuck." No column for "near an obstacle." These are semantic concepts that require understanding the *content* of the recordings. v5 adds this capability without adding any new GCP services.

### 7.2 Architecture

```
New file uploaded to GCS
        │
        ▼
Cloud Run: index_mcap_file()
        │
        ├── index_in_database()     ← existing, unchanged
        │
        └── vlm_tagger.tag_file()  ← NEW, runs after insert
            │
            ├── build_file_description(meta)
            │   └── constructs rich text from metadata fields
            │       (robot_id, priority, topics, anomaly_flags,
            │        data_type, size, firmware, timestamps)
            │
            ├── Gemini 3 Flash
            │   └── generate 1-2 sentence semantic tag
            │
            ├── gemini-embedding-001
            │   └── 1536-dim embedding vector
            │       (output_dimensionality=1536 for IVFFlat compatibility)
            │
            └── UPDATE robot_files
                SET vlm_description = ..., embedding = ...
                WHERE gcs_uri = ...
```

### 7.3 Why It Works on Simulated Data

The VLM tagger does not read sensor payloads. It builds descriptions from file metadata — fields that are populated identically for simulated and real recordings. A simulated collision file produces:

> "Priority-0 CRITICAL MCAP recording from robot 'robot-local-001' session 's846dd5c3'. Sensor topics: anomaly. ANOMALIES DETECTED: collision_imu. Data type: anomaly. File size: 4.2 MB. Firmware: v5. Schema: v5."

This description is semantically rich enough for meaningful similarity search. When real hardware produces actual camera frames and LiDAR scans, the same pipeline produces richer tags automatically.

### 7.4 The Free Tier

| Model | Use | Free limit |
|---|---|---|
| `gemini-3-flash-preview` | Description generation | 30 RPM / 1,500 RPD |
| `gemini-embedding-001` | 1536-dim vector | 100 RPM / 1,500 RPD |

At 2 API calls per file, the free tier covers **750 new files per day** with zero cost. The backfill script throttles to 1 file per 3 seconds to stay safely under rate limits.

### 7.5 Failure Isolation

VLM tagging is completely isolated from the core pipeline:

```python
# In index_mcap_file() — Cloud Run entry point
try:
    meta = read_mcap_trailer(bucket_name, object_name, file_size)
    index_in_database(meta, file_size)      # ← NEVER affected by VLM
except Exception as exc:
    raise                                   # ← triggers Eventarc retry

# VLM runs AFTER the index write succeeds
try:
    description, embedding = vlm_tagger.tag_file(meta)
    if description and embedding:
        _store_embedding(meta["gcs_uri"], description, embedding)
except Exception as exc:
    logger.warning("VLM tagging failed (non-fatal): %s", exc)
    # ← file is indexed, just without embedding
```

A Gemini outage means files are indexed without embeddings. It never means files are not indexed.

### 7.6 Semantic Search Query

```python
# Query embedding uses RETRIEVAL_QUERY task type
# Document embeddings use RETRIEVAL_DOCUMENT task type
# This asymmetry is intentional — improves retrieval quality

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="battery critical events near obstacles",
    config={"output_dimensionality": 1536},
)
query_vector = np.array(result.embeddings[0].values, dtype=np.float32)

# Cosine similarity search via pgvector <=> operator
SELECT gcs_uri, vlm_description,
       1 - (embedding <=> %s) AS similarity
FROM robot_files
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s
LIMIT 10
```

The `<=>` operator performs approximate nearest-neighbor search using the IVFFlat index. Sub-50ms on 6,000 rows.

---

## 8. Observability: Prometheus + Grafana

### 8.1 Metrics Exported (`metrics_exporter.py`)

Eight core KPIs are exposed at `localhost:9100/metrics`:

| Metric | What It Shows |
|---|---|
| `robot_cpu_usage_pct` | Pipeline CPU consumption (should stay <20%) |
| `robot_token_bucket_level` | Available upload bandwidth tokens |
| `robot_queue_depth_bytes` | Bytes waiting to be uploaded |
| `robot_disk_usage_pct` | Local storage utilization |
| `robot_anomaly_triggers_total` | Cumulative anomaly events by rule ID |
| `robot_upload_latency_seconds` | End-to-end upload time per file |
| `robot_thermal_cutoff_active` | 1 when thermal gate is blocking uploads |
| `robot_mmcf_backoff_total` | Times MMCF cost function triggered backoff |

### 8.2 Alert Rules (`monitoring/prometheus_rules.yml`)

11 alert rules covering:
- `PipelineCPUCritical` — pipeline exceeds 18% CPU for >2 minutes
- `StorageQuotaCritical` — local disk >90%
- `ThermalCutoffActive` — uploads suspended due to temperature
- `UploadQueueDepthHigh` — queue >500MB for >5 minutes
- `AnomalyRateHigh` — >10 anomaly triggers in 60 seconds
- `P0DataStuck` — any P0 file in STUCK state for >30 minutes

### 8.3 Grafana Dashboard

Pre-provisioned dashboard at port 3000 with:
- Robot selector dropdown (multi-robot fleet support)
- Real-time bandwidth throttle visualization
- Anomaly trigger timeline with rule breakdown
- Storage watermark with eviction event markers
- Upload success rate and latency percentiles

---

## 9. Demo Walkthrough

### Preparation (before audience arrives)

```bash
cd ~/robot-pipeline
set -a && source .env && set +a

# Start the Cloud SQL proxy
./cloud-sql-proxy data-pipeline-cz78:us-central1:robot-pipeline-pg --port=5433 &
sleep 4

# Verify DB connection
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 \
  -U pipeline -d robot_pipeline \
  -c "SELECT * FROM v_vlm_coverage;"
```

### Opening (2 min) — Set the Stakes

Do not open with code. Open with the problem:

> "A warehouse robot running SLAM at 60Hz just crashed into a shelf. You need the sensor data from 10 seconds before impact. The robot has a 20% CPU budget, a 4G connection that's currently at 80% capacity, and the CPU is at 79°C. How does the data get off the robot?"

Then show the architecture diagram and walk the data flow in one sentence per component.

### Act 1: The Edge (5 min)

Start the simulator:
```bash
GCS_BUCKET=robot-data-pipeline-cz78-demo DEMO_MODE=1 \
docker compose up -d robot prometheus alertmanager
```

While it starts, explain the three gates of the bandwidth controller. Draw the priority flowchart on a whiteboard or show it from the README. Emphasize: **P0 data bypasses all three gates. A collision log always gets through.**

Show the CPU governor constraint: "This process cannot physically exceed 20% CPU. Not by policy — by kernel enforcement."

### Act 2: Live Metrics (8 min)

Start Grafana and open port 3000. Wait 2 minutes for anomalies to fire, then point to:

1. **Token bucket level** — watch it drain as uploads happen, refill as bandwidth clears
2. **Anomaly trigger panel** — when `battery_critical` fires, explain: "the detector just promoted 30 seconds of pre-event data to P0. That data is now uploading ahead of everything else regardless of thermal or bandwidth state"
3. **Queue depth** — if it rises: "the MMCF cost function is currently blocking P1-P2 uploads. P0 is still flowing"

Check the database is being populated:
```bash
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 \
  -U pipeline -d robot_pipeline \
  -c "SELECT priority, COUNT(*), MAX(indexed_at) FROM robot_files GROUP BY priority ORDER BY priority;"
```

### Act 3: The New Feature — Semantic Search (5 min)

Transition with:
> "We can now query the database in milliseconds by robot, by anomaly type, by time range. But what if you don't know exactly what you're looking for? What if you want to ask a question in plain English?"

Show VLM coverage:
```bash
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 \
  -U pipeline -d robot_pipeline \
  -c "SELECT * FROM v_vlm_coverage;"
```

Show sample descriptions:
```bash
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 \
  -U pipeline -d robot_pipeline \
  -c "SELECT priority, vlm_description FROM robot_files WHERE embedding IS NOT NULL ORDER BY priority LIMIT 5;"
```

Read one out loud. "Gemini wrote that. For every file. Automatically."

Run the live searches:
```bash
python scripts/semantic_search.py "battery critical events"
python scripts/semantic_search.py "motor temperature overheating high priority"
python scripts/semantic_search.py "lidar obstacle robot stuck navigation"
python scripts/semantic_search.py "priority zero critical anomaly"
```

Point to the similarity scores. "0.87 similarity — that file is about battery criticality even though the query didn't use the exact word 'battery_critical'. The model understood the meaning."

Show auto-tagging is live:
```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="mcap-indexer" AND textPayload:"VLM tagged"' \
  --project=data-pipeline-cz78 --limit=5 --format="value(textPayload)"
```

"Every new upload — from this robot or any robot — is indexed and semantically tagged in the same Cloud Run invocation. Zero additional steps."

### Before/After Summary

| Capability | Before v5 | After v5 |
|---|---|---|
| Find collision data | Scan GCS bucket manually (minutes) | `semantic_search.py "collision"` (seconds) |
| Query by sensor type | SQL: `WHERE 'lidar' = ANY(topics)` | Natural language supported |
| New file discoverable | After indexing | After indexing + tagged + searchable |
| Infrastructure cost | Cloud SQL + Cloud Run | Same — pgvector on existing Cloud SQL |
| Tagging cost | N/A | Free tier (750 files/day) |

---

## 10. Architecture Decisions & Tradeoffs

### Why cgroups instead of `nice`?

`nice` affects scheduling priority but not CPU quota. A `nice 19` process will still consume 100% CPU when the system is otherwise idle, causing thermal throttling that degrades SLAM accuracy. `cfs_quota_us` is an absolute ceiling enforced by the kernel regardless of system load.

### Why SQLite instead of a proper database?

The edge device has no network-accessible database and cannot run a server process within the CPU budget. SQLite in WAL mode with proper page sizing achieves the required durability guarantees (ACID, crash recovery) at essentially zero overhead. The manifest DB handles tens of thousands of file state transitions per day on an eMMC device without wear issues.

### Why MCAP format?

MCAP is the robotics industry standard for multi-modal sensor recordings. It supports heterogeneous message types (LiDAR, camera, IMU, logs) in a single file with indexed access, checksumming, and a trailer that can be read without scanning the whole file. The 64KB trailer read in the cloud indexer is only possible because of MCAP's trailer design.

### Why pgvector instead of Pinecone / Weaviate / Qdrant?

The project already had Cloud SQL. Adding `CREATE EXTENSION vector` is one migration line. No new GCP service, no new VPC peering, no new IAM roles, no new billing dimension. IVFFlat on 6,000–100,000 rows has sub-50ms query latency — a dedicated vector database would not improve this meaningfully until the dataset reaches millions of rows. The operational simplicity gain is enormous.

### Why Gemini instead of a local embedding model?

Two constraints: the Jetson SoC is already at its 20% CPU budget, and running inference would violate it. Second, the free Gemini API tier covers the entire backfill and ongoing indexing workload at zero cost. At scale, switching to Vertex AI `text-embedding-004` costs approximately $0.00002 per file — negligible. The only change would be two lines in `vlm_tagger.py`.

### Why not embed at upload time on the robot?

Same CPU budget constraint. The embedding model runs in Cloud Run where it does not affect robot performance. The slight latency (seconds between upload and searchability) is acceptable for the retrieval use case.

### Why IVFFlat and not HNSW?

`gemini-embedding-001` with `output_dimensionality=1536` fits within IVFFlat's 2000-dimension limit. IVFFlat is faster to build and uses less memory than HNSW for datasets under ~1M rows. For larger fleets, switching to HNSW requires changing one line in the migration SQL.

---

## 11. Project Structure

```
robot-pipeline/
├── robot_agent/
│   ├── core/
│   │   ├── upload_agent.py         5-stage FSM, hardware-level resumable upload,
│   │   │                           streaming CRC32C
│   │   ├── anomaly_detector.py     3-layer detection: YAML rules, Z-score, Isolation Forest
│   │   ├── bandwidth_limiter.py    Thermal cutoff, MMCF cost function, token bucket
│   │   ├── ring_buffer.py          256MB ring buffer, 3-stage adaptive backpressure
│   │   ├── mcap_writer.py          2-phase non-blocking MCAP writer
│   │   ├── eviction_manager.py     4-stage cascade eviction, Zstd dict compression
│   │   ├── manifest_db.py          SQLite WAL state machine, orphan reconciliation
│   │   ├── codec_utils.py          H.265 short-GOP encoder, LiDAR bit-shuffle+Zstd
│   │   ├── cpu_governor.py         Linux cgroups CFS quota enforcement
│   │   └── metrics_exporter.py     8 Prometheus KPIs
│   ├── main.py                     Component topology and scheduler
│   ├── simulator.py                LiDAR point clouds, H.265 frames, anomaly scripts
│   └── Dockerfile
│
├── cloud/
│   ├── functions/
│   │   └── mcap_indexer/
│   │       ├── main.py             Cloud Run indexer + VLM tagging entry point
│   │       ├── vlm_tagger.py       Gemini description + embedding generation
│   │       └── requirements.txt
│   └── terraform/
│       ├── database.tf             Cloud SQL, VPC, Secret Manager
│       ├── main.tf                 GCS, Eventarc, service accounts
│       ├── vlm_tagging.tf          Gemini API key secret + Cloud Run patch
│       └── variables.tf
│
├── schema/
│   ├── init.sql                    Full schema: tables, GIN indexes, views
│   └── migration_vlm.sql           pgvector extension, embedding column, IVFFlat index
│
├── scripts/
│   ├── backfill_embeddings.py      Batch tag existing files, free-tier throttled,
│   │                               graceful stop via /tmp/stop_backfill
│   └── semantic_search.py          CLI semantic search with ranked results
│
├── monitoring/
│   ├── prometheus.yml              Scrape config
│   ├── prometheus_rules.yml        11 alert rules
│   └── grafana/
│       ├── dashboards/             Pre-provisioned robot pipeline dashboard
│       └── datasources/            Prometheus datasource config
│
├── config/
│   ├── pipeline.yaml               Thresholds, rotation sizes, MMCF weights
│   └── priority_rules.yaml         YAML anomaly rules with pre/post windows
│
├── deploy/
│   └── robot-pipeline.service      systemd unit with watchdog
│
├── tests/
│   ├── test_pipeline_v4.py         Unit tests including power-loss simulation
│   └── test_e2e.py                 End-to-end integration tests
│
├── demo-runbook.sh                 Full demo script with all steps automated
└── .env                            Environment config (not committed)
```

---

## 12. Quick Start

### Local Simulation (no cloud required)

```bash
# Copy environment config
cp .env.example .env

# Start the full stack: robot agent, Prometheus, Grafana
GCS_BUCKET=local-test DEMO_MODE=1 docker compose up

# Open Grafana at http://localhost:3000
# Watch anomalies fire and bandwidth throttle in real time
```

### Cloud Deployment

```bash
# Deploy infrastructure
cd cloud/terraform
terraform apply -var-file=demo.tfvars -var="gemini_api_key=AIza..."

# Build and push the indexer
cd ../functions/mcap_indexer
docker build -t us-central1-docker.pkg.dev/$PROJECT/robot-pipeline/mcap-indexer:latest .
docker push us-central1-docker.pkg.dev/$PROJECT/robot-pipeline/mcap-indexer:latest

# Run database migrations
psql -h $DB_HOST -U pipeline -d robot_pipeline -f schema/init.sql
psql -h $DB_HOST -U pipeline -d robot_pipeline -f schema/migration_vlm.sql
```

### Semantic Search (after backfill)

```bash
# Backfill existing files (resumes automatically if interrupted)
nohup python scripts/backfill_embeddings.py > backfill.log 2>&1 &

# Stop gracefully
touch /tmp/stop_backfill

# Search
python scripts/semantic_search.py "battery critical events"
python scripts/semantic_search.py "motor temperature anomaly high priority"
python scripts/semantic_search.py "lidar obstacle robot stuck"
```

---

*Built with: Python · PostgreSQL + pgvector · Google Cloud Storage · Cloud Run · Eventarc · Terraform · Prometheus · Grafana · Gemini API · MCAP format*
