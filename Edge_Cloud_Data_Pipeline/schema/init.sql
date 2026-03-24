-- schema/init.sql — PostgreSQL robot_files index (v4)
-- Cloud-side: enables millisecond queries over entire fleet's MCAP metadata
-- Without this: data scientists must scan S3/GCS bucket (seconds+, costly)
-- With this: query robot_files table (milliseconds, cents)

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── robot_files: primary index of all uploaded MCAP files ─────────────────
CREATE TABLE IF NOT EXISTS robot_files (
    id              BIGSERIAL PRIMARY KEY,
    robot_id        TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    gcs_uri         TEXT UNIQUE NOT NULL,    -- gs://bucket/prefix/session/filename.mcap
    filename        TEXT,
    data_type       TEXT,
    topics          TEXT[],                  -- array: ['lidar', 'imu', 'camera']
    time_start      TIMESTAMPTZ,             -- first message timestamp in file
    time_end        TIMESTAMPTZ,             -- last message timestamp in file
    anomaly_flags   TEXT[],                  -- array: ['battery_critical', 'collision']
    schema_version  TEXT,                    -- MCAP IDL version (v4: detect drift across firmware)
    firmware_ver    TEXT,                    -- robot firmware version at capture time
    priority        SMALLINT DEFAULT 3,      -- 0=critical, 1=high, 2=normal, 3=low
    size_bytes      BIGINT DEFAULT 0,
    indexed_at      TIMESTAMPTZ DEFAULT NOW(),
    -- GCS lifecycle state
    storage_class   TEXT DEFAULT 'STANDARD', -- STANDARD | NEARLINE | COLDLINE | ARCHIVE
    expires_at      TIMESTAMPTZ              -- NULL = permanent
);

-- ── Indexes for common query patterns ──────────────────────────────────────
-- "Give me all P0 anomaly files from this robot in the last 7 days"
CREATE INDEX IF NOT EXISTS idx_robot_time
    ON robot_files (robot_id, time_start DESC);

-- "All collision events across fleet today"
CREATE INDEX IF NOT EXISTS idx_anomaly_flags
    ON robot_files USING GIN (anomaly_flags);

-- "LiDAR files only"
CREATE INDEX IF NOT EXISTS idx_topics
    ON robot_files USING GIN (topics);

-- "Recently indexed"
CREATE INDEX IF NOT EXISTS idx_indexed_at
    ON robot_files (indexed_at DESC);

-- "Priority queue for manual review"
CREATE INDEX IF NOT EXISTS idx_priority_time
    ON robot_files (priority, time_start DESC);

-- ── schema_drift: detect IDL version changes across firmware ──────────────
-- (v4 net-new: Lambda reads schema_version from MCAP trailer without opening file)
CREATE TABLE IF NOT EXISTS schema_drift_events (
    id              BIGSERIAL PRIMARY KEY,
    robot_id        TEXT NOT NULL,
    old_schema_ver  TEXT,
    new_schema_ver  TEXT,
    firmware_ver    TEXT,
    detected_at     TIMESTAMPTZ DEFAULT NOW(),
    file_gcs_uri    TEXT
);

-- ── fleet_robots: robot registry ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fleet_robots (
    robot_id        TEXT PRIMARY KEY,
    mac_address     TEXT,
    firmware_ver    TEXT,
    last_seen       TIMESTAMPTZ,
    last_upload     TIMESTAMPTZ,
    total_files     BIGINT DEFAULT 0,
    total_bytes     BIGINT DEFAULT 0,
    status          TEXT DEFAULT 'active'   -- 'active' | 'offline' | 'maintenance'
);

-- ── upload_events: audit log ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS upload_events (
    id              BIGSERIAL PRIMARY KEY,
    robot_id        TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    filename        TEXT NOT NULL,
    event_type      TEXT NOT NULL,          -- 'uploaded' | 'verified' | 'indexed' | 'error'
    priority        SMALLINT,
    size_bytes      BIGINT,
    latency_ms      INTEGER,                -- upload latency
    error_msg       TEXT,
    occurred_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_events_robot ON upload_events (robot_id, occurred_at DESC);

-- ── bandwidth_log: fleet-wide bandwidth accounting ────────────────────────
CREATE TABLE IF NOT EXISTS fleet_bandwidth_log (
    id              BIGSERIAL PRIMARY KEY,
    robot_id        TEXT NOT NULL,
    day             DATE NOT NULL,
    bytes_uploaded  BIGINT DEFAULT 0,
    file_count      INTEGER DEFAULT 0,
    p0_bytes        BIGINT DEFAULT 0,       -- critical data (uncapped)
    p1_bytes        BIGINT DEFAULT 0,
    p2_bytes        BIGINT DEFAULT 0,
    p3_bytes        BIGINT DEFAULT 0,
    UNIQUE (robot_id, day)
);

-- ── Views for common dashboards ────────────────────────────────────────────

-- Fleet anomaly summary (last 24h)
CREATE OR REPLACE VIEW v_fleet_anomalies_24h AS
SELECT
    robot_id,
    unnest(anomaly_flags) AS anomaly_type,
    COUNT(*) AS occurrences,
    SUM(size_bytes) AS total_bytes,
    MAX(time_start) AS latest_event
FROM robot_files
WHERE time_start > NOW() - INTERVAL '24 hours'
  AND cardinality(anomaly_flags) > 0
GROUP BY robot_id, anomaly_type
ORDER BY occurrences DESC;

-- Schema drift detection (critical for cross-firmware compatibility)
CREATE OR REPLACE VIEW v_schema_versions AS
SELECT
    firmware_ver,
    schema_version,
    COUNT(*) AS file_count,
    MIN(indexed_at) AS first_seen,
    MAX(indexed_at) AS last_seen
FROM robot_files
GROUP BY firmware_ver, schema_version
ORDER BY last_seen DESC;

-- Storage cost attribution by priority
CREATE OR REPLACE VIEW v_storage_by_priority AS
SELECT
    priority,
    CASE priority
        WHEN 0 THEN 'P0 Critical'
        WHEN 1 THEN 'P1 High'
        WHEN 2 THEN 'P2 Normal'
        WHEN 3 THEN 'P3 Low/Bulk'
    END AS priority_label,
    COUNT(*) AS file_count,
    SUM(size_bytes) / 1e9 AS total_gb,
    ROUND(SUM(size_bytes) / 1e9 * 0.023, 2) AS est_monthly_cost_usd  -- GCS Standard pricing
FROM robot_files
GROUP BY priority
ORDER BY priority;

-- ── Seed default robot ────────────────────────────────────────────────────
INSERT INTO fleet_robots (robot_id, firmware_ver, status)
VALUES ('robot-demo', 'v4.0.0', 'active')
ON CONFLICT (robot_id) DO NOTHING;
