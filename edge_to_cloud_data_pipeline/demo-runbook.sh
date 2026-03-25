#!/usr/bin/env bash
# Robot Pipeline v5 — Full Demo Runbook
# Project: data-pipeline-cz78 | Region: us-central1
# NEW in v5: Vector DB + VLM Tagging (pgvector + Gemini)
# Run all commands from: ~/robot-pipeline in Cloud Shell

# ============================================================
# STEP 0: LOAD ENVIRONMENT (NEW — run this first every session)
# ============================================================
cd ~/robot-pipeline
set -a && source .env && set +a

# Verify everything is loaded
echo "GCS_BUCKET   = $GCS_BUCKET"
echo "POSTGRES_HOST= $POSTGRES_HOST"
echo "GEMINI_API_KEY set: $([ -n "$GEMINI_API_KEY" ] && echo YES || echo NO)"
# Expected: all 3 lines populated


# ============================================================
# STEP 1: VERIFY YOU'RE IN THE RIGHT DIRECTORY
# ============================================================
pwd   # should show /home/cz78illinoisedu/robot-pipeline


# ============================================================
# STEP 2: CLEAN START — KILL EVERYTHING FIRST
# ============================================================
docker compose down --remove-orphans
docker rm -f $(docker ps -aq --filter "name=robot") 2>/dev/null || true
docker volume rm robot-pipeline_prometheus_data 2>/dev/null || true
docker volume rm robot-pipeline_grafana_data 2>/dev/null || true


# ============================================================
# STEP 3: START CLOUD SQL PROXY (NEW — required for DB access)
# ============================================================
pkill -f cloud-sql-proxy 2>/dev/null || true
sleep 2

./cloud-sql-proxy data-pipeline-cz78:us-central1:robot-pipeline-pg \
  --port=5433 &

sleep 4
echo "Proxy status:"
curl -s http://127.0.0.1:5433 2>/dev/null || echo "Proxy running (no HTTP endpoint — that's normal)"
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 \
  -U pipeline -d robot_pipeline \
  -c "SELECT 'DB connected' AS status;"
# Expected: DB connected


# ============================================================
# STEP 4: START PROMETHEUS + ALERTMANAGER
# ============================================================
docker compose up -d prometheus alertmanager

sleep 5
curl -s http://localhost:9090/-/healthy
# Expected: "Prometheus Server is Healthy."


# ============================================================
# STEP 5: START ROBOT AGENT IN DEMO MODE
# ============================================================
GCS_BUCKET=robot-data-pipeline-cz78-demo \
DEMO_MODE=1 \
docker compose up -d robot

sleep 15
docker logs robot-agent 2>&1 | grep -E "Demo mode|SIM|ANOMALY" | head -5
# Expected:
#   Demo mode: starting simulator
#   [SIM] Anomaly scheduled: fast_battery_drain in 30s


# ============================================================
# STEP 6: START GRAFANA
# ============================================================
docker stop grafana 2>/dev/null; docker rm grafana 2>/dev/null

docker run -d \
  --name grafana \
  --network robot-pipeline_default \
  -p 3000:3000 \
  -e GF_SERVER_ROOT_URL="https://3000-cs-51342460920-default.cs-us-east1-rtep.cloudshell.dev/" \
  -e GF_SERVER_DOMAIN="3000-cs-51342460920-default.cs-us-east1-rtep.cloudshell.dev" \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  -e GF_SECURITY_CSRF_ALWAYS_CHECK=false \
  -e GF_SECURITY_CSRF_TRUSTED_ORIGINS="3000-cs-51342460920-default.cs-us-east1-rtep.cloudshell.dev" \
  -e GF_SECURITY_COOKIE_SAMESITE=none \
  -e GF_SECURITY_COOKIE_SECURE=true \
  -e GF_AUTH_ANONYMOUS_ENABLED=true \
  -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
  -e GF_ANALYTICS_REPORTING_ENABLED=false \
  -v robot-pipeline_grafana_data:/var/lib/grafana \
  -v ~/robot-pipeline/monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro \
  -v ~/robot-pipeline/monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro \
  grafana/grafana:10.4.0


# ============================================================
# STEP 7: VERIFY ALL SERVICES ARE UP
# ============================================================
sleep 20
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
# Expected:
#   robot-agent    Up X min (healthy)   0.0.0.0:9100->9100/tcp
#   prometheus     Up X min             0.0.0.0:9090->9090/tcp
#   grafana        Up X min (healthy)   0.0.0.0:3000->3000/tcp
#   alertmanager   Up X min             0.0.0.0:9093->9093/tcp

echo "=== Robot ===" && curl -s http://localhost:9100/metrics | grep "^robot_cpu" | head -1
echo "=== Prometheus ===" && curl -s http://localhost:9090/-/healthy
echo "=== Grafana ===" && curl -s http://localhost:3000/api/health
echo "=== Alertmanager ===" && curl -s http://localhost:9093/-/healthy


# ============================================================
# STEP 8: VERIFY PROMETHEUS IS SCRAPING
# ============================================================
sleep 30
curl -s "http://localhost:9090/api/v1/targets" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
for t in d['data']['activeTargets']:
    print('State:', t['health'])
    print('URL:', t['scrapeUrl'])
    print('Error:', t.get('lastError','none'))
"
# Expected: State: up | URL: http://robot-agent:9100/metrics | Error: none


# ============================================================
# STEP 9: VERIFY METRICS AND ANOMALIES ARE FLOWING
# ============================================================
sleep 90
curl -s "http://localhost:9090/api/v1/query?query=robot_anomaly_triggers_total" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
total=sum(float(r['value'][1]) for r in d['data']['result'])
print(f'Total anomaly events: {total}')
for r in d['data']['result']:
    print(f'  {r[\"metric\"][\"rule_id\"]}: {r[\"value\"][1]}')
"
# Expected: Total anomaly events: 40+ with battery_critical, motor_temp, etc.


# ============================================================
# STEP 10: CHECK DATABASE IS POPULATED
# ============================================================
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 \
  -U pipeline -d robot_pipeline << 'SQL'
SELECT priority, COUNT(*) as files,
  ROUND(SUM(size_bytes)/1024/1024.0,1) as mb,
  MAX(indexed_at) as latest
FROM robot_files
GROUP BY priority
ORDER BY priority;
SQL
# Expected: 4 rows showing p0/p1/p2/p3 with file counts and sizes


# ============================================================
# STEP 11: CHECK GCS BUCKET HAS FILES
# ============================================================
gcloud storage ls gs://robot-data-pipeline-cz78-demo/ --recursive 2>/dev/null | head -10
for p in p0_critical p1_high p2_normal p3_low; do
  echo -n "$p: "
  gcloud storage ls "gs://robot-data-pipeline-cz78-demo/**/$p/**" 2>/dev/null | wc -l
done


# ============================================================
# STEP 12: VLM TAGGING — CHECK COVERAGE (NEW)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FEATURE: Vector DB + VLM Tagging"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 \
  -U pipeline -d robot_pipeline \
  -c "SELECT * FROM v_vlm_coverage;"
# Expected: tagged_files > 0, pct_tagged shows progress

# Show a few example VLM descriptions
echo ""
echo "=== Sample VLM-generated descriptions ==="
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 \
  -U pipeline -d robot_pipeline << 'SQL'
SELECT
  priority,
  filename,
  vlm_description
FROM robot_files
WHERE vlm_description IS NOT NULL
ORDER BY priority ASC, indexed_at DESC
LIMIT 6;
SQL
# Expected: Gemini-generated 1-2 sentence tags per file, searchable by meaning


# ============================================================
# STEP 13: VLM TAGGING — BACKFILL STATUS (NEW)
# ============================================================

# Check if backfill is running
echo ""
echo "=== Backfill job status ==="
if pgrep -f backfill_embeddings > /dev/null; then
  echo "Backfill is RUNNING"
  tail -3 backfill.log 2>/dev/null
else
  echo "Backfill is NOT running"
  echo "To start: nohup python scripts/backfill_embeddings.py > backfill.log 2>&1 &"
  echo "To stop:  touch /tmp/stop_backfill"
fi

# Show embedding index health
echo ""
echo "=== pgvector IVFFlat index ==="
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 \
  -U pipeline -d robot_pipeline \
  -c "SELECT indexname, indexdef FROM pg_indexes WHERE tablename='robot_files' AND indexname='idx_embedding_cosine';"
# Expected: idx_embedding_cosine | USING ivfflat (embedding vector_cosine_ops)


# ============================================================
# STEP 14: VLM TAGGING — LIVE SEMANTIC SEARCH DEMO (NEW)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DEMO: Natural Language Semantic Search over 6000+ MCAP files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Search 1: Safety critical
echo ""
echo ">>> Query: battery critical events"
python scripts/semantic_search.py "battery critical events" --limit 3

# Search 2: Sensor anomaly
echo ""
echo ">>> Query: motor temperature overheating"
python scripts/semantic_search.py "motor temperature overheating" --limit 3

# Search 3: Navigation
echo ""
echo ">>> Query: lidar obstacle robot stuck navigation"
python scripts/semantic_search.py "lidar obstacle robot stuck navigation" --limit 3

# Search 4: Highest priority files
echo ""
echo ">>> Query: priority zero critical anomaly"
python scripts/semantic_search.py "priority zero critical anomaly" --limit 3


# ============================================================
# STEP 15: VLM TAGGING — NEW FILES AUTO-TAGGED (NEW)
# ============================================================
echo ""
echo "=== Verifying new files get auto-tagged by Cloud Run ==="
echo "Check the mcap-indexer Cloud Run logs to see VLM tagging happening:"
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="mcap-indexer" AND textPayload:"VLM tagged"' \
  --project=data-pipeline-cz78 \
  --limit=5 \
  --format="value(textPayload)"
# Expected: "VLM tagged: <filename> (1536-dim embedding)"
# This proves new uploads are tagged automatically — no manual steps needed


# ============================================================
# STEP 16: OPEN THE DASHBOARDS
# ============================================================
# In Cloud Shell — click "Web Preview" icon (top right) → "Change port"

# Grafana dashboard:     port 3000  (no login — anonymous admin)
# Prometheus queries:    port 9090
# Alertmanager alerts:   port 9093
# Raw robot metrics:     port 9100  (add /metrics to URL)

# In Grafana:
#   1. Dashboards → Robot Pipeline folder → Robot Data Pipeline
#   2. Set time range: Last 15 minutes
#   3. Robot dropdown top-left: select robot-local-001
#   4. Auto-refresh: 15s (already set)


# ============================================================
# STEP 17: DEMO TALKING POINTS
# ============================================================
# 1. Pipeline flow: robot → GCS → Eventarc → Cloud Run → Postgres
#    Show: event log in web dashboard, robot logs, GCS bucket
#
# 2. Anomaly detection: 9 rule types, YAML + z-score detectors
#    Show: Grafana "Anomaly Triggers" panel spiking in real time
#
# 3. Priority system: P0 always uploaded first, bypasses daily cap
#    Show: GCS folder structure p0_critical/ p1_high/ p2_normal/ p3_low/
#
# 4. Observability: Prometheus scraping 8 golden KPIs every 15s
#    Show: Prometheus Explore → query robot_token_bucket_level
#
# 5. Alert rules: 11 rules in prometheus_rules.yml
#    Show: Alertmanager at port 9093, explain PipelineCPUCritical etc.
#
# 6. Burst test: web dashboard → BURST TEST button
#    Demonstrates: backpressure, Cloud Run autoscaling, quota bypass
#
# ── NEW v5 TALKING POINTS ────────────────────────────────────────────
#
# 7. Vector DB: pgvector on existing Cloud SQL — no new infrastructure
#    Show: \d robot_files in psql — embedding VECTOR(1536) column
#    "We added semantic search without adding a single new GCP service"
#
# 8. VLM tagging: Gemini 3 Flash generates descriptions per file
#    Show: Step 12 output — real Gemini-written tags for each MCAP file
#    "Every file gets a human-readable semantic tag automatically"
#
# 9. Auto-tagging on upload: new files tagged in Cloud Run, zero ops
#    Show: Step 15 Cloud Run logs — "VLM tagged" appearing in real time
#    "Upload a file → indexed → tagged → searchable, all in one pipeline"
#
# 10. Natural language search across 6000+ files
#     Show: Step 14 live search queries returning ranked results
#     "Find 'battery critical events near obstacles' — no SQL needed"
#     Compare: before = scan GCS bucket manually; after = semantic search


# ============================================================
# QUICK RESTART IF SOMETHING BREAKS
# ============================================================

# If Cloud SQL proxy dies:
pkill -f cloud-sql-proxy 2>/dev/null
./cloud-sql-proxy data-pipeline-cz78:us-central1:robot-pipeline-pg --port=5433 &

# If env vars are lost:
set -a && source .env && set +a

# If Grafana shows "origin not allowed":
docker restart grafana && sleep 15 && curl -s http://localhost:3000/api/health

# If Prometheus shows no data (state: down):
curl -s -X POST http://localhost:9090/-/reload
docker restart prometheus

# If robot agent stops simulating:
docker rm -f robot-agent
GCS_BUCKET=robot-data-pipeline-cz78-demo DEMO_MODE=1 docker compose up -d robot

# If backfill needs to be started/stopped:
nohup python scripts/backfill_embeddings.py > backfill.log 2>&1 &  # start
touch /tmp/stop_backfill                                             # stop gracefully

# Check everything at once:
docker ps --format "table {{.Names}}\t{{.Status}}"
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 \
  -U pipeline -d robot_pipeline \
  -c "SELECT * FROM v_vlm_coverage;"
