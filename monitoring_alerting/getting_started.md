# 🚀 Getting Started - Monitoring & Alerting System

This guide will get your monitoring system up and running step-by-step.

---

## 📋 Prerequisites

- **Docker** 20.10+ and **Docker Compose** v2.0+
- **8GB RAM** minimum (16GB recommended)
- **50GB free disk space**
- **Git** (optional, for version control)

---

## 🏗️ Project Structure

Your project should have this structure:

```
project-04-monitoring-alerting/
├── docker-compose.yml
├── Dockerfile.mlapi
│
├── src/
│   ├── instrumentation.py
│   └── app.py
│
├── prometheus/
│   ├── prometheus.yml
│   └── alerts.yml
│
├── alertmanager/
│   └── alertmanager.yml
│
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── datasources.yml
│   │   └── dashboards/
│   │       └── dashboards.yml
│   └── dashboards/
│       └── (dashboard JSON files go here)
│
├── elasticsearch/
│   └── logstash.conf
│
├── filebeat/
│   └── filebeat.yml
│
├── logs/
│   └── (application logs)
│
└── tests/
    └── test_metrics.py
```

---

## ⚡ Quick Start (5 Steps)

### Step 1: Create Project Directory

```bash
mkdir project-04-monitoring-alerting
cd project-04-monitoring-alerting
```

### Step 2: Create All Required Files

Copy all the configuration files I provided into their correct locations:

- `docker-compose.yml` → root
- `Dockerfile.mlapi` → root
- `src/instrumentation.py` → src/
- `src/app.py` → src/
- `prometheus/prometheus.yml` → prometheus/
- `prometheus/alerts.yml` → prometheus/
- `alertmanager/alertmanager.yml` → alertmanager/
- `grafana/provisioning/datasources/datasources.yml` → grafana/provisioning/datasources/
- `grafana/provisioning/dashboards/dashboards.yml` → grafana/provisioning/dashboards/
- `elasticsearch/logstash.conf` → elasticsearch/
- `filebeat/filebeat.yml` → filebeat/

### Step 3: Run Configuration Fix Script

```bash
# Make scripts executable
chmod +x fix-config.sh
chmod +x setup-complete.sh
chmod +x test-monitoring.sh

# Run the fix script to verify everything
./fix-config.sh
```

This will check all files and create any missing directories.

### Step 4: Start the Stack

```bash
# Start all services
docker-compose up -d

# Wait 1-2 minutes for services to start
# Watch the logs
docker-compose logs -f
```

Press Ctrl+C when you see these messages:
- Elasticsearch: `"Cluster health status changed"`
- Grafana: `"HTTP Server Listen"`
- Prometheus: `"Server is ready to receive web requests"`

### Step 5: Verify Everything Works

```bash
# Run the test script
./test-monitoring.sh
```

---

## 🌐 Access the Services

After starting, access these URLs:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3000 | admin / admin123 |
| **Prometheus** | http://localhost:9090 | None |
| **Kibana** | http://localhost:5601 | None |
| **Alertmanager** | http://localhost:9093 | None |
| **ML API** | http://localhost:5000 | None |
| **Elasticsearch** | http://localhost:9200 | None |

---

## ✅ Verification Checklist

### 1. Check All Services Are Running

```bash
docker-compose ps
```

Expected output: All services show "Up" or "healthy"

### 2. Test Each Service

```bash
# Prometheus
curl http://localhost:9090/-/healthy
# Should return: Prometheus is Healthy.

# Grafana
curl http://localhost:3000/api/health
# Should return: {"commit":"...","database":"ok",...}

# ML API
curl http://localhost:5000/health
# Should return: {"status":"healthy","timestamp":"..."}

# Elasticsearch
curl http://localhost:9200/_cluster/health
# Should return: {"status":"green",...}
```

### 3. Check Prometheus Targets

1. Go to http://localhost:9090/targets
2. You should see these targets as "UP":
   - prometheus (localhost:9090)
   - node-exporter (node-exporter:9100)
   - ml-api (ml-api:5000)

### 4. Check Grafana Datasource

1. Go to http://localhost:3000
2. Login (admin / admin123)
3. Click ☰ → Connections → Data sources
4. You should see "Prometheus" datasource

If it's not there:
- Click "Add data source"
- Select "Prometheus"
- URL: `http://prometheus:9090`
- Click "Save & Test"

### 5. Test ML API

```bash
# Make a prediction request
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}}'

# Check metrics
curl http://localhost:5000/metrics
```

---

## 🧪 Generate Test Data

### Create Metrics

```bash
# Generate 50 requests
for i in {1..50}; do
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d "{\"features\": {\"feature1\": $i, \"feature2\": $((i*2)), \"feature3\": $((i*3))}}"
  sleep 0.1
done
```

### View Metrics in Prometheus

1. Go to http://localhost:9090
2. Click "Graph"
3. Enter these queries:

```promql
# Request rate
rate(http_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Model predictions by class
sum by (prediction_class) (model_predictions_total)

# CPU usage
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

### View Metrics in Grafana

1. Go to http://localhost:3000
2. Click "Explore" (compass icon)
3. Select "Prometheus" datasource
4. Enter the same queries above
5. Click "Run query"

---

## 📊 Create Your First Dashboard

### In Grafana:

1. Click "+" → "Dashboard"
2. Click "Add visualization"
3. Select "Prometheus"
4. Enter query: `rate(http_requests_total[5m])`
5. Set visualization to "Time series"
6. Click "Apply"
7. Add more panels with different metrics
8. Click "Save dashboard"

---

## 🚨 Trigger an Alert

### Test Error Alert

```bash
# Generate errors to trigger HighErrorRate alert
for i in {1..50}; do 
  curl http://localhost:5000/error
done

# Wait 5 minutes, then check alerts
```

View alerts:
- Prometheus: http://localhost:9090/alerts
- Alertmanager: http://localhost:9093

---

## 🐛 Troubleshooting

### Services Won't Start

```bash
# Check logs for specific service
docker-compose logs [service-name]

# Common issues:
# - Port already in use: Change port in docker-compose.yml
# - Out of memory: Increase Docker memory limit
# - Permission denied: Fix file permissions
```

### Kibana Fails with "elastic user forbidden"

This is already fixed in the updated docker-compose.yml. If you still see it:

```bash
# Stop everything
docker-compose down

# Make sure docker-compose.yml has this for Kibana:
# environment:
#   - XPACK_SECURITY_ENABLED=false

# Restart
docker-compose up -d
```

### Grafana - No Prometheus Datasource

```bash
# Check if file exists
ls grafana/provisioning/datasources/datasources.yml

# If missing, run:
./fix-config.sh

# Restart Grafana
docker-compose restart grafana
```

### ML API - 404 Not Found

```bash
# Check logs
docker-compose logs ml-api

# Rebuild container
docker-compose build ml-api
docker-compose up -d ml-api

# Test endpoints (note: no /tcp!)
curl http://localhost:5000/health
curl http://localhost:5000/metrics
```

### No Logs in Kibana

```bash
# Wait 5 minutes for logs to appear
# Then check if Elasticsearch has data
curl http://localhost:9200/logs-*/_count

# If count is 0, check Logstash
docker-compose logs logstash

# Check if logs are being written
docker-compose exec ml-api ls -la /var/log/app/
```

---

## 🎯 What's Next?

After your system is running:

1. **Create More Dashboards**
   - Infrastructure monitoring
   - ML model performance
   - Business metrics

2. **Configure Alerts**
   - Add Slack integration
   - Set up PagerDuty
   - Fine-tune thresholds

3. **Monitor Your Real ML Models**
   - Replace demo API with your model
   - Add custom metrics
   - Implement data drift detection

4. **Scale the System**
   - Add more Prometheus instances
   - Scale Elasticsearch cluster
   - Deploy to Kubernetes

---

## 📚 Additional Resources

- **QUICKSTART.md** - Detailed setup guide
- **PROJECT_SUMMARY.md** - Complete project overview
- **test-monitoring.sh** - Automated testing
- **fix-config.sh** - Configuration verification

---

## 🆘 Getting Help

If you're stuck:

1. Run `./fix-config.sh` to check configuration
2. Run `./test-monitoring.sh` to test services
3. Check logs: `docker-compose logs [service]`
4. Verify ports: `netstat -tuln | grep [port]`

---

**You're all set!** 🎉

Your monitoring system is now ready to track metrics, logs, and alerts from your ML applications!
