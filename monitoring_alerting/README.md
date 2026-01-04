# Project 04: Monitoring & Alerting System

---

## Overview

Build a comprehensive monitoring and alerting system for ML infrastructure covering application metrics, infrastructure health, model performance, and business KPIs. This project introduces production observability: logs, metrics, traces, alerting, and incident response.

### What You'll Build

A complete observability stack that monitors:
- **Infrastructure**: CPU, memory, disk, network usage
- **Applications**: Request rates, latency, error rates
- **ML Models**: Predictions/sec, inference latency, accuracy, drift
- **Business Metrics**: SLAs, user requests, success rates

### Real-World Context

Your ML systems are now in production (Projects 1-3), but you're flying blind without proper monitoring. When things break at 3 AM, you need to know immediately. When models drift, you need alerts. When infrastructure is stressed, you need visibility. This project simulates building a production-grade observability stack.

### Learning Outcomes

By completing this project, you will:

1. Deploy monitoring infrastructure (Prometheus, Grafana, ELK Stack)
2. Instrument applications with metrics, logs, and traces
3. Create operational dashboards for different stakeholders
4. Implement alerting rules for proactive incident detection
5. Set up log aggregation and analysis pipelines
6. Monitor ML-specific metrics (drift, data quality, model performance)
7. Build on-call runbooks for incident response
8. Integrate with incident management (PagerDuty, Opsgenie)

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Metrics Collection** | Prometheus 2.47+ | Time-series metrics storage |
| **Alerting** | Alertmanager 0.26+ | Alert routing and notification |
| **Visualization** | Grafana 10.2+ | Dashboards and visualization |
| **Log Storage** | Elasticsearch 8.11+ | Log aggregation and search |
| **Log Processing** | Logstash 8.11+ | Log parsing and forwarding |
| **Log Visualization** | Kibana 8.11+ | Log search and visualization |
| **Log Shipping** | Filebeat 8.11+ | Log collection from nodes |
| **Metrics Exporter** | Node Exporter | Infrastructure metrics |
| **Application Instrumentation** | prometheus-client (Python) | Application metrics |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring Architecture                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      Data Sources                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Kubernetes  │  │ Applications│  │ ML Models   │             │
│  │ Cluster     │  │ (API/Web)   │  │ (Inference) │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │ Metrics         │ Metrics + Logs  │ Metrics + Logs    │
└─────────┼─────────────────┼─────────────────┼────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Collection Layer                              │
│  ┌──────────────────────────┐    ┌─────────────────────────┐    │
│  │      Prometheus          │    │   Filebeat/Fluentd      │    │
│  │  - Scrape metrics        │    │   - Collect logs        │    │
│  │  - Store time-series     │    │   - Ship to Logstash    │    │
│  │  - Evaluate alerts       │    │                         │    │
│  └────────┬─────────────────┘    └───────────┬─────────────┘    │
└───────────┼───────────────────────────────────┼──────────────────┘
            │                                   │
            ▼                                   ▼
┌──────────────────────────┐      ┌────────────────────────────┐
│     Alertmanager         │      │      Logstash              │
│  - Route alerts          │      │  - Parse logs              │
│  - De-duplicate          │      │  - Transform & enrich      │
│  - Send notifications    │      │  - Forward to ES           │
└────────┬─────────────────┘      └──────────┬─────────────────┘
         │                                   │
         │                                   ▼
         │                        ┌────────────────────────────┐
         │                        │    Elasticsearch           │
         │                        │  - Store logs              │
         │                        │  - Index & search          │
         │                        └──────────┬─────────────────┘
         │                                   │
         ▼                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Visualization Layer                            │
│  ┌──────────────────────┐         ┌──────────────────────┐      │
│  │      Grafana         │         │       Kibana         │      │
│  │  - Dashboards        │         │  - Log search        │      │
│  │  - Metrics viz       │         │  - Log analysis      │      │
│  └──────────────────────┘         └──────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
            │                                   │
            ▼                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Notification Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Email   │  │  Slack   │  │PagerDuty │  │ Webhook  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
project-04-monitoring-alerting/
├── README.md                          # This file
├── requirements.md                    # Detailed requirements and SLIs/SLOs
├── architecture.md                    # Observability stack architecture
├── src/
│   ├── instrumentation.py            # Prometheus metrics instrumentation (STUB)
│   └── custom_metrics.py             # Custom ML metrics (STUB)
├── prometheus/
│   ├── prometheus.yml                # Prometheus config (STUB)
│   └── alerts.yml                    # Alerting rules (STUB)
├── grafana/
│   ├── dashboards/
│   │   └── ml-overview.json         # ML dashboard template
│   └── datasources.yml              # Grafana datasource config
├── elasticsearch/
│   ├── logstash.conf                # Logstash pipeline (STUB)
│   └── kibana-dashboard.ndjson      # Kibana dashboard export
├── tests/
│   └── test_metrics.py              # Metrics tests (STUB)
├── docker-compose.yml               # Full monitoring stack
├── .env.example                     # Monitoring configuration
└── docs/
    ├── SETUP.md                     # Setup instructions
    ├── RUNBOOK.md                   # On-call runbook
    └── TROUBLESHOOTING.md           # Common issues
```

---

## Getting Started

### Step 1: Review Requirements

Start by reading `requirements.md` to understand:
- Functional requirements (metrics, dashboards, alerts, logging)
- Non-functional requirements (performance, reliability, scalability)
- Success criteria

### Step 2: Study Architecture

Review `architecture.md` to understand:
- Monitoring stack components
- Data flow (metrics, logs, alerts)
- Integration points

### Step 3: Set Up Infrastructure

1. Copy `.env.example` to `.env` and configure
2. Start the monitoring stack with Docker Compose:
   ```bash
   docker-compose up -d
   ```
3. Verify all services are running:
   ```bash
   docker-compose ps
   ```

### Step 4: Complete Code Stubs

Work through the TODO comments in each stub file:

1. **src/instrumentation.py**: Implement Prometheus metrics
2. **src/custom_metrics.py**: Add ML-specific metrics
3. **prometheus/prometheus.yml**: Configure scrape targets
4. **prometheus/alerts.yml**: Define alert rules
5. **elasticsearch/logstash.conf**: Set up log processing
6. **tests/test_metrics.py**: Write metric tests

### Step 5: Create Dashboards

1. Access Grafana at `http://localhost:3000` (admin/admin123)
2. Import or create dashboards:
   - Infrastructure Dashboard
   - Application Dashboard
   - ML Model Dashboard
   - Business Dashboard
3. Save dashboard JSON to `grafana/dashboards/`

### Step 6: Test Alerting

1. Trigger test alerts (simulate high CPU, errors, drift)
2. Verify alerts fire correctly
3. Check alert routing to different channels
4. Document alert response in runbooks

### Step 7: Documentation

Complete the documentation:
- `docs/SETUP.md`: Setup and deployment guide
- `docs/RUNBOOK.md`: On-call incident response procedures
- `docs/TROUBLESHOOTING.md`: Common issues and solutions

---

## Key Concepts

### Metrics Types

1. **Counter**: Monotonically increasing value (total requests, errors)
2. **Gauge**: Current value (CPU usage, memory, active connections)
3. **Histogram**: Distribution of values (request duration, response size)
4. **Summary**: Quantiles over sliding time window (P50, P95, P99 latency)

### The Four Golden Signals

Monitor these for every service:

1. **Latency**: How long requests take
2. **Traffic**: How much demand on the system
3. **Errors**: Rate of failed requests
4. **Saturation**: How "full" the service is

### Alert Design Principles

- **Actionable**: Every alert should require human action
- **Meaningful**: Alert on symptoms, not causes
- **Contextual**: Include enough information to triage
- **Severity-based**: Critical vs Warning vs Info
- **De-duplicated**: Avoid alert storms

### SLIs, SLOs, SLAs

- **SLI (Service Level Indicator)**: Measurement (e.g., 99% of requests < 200ms)
- **SLO (Service Level Objective)**: Target (e.g., maintain 99.9% uptime)
- **SLA (Service Level Agreement)**: Contract with consequences if broken


---

## Testing Your Implementation

### Metrics Testing

```bash
# Check Prometheus is scraping targets
curl http://localhost:9090/api/v1/targets

# Query a metric
curl http://localhost:9090/api/v1/query?query=up

# Test application metrics endpoint
curl http://localhost:5000/metrics
```

### Alert Testing

```bash
# Simulate high CPU
stress-ng --cpu 8 --timeout 60s

# Trigger error rate alert
for i in {1..100}; do curl http://localhost:5000/error; done

# Check active alerts
curl http://localhost:9090/api/v1/alerts
```

### Log Testing

```bash
# Send test logs
echo '{"level": "error", "message": "Test error"}' | \
  curl -X POST -H "Content-Type: application/json" \
  -d @- http://localhost:5000/

# Search logs in Elasticsearch
curl http://localhost:9200/_search?q=level:error
```

---

## Common Pitfalls

1. **Alert Fatigue**: Too many noisy alerts → Team ignores them
   - **Solution**: Start with critical alerts only, tune thresholds

2. **No Alert Testing**: Alerts don't fire when needed
   - **Solution**: Test alerts before production, simulate failures

3. **Poor Dashboard Design**: Too much information, unclear visualizations
   - **Solution**: One dashboard per audience, clear purpose

4. **Missing Context**: Alerts without enough info to triage
   - **Solution**: Include relevant labels, annotations, runbook links

5. **No Retention Policy**: Running out of storage
   - **Solution**: Configure retention (30 days metrics, 90 days logs)

6. **Ignoring ML Metrics**: Only monitoring infrastructure
   - **Solution**: Monitor model performance, drift, data quality

---


