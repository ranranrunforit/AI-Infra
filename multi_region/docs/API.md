# API Documentation: Multi-Region Platform

The platform exposes several internal endpoints for monitoring, management, and debugging. Note that these are primarily management APIs; the actual ML inference APIs would be defined by the specific models deployed.

## Base URL
Default: `http://localhost:8080` (or the configured service port)

---

## 1. System Health & Monitoring

### `GET /health`
Returns the aggregated health status of the entire multi-region platform.

**Response:**
```json
{
  "status": "healthy", // "healthy", "degraded", "unhealthy"
  "primary_region": "us-west-2",
  "regions": {
    "us-west-2": {
      "status": "active",
      "latency": "45ms"
    },
    "eu-west-1": {
      "status": "standby",
      "latency": "120ms"
    }
  },
  "timestamp": "2023-10-27T10:00:00Z"
}
```

### `GET /metrics`
Exposes Prometheus metrics for scraping.

**Output Format**: Standard Prometheus text format.

**Key Metrics**:
*   `region_health_status{region="us-west-2"}`: 1 for healthy, 0 for unhealthy.
*   `failover_events_total`: Counter of failover actions.
*   `model_replication_total`: Counter of synced models.
*   `active_requests_total`: Gauge of current inflight requests.

---

## 2. Failover Management

### `POST /failover/trigger`
Manually trigger a failover event. Useful for testing or planned maintenance.

**Body:**
```json
{
  "target_region": "eu-west-1",
  "reason": "maintenance",
  "drain_seconds": 60
}
```

**Response:**
```json
{
  "job_id": "failover-12345",
  "status": "initiated",
  "message": "Failover to eu-west-1 started. DNS propagation may take 60s."
}
```

### `GET /failover/history`
Retrieve a log of past failover events.

**Response:**
```json
[
  {
    "id": "evt-001",
    "timestamp": "2023-10-26T15:30:00Z",
    "source": "us-west-2",
    "target": "eu-west-1",
    "cause": "health_check_failure",
    "duration_ms": 4500
  }
]
```

---

## 3. Cost Analysis

### `GET /cost/report`
Get the latest cost analysis report.

**Parameters:**
*   `start_date` (optional, YYYY-MM-DD): default start of month
*   `end_date` (optional, YYYY-MM-DD): default today

**Response:**
```json
{
  "period": { "start": "2023-10-01", "end": "2023-10-27" },
  "total_cost": 1250.45,
  "currency": "USD",
  "breakdown": {
    "aws": 800.00,
    "gcp": 300.20,
    "azure": 150.25
  },
  "forecast_eom": 1450.00
}
```

### `GET /cost/optimize`
Get actionable optimization recommendations.

**Response:**
```json
[
  {
    "resource_id": "i-0123456789abcdef0",
    "provider": "aws",
    "action": "resize",
    "reason": "CPU utilization < 5% for 7 days",
    "estimated_savings_monthly": 45.00
  }
]
```

---

## 4. Replication

### `POST /replication/sync`
Force an immediate synchronization cycle for models or data.

**Body:**
```json
{
  "type": "model", // or "data"
  "force": true
}
```

**Response:**
```json
{
  "sync_job_id": "sync-9988",
  "status": "queued"
}
```
