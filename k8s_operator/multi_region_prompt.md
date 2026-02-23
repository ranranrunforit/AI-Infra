# Reimplementation Prompt: Multi-Region ML Platform (multi_region)

## Goal

Build a **production-ready multi-cloud, multi-region ML serving platform** that spans AWS, GCP, and Azure. The platform provides automatic failover, cross-region model replication, cost analysis, and unified monitoring for ML workloads deployed globally.

---

## Tech Stack

- **Language**: Python 3.11+ (async / `asyncio`)
- **HTTP**: `aiohttp` for async HTTP client/server
- **Cloud SDKs**: `boto3`, `aioboto3`, `google-cloud-storage`, `google-cloud-bigquery`, `azure-storage-blob`, `azure-mgmt-costmanagement`
- **Metrics**: `prometheus_client`
- **Infrastructure**: Terraform ≥ 1.5 (AWS EKS, GCP GKE, Azure AKS)
- **Containerization**: Docker + Docker Compose
- **Retry**: `tenacity`

**[requirements.txt](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/requirements.txt)**:
```
aiohttp>=3.9.0
aioboto3>=12.0.0
boto3>=1.33.0
google-cloud-storage>=2.13.0
google-cloud-bigquery>=3.13.0
azure-storage-blob>=12.19.0
azure-mgmt-costmanagement>=4.0.0
azure-identity>=1.15.0
prometheus_client>=0.19.0
tenacity>=8.2.3
pyyaml>=6.0
kubernetes>=28.1.0
```

---

## Project Structure

```
multi_region/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py                        # Async entrypoint
│   ├── failover/
│   │   ├── __init__.py
│   │   ├── failover_controller.py      # Region health + auto-failover
│   │   ├── health_checker.py           # HTTP + K8s health checks
│   │   └── dns_updater.py              # Route53 DNS management
│   ├── replication/
│   │   ├── __init__.py
│   │   ├── model_replicator.py         # Cross-region model sync (S3/GCS/Azure)
│   │   ├── data_sync.py                # Dataset replication
│   │   └── config_sync.py             # Config replication
│   ├── cost/
│   │   ├── __init__.py
│   │   ├── cost_analyzer.py            # Multi-cloud cost aggregation
│   │   └── optimizer.py               # Cost recommendations
│   └── monitoring/
│       ├── __init__.py
│       ├── metrics_aggregator.py       # Prometheus federation
│       └── alerting.py                 # AlertManager + notifications
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── terraform.tfvars.example
│   └── modules/
│       ├── aws/                        # EKS + VPC + S3 + ECR
│       ├── gcp/                        # GKE + VPC + GCS + Artifact Registry
│       ├── azure/                      # AKS + VNet + Blob + ACR
│       └── dns/                        # Route53 with failover routing
├── kubernetes/
│   ├── base/
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml             # Main platform deployment
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   └── overlays/
│       ├── us-west-2/                  # AWS region-specific patches
│       ├── eu-west-1/                  # GCP region-specific patches
│       └── ap-south-1/                 # Azure region-specific patches
├── docker/
│   ├── docker-compose.yml              # Local dev with all services
│   └── Dockerfile
├── scripts/
│   ├── deploy.sh                       # Full deployment script
│   ├── health_check.sh
│   ├── verify_dns.sh
│   └── check_monitoring.sh
├── tests/
│   ├── test_replication.py
│   ├── test_failover.py
│   └── test_cost.py
└── docs/
    ├── ARCHITECTURE.md
    ├── DEPLOYMENT.md
    └── TROUBLESHOOTING.md
```

---

## Configuration

The platform uses a **config dictionary** loaded from environment variables (with optional YAML file override):

```python
config = {
    "regions": [
        {
            "name": "us-west-2",           # AWS primary
            "provider": "aws",
            "endpoint": "...",              # ML serving endpoint hostname
            "k8s_context": "...",           # kubectl context name
            "prometheus_url": "http://...:9091",
            "bucket": "ml-platform-models-us-west-2"  # optional override
        },
        {
            "name": "eu-west-1",           # GCP secondary
            "provider": "gcp",
            "endpoint": "...",
            "k8s_context": "...",
            "prometheus_url": "http://...:9091"
        },
        {
            "name": "ap-south-1",          # Azure tertiary
            "provider": "azure",
            "endpoint": "...",
            "k8s_context": "...",
            "prometheus_url": "http://...:9091",
            "connection_string": "...",    # Azure storage connection string
            "container": "ml-models"
        }
    ],
    "primary_region": "us-west-2",
    "failover_enabled": True,
    "aws_region": "us-west-2",
    "gcp_project_id": "...",
    "gcp_billing_table": "project.dataset.table",  # BigQuery table
    "azure_subscription_id": "...",
    "health_check": {
        "endpoint_path": "/health",
        "timeout_seconds": 5,
        "interval_seconds": 10,
        "failure_threshold": 3,
        "success_threshold": 2
    }
}
```

Env var names: `REGION_1_NAME`, `REGION_1_ENDPOINT`, `REGION_1_K8S_CONTEXT`, `REGION_1_PROMETHEUS_URL`, `PRIMARY_REGION`, `FAILOVER_ENABLED`, `AWS_REGION`, `GCP_PROJECT_ID`, `GCP_BILLING_TABLE`, `AZURE_SUBSCRIPTION_ID`

---

## src/main.py — Async Entry Point

```python
async def main():
    config = load_config()
    
    # Initialize in dependency order:
    metrics_aggregator = MetricsAggregator(config)
    config['registry'] = metrics_aggregator.registry   # share Prometheus registry
    
    alert_manager = AlertManager(config)
    failover_controller = FailoverController(config, alert_manager=alert_manager)
    model_replicator = ModelReplicator(config, registry=metrics_aggregator.registry)
    data_sync = DataSync(config)
    cost_analyzer = CostAnalyzer(config)
    
    # Launch concurrent background tasks:
    tasks = [
        asyncio.create_task(failover_controller.continuous_monitoring(), name="FailoverMonitor"),
        asyncio.create_task(model_replicator.continuous_replication(), name="ModelReplication"),
        asyncio.create_task(data_sync.continuous_sync(), name="DataSync"),
        asyncio.create_task(periodic_cost_report(cost_analyzer), name="CostReport"),
        asyncio.create_task(metrics_aggregator.continuous_collection(), name="MetricsCollection"),
        asyncio.create_task(start_web_servers(metrics_aggregator), name="WebServers"),
    ]
    
    await asyncio.gather(*tasks, return_exceptions=True)
```

**Web server** ([start_web_servers](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/main.py#139-179)):
- Health server on port **8080**: `GET /health` and `GET /ready` → `200 OK`
- Metrics server on port **9090**: `GET /metrics` → Prometheus exposition format from `metrics_aggregator.registry`

**Periodic cost report** ([periodic_cost_report](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/main.py#180-201)):
- Every 24 hours: call `await analyzer.get_costs(yesterday, today)` then `await analyzer.generate_report(...)`
- Log: `Estimated daily cost: $X`
- Sleep 86400s between cycles; catch `asyncio.CancelledError` to exit cleanly

**Signal handlers**: Register `SIGINT` and `SIGTERM` to cancel all tasks gracefully.

**Windows compatibility**: Add `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())` on win32.

---

## src/failover/failover_controller.py

### Data Models (dataclasses/enums)

```python
class RegionHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class FailoverStrategy(Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    GRACEFUL = "graceful"
    IMMEDIATE = "immediate"

@dataclass
class RegionStatus:
    region_name: str
    provider: str
    health: RegionHealth
    endpoint: str
    last_check: str         # ISO 8601
    response_time_ms: float
    error_rate: float
    active_connections: int
    cpu_utilization: float
    memory_utilization: float
    is_primary: bool = False
    consecutive_failures: int = 0

@dataclass
class FailoverEvent:
    event_id: str           # f"failover-{timestamp}"
    timestamp: str
    source_region: str
    target_region: str
    reason: str
    strategy: FailoverStrategy
    status: str             # "initiated" / "completed" / "failed"
    duration_seconds: Optional[float] = None
    affected_connections: int = 0
    error_message: Optional[str] = None
    automated: bool = True

@dataclass
class HealthCheckConfig:
    endpoint_path: str = "/health"
    timeout_seconds: int = 5
    interval_seconds: int = 10
    failure_threshold: int = 3
    success_threshold: int = 2
    expected_status_code: int = 200
```

### HealthChecker

```python
class HealthChecker:
    async def check_endpoint(self, endpoint: str) -> Dict:
        # STANDALONE MODE: If endpoint == "localhost" or starts with "127.0.0.1"
        #   → simulate: sleep 5ms, return {healthy: True, status_code: 200, response_time_ms: 5, error: None}
        # REAL MODE: HTTP GET http://{endpoint}{config.endpoint_path}
        #   → return {healthy: status==200, status_code, response_time_ms, error}
        #   → on timeout → {healthy: False, error: 'timeout'}
        #   → on other exception → {healthy: False, error: str(e)}

    async def check_kubernetes_cluster(self, region_config: Dict) -> Dict:
        # STANDALONE MODE: If k8s_context == 'local' or absent
        #   → return {healthy: True, ready_nodes: 1, total_nodes: 1, system_pods_running: True}
        # REAL MODE: Use kubernetes SDK, load context, check node readiness and control-plane pods
        #   → return {healthy: ready_nodes > 0 and system_pods_running, ready_nodes, total_nodes, ...}
        #   → on exception → {healthy: True} (don't fail in standalone mode)
```

### FailoverController

```python
class FailoverController:
    def __init__(self, config: Dict, alert_manager=None):
        # Initialize self.regions: Dict[str, RegionStatus]
        # Initialize HealthChecker with config.health_check settings
        # Optional Prometheus metrics (from shared registry):
        #   - failover_events_total (Counter, labels: source_region, target_region, reason, status)
        #   - region_health_status (Gauge, labels: region) — 1=healthy, 0=unhealthy

    async def check_region_health(self, region_name: str) -> RegionStatus:
        # 1. check_endpoint + check_kubernetes_cluster in parallel
        # 2. Health determination:
        #    - Both healthy + response < 200ms → HEALTHY
        #    - Both healthy + response 200-500ms → DEGRADED
        #    - Either unhealthy → increment consecutive_failures
        #      - consecutive_failures >= failure_threshold → UNHEALTHY
        #      - else → DEGRADED
        # 3. Update region_health_status Prometheus gauge

    async def check_all_regions(self) -> Dict[str, RegionStatus]:
        # asyncio.gather all per-region health checks concurrently

    def should_trigger_failover(self) -> Optional[str]:
        # Returns reason string or None
        # Triggers if primary is UNHEALTHY → "primary_unhealthy"
        # Triggers if primary is DEGRADED and consecutive_failures >= 5 → "primary_degraded"
        # Triggers if primary error_rate > 0.1 → "high_error_rate"

    def select_failover_target(self, exclude_regions=None) -> Optional[str]:
        # Prefer HEALTHY regions (not primary, not excluded)
        # Fall back to DEGRADED if no HEALTHY candidates
        # Among candidates, pick region with lowest response_time_ms
        # Return None if no candidates

    async def execute_failover(self, target_region, reason, strategy=FailoverStrategy.AUTOMATIC) -> FailoverEvent:
        # 1. Create FailoverEvent with status="initiated"
        # 2. Update self.current_active_region = target_region
        # 3. Flip is_primary flags in self.regions
        # 4. Send alert via alert_manager.send_alert(severity="critical", ...)
        # 5. Set event.status = "completed", record duration
        # 6. On exception: event.status = "failed", record error_message
        # 7. Append to self.failover_history
        # 8. Increment failover_events_total counter

    async def continuous_monitoring(self):
        # Loop every health_check.interval_seconds:
        # 1. check_all_regions()
        # 2. should_trigger_failover() → if reason: select_failover_target() → execute_failover()
        # 3. Handle exceptions gracefully (log, continue loop)
```

---

## src/replication/model_replicator.py

### Storage Adapters

Implement a common **base class** [CloudStorageAdapter](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/replication/model_replicator.py#56-77) with:
```python
async def upload(source_path: str, destination: str) -> bool
async def download(source: str, destination_path: str) -> bool
async def list_objects(prefix: str) -> List[str]
async def get_metadata(object_key: str) -> Dict  # {size, etag, last_modified, metadata}
async def delete(object_key: str) -> bool
```

**S3Adapter** (`provider='aws'`, [bucket](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/replication/model_replicator.py#159-164), [region](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/failover/failover_controller.py#331-339)):
- Use `aioboto3.Session()` — lazy init (create `_session` on first use)
- `async with session.client('s3') as s3:` for each operation
- [upload](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/replication/model_replicator.py#234-246) → `s3.upload_file`, [download](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/replication/model_replicator.py#104-113) → `s3.download_file`
- [list_objects](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/replication/model_replicator.py#185-190) → paginator `list_objects_v2`, yield all `Key` values

**GCSAdapter** (`provider='gcp'`, `bucket_name`):
- Lazy init: `_client = gcs_storage.Client()`, `_bucket = client.bucket(bucket_name)`
- Use `asyncio.to_thread(blob.upload_from_filename, ...)` etc. for blocking calls

**AzureBlobAdapter** (`provider='azure'`, `connection_string`, [container](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#118-176)):
- Lazy init: `BlobServiceClient.from_connection_string(connection_string)`
- Use `asyncio.to_thread(blob_client.upload_blob, data, overwrite=True)` etc.

### Data Models

```python
@dataclass
class ModelMetadata:
    model_id: str
    version: str
    checksum: str        # SHA256 hex
    size_bytes: int
    timestamp: str       # ISO 8601
    source_region: str
    target_regions: List[str]
    format: str          # pytorch / tensorflow / onnx
    framework_version: str
    tags: Dict[str, str]

@dataclass
class ReplicationStatus:
    model_id: str
    version: str
    source_region: str
    target_region: str
    status: str          # pending / in_progress / completed / failed
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    bytes_transferred: int
```

### ModelReplicator

```python
class ModelReplicator:
    def __init__(self, config: Dict, registry=None):
        # self.adapters: Dict[str, CloudStorageAdapter] — keyed by region name
        # Initialize adapters based on provider (aws→S3, gcp→GCS, azure→AzureBlob)
        # Optional Prometheus metrics (shared registry):
        #   - model_replication_total (Counter, labels: source_region, target_region, status)
        #   - model_replication_duration_seconds (Histogram, labels: source_region, target_region)

    @staticmethod
    async def compute_checksum(file_path: str) -> str:
        # SHA256 of file contents (read in 8192-byte chunks)

    async def register_model(self, model_path, model_id, version, source_region,
                              target_regions, format="pytorch", framework_version="2.0.0", tags=None) -> ModelMetadata:
        # Compute checksum, get file size, build and return ModelMetadata

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def replicate_model(self, model_path: str, metadata: ModelMetadata) -> Dict[str, ReplicationStatus]:
        # 1. Upload model binary to source: models/{model_id}/{version}/model.bin
        # 2. Upload metadata JSON to source: models/{model_id}/{version}/metadata.json
        # 3. asyncio.gather all _replicate_to_region tasks for each target_region
        # 4. Return dict: {region_name: ReplicationStatus}

    async def _replicate_to_region(self, source_key, metadata_key, metadata, target_region) -> ReplicationStatus:
        # 1. Download model and metadata from source to /tmp files
        # 2. Verify SHA256 checksum matches metadata.checksum
        # 3. Upload both to target adapter
        # 4. Verify upload: check target metadata size == metadata.size_bytes
        # 5. Record Prometheus metrics (counter + histogram)
        # 6. Cleanup /tmp files (Path.unlink(missing_ok=True))
        # 7. Return ReplicationStatus with status="completed" or "failed"

    async def sync_models(self) -> Dict[str, List[str]]:
        # 1. List models/*.../model.bin in each region
        # 2. Find union of all model keys
        # 3. For each region, find missing keys
        # 4. For missing keys: find source region, download metadata.json, call _replicate_to_region
        # 5. Return sync_plan: {region: [missing_keys]}

    async def continuous_replication(self, interval_seconds: int = 300):
        # Loop: sync_models() every 300s (5 minutes), handle exceptions
```

---

## src/cost/cost_analyzer.py

### Data Models

```python
@dataclass
class CostData:
    service: str
    region: str
    provider: str     # aws / gcp / azure
    amount: Decimal
    currency: str
    start_date: str
    end_date: str
    unit: str         # hours / GB / requests / USD
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class CostReport:
    report_id: str    # f"cost-report-{timestamp}"
    start_date: str
    end_date: str
    total_cost: Decimal
    currency: str
    costs_by_provider: Dict[str, Decimal]
    costs_by_region: Dict[str, Decimal]
    costs_by_service: Dict[str, Decimal]
    daily_costs: List[Tuple[str, Decimal]]    # sorted by date
    anomalies: List[Dict]
    trends: Dict[str, float]                  # {daily_change, weekly_change}
```

### CostAnalyzer

```python
class CostAnalyzer:
    def __init__(self, config: Dict):
        # Lazy-init all three cloud clients:
        # self.aws_ce = boto3.client('ce', region_name=...) or None
        # self.gcp_billing = billing_v1.CloudBillingClient() or None
        # self.azure_cost = CostManagementClient(credential, subscription_id) or None
        # Use try/except for each; log warning if unavailable

    async def get_aws_costs(self, start_date, end_date) -> List[CostData]:
        # Use Cost Explorer: get_cost_and_usage, DAILY granularity,
        # Metric=UnblendedCost, GroupBy=[SERVICE, REGION]
        # Skip if self.aws_ce is None

    async def get_gcp_costs(self, start_date, end_date) -> List[CostData]:
        # Use BigQuery: query gcp_billing_table for cost by service+region
        # SQL: SELECT service.description, location.location, SUM(cost), currency FROM ... WHERE usage_start_time >= ...
        # Skip if gcp_billing_table not configured

    async def get_azure_costs(self, start_date, end_date) -> List[CostData]:
        # Use CostManagement query.usage with QueryDefinition(type=ActualCost, granularity=Daily)
        # GroupBy ServiceName + ResourceLocation
        # Skip if self.azure_cost is None

    async def get_costs(self, start_date, end_date) -> List[CostData]:
        # asyncio.gather all three in parallel (return_exceptions=True)
        # Combine results into self.cost_data

    async def generate_report(self, start_date, end_date) -> CostReport:
        # get_costs() then generate_cost_report()

    def generate_cost_report(self, start_date, end_date) -> CostReport:
        # Aggregate self.cost_data into CostReport fields
        # _detect_anomalies: group by service, find values >2 std dev from mean
        # _calculate_trends: daily_change and weekly_change percentages
```

---

## src/monitoring/metrics_aggregator.py

### Data Models

```python
@dataclass
class RegionMetrics:
    region: str
    timestamp: str
    request_rate: float          # req/s
    error_rate: float            # 0.0–1.0
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    cpu_utilization: float       # 0–100
    memory_utilization: float    # 0–100
    active_pods: int
    total_pods: int
```

### MetricsAggregator

```python
class MetricsAggregator:
    def __init__(self, config: Dict):
        self.registry = CollectorRegistry()   # independent from default registry
        # Initialize Prometheus gauges (on self.registry):
        #   multiregion_request_rate (labels: region)
        #   multiregion_error_rate (labels: region)
        #   multiregion_latency_ms (labels: region, percentile)

    async def query_prometheus(self, prometheus_url: str, query: str) -> Optional[Dict]:
        # GET {prometheus_url}/api/v1/query?query=...
        # Return response['data'] or None on timeout/connection error/non-200

    async def collect_region_metrics(self, region_config: Dict) -> Optional[RegionMetrics]:
        # If Prometheus unavailable → _create_default_metrics (simulated values)
        # PromQL queries:
        #   request_rate: sum(rate(http_requests_total[5m]))
        #   error_rate: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
        #   p50/p95/p99: histogram_quantile(0.50/0.95/0.99, rate(http_request_duration_seconds_bucket[5m]))
        #   cpu: avg(rate(container_cpu_usage_seconds_total[5m]))
        #   memory: avg(container_memory_usage_bytes / container_spec_memory_limit_bytes)
        #   pods: count(kube_pod_info)
        # Update Prometheus gauges from collected values

    def _create_default_metrics(self, region: str) -> RegionMetrics:
        # Use random.uniform() to simulate realistic-looking metrics:
        # request_rate: 50–150 req/s, error_rate: 0.001–0.01
        # p50: 10–30ms, p95: 50–100ms, p99: 100–200ms
        # cpu: 20–60%, memory: 30–70%, pods: 3

    async def collect_all_metrics(self) -> Dict[str, RegionMetrics]:
        # asyncio.gather all regions concurrently

    def get_global_metrics(self) -> Dict:
        # Aggregate: total_request_rate, avg_error_rate, avg_p99_latency

    async def continuous_collection(self, interval_seconds: int = 60):
        # Loop: collect_all_metrics() every 60s
```

---

## src/monitoring/alerting.py — AlertManager

```python
class AlertManager:
    async def send_alert(self, severity: str, title: str, message: str, tags: Dict = None):
        # Log the alert at appropriate level (severity=critical → logger.critical, etc.)
        # Optionally send to webhook URLs from config (e.g., Slack, PagerDuty)
        # severity options: info / warning / critical
```

---

## src/replication/data_sync.py — DataSync

```python
class DataSync:
    async def continuous_sync(self, interval_seconds: int = 600):
        # Loop: discover and sync datasets across regions every 10 minutes
        # Similar pattern to ModelReplicator but for datasets/artifacts
```

---

## Terraform Infrastructure

### [terraform/variables.tf](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/terraform/variables.tf)

```hcl
variable "project_name" { default = "ml-platform" }
variable "environment"  { default = "prod" }
variable "domain_name"  { default = "example.com" }
variable "gcp_project_id" {}
variable "use_spot_instances" { default = true }
variable "enable_gpu"         { default = false }
variable "enable_weighted_routing" { default = true }
```

### [terraform/main.tf](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/terraform/main.tf)

Call four modules:
```hcl
module "aws"   { source = "./modules/aws"   ... }
module "gcp"   { source = "./modules/gcp"   ... }
module "azure" { source = "./modules/azure" ... }
module "dns"   { source = "./modules/dns"   ... }
```

### `terraform/modules/aws/`
- **EKS cluster** with multi-AZ managed node groups
- **VPC** with public + private subnets, NAT gateway
- **S3 buckets**: models, logs (with versioning enabled)
- **ECR** repository for container images
- **IAM roles** for node groups

### `terraform/modules/gcp/`
- **GKE cluster** with auto-scaling node pools
- **VPC** with Cloud NAT
- **GCS buckets** for models/data
- **Artifact Registry** for containers
- **Workload Identity** for pod IAM binding

### `terraform/modules/azure/`
- **AKS cluster** with spot VM node pools
- **VNet** + subnets
- **Azure Storage Account** + Blob containers
- **ACR** (Azure Container Registry)
- **Managed Identity** for AKS

### `terraform/modules/dns/`
- **Route53 hosted zone** for `domain_name`
- **Failover routing**: primary → `us-west-2`, secondary → EU/AP
- **Latency-based routing** records
- **Health checks** for each region endpoint

---

## Kubernetes Manifests

### `kubernetes/base/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-platform-config
data:
  config.yaml: |
    server:
      port: 8080
      host: "0.0.0.0"
    replication:
      enabled: true
      sync_interval_seconds: 300
    failover:
      enabled: true
      health_check_interval_seconds: 10
```

### `kubernetes/base/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-platform
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: ml-platform
          image: ml-platform:latest
          command: ["python", "src/main.py"]
          ports:
            - containerPort: 8080   # health
            - containerPort: 9090   # metrics
          env:
            - name: PRIMARY_REGION
              value: us-west-2
            - name: FAILOVER_ENABLED
              value: "true"
          livenessProbe:
            httpGet: {path: /health, port: 8080}
          readinessProbe:
            httpGet: {path: /ready, port: 8080}
```

---

## Docker Compose (Local Development)

```yaml
# docker/docker-compose.yml
version: "3.8"
services:
  ml-platform:
    build: ..
    ports:
      - "8080:8080"   # health
      - "9090:9090"   # metrics
    environment:
      - PRIMARY_REGION=us-west-2
      - FAILOVER_ENABLED=true
      - REGION_1_NAME=us-west-2
      - REGION_1_ENDPOINT=localhost
      - REGION_2_NAME=eu-west-1
      - REGION_2_ENDPOINT=localhost
      - REGION_3_NAME=ap-south-1
      - REGION_3_ENDPOINT=localhost

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

---

## Key Design Patterns

1. **Everything is async** — all I/O uses `asyncio`, `aiohttp`, `asyncio.to_thread()` for blocking SDK calls
2. **Standalone mode** — services work locally without real cloud credentials; health checker simulates healthy responses for `localhost` endpoints; MetricsAggregator falls back to random simulated metrics
3. **Lazy initialization** — cloud SDK clients are initialized lazily (not at import time) to avoid credential errors on startup
4. **Graceful degradation** — each cloud provider in CostAnalyzer and ModelReplicator is optional; missing credentials emit a `logger.warning` and are skipped
5. **Shared Prometheus registry** — [MetricsAggregator](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/monitoring/metrics_aggregator.py#37-286) creates a `CollectorRegistry()` and passes it to other services to avoid metric name conflicts
6. **Retry on replication** — use `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))` on [replicate_model](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/replication/model_replicator.py#398-451)
7. **Concurrent health checks** — all region checks run concurrently with `asyncio.gather`
8. **Failover target selection** — prefer lowest-latency healthy region; fall back to degraded if none

---

## Performance Targets

- Global P99 Latency: < 300ms
- Availability: 99.95% (45 min/month downtime max)
- Failover Time: < 60 seconds
- Replication Lag: < 5 minutes
- Cost Savings vs single-region: ~30% (via spot instances + multi-cloud arbitrage)
