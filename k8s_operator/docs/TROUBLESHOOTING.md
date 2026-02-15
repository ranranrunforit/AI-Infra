# Troubleshooting Guide

This guide helps diagnose and resolve common issues encountered in the Multi-Region ML Platform.

## 1. Cloud Connectivity Issues

### Issue: `CredentialsError` or `AuthFailure`
**Symptoms**: Services crash on startup; logs show "Unable to locate credentials".
**Diagnosis**:
*   The application cannot find cloud provider credentials.
**Solution**:
*   **AWS**: Check `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` or `~/.aws/credentials`. If on EKS, check IRSA service account annotation.
*   **GCP**: Check `GOOGLE_APPLICATION_CREDENTIALS` points to a valid JSON key file. If on GKE, check Workload Identity.
*   **Azure**: Check `AZURE_SUBSCRIPTION_ID`, `AZURE_CLIENT_ID`, etc. or run `az login` locally.

### Issue: `AccessDenied` accessing Object Storage
**Symptoms**: `ModelReplicator` logs permission errors when listing/copying objects.
**Solution**:
1.  Verify the IAM Role/Service Account has `Storage Object Admin` or `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` permissions.
2.  Check if the bucket name in `config.yaml` matches the actual cloud resource.

## 2. Kubernetes & Deployment

### Issue: Pods stuck in `CrashLoopBackOff`
**Diagnosis**:
```bash
kubectl logs -n ml-serving <pod-name> --previous
```
**Common Causes**:
*   Missing environment variables (Secrets/ConfigMaps not applied).
*   Application code error (Syntax error, Import error).
*   OOMKilled (Memory limit too low).

### Issue: Services not accessible via LoadBalancer
**Symptoms**: External IP is `<pending>` forever.
**Solution**:
*   **AWS**: Ensure subnets are tagged `kubernetes.io/role/elb = 1`.
*   **GCP/Azure**: Check quota limits for Static IPs using the cloud console.

## 3. Failover & DNS

### Issue: DNS not failing over to Secondary
**Symptoms**: Primary is down, but `ml.example.com` still resolves to Primary IP.
**Solution**:
1.  Check `FailoverController` logs:
    ```bash
    kubectl logs -l app=failover-controller
    ```
    Ensure it detects the failure ("Region us-west-2 is UNHEALTHY").
2.  Check TTL: Route53 records have a TTL (e.g., 60s). Propagation takes time.
3.  Check Route53 Health Checks: Are they properly associated with the Record Set?

### Issue: Flapping (Rapid switching between regions)
**Symptoms**: Traffic keeps moving between regions every minute.
**Solution**:
*   Increase `failure_threshold` in `config.yaml` (e.g., from 3 to 5).
*   Increase `health_check_interval_seconds`.

## 4. Cost Analysis

### Issue: Empty Cost Reports
**Symptoms**: `CostAnalyzer` returns 0 cost or empty lists.
**Solution**:
*   **AWS**: Cost Explorer API takes ~24h to populate data. Real-time data is not available.
*   **Permissions**: Ensure the credentials have `ce:GetCostAndUsage` (AWS), `billing.accounts.getSpendingInformation` (GCP).

## 5. Metrics & Alerting

### Issue: Missing Metrics in Prometheus
**Solution**:
*   Verify the application exposes metrics at `/metrics` (or port 9090).
    ```bash
    curl http://localhost:9090/metrics
    ```
*   Check ServiceMonitor configuration if using Prometheus Operator.

## 6. General Debugging Tools

### Run Verify Script
The built-in verification script checks most connectivity issues:
```bash
python multi_region/scripts/verify_cloud.py
```

### Check Logs
```bash
# Application Logs
kubectl logs -f deployment/ml-platform -n ml-serving

# Previous Logs (if crashed)
kubectl logs -p deployment/ml-platform -n ml-serving
```
