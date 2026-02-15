# Main Terraform Outputs

# AWS Outputs
output "aws_cluster_endpoint" {
  description = "AWS EKS cluster endpoint"
  value       = module.aws_us_west_2.cluster_endpoint
}

output "aws_cluster_name" {
  description = "AWS EKS cluster name"
  value       = module.aws_us_west_2.cluster_name
}

output "aws_models_bucket" {
  description = "AWS S3 bucket for models"
  value       = module.aws_us_west_2.models_bucket_name
}

output "aws_ecr_repository" {
  description = "AWS ECR repository URL"
  value       = module.aws_us_west_2.ecr_repository_url
}

# GCP Outputs
output "gcp_cluster_endpoint" {
  description = "GCP GKE cluster endpoint"
  value       = module.gcp_eu_west_1.cluster_endpoint
}

output "gcp_cluster_name" {
  description = "GCP GKE cluster name"
  value       = module.gcp_eu_west_1.cluster_name
}

output "gcp_models_bucket" {
  description = "GCP GCS bucket for models"
  value       = module.gcp_eu_west_1.models_bucket_name
}

output "gcp_artifact_registry" {
  description = "GCP Artifact Registry ID"
  value       = module.gcp_eu_west_1.artifact_registry_id
}

# Azure Outputs
output "azure_cluster_endpoint" {
  description = "Azure AKS cluster endpoint"
  value       = module.azure_ap_south_1.cluster_endpoint
}

output "azure_cluster_name" {
  description = "Azure AKS cluster name"
  value       = module.azure_ap_south_1.cluster_name
}

output "azure_models_storage" {
  description = "Azure Storage account for models"
  value       = module.azure_ap_south_1.models_storage_account_name
}

output "azure_container_registry" {
  description = "Azure Container Registry login server"
  value       = module.azure_ap_south_1.container_registry_login_server
}

# DNS Outputs
output "primary_endpoint" {
  description = "Primary global endpoint"
  value       = module.dns.primary_endpoint
}

output "latency_based_endpoint" {
  description = "Latency-based routing endpoint"
  value       = module.dns.latency_based_endpoint
}

output "regional_endpoints" {
  description = "Regional-specific endpoints"
  value       = module.dns.regional_endpoints
}

output "certificate_arn" {
  description = "ACM certificate ARN"
  value       = module.dns.certificate_arn
}

# Infrastructure Outputs
output "replication_metadata_bucket" {
  description = "S3 bucket for replication metadata"
  value       = aws_s3_bucket.replication_metadata.id
}

output "distributed_lock_table" {
  description = "DynamoDB table for distributed locking"
  value       = aws_dynamodb_table.distributed_lock.name
}

output "central_log_group" {
  description = "CloudWatch log group for centralized logging"
  value       = aws_cloudwatch_log_group.central.name
}

output "alerts_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = aws_sns_topic.alerts.arn
}

# Cluster Configuration for kubectl
output "cluster_configs" {
  description = "Commands to configure kubectl for each cluster"
  value = {
    aws = "aws eks update-kubeconfig --region us-west-2 --name ${module.aws_us_west_2.cluster_name}"
    gcp = "gcloud container clusters get-credentials ${module.gcp_eu_west_1.cluster_name} --region europe-west1 --project ${var.gcp_project_id}"
    azure = "az aks get-credentials --resource-group ${module.azure_ap_south_1.resource_group_name} --name ${module.azure_ap_south_1.cluster_name}"
  }
  sensitive = true
}
