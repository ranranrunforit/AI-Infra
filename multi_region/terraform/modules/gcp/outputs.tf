# GCP Module Outputs

output "cluster_id" {
  description = "GKE cluster ID"
  value       = google_container_cluster.main.id
}

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.main.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.main.endpoint
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.main.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.main.name
}

output "subnet_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.main.name
}

output "models_bucket_name" {
  description = "GCS bucket name for models"
  value       = google_storage_bucket.models.name
}

output "models_bucket_url" {
  description = "GCS bucket URL for models"
  value       = google_storage_bucket.models.url
}

output "logs_bucket_name" {
  description = "GCS bucket name for logs"
  value       = google_storage_bucket.logs.name
}

output "artifact_registry_id" {
  description = "Artifact Registry repository ID"
  value       = google_artifact_registry_repository.ml_serving.id
}

output "service_account_email" {
  description = "Service account email for ML serving"
  value       = google_service_account.ml_serving.email
}

output "region" {
  description = "GCP region"
  value       = var.region
}

output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}
