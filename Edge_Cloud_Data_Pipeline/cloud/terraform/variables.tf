# cloud/terraform/variables.tf

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment: demo | staging | production"
  type        = string
  default     = "demo"
}

variable "gcs_bucket_name" {
  description = "GCS bucket name for robot MCAP data"
  type        = string
  default     = "robot-pipeline-v4-demo"
}

variable "db_tier" {
  description = "Cloud SQL machine type"
  type        = string
  default     = "db-f1-micro"   # $7/mo; use db-g1-small ($25/mo) for production
}

variable "postgres_password" {
  description = "PostgreSQL pipeline user password"
  type        = string
  sensitive   = true
}

variable "db_backend" {
  description = "Database backend to use: 'cloudsql'"
  type        = string
  default     = "cloudsql"
}
