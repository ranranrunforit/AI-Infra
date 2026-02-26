# =============================================================================
# GCP Terraform - Main Configuration
# Project 303: Enterprise LLM Platform with RAG
#
# This deploys the GCP infrastructure alongside the original AWS design.
# Original AWS Terraform is kept at: terraform/modules/gpu-nodes/
#
# Deploys:
#   - Cloud Run service for RAG API
#   - GCS bucket for documents
#   - Service accounts & IAM
#   - (Optional) GKE Autopilot cluster for Qdrant
#   - (Optional) Compute Engine Spot VM for vLLM self-hosting
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }

  # Uncomment to use GCS backend for state (recommended for teams)
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "project-303/gcp"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ── Enable required GCP APIs ──────────────────────────────────────────────────

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",                 # Cloud Run
    "artifactregistry.googleapis.com",     # Docker image registry
    "storage.googleapis.com",             # Cloud Storage (GCS)
    "secretmanager.googleapis.com",       # Secret Manager for API keys
    "container.googleapis.com",           # GKE (for Qdrant)
    "compute.googleapis.com",             # Compute Engine (for vLLM VM)
    "generativelanguage.googleapis.com",  # Gemini API
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# ── Artifact Registry (Docker images) ────────────────────────────────────────

resource "google_artifact_registry_repository" "rag_platform" {
  location      = var.region
  repository_id = "rag-platform"
  description   = "Project 303 LLM RAG Platform Docker images"
  format        = "DOCKER"

  depends_on = [google_project_service.apis]
}

# ── Secret Manager - Store API keys securely ──────────────────────────────────

resource "google_secret_manager_secret" "google_api_key" {
  secret_id = "rag-google-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "rag_api_key" {
  secret_id = "rag-service-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# ─────────────────────────────────────────────────────────────────────────────
# Outputs
# ─────────────────────────────────────────────────────────────────────────────

output "cloud_run_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.rag_api.uri
}

output "qdrant_internal_ip" {
  description = "Qdrant VM internal IP (if deployed)"
  value       = var.enable_qdrant_vm ? google_compute_instance.qdrant[0].network_interface[0].network_ip : "N/A"
}

output "artifact_registry_repo" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/rag-platform"
}

output "docker_push_command" {
  description = "Command to push RAG API image"
  value       = "docker tag rag-api:latest ${var.region}-docker.pkg.dev/${var.project_id}/rag-platform/rag-api:latest && docker push ${var.region}-docker.pkg.dev/${var.project_id}/rag-platform/rag-api:latest"
}

output "add_gemini_key_command" {
  description = "Command to set Gemini API key in Secret Manager"
  value       = "echo -n 'YOUR_GEMINI_API_KEY' | gcloud secrets versions add rag-google-api-key --data-file=-"
}
