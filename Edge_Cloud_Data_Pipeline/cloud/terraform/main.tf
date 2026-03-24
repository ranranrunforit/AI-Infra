# cloud/terraform/main.tf — Robot Pipeline v4 GCP Infrastructure

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
  # Remote state: enable for team usage
  # backend "gcs" {
  #   bucket = "YOUR_TFSTATE_BUCKET"
  #   prefix = "robot-pipeline/tfstate"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ── Enable required APIs ──────────────────────────────────────────────────────
resource "google_project_service" "apis" {
  for_each = toset([
    "cloudfunctions.googleapis.com",
    "cloudbuild.googleapis.com",
    "eventarc.googleapis.com",
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
    "cloudtrace.googleapis.com",
    "artifactregistry.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# ── Service Account for Cloud Functions ──────────────────────────────────────
resource "google_service_account" "pipeline_fn" {
  account_id   = "robot-pipeline-fn"
  display_name = "Robot Pipeline Cloud Functions SA"
}

resource "google_project_iam_member" "fn_gcs_reader" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.pipeline_fn.email}"
}

resource "google_project_iam_member" "fn_cloudsql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.pipeline_fn.email}"
}

resource "google_project_iam_member" "fn_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.pipeline_fn.email}"
}

# ── Service Account for Robot Agents ─────────────────────────────────────────
resource "google_service_account" "robot_agent" {
  account_id   = "robot-agent"
  display_name = "Robot Edge Agent SA (upload to GCS)"
}

resource "google_project_iam_member" "robot_gcs_writer" {
  project = var.project_id
  role    = "roles/storage.objectCreator"
  member  = "serviceAccount:${google_service_account.robot_agent.email}"
}

resource "google_service_account" "pipeline_robot" {
  account_id   = "pipeline-robot"
  display_name = "Pipeline Robot Service Account"
  project      = var.project_id
}

resource "google_project_iam_member" "fn_run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.pipeline_fn.email}"
}

resource "google_project_iam_member" "eventarc_agent_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-eventarc.iam.gserviceaccount.com"
}
data "google_project" "project" {
  project_id = var.project_id
}

resource "google_project_iam_member" "fn_eventarc_receiver" {
  project = var.project_id
  role    = "roles/eventarc.eventReceiver"
  member  = "serviceAccount:${google_service_account.pipeline_fn.email}"
}

resource "google_project_iam_member" "gcs_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${data.google_storage_project_service_account.gcs_sa.email_address}"
}

data "google_storage_project_service_account" "gcs_sa" {
  project = var.project_id
}
