# cloud/terraform/gcs.tf — GCS bucket with v4 lifecycle policies
# S3-lifecycle equivalent: STANDARD → NEARLINE → COLDLINE → expiry per priority tier

# ── Main robot data bucket ─────────────────────────────────────────────────
resource "google_storage_bucket" "robot_data" {
  name                        = var.gcs_bucket_name
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy               = false   # Protect production data

  versioning {
    enabled = false   # No versioning on robot data (cost optimization)
  }

  lifecycle_rule {
    condition {
      matches_prefix  = ["robot-data/"]
      matches_suffix  = [".mcap.zst"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 30
      matches_prefix = ["robot-data/p0_critical/"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
    condition {
      age = 365
      matches_prefix = ["robot-data/p0_critical/"]
    }
  }

  # P2/P3 bulk telemetry: expire after 90 days
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age            = 90
      matches_prefix = ["robot-data/p3_low/", "robot-data/p2_normal/"]
    }
  }

  # P1 high: NEARLINE after 30d
  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age            = 30
      matches_prefix = ["robot-data/p1_high/"]
    }
  }

  # Camera data: NEARLINE after 7d → COLDLINE after 90d (v2 retention policy)
  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age            = 7
      matches_prefix = ["robot-data/camera/"]
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
    condition {
      age            = 90
      matches_prefix = ["robot-data/camera/"]
    }
  }

  # Incomplete multipart uploads: abort after 7 days
  lifecycle_rule {
    action {
      type = "AbortIncompleteMultipartUpload"
    }
    condition {
      age = 7
    }
  }

  cors {
    origin          = ["https://app.foxglove.dev", "http://localhost:8080"]
    method          = ["GET", "HEAD"]
    response_header = ["Content-Type", "Range"]
    max_age_seconds = 3600
  }

  labels = {
    environment = var.environment
    project     = "robot-pipeline-v4"
    managed-by  = "terraform"
  }
}

# ── Eventarc trigger: GCS → Cloud Function ────────────────────────────────
# Triggers index_mcap_file function on every new .mcap file upload
resource "google_eventarc_trigger" "mcap_upload" {
  name     = "robot-mcap-upload-trigger"
  location = var.region

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }
  matching_criteria {
    attribute = "bucket"
    value     = google_storage_bucket.robot_data.name
  }

  destination {
    cloud_run_service {
      service = google_cloud_run_v2_service.mcap_indexer.name
      region  = var.region
    }
  }

  service_account = google_service_account.pipeline_fn.email
  depends_on      = [google_project_service.apis]
}

output "gcs_bucket_name" {
  value = google_storage_bucket.robot_data.name
}

output "gcs_bucket_url" {
  value = "gs://${google_storage_bucket.robot_data.name}"
}
