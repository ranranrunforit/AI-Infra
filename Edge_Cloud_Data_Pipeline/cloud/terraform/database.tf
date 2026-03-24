# cloud/terraform/database.tf
#
# Database backend:
#
#   db_backend = "cloudsql"      (default)
#     Creates Cloud SQL Postgres with private VPC.
#     Cost: ~$7/mo for db-f1-micro (free tier does not exist).
#     Best for: production fleets, sub-millisecond latency, no cold starts.
#
# Usage in demo.tfvars:

# Usage in production.tfvars:
#   db_backend = "cloudsql"
#   postgres_password = "your-strong-password"

locals {
  use_cloudsql    = var.db_backend == "cloudsql"
}

# ── Cloud SQL (only created when db_backend = "cloudsql") ─────────────────────

resource "google_sql_database_instance" "robot_pipeline" {
  count            = local.use_cloudsql ? 1 : 0
  name             = "robot-pipeline-pg"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier              = var.db_tier
    availability_type = "ZONAL"
    disk_size         = 20
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    database_flags {
      name  = "max_connections"
      value = "100"
    }
    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"
    }

    backup_configuration {
      enabled    = true
      start_time = "03:00"
      backup_retention_settings {
        retained_backups = 7
      }
    }

    ip_configuration {
      ipv4_enabled    = true   # Required for Cloud SQL Auth Proxy in Cloud Run
      private_network = google_compute_network.pipeline_vpc[0].id
    }

    insights_config {
      query_insights_enabled = true
      query_string_length    = 1024
    }
  }

  deletion_protection = var.environment == "production"
  depends_on = [
    google_project_service.apis,
    google_service_networking_connection.private_vpc_connection[0],
  ]
}

resource "google_sql_database" "robot_pipeline" {
  count    = local.use_cloudsql ? 1 : 0
  name     = "robot_pipeline"
  instance = google_sql_database_instance.robot_pipeline[0].name
}

resource "google_sql_user" "pipeline" {
  count    = local.use_cloudsql ? 1 : 0
  name     = "pipeline"
  instance = google_sql_database_instance.robot_pipeline[0].name
  password = var.postgres_password
}

# ── VPC (only when using Cloud SQL private IP) ────────────────────────────────

resource "google_compute_network" "pipeline_vpc" {
  count                   = local.use_cloudsql ? 1 : 0
  name                    = "robot-pipeline-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "pipeline_subnet" {
  count         = local.use_cloudsql ? 1 : 0
  name          = "robot-pipeline-subnet"
  ip_cidr_range = "10.10.0.0/24"
  region        = var.region
  network       = google_compute_network.pipeline_vpc[0].id
}

resource "google_compute_global_address" "private_ip_range" {
  count         = local.use_cloudsql ? 1 : 0
  name          = "robot-pipeline-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.pipeline_vpc[0].id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  count                   = local.use_cloudsql ? 1 : 0
  network                 = google_compute_network.pipeline_vpc[0].id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_range[0].name]
}

# ── Secrets ────────────────────────────────────────────────────────────────────

resource "google_secret_manager_secret" "pg_password" {
  count     = local.use_cloudsql ? 1 : 0
  secret_id = "robot-pipeline-pg-password"
  replication {
  auto {}
}
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "pg_password" {
  count       = local.use_cloudsql ? 1 : 0
  secret      = google_secret_manager_secret.pg_password[0].id
  secret_data = var.postgres_password
}



# ── Artifact Registry ─────────────────────────────────────────────────────────

resource "google_artifact_registry_repository" "robot_pipeline" {
  location      = var.region
  repository_id = "robot-pipeline"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# ── Cloud Run MCAP Indexer ────────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "mcap_indexer" {
  name     = "mcap-indexer"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  template {
    service_account = google_service_account.pipeline_fn.email

    containers {
      image = "us-central1-docker.pkg.dev/${var.project_id}/robot-pipeline/mcap-indexer:latest"

      # Which backend to use
      env {
        name  = "DB_BACKEND"
        value = var.db_backend
      }

      # Cloud SQL backend vars (only meaningful when db_backend=cloudsql)
      dynamic "env" {
        for_each = local.use_cloudsql ? [1] : []
        content {
          name  = "POSTGRES_HOST"
          value = "/cloudsql/data-pipeline-cz78:us-central1:robot-pipeline-pg"
        }
      }
      dynamic "env" {
        for_each = local.use_cloudsql ? [1] : []
        content {
          name  = "POSTGRES_DB"
          value = "robot_pipeline"
        }
      }
      dynamic "env" {
        for_each = local.use_cloudsql ? [1] : []
        content {
          name  = "POSTGRES_USER"
          value = "pipeline"
        }
      }
      dynamic "env" {
        for_each = local.use_cloudsql ? [1] : []
        content {
          name = "POSTGRES_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.pg_password[0].secret_id
              version = "latest"
            }
          }
        }
      }



      env {
        name  = "GCS_BUCKET"
        value = var.gcs_bucket_name
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = ["data-pipeline-cz78:us-central1:robot-pipeline-pg"]
      }
    }

    # VPC access only needed for Cloud SQL private IP connectivity
    dynamic "vpc_access" {
      for_each = local.use_cloudsql ? [1] : []
      content {
        network_interfaces {
          network    = google_compute_network.pipeline_vpc[0].id
          subnetwork = google_compute_subnetwork.pipeline_subnet[0].id
        }
        egress = "PRIVATE_RANGES_ONLY"
      }
    }
  }

  depends_on = [google_project_service.apis]
}

# ── Dead-letter queue for failed indexer invocations ─────────────────────────

resource "google_pubsub_topic" "indexer_dlq" {
  name = "robot-indexer-dead-letter"
  labels = {
    project = "robot-pipeline-v4"
  }
}

resource "google_pubsub_subscription" "indexer_dlq_sub" {
  name  = "robot-indexer-dead-letter-sub"
  topic = google_pubsub_topic.indexer_dlq.name

  message_retention_duration = "604800s"  # 7 days
  retain_acked_messages      = true

  expiration_policy {
    ttl = ""  # never expire
  }
}

# ── Outputs ────────────────────────────────────────────────────────────────────

output "cloud_sql_private_ip" {
  value     = local.use_cloudsql ? google_sql_database_instance.robot_pipeline[0].private_ip_address : "N/A"
  sensitive = true
}

output "mcap_indexer_url" {
  value = google_cloud_run_v2_service.mcap_indexer.uri
}

output "db_backend_active" {
  value = var.db_backend
}

output "robot_agent_env" {
  description = "Environment variables to export on each robot"
  value       = <<-ENV
    export GCS_BUCKET=$${var.gcs_bucket_name}
    export ROBOT_ID=robot-001
    export POSTGRES_HOST=$${google_sql_database_instance.robot_pipeline[0].private_ip_address}
  ENV
}

output "init_sql_command" {
  description = "SQL init command for the selected backend"
  value = local.use_cloudsql ? (
    "psql -h $(terraform output -raw cloud_sql_private_ip) -U pipeline -d robot_pipeline < ../../schema/init.sql"
  ) : ""
}
