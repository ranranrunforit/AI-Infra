# =============================================================================
# GCP Terraform - Cloud Run + Qdrant VM + vLLM VM
# Project 303: Enterprise LLM RAG Platform
# =============================================================================

# ── Service Account for Cloud Run ─────────────────────────────────────────────

resource "google_service_account" "rag_api" {
  account_id   = "rag-api-sa"
  display_name = "RAG API Service Account"
  description  = "Service account for the RAG API Cloud Run service"
}

# IAM: Allow Cloud Run SA to access secrets
resource "google_secret_manager_secret_iam_member" "rag_api_gemini_key" {
  secret_id = google_secret_manager_secret.google_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.rag_api.email}"
}

resource "google_secret_manager_secret_iam_member" "rag_api_service_key" {
  secret_id = google_secret_manager_secret.rag_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.rag_api.email}"
}

# IAM: Allow Cloud Run SA to write to GCS
resource "google_project_iam_member" "rag_api_gcs" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.rag_api.email}"
}

# ── GCS Bucket for Documents ──────────────────────────────────────────────────

resource "google_storage_bucket" "documents" {
  name          = "${var.project_id}-rag-documents"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = var.tags
}

# ── Cloud Run Service (RAG API) ───────────────────────────────────────────────

resource "google_cloud_run_v2_service" "rag_api" {
  name     = "rag-api"
  location = var.region

  # Allow unauthenticated access (API key auth handled by the app)
  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.rag_api.email

    scaling {
      min_instance_count = var.cloud_run_min_instances
      max_instance_count = var.cloud_run_max_instances
    }

    containers {
      # Image from Artifact Registry
      image = "${var.region}-docker.pkg.dev/${var.project_id}/rag-platform/rag-api:latest"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cloud_run_cpu
          memory = var.cloud_run_memory
        }
        startup_cpu_boost = true  # Faster cold starts
      }

      # ── Environment Variables ───────────────────────────────────────────────
      env {
        name  = "LLM_BACKEND"
        value = var.llm_backend
      }

      env {
        name  = "GEMINI_MODEL"
        value = var.gemini_model
      }

      env {
        name  = "EMBEDDING_MODEL"
        value = var.embedding_model
      }

      env {
        name  = "QDRANT_HOST"
        value = var.enable_qdrant_vm ? google_compute_instance.qdrant[0].network_interface[0].network_ip : "localhost"
      }

      env {
        name  = "ENABLE_RERANKING"
        value = "true"
      }

      env {
        name  = "ENABLE_PII_DETECTION"
        value = "true"
      }

      # ── Secrets from Secret Manager ─────────────────────────────────────────
      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.google_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.rag_api_key.secret_id
            version = "latest"
          }
        }
      }

      # ── Health check ────────────────────────────────────────────────────────
      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        period_seconds        = 10
        failure_threshold     = 10
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        period_seconds    = 30
        failure_threshold = 3
      }
    }

    labels = var.tags
  }

  depends_on = [
    google_project_service.apis,
    google_artifact_registry_repository.rag_platform,
  ]
}

# Allow unauthenticated invocations of Cloud Run
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = google_cloud_run_v2_service.rag_api.project
  location = google_cloud_run_v2_service.rag_api.location
  name     = google_cloud_run_v2_service.rag_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Qdrant VM (optional, for persistent vector store) ────────────────────────

resource "google_compute_instance" "qdrant" {
  count = var.enable_qdrant_vm ? 1 : 0

  name         = "qdrant-vector-db"
  machine_type = var.qdrant_machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 50   # GB - for vector data storage
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = "default"
    # No external IP for security (Cloud Run uses internal IP)
  }

  # Startup script: install Docker and run Qdrant
  metadata_startup_script = <<-STARTUP
    #!/bin/bash
    set -e
    # Install Docker
    curl -fsSL https://get.docker.com | sh

    # Create data directory
    mkdir -p /data/qdrant

    # Run Qdrant
    docker run -d \
      --name qdrant \
      --restart unless-stopped \
      -p 6333:6333 \
      -p 6334:6334 \
      -v /data/qdrant:/qdrant/storage \
      qdrant/qdrant:v1.9.0

    echo "Qdrant started successfully"
  STARTUP

  metadata = {
    enable-osconfig = "TRUE"
  }

  service_account {
    email  = google_service_account.rag_api.email
    scopes = ["cloud-platform"]
  }

  labels = var.tags

  tags = ["qdrant-server", "allow-internal"]
}

# Firewall rule: allow Cloud Run to reach Qdrant
resource "google_compute_firewall" "allow_qdrant_internal" {
  count = var.enable_qdrant_vm ? 1 : 0

  name    = "allow-qdrant-internal"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["6333", "6334"]
  }

  source_ranges = ["10.0.0.0/8"]  # Internal VPC only
  target_tags   = ["qdrant-server"]
}

# ── vLLM Spot VM (optional, for self-hosted Llama 3 / Mistral) ───────────────

resource "google_compute_instance" "vllm" {
  count = var.enable_vllm_vm ? 1 : 0

  name         = "vllm-inference-server"
  machine_type = var.vllm_machine_type  # g2-standard-4: 1x L4 GPU
  zone         = var.zone

  # Spot (preemptible) for ~70% cost savings
  scheduling {
    preemptible         = true
    automatic_restart   = false
    on_host_maintenance = "TERMINATE"
    provisioning_model  = "SPOT"
  }

  boot_disk {
    initialize_params {
      image = "projects/deeplearning-platform-release/global/images/family/common-cu121-debian-11-py310"
      size  = 100  # GB - for model storage
      type  = "pd-ssd"
    }
  }

  # GPU attachment
  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  network_interface {
    network = "default"
    # No external IP for security
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  # Startup script: install vLLM and start serving
  metadata_startup_script = <<-STARTUP
    #!/bin/bash
    set -e

    # Install vLLM (requires CUDA)
    pip install vllm>=0.4.0

    # Start vLLM server
    # Using Mistral 7B by default (fits in 1x L4 GPU)
    python -m vllm.entrypoints.openai.api_server \
      --model ${var.vllm_model} \
      --host 0.0.0.0 \
      --port 8000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 64 \
      --trust-remote-code \
      --dtype bfloat16 \
      ${var.vllm_hf_token != "" ? "--token ${var.vllm_hf_token}" : ""} \
      &

    echo "vLLM server starting with model: ${var.vllm_model}"
  STARTUP

  service_account {
    email  = google_service_account.rag_api.email
    scopes = ["cloud-platform"]
  }

  labels = merge(var.tags, {
    workload = "vllm-inference"
    model    = replace(var.vllm_model, "/", "-")
  })

  tags = ["vllm-server", "allow-internal"]

  depends_on = [google_project_service.apis]
}

# Firewall rule: allow Cloud Run to reach vLLM
resource "google_compute_firewall" "allow_vllm_internal" {
  count = var.enable_vllm_vm ? 1 : 0

  name    = "allow-vllm-internal"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["10.0.0.0/8"]
  target_tags   = ["vllm-server"]
}
