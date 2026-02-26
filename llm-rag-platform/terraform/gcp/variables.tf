variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run and GCS"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for Compute Engine instances"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Deployment environment (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "gemini_model" {
  description = "Gemini model to use"
  type        = string
  default     = "gemini-2.0-flash"
}

variable "embedding_model" {
  description = "HuggingFace embedding model"
  type        = string
  default     = "BAAI/bge-small-en-v1.5"
}

variable "llm_backend" {
  description = "LLM backend: gemini or vllm"
  type        = string
  default     = "gemini"

  validation {
    condition     = contains(["gemini", "vllm"], var.llm_backend)
    error_message = "llm_backend must be 'gemini' or 'vllm'"
  }
}

# ── Cloud Run configuration ───────────────────────────────────────────────────

variable "cloud_run_min_instances" {
  description = "Minimum Cloud Run instances (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "cloud_run_max_instances" {
  description = "Maximum Cloud Run instances"
  type        = number
  default     = 5
}

variable "cloud_run_cpu" {
  description = "Cloud Run CPU allocation"
  type        = string
  default     = "2"
}

variable "cloud_run_memory" {
  description = "Cloud Run memory allocation"
  type        = string
  default     = "2Gi"
}

# ── Qdrant VM (optional, for persistent vector store) ────────────────────────

variable "enable_qdrant_vm" {
  description = "Deploy a dedicated e2-medium VM for Qdrant vector database"
  type        = bool
  default     = false  # By default, Qdrant runs in-memory in Cloud Run
}

variable "qdrant_machine_type" {
  description = "Machine type for Qdrant VM"
  type        = string
  default     = "e2-medium"  # 1 vCPU, 4GB RAM (~$30/month), or e2-micro for free tier
}

# ── vLLM VM (optional, for self-hosted LLM) ──────────────────────────────────

variable "enable_vllm_vm" {
  description = "Deploy a GPU Spot VM for vLLM (Llama 3 / Mistral)"
  type        = bool
  default     = false  # Off by default; uses Gemini API instead
}

variable "vllm_machine_type" {
  description = "Machine type for vLLM VM (must have GPU)"
  type        = string
  default     = "g2-standard-4"  # 1x L4 GPU, 4 vCPU, 16GB RAM (~$0.70/hour spot)
}

variable "vllm_model" {
  description = "Model to serve with vLLM"
  type        = string
  default     = "mistralai/Mistral-7B-Instruct-v0.3"
}

variable "vllm_hf_token" {
  description = "HuggingFace token for downloading gated models (leave empty for public models)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "tags" {
  description = "Labels to apply to all GCP resources"
  type        = map(string)
  default = {
    project     = "project-303-llm-rag"
    managed_by  = "terraform"
  }
}
