# GCP Module Variables

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
}

variable "subnet_cidr" {
  description = "CIDR block for subnet"
  type        = string
  default     = "10.1.0.0/20"
}

variable "pods_cidr" {
  description = "CIDR block for pods"
  type        = string
  default     = "10.2.0.0/16"
}

variable "services_cidr" {
  description = "CIDR block for services"
  type        = string
  default     = "10.3.0.0/20"
}

variable "master_cidr" {
  description = "CIDR block for GKE master"
  type        = string
  default     = "172.16.0.0/28"
}

variable "machine_type" {
  description = "Machine type for nodes"
  type        = string
  default     = "n1-standard-4"
}

variable "node_count_per_zone" {
  description = "Number of nodes per zone"
  type        = number
  default     = 1
}

variable "node_min_count" {
  description = "Minimum number of nodes"
  type        = number
  default     = 1
}

variable "node_max_count" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
}

variable "use_preemptible" {
  description = "Use preemptible instances for cost optimization"
  type        = bool
  default     = true
}

variable "enable_gpu" {
  description = "Enable GPU node pool"
  type        = bool
  default     = false
}

variable "gpu_machine_type" {
  description = "Machine type for GPU nodes"
  type        = string
  default     = "n1-standard-4"
}

variable "gpu_type" {
  description = "GPU accelerator type"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count_per_node" {
  description = "Number of GPUs per node"
  type        = number
  default     = 1
}

variable "gpu_node_count" {
  description = "Number of GPU nodes per zone"
  type        = number
  default     = 1
}

variable "gpu_node_min_count" {
  description = "Minimum GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_node_max_count" {
  description = "Maximum GPU nodes"
  type        = number
  default     = 5
}

variable "release_channel" {
  description = "GKE release channel"
  type        = string
  default     = "REGULAR"
}

variable "k8s_namespace" {
  description = "Kubernetes namespace for workload identity"
  type        = string
  default     = "default"
}

variable "k8s_service_account" {
  description = "Kubernetes service account for workload identity"
  type        = string
  default     = "ml-serving"
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
