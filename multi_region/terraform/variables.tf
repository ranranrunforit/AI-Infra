# Main Terraform Variables

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "ml-platform"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "domain_name" {
  description = "Domain name for the platform"
  type        = string
}

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
}

variable "k8s_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

# AWS Configuration
variable "aws_node_instance_types" {
  description = "AWS EKS node instance types"
  type        = list(string)
  default     = ["t3.xlarge", "t3.2xlarge"]
}

variable "aws_node_count" {
  description = "AWS EKS desired node count"
  type        = number
  default     = 3
}

variable "aws_node_min_count" {
  description = "AWS EKS minimum node count"
  type        = number
  default     = 1
}

variable "aws_node_max_count" {
  description = "AWS EKS maximum node count"
  type        = number
  default     = 10
}

# GCP Configuration
variable "gcp_machine_type" {
  description = "GCP GKE machine type"
  type        = string
  default     = "n1-standard-4"
}

variable "gcp_node_count" {
  description = "GCP GKE nodes per zone"
  type        = number
  default     = 1
}

variable "gcp_node_min_count" {
  description = "GCP GKE minimum node count"
  type        = number
  default     = 1
}

variable "gcp_node_max_count" {
  description = "GCP GKE maximum node count"
  type        = number
  default     = 10
}

# Azure Configuration
variable "azure_node_vm_size" {
  description = "Azure AKS node VM size"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "azure_node_count" {
  description = "Azure AKS initial node count"
  type        = number
  default     = 3
}

variable "azure_node_min_count" {
  description = "Azure AKS minimum node count"
  type        = number
  default     = 1
}

variable "azure_node_max_count" {
  description = "Azure AKS maximum node count"
  type        = number
  default     = 10
}

# Common Configuration
variable "use_spot_instances" {
  description = "Use spot/preemptible instances for cost optimization"
  type        = bool
  default     = true
}

variable "enable_gpu" {
  description = "Enable GPU node pools"
  type        = bool
  default     = false
}

variable "enable_weighted_routing" {
  description = "Enable weighted DNS routing for A/B testing"
  type        = bool
  default     = false
}

variable "alarm_sns_topic_arn" {
  description = "SNS topic ARN for alarms"
  type        = string
  default     = null
}

variable "tags" {
  description = "Additional tags for all resources"
  type        = map(string)
  default     = {}
}
