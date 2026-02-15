# AWS Module Variables

variable "region" {
  description = "AWS region"
  type        = string
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
}

variable "k8s_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "Instance types for node group"
  type        = list(string)
  default     = ["t3.xlarge", "t3.2xlarge"]
}

variable "node_desired_size" {
  description = "Desired number of nodes"
  type        = number
  default     = 3
}

variable "node_min_size" {
  description = "Minimum number of nodes"
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
}

variable "use_spot_instances" {
  description = "Use spot instances for cost optimization"
  type        = bool
  default     = true
}

variable "enable_gpu" {
  description = "Enable GPU node group"
  type        = bool
  default     = false
}

variable "gpu_instance_types" {
  description = "GPU instance types"
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge"]
}

variable "gpu_node_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 5
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "CIDR blocks that can access the cluster endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
