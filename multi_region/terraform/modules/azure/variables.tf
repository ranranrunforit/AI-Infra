# Azure Module Variables

variable "location" {
  description = "Azure location"
  type        = string
}

variable "cluster_name" {
  description = "Name of the AKS cluster"
  type        = string
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
}

variable "vnet_cidr" {
  description = "CIDR block for VNet"
  type        = string
  default     = "10.4.0.0/16"
}

variable "aks_subnet_cidr" {
  description = "CIDR block for AKS subnet"
  type        = string
  default     = "10.4.0.0/20"
}

variable "service_cidr" {
  description = "CIDR block for Kubernetes services"
  type        = string
  default     = "10.5.0.0/20"
}

variable "dns_service_ip" {
  description = "IP address for Kubernetes DNS service"
  type        = string
  default     = "10.5.0.10"
}

variable "k8s_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_vm_size" {
  description = "VM size for nodes"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "node_count" {
  description = "Initial node count"
  type        = number
  default     = 3
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

variable "enable_gpu" {
  description = "Enable GPU node pool"
  type        = bool
  default     = false
}

variable "gpu_vm_size" {
  description = "VM size for GPU nodes"
  type        = string
  default     = "Standard_NC6s_v3"
}

variable "gpu_node_count" {
  description = "Initial GPU node count"
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

variable "use_spot_instances" {
  description = "Use spot instances for GPU nodes"
  type        = bool
  default     = true
}

variable "enable_spot_pool" {
  description = "Enable separate spot instance pool"
  type        = bool
  default     = true
}

variable "spot_vm_size" {
  description = "VM size for spot nodes"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "spot_max_count" {
  description = "Maximum spot nodes"
  type        = number
  default     = 10
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
