variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "node_role_arn" {
  description = "IAM role ARN for node groups"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for node groups"
  type        = list(string)
}

variable "kms_key_id" {
  description = "KMS key ID for EBS encryption"
  type        = string
}

variable "a100_desired_size" {
  description = "Desired number of A100 nodes"
  type        = number
  default     = 2
}

variable "a100_min_size" {
  description = "Minimum number of A100 nodes"
  type        = number
  default     = 1
}

variable "a100_max_size" {
  description = "Maximum number of A100 nodes"
  type        = number
  default     = 4
}

variable "l40s_desired_size" {
  description = "Desired number of L40S nodes"
  type        = number
  default     = 1
}

variable "l40s_min_size" {
  description = "Minimum number of L40S nodes"
  type        = number
  default     = 0
}

variable "l40s_max_size" {
  description = "Maximum number of L40S nodes"
  type        = number
  default     = 3
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
