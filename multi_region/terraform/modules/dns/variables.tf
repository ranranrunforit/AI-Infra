# DNS Module Variables

variable "domain_name" {
  description = "Domain name for the service"
  type        = string
}

variable "service_name" {
  description = "Name of the service"
  type        = string
}

variable "region_endpoints" {
  description = "Map of region identifiers to their endpoint configurations"
  type = map(object({
    endpoint   = string
    aws_region = string
    weight     = number
  }))
}

variable "default_region" {
  description = "Default region for fallback"
  type        = string
}

variable "health_check_path" {
  description = "Path for health checks"
  type        = string
  default     = "/health"
}

variable "enable_weighted_routing" {
  description = "Enable weighted routing for A/B testing"
  type        = bool
  default     = false
}

variable "alarm_sns_topic_arn" {
  description = "SNS topic ARN for health check alarms"
  type        = string
  default     = null
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
