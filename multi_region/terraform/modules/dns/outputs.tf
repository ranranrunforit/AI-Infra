# DNS Module Outputs

output "zone_id" {
  description = "Route53 hosted zone ID"
  value       = data.aws_route53_zone.main.zone_id
}

output "primary_endpoint" {
  description = "Primary DNS endpoint"
  value       = "${var.service_name}.${var.domain_name}"
}

output "latency_based_endpoint" {
  description = "Latency-based routing endpoint"
  value       = "lb.${var.service_name}.${var.domain_name}"
}

output "weighted_endpoint" {
  description = "Weighted routing endpoint"
  value       = var.enable_weighted_routing ? "weighted.${var.service_name}.${var.domain_name}" : null
}

output "health_check_ids" {
  description = "Map of region to health check IDs"
  value       = { for k, v in aws_route53_health_check.regions : k => v.id }
}

output "certificate_arn" {
  description = "ACM certificate ARN"
  value       = aws_acm_certificate.main.arn
}

output "regional_endpoints" {
  description = "Regional-specific endpoints"
  value = {
    us = contains(keys(var.region_endpoints), "us-west-2") ? "us.${var.service_name}.${var.domain_name}" : null
    eu = contains(keys(var.region_endpoints), "eu-west-1") ? "eu.${var.service_name}.${var.domain_name}" : null
    ap = contains(keys(var.region_endpoints), "ap-south-1") ? "ap.${var.service_name}.${var.domain_name}" : null
  }
}
