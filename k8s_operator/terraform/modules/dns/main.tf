# DNS Module for Global Load Balancing
# Manages Route53 for multi-region traffic routing

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Route53 Hosted Zone (assumes existing zone)
data "aws_route53_zone" "main" {
  name         = var.domain_name
  private_zone = false
}

# Health checks for each region
resource "aws_route53_health_check" "regions" {
  for_each          = var.region_endpoints
  fqdn              = each.value.endpoint
  port              = 443
  type              = "HTTPS"
  resource_path     = var.health_check_path
  failure_threshold = 3
  request_interval  = 30

  tags = {
    Name   = "${var.service_name}-${each.key}"
    Region = each.key
  }
}

# CloudWatch alarms for health checks
resource "aws_cloudwatch_metric_alarm" "health_check" {
  for_each = var.region_endpoints

  alarm_name          = "${var.service_name}-${each.key}-health"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "HealthCheckStatus"
  namespace           = "AWS/Route53"
  period              = "60"
  statistic           = "Minimum"
  threshold           = "1"
  alarm_description   = "Health check for ${each.key} region"
  treat_missing_data  = "breaching"

  dimensions = {
    HealthCheckId = aws_route53_health_check.regions[each.key].id
  }

  alarm_actions = var.alarm_sns_topic_arn != null ? [var.alarm_sns_topic_arn] : []
}

# Primary DNS record with geolocation routing
resource "aws_route53_record" "primary" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "${var.service_name}.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_route53_record.us_west_2[0].fqdn
    zone_id                = data.aws_route53_zone.main.zone_id
    evaluate_target_health = true
  }

  set_identifier = "primary"

  failover_routing_policy {
    type = "PRIMARY"
  }

  health_check_id = aws_route53_health_check.regions["us-west-2"].id
}

# Secondary DNS record for failover
resource "aws_route53_record" "secondary" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "${var.service_name}.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_route53_record.eu_west_1[0].fqdn
    zone_id                = data.aws_route53_zone.main.zone_id
    evaluate_target_health = true
  }

  set_identifier = "secondary"

  failover_routing_policy {
    type = "SECONDARY"
  }

  health_check_id = aws_route53_health_check.regions["eu-west-1"].id
}

# Geolocation routing for North America
resource "aws_route53_record" "us_west_2" {
  count   = contains(keys(var.region_endpoints), "us-west-2") ? 1 : 0
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "us.${var.service_name}.${var.domain_name}"
  type    = "CNAME"
  ttl     = 60
  records = [var.region_endpoints["us-west-2"].endpoint]

  set_identifier = "us-west-2"

  geolocation_routing_policy {
    continent = "NA"
  }

  health_check_id = aws_route53_health_check.regions["us-west-2"].id
}

# Geolocation routing for Europe
resource "aws_route53_record" "eu_west_1" {
  count   = contains(keys(var.region_endpoints), "eu-west-1") ? 1 : 0
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "eu.${var.service_name}.${var.domain_name}"
  type    = "CNAME"
  ttl     = 60
  records = [var.region_endpoints["eu-west-1"].endpoint]

  set_identifier = "eu-west-1"

  geolocation_routing_policy {
    continent = "EU"
  }

  health_check_id = aws_route53_health_check.regions["eu-west-1"].id
}

# Geolocation routing for Asia
resource "aws_route53_record" "ap_south_1" {
  count   = contains(keys(var.region_endpoints), "ap-south-1") ? 1 : 0
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "ap.${var.service_name}.${var.domain_name}"
  type    = "CNAME"
  ttl     = 60
  records = [var.region_endpoints["ap-south-1"].endpoint]

  set_identifier = "ap-south-1"

  geolocation_routing_policy {
    continent = "AS"
  }

  health_check_id = aws_route53_health_check.regions["ap-south-1"].id
}

# Latency-based routing records
resource "aws_route53_record" "latency_based" {
  for_each = var.region_endpoints

  zone_id = data.aws_route53_zone.main.zone_id
  name    = "lb.${var.service_name}.${var.domain_name}"
  type    = "CNAME"
  ttl     = 60
  records = [each.value.endpoint]

  set_identifier = each.key

  latency_routing_policy {
    region = each.value.aws_region
  }

  health_check_id = aws_route53_health_check.regions[each.key].id
}

# Weighted routing for A/B testing and canary deployments
resource "aws_route53_record" "weighted" {
  for_each = var.enable_weighted_routing ? var.region_endpoints : {}

  zone_id = data.aws_route53_zone.main.zone_id
  name    = "weighted.${var.service_name}.${var.domain_name}"
  type    = "CNAME"
  ttl     = 60
  records = [each.value.endpoint]

  set_identifier = each.key

  weighted_routing_policy {
    weight = each.value.weight
  }

  health_check_id = aws_route53_health_check.regions[each.key].id
}

# Default fallback record
resource "aws_route53_record" "default" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "${var.service_name}.${var.domain_name}"
  type    = "CNAME"
  ttl     = 300
  records = [var.region_endpoints[var.default_region].endpoint]

  set_identifier = "default"

  geolocation_routing_policy {
    location = "*"
  }
}

# ACM Certificate for HTTPS (wildcard)
resource "aws_acm_certificate" "main" {
  domain_name               = "*.${var.domain_name}"
  subject_alternative_names = [var.domain_name]
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${var.service_name}-cert"
  }
}

# ACM Certificate validation
resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.main.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = data.aws_route53_zone.main.zone_id
}

resource "aws_acm_certificate_validation" "main" {
  certificate_arn         = aws_acm_certificate.main.arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]
}
