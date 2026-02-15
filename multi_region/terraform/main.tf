# Main Terraform Configuration for Multi-Region ML Platform
# Orchestrates AWS, GCP, and Azure deployments

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }

  backend "s3" {
    bucket         = "your-terraform-state-bucket"
    key            = "multi-region-ml-platform/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Provider configurations
provider "aws" {
  region = "us-west-2"
  alias  = "us_west_2"

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = "europe-west1"
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# AWS US-WEST-2 Region
module "aws_us_west_2" {
  source = "./modules/aws"

  providers = {
    aws = aws.us_west_2
  }

  region              = "us-west-2"
  cluster_name        = "${var.project_name}-us-west-2"
  environment         = var.environment
  vpc_cidr            = "10.0.0.0/16"
  availability_zones  = ["us-west-2a", "us-west-2b", "us-west-2c"]
  k8s_version         = var.k8s_version
  node_instance_types = var.aws_node_instance_types
  node_desired_size   = var.aws_node_count
  node_min_size       = var.aws_node_min_count
  node_max_size       = var.aws_node_max_count
  use_spot_instances  = var.use_spot_instances
  enable_gpu          = var.enable_gpu
}

# GCP EU-WEST-1 Region
module "gcp_eu_west_1" {
  source = "./modules/gcp"

  project_id           = var.gcp_project_id
  region               = "europe-west1"
  cluster_name         = "${var.project_name}-eu-west-1"
  environment          = var.environment
  subnet_cidr          = "10.1.0.0/20"
  pods_cidr            = "10.2.0.0/16"
  services_cidr        = "10.3.0.0/20"
  master_cidr          = "172.16.0.0/28"
  machine_type         = var.gcp_machine_type
  node_count_per_zone  = var.gcp_node_count
  node_min_count       = var.gcp_node_min_count
  node_max_count       = var.gcp_node_max_count
  use_preemptible      = var.use_spot_instances
  enable_gpu           = var.enable_gpu
}

# Azure AP-SOUTH-1 Region
module "azure_ap_south_1" {
  source = "./modules/azure"

  location        = "centralindia"
  cluster_name    = "${var.project_name}-ap-south-1"
  environment     = var.environment
  vnet_cidr       = "10.4.0.0/16"
  aks_subnet_cidr = "10.4.0.0/20"
  service_cidr    = "10.5.0.0/20"
  dns_service_ip  = "10.5.0.10"
  k8s_version     = var.k8s_version
  node_vm_size    = var.azure_node_vm_size
  node_count      = var.azure_node_count
  node_min_count  = var.azure_node_min_count
  node_max_count  = var.azure_node_max_count
  enable_gpu      = var.enable_gpu
  enable_spot_pool = var.use_spot_instances
}

# DNS and Global Load Balancing
module "dns" {
  source = "./modules/dns"

  providers = {
    aws = aws.us_west_2
  }

  domain_name    = var.domain_name
  service_name   = var.project_name
  default_region = "us-west-2"

  region_endpoints = {
    "us-west-2" = {
      endpoint   = module.aws_us_west_2.cluster_endpoint
      aws_region = "us-west-2"
      weight     = 50
    }
    "eu-west-1" = {
      endpoint   = module.gcp_eu_west_1.cluster_endpoint
      aws_region = "eu-west-1"
      weight     = 30
    }
    "ap-south-1" = {
      endpoint   = module.azure_ap_south_1.cluster_endpoint
      aws_region = "ap-south-1"
      weight     = 20
    }
  }

  health_check_path       = "/health"
  enable_weighted_routing = var.enable_weighted_routing
  alarm_sns_topic_arn     = var.alarm_sns_topic_arn
}

# S3 bucket for cross-region replication metadata
resource "aws_s3_bucket" "replication_metadata" {
  provider = aws.us_west_2
  bucket   = "${var.project_name}-replication-metadata"

  tags = {
    Purpose = "Cross-region replication tracking"
  }
}

resource "aws_s3_bucket_versioning" "replication_metadata" {
  provider = aws.us_west_2
  bucket   = aws_s3_bucket.replication_metadata.id

  versioning_configuration {
    status = "Enabled"
  }
}

# DynamoDB table for distributed locking
resource "aws_dynamodb_table" "distributed_lock" {
  provider     = aws.us_west_2
  name         = "${var.project_name}-distributed-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  ttl {
    attribute_name = "TimeToLive"
    enabled        = true
  }

  tags = {
    Purpose = "Distributed locking for multi-region coordination"
  }
}

# CloudWatch Log Group for centralized logging
resource "aws_cloudwatch_log_group" "central" {
  provider          = aws.us_west_2
  name              = "/aws/multi-region/${var.project_name}"
  retention_in_days = 30

  tags = {
    Purpose = "Centralized logging"
  }
}

# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  provider = aws.us_west_2
  name     = "${var.project_name}-alerts"

  tags = {
    Purpose = "Multi-region alerts"
  }
}

# VPC Peering (AWS to AWS if multiple AWS regions)
# Commented out as we only have one AWS region, but included as example
# resource "aws_vpc_peering_connection" "aws_to_aws" {
#   vpc_id      = module.aws_us_west_2.vpc_id
#   peer_vpc_id = module.aws_us_east_1.vpc_id
#   auto_accept = true
# }
