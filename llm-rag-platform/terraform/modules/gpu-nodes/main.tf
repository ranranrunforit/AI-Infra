# GPU Node Groups for LLM Inference
# Supports A100 GPUs for Llama 3 70B and L40S for Mistral 7B

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# A100 Node Group for Llama 3 70B (8x GPUs per node)
resource "aws_eks_node_group" "a100_llama_70b" {
  cluster_name    = var.cluster_name
  node_group_name = "${var.cluster_name}-a100-llama-70b"
  node_role_arn   = var.node_role_arn
  subnet_ids      = var.private_subnet_ids

  instance_types = ["p4d.24xlarge"] # 8x A100 40GB GPUs, 96 vCPUs, 1.1TB RAM
  capacity_type  = "ON_DEMAND"      # Critical workload, no spot instances

  scaling_config {
    desired_size = var.a100_desired_size
    min_size     = var.a100_min_size
    max_size     = var.a100_max_size
  }

  # GPU-optimized AMI with CUDA drivers
  launch_template {
    id      = aws_launch_template.a100.id
    version = "$Latest"
  }

  labels = {
    "workload-type"     = "llm-inference"
    "gpu-type"          = "a100"
    "model"             = "llama-3-70b"
    "node-group"        = "a100-llama-70b"
    "nvidia.com/gpu"    = "true"
  }

  taints {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_name}-a100-llama-70b"
      Purpose = "LLM Inference - Llama 3 70B"
      CostCenter = "ml-inference"
    }
  )

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }

  depends_on = [var.node_role_arn]
}

# L40S Node Group for Mistral 7B (4x GPUs per node)
resource "aws_eks_node_group" "l40s_mistral_7b" {
  cluster_name    = var.cluster_name
  node_group_name = "${var.cluster_name}-l40s-mistral-7b"
  node_role_arn   = var.node_role_arn
  subnet_ids      = var.private_subnet_ids

  instance_types = ["g5.12xlarge"] # 4x A10G GPUs (L40S equivalent), 48 vCPUs, 192GB RAM
  capacity_type  = "SPOT"          # Cost optimization for smaller model

  scaling_config {
    desired_size = var.l40s_desired_size
    min_size     = var.l40s_min_size
    max_size     = var.l40s_max_size
  }

  launch_template {
    id      = aws_launch_template.l40s.id
    version = "$Latest"
  }

  labels = {
    "workload-type"     = "llm-inference"
    "gpu-type"          = "l40s"
    "model"             = "mistral-7b"
    "node-group"        = "l40s-mistral-7b"
    "nvidia.com/gpu"    = "true"
    "spot-instance"     = "true"
  }

  taints {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_name}-l40s-mistral-7b"
      Purpose = "LLM Inference - Mistral 7B (Cost Optimized)"
      CostCenter = "ml-inference"
    }
  }

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }
}

# Launch Template for A100 nodes
resource "aws_launch_template" "a100" {
  name_prefix = "${var.cluster_name}-a100-"
  description = "Launch template for A100 GPU nodes"

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 500 # Large volume for model storage
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      delete_on_termination = true
      encrypted             = true
      kms_key_id            = var.kms_key_id
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required" # IMDSv2 only
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  monitoring {
    enabled = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data_a100.sh", {
    cluster_name = var.cluster_name
    cuda_version = "12.2"
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(
      var.tags,
      {
        Name = "${var.cluster_name}-a100-node"
        NodeType = "gpu-a100"
      }
    )
  }
}

# Launch Template for L40S nodes
resource "aws_launch_template" "l40s" {
  name_prefix = "${var.cluster_name}-l40s-"
  description = "Launch template for L40S GPU nodes"

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 300
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      delete_on_termination = true
      encrypted             = true
      kms_key_id            = var.kms_key_id
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  monitoring {
    enabled = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data_l40s.sh", {
    cluster_name = var.cluster_name
    cuda_version = "12.2"
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(
      var.tags,
      {
        Name = "${var.cluster_name}-l40s-node"
        NodeType = "gpu-l40s"
      }
    )
  }
}

# GPU Feature Discovery DaemonSet (automatic GPU labeling)
resource "kubernetes_daemonset" "nvidia_device_plugin" {
  metadata {
    name      = "nvidia-device-plugin-daemonset"
    namespace = "kube-system"
  }

  spec {
    selector {
      match_labels = {
        name = "nvidia-device-plugin-ds"
      }
    }

    template {
      metadata {
        labels = {
          name = "nvidia-device-plugin-ds"
        }
      }

      spec {
        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        priority_class_name = "system-node-critical"

        container {
          name  = "nvidia-device-plugin-ctr"
          image = "nvcr.io/nvidia/k8s-device-plugin:v0.14.1"

          security_context {
            allow_privilege_escalation = false
            capabilities {
              drop = ["ALL"]
            }
          }

          env {
            name  = "FAIL_ON_INIT_ERROR"
            value = "false"
          }

          volume_mount {
            name       = "device-plugin"
            mount_path = "/var/lib/kubelet/device-plugins"
          }
        }

        volume {
          name = "device-plugin"
          host_path {
            path = "/var/lib/kubelet/device-plugins"
          }
        }

        node_selector = {
          "nvidia.com/gpu" = "true"
        }
      }
    }
  }
}

# Outputs
output "a100_node_group_id" {
  description = "A100 node group ID"
  value       = aws_eks_node_group.a100_llama_70b.id
}

output "l40s_node_group_id" {
  description = "L40S node group ID"
  value       = aws_eks_node_group.l40s_mistral_7b.id
}

output "a100_node_group_status" {
  description = "A100 node group status"
  value       = aws_eks_node_group.a100_llama_70b.status
}

output "l40s_node_group_status" {
  description = "L40S node group status"
  value       = aws_eks_node_group.l40s_mistral_7b.status
}
