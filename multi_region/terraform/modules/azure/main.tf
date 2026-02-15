# Azure Region Module for Multi-Region ML Platform
# Provisions AKS cluster, Azure Storage, and supporting infrastructure

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.cluster_name}-rg"
  location = var.location

  tags = {
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "${var.cluster_name}-vnet"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = [var.vnet_cidr]

  tags = {
    Environment = var.environment
  }
}

# Subnet for AKS
resource "azurerm_subnet" "aks" {
  name                 = "${var.cluster_name}-aks-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.aks_subnet_cidr]
}

# Log Analytics Workspace for AKS monitoring
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.cluster_name}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = {
    Environment = var.environment
  }
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = var.cluster_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = var.cluster_name
  kubernetes_version  = var.k8s_version

  default_node_pool {
    name                = "default"
    vm_size             = var.node_vm_size
    enable_auto_scaling = true
    min_count           = var.node_min_count
    max_count           = var.node_max_count
    node_count          = var.node_count
    vnet_subnet_id      = azurerm_subnet.aks.id
    os_disk_size_gb     = 100
    type                = "VirtualMachineScaleSets"

    node_labels = {
      role        = "ml-worker"
      environment = var.environment
    }

    tags = {
      Environment = var.environment
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    load_balancer_sku = "standard"
    service_cidr      = var.service_cidr
    dns_service_ip    = var.dns_service_ip
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  azure_policy_enabled = true

  tags = {
    Environment = var.environment
  }
}

# GPU Node Pool (optional)
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  count                 = var.enable_gpu ? 1 : 0
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.gpu_vm_size
  enable_auto_scaling   = true
  min_count             = var.gpu_node_min_count
  max_count             = var.gpu_node_max_count
  node_count            = var.gpu_node_count
  vnet_subnet_id        = azurerm_subnet.aks.id
  os_disk_size_gb       = 200
  priority              = var.use_spot_instances ? "Spot" : "Regular"
  eviction_policy       = var.use_spot_instances ? "Delete" : null
  spot_max_price        = var.use_spot_instances ? -1 : null

  node_labels = {
    role        = "ml-gpu-worker"
    environment = var.environment
  }

  node_taints = [
    "nvidia.com/gpu=true:NoSchedule"
  ]

  tags = {
    Environment = var.environment
  }
}

# Spot Node Pool for cost optimization
resource "azurerm_kubernetes_cluster_node_pool" "spot" {
  count                 = var.enable_spot_pool ? 1 : 0
  name                  = "spot"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.spot_vm_size
  enable_auto_scaling   = true
  min_count             = 0
  max_count             = var.spot_max_count
  node_count            = 1
  vnet_subnet_id        = azurerm_subnet.aks.id
  priority              = "Spot"
  eviction_policy       = "Delete"
  spot_max_price        = -1 # Pay up to regular price

  node_labels = {
    role        = "ml-worker-spot"
    environment = var.environment
  }

  tags = {
    Environment = var.environment
  }
}

# Storage Account for Models
resource "azurerm_storage_account" "models" {
  name                     = lower(replace("${var.cluster_name}models${var.location}", "/[^a-z0-9]/", ""))
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"

  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = 7
    }
  }

  tags = {
    Environment = var.environment
    Purpose     = "ml-models"
  }
}

resource "azurerm_storage_container" "models" {
  name                  = "ml-models"
  storage_account_name  = azurerm_storage_account.models.name
  container_access_type = "private"
}

# Storage Account for Logs
resource "azurerm_storage_account" "logs" {
  name                     = lower(replace("${var.cluster_name}logs${var.location}", "/[^a-z0-9]/", ""))
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"

  blob_properties {
    delete_retention_policy {
      days = 90
    }
  }

  tags = {
    Environment = var.environment
    Purpose     = "logs"
  }
}

resource "azurerm_storage_container" "logs" {
  name                  = "logs"
  storage_account_name  = azurerm_storage_account.logs.name
  container_access_type = "private"
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = lower(replace("${var.cluster_name}acr", "/[^a-z0-9]/", ""))
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Standard"
  admin_enabled       = false

  tags = {
    Environment = var.environment
  }
}

# Role assignment for AKS to pull from ACR
resource "azurerm_role_assignment" "aks_acr" {
  principal_id                     = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.main.id
  skip_service_principal_aad_check = true
}

# Managed Identity for ML workloads
resource "azurerm_user_assigned_identity" "ml_serving" {
  name                = "${var.cluster_name}-ml-serving-identity"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  tags = {
    Environment = var.environment
  }
}

# Role assignments for managed identity
resource "azurerm_role_assignment" "ml_serving_storage" {
  scope                = azurerm_storage_account.models.id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_user_assigned_identity.ml_serving.principal_id
}

resource "azurerm_role_assignment" "ml_serving_acr" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.ml_serving.principal_id
}

# Network Security Group
resource "azurerm_network_security_group" "aks" {
  name                = "${var.cluster_name}-aks-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "allow-https"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = {
    Environment = var.environment
  }
}

resource "azurerm_subnet_network_security_group_association" "aks" {
  subnet_id                 = azurerm_subnet.aks.id
  network_security_group_id = azurerm_network_security_group.aks.id
}
