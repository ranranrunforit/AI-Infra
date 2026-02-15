# Azure Module Outputs

output "cluster_id" {
  description = "AKS cluster ID"
  value       = azurerm_kubernetes_cluster.main.id
}

output "cluster_name" {
  description = "AKS cluster name"
  value       = azurerm_kubernetes_cluster.main.name
}

output "cluster_endpoint" {
  description = "AKS cluster endpoint"
  value       = azurerm_kubernetes_cluster.main.kube_config[0].host
}

output "cluster_ca_certificate" {
  description = "AKS cluster CA certificate"
  value       = azurerm_kubernetes_cluster.main.kube_config[0].cluster_ca_certificate
  sensitive   = true
}

output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}

output "vnet_id" {
  description = "VNet ID"
  value       = azurerm_virtual_network.main.id
}

output "subnet_id" {
  description = "AKS subnet ID"
  value       = azurerm_subnet.aks.id
}

output "models_storage_account_name" {
  description = "Storage account name for models"
  value       = azurerm_storage_account.models.name
}

output "models_storage_account_primary_blob_endpoint" {
  description = "Primary blob endpoint for models storage"
  value       = azurerm_storage_account.models.primary_blob_endpoint
}

output "logs_storage_account_name" {
  description = "Storage account name for logs"
  value       = azurerm_storage_account.logs.name
}

output "container_registry_name" {
  description = "Container registry name"
  value       = azurerm_container_registry.main.name
}

output "container_registry_login_server" {
  description = "Container registry login server"
  value       = azurerm_container_registry.main.login_server
}

output "managed_identity_id" {
  description = "Managed identity ID for ML serving"
  value       = azurerm_user_assigned_identity.ml_serving.id
}

output "managed_identity_client_id" {
  description = "Managed identity client ID"
  value       = azurerm_user_assigned_identity.ml_serving.client_id
}

output "location" {
  description = "Azure location"
  value       = var.location
}

output "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID"
  value       = azurerm_log_analytics_workspace.main.id
}
