# cloud/terraform/pubsub.tf — P0 real-time alert channel
#
# Robot agents publish P0 anomaly events to this topic via a single HTTPS call
# (~50-80ms round-trip).  This is faster than waiting for GCS upload + Eventarc
# trigger, giving the fleet dashboard near-real-time P0 visibility.
#
# Why Pub/Sub instead of MQTT:
#   - No broker to run/maintain
#   - Native GCP integration (push subscriptions → Cloud Run)
#   - QoS-1 semantics built-in
#   - Single HTTPS call from robot — works through any corporate proxy

resource "google_pubsub_topic" "p0_alerts" {
  name = "robot-p0-alerts"

  message_retention_duration = "86400s"   # 24h retention

  labels = {
    environment = var.environment
    project     = "robot-pipeline-v4"
    managed-by  = "terraform"
  }
}

# Push subscription → Cloud Run for real-time P0 dashboard updates
resource "google_pubsub_subscription" "p0_alerts_push" {
  name  = "robot-p0-alerts-push"
  topic = google_pubsub_topic.p0_alerts.id

  push_config {
    push_endpoint = "${google_cloud_run_v2_service.mcap_indexer.uri}/p0-alert"

    oidc_token {
      service_account_email = google_service_account.pipeline_fn.email
    }
  }

  ack_deadline_seconds       = 30
  message_retention_duration = "86400s"
  retain_acked_messages      = false

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  labels = {
    environment = var.environment
    project     = "robot-pipeline-v4"
  }
}

# IAM: robot service account can publish to the topic
resource "google_pubsub_topic_iam_member" "robot_publisher" {
  topic  = google_pubsub_topic.p0_alerts.id
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.pipeline_robot.email}"
}

output "pubsub_topic" {
  value = google_pubsub_topic.p0_alerts.id
}
