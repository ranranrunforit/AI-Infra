# cloud/terraform/demo.tfvars
# Usage: terraform apply -var-file=demo.tfvars

project_id        = "data-pipeline-cz78"   # Replace with your project
region            = "us-central1"
environment       = "demo"
gcs_bucket_name   = "robot-data-pipeline-cz78-demo"
db_tier           = "db-f1-micro"
postgres_password = "robotdb-pass-2026!x"
db_backend = "cloudsql"

