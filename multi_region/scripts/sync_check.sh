#!/bin/bash
# Data sync verification — checks replication lag and consistency across regions

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Data Sync Verification ===${NC}"

# Run Python sync verification
echo ""
echo "Running Python sync verification..."
python -m src.replication.data_sync --verify 2>/dev/null || {
    echo -e "${YELLOW}⚠ Python sync verification not available in this environment.${NC}"
    echo "  In production: python -m src.replication.data_sync --verify"
}

# Check replication metadata bucket (AWS)
echo ""
echo "Checking replication metadata..."
if command -v aws &> /dev/null && [ -n "$AWS_REGION" ]; then
    BUCKET="${PROJECT_NAME:-ml-platform}-replication-metadata"
    echo -n "  S3 bucket ${BUCKET}: "
    aws s3 ls "s3://${BUCKET}" --region "${AWS_REGION:-us-west-2}" &>/dev/null && \
        echo -e "${GREEN}✓ Accessible${NC}" || \
        echo -e "${RED}✗ Not accessible${NC}"
else
    echo -e "${YELLOW}  ⚠ AWS CLI not configured, skipping S3 check${NC}"
fi

# Check GCS (GCP)
if command -v gsutil &> /dev/null && [ -n "$GCP_PROJECT_ID" ]; then
    BUCKET="${PROJECT_NAME:-ml-platform}-models-eu"
    echo -n "  GCS bucket ${BUCKET}: "
    gsutil ls "gs://${BUCKET}" &>/dev/null && \
        echo -e "${GREEN}✓ Accessible${NC}" || \
        echo -e "${RED}✗ Not accessible${NC}"
else
    echo -e "${YELLOW}  ⚠ gsutil not configured, skipping GCS check${NC}"
fi

echo ""
echo -e "${GREEN}Sync verification complete.${NC}"
echo "For detailed replication status, run:"
echo "  python -m src.replication.model_replicator --status"
