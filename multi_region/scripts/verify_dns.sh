#!/bin/bash
# Verify DNS configuration for all regions

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

DOMAIN=${1:-"example.com"}
SERVICE="ml"

echo -e "${YELLOW}=== DNS Verification for ${SERVICE}.${DOMAIN} ===${NC}"

# Check if dig is available
if ! command -v dig &> /dev/null; then
    echo -e "${RED}dig is required. Install bind-utils / dnsutils.${NC}"
    exit 1
fi

# Check DNS resolution
echo ""
echo "Resolving ${SERVICE}.${DOMAIN}..."
RESOLVED=$(dig +short ${SERVICE}.${DOMAIN} 2>/dev/null)

if [ -z "$RESOLVED" ]; then
    echo -e "${RED}✗ DNS resolution failed for ${SERVICE}.${DOMAIN}${NC}"
    echo "  Ensure Route53 hosted zone is configured and domain is delegated."
    exit 1
else
    echo -e "${GREEN}✓ Resolved to: ${RESOLVED}${NC}"
fi

# Check health check endpoints for each region
REGIONS=("us-west-2" "eu-west-1" "ap-south-1")
ENDPOINTS=(
    "${REGION_1_ENDPOINT:-localhost:8080}"
    "${REGION_2_ENDPOINT:-localhost:8080}"
    "${REGION_3_ENDPOINT:-localhost:8080}"
)

echo ""
echo "Checking regional health endpoints..."
for i in "${!REGIONS[@]}"; do
    region="${REGIONS[$i]}"
    endpoint="${ENDPOINTS[$i]}"
    echo -n "  ${region} (${endpoint}/health): "
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "http://${endpoint}/health" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✓ HTTP ${HTTP_CODE}${NC}"
    else
        echo -e "${RED}✗ HTTP ${HTTP_CODE}${NC}"
    fi
done

echo ""
echo -e "${GREEN}DNS verification complete.${NC}"
