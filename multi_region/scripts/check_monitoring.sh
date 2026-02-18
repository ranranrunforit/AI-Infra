#!/bin/bash
# Check monitoring stack health across all regions

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Monitoring Stack Health Check ===${NC}"

PROMETHEUS_URLS=(
    "${REGION_1_PROMETHEUS_URL:-http://localhost:9091}"
    "${REGION_2_PROMETHEUS_URL:-http://localhost:9091}"
    "${REGION_3_PROMETHEUS_URL:-http://localhost:9091}"
)
REGION_NAMES=("us-west-2" "eu-west-1" "ap-south-1")

echo ""
echo "Checking Prometheus endpoints..."
for i in "${!REGION_NAMES[@]}"; do
    region="${REGION_NAMES[$i]}"
    url="${PROMETHEUS_URLS[$i]}"
    echo -n "  ${region} Prometheus (${url}/-/healthy): "
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "${url}/-/healthy" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✓ Healthy${NC}"
    else
        echo -e "${RED}✗ Unreachable (HTTP ${HTTP_CODE})${NC}"
    fi
done

echo ""
echo "Checking Grafana..."
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "${GRAFANA_URL}/api/health" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo -e "  ${GREEN}✓ Grafana healthy at ${GRAFANA_URL}${NC}"
else
    echo -e "  ${RED}✗ Grafana unreachable (HTTP ${HTTP_CODE})${NC}"
fi

echo ""
echo "Checking application metrics endpoint..."
APP_URL="${APP_METRICS_URL:-http://localhost:9090}"
METRICS=$(curl -s --connect-timeout 5 "${APP_URL}/metrics" 2>/dev/null | head -5)
if [ -n "$METRICS" ]; then
    echo -e "  ${GREEN}✓ Metrics endpoint responding${NC}"
    echo "  Sample metrics:"
    echo "$METRICS" | sed 's/^/    /'
else
    echo -e "  ${RED}✗ Metrics endpoint not responding${NC}"
fi

echo ""
echo -e "${GREEN}Monitoring check complete.${NC}"
