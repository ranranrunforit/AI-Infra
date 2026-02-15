#!/bin/bash
# Health Check Script for Multi-Region Platform

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Multi-Region Health Check ===${NC}\n"

REGIONS=("us-west-2" "eu-west-1" "ap-south-1")
ENDPOINTS=(
    "ml-platform-us-west-2.example.com"
    "ml-platform-eu-west-1.example.com"
    "ml-platform-ap-south-1.example.com"
)

for i in "${!REGIONS[@]}"; do
    region="${REGIONS[$i]}"
    endpoint="${ENDPOINTS[$i]}"

    echo -e "${YELLOW}Checking $region ($endpoint)${NC}"

    # Check HTTP health endpoint
    status_code=$(curl -s -o /dev/null -w "%{http_code}" https://$endpoint/health || echo "000")

    if [ "$status_code" == "200" ]; then
        echo -e "${GREEN}✓ Health endpoint responding${NC}"
    else
        echo -e "${RED}✗ Health endpoint failed (status: $status_code)${NC}"
    fi

    # Check response time
    response_time=$(curl -s -o /dev/null -w "%{time_total}" https://$endpoint/health 2>/dev/null || echo "0")
    echo "  Response time: ${response_time}s"

    # Check Kubernetes cluster
    echo "  Kubernetes status:"
    kubectl config use-context $region 2>/dev/null
    kubectl get pods -n ml-platform --no-headers 2>/dev/null | wc -l | xargs echo "    Pods running:"

    echo ""
done

echo -e "${GREEN}Health check complete${NC}"
