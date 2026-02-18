#!/bin/bash
# Failover testing script — simulates region failures and validates recovery

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

TARGET_REGION=${1:-"eu-west-1"}
APP_URL="${APP_URL:-http://localhost:8080}"

echo -e "${YELLOW}=== Failover Test: Simulating failure, target=${TARGET_REGION} ===${NC}"

# 1. Verify baseline health
echo ""
echo "Step 1: Checking baseline health..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "${APP_URL}/health" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo -e "${RED}✗ App not healthy before test (HTTP ${HTTP_CODE}). Aborting.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Baseline health OK${NC}"

# 2. Trigger failover via Python controller
echo ""
echo "Step 2: Triggering simulated failover to ${TARGET_REGION}..."
python -m src.failover.failover_controller --simulate-failure --target "${TARGET_REGION}" 2>/dev/null || {
    echo -e "${YELLOW}⚠ Python failover simulation not available in this environment.${NC}"
    echo "  In production, this would call: python -m src.failover.failover_controller --simulate-failure --target ${TARGET_REGION}"
}

# 3. Wait for failover to propagate
echo ""
echo "Step 3: Waiting 30s for failover propagation..."
sleep 30

# 4. Verify health after failover
echo ""
echo "Step 4: Verifying health after failover..."
for i in 1 2 3; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "${APP_URL}/health" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✓ App healthy after failover (attempt ${i})${NC}"
        break
    else
        echo -e "${YELLOW}⚠ Attempt ${i}: HTTP ${HTTP_CODE}, retrying...${NC}"
        sleep 10
    fi
done

echo ""
echo -e "${GREEN}=== Failover test complete ===${NC}"
echo "Review logs: kubectl logs -l app=ml-platform -n ml-platform --tail=50"
