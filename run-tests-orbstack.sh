#!/bin/bash

# Script to run tests with OrbStack for improved performance
# OrbStack provides faster file syncing, lower resource usage, and optimized networking

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting test environment with OrbStack...${NC}"

# Switch to OrbStack context
echo -e "${YELLOW}🔄 Switching to OrbStack Docker context...${NC}"
docker context use orbstack

# Verify we're using OrbStack
CURRENT_CONTEXT=$(docker context show)
if [ "$CURRENT_CONTEXT" != "orbstack" ]; then
    echo -e "${RED}❌ Failed to switch to OrbStack context${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Using OrbStack context${NC}"

# Check if .env file exists and export API keys
if [ -f .env ]; then
    export $(cat .env | grep -E '^(OPENAI_API_KEY|GROQ_API_KEY|LOGFIRE_TOKEN)=' | xargs)
else
    echo -e "${YELLOW}⚠️  Warning: .env file not found. API keys may be missing.${NC}"
fi

# Parse command line arguments
TEST_FILE=${1:-"tests/test_enhanced_nodes.py"}
COMPOSE_FILE="docker-compose.orbstack.yml"
PROJECT_NAME="graphiti-orbstack-tests"

# OrbStack-specific optimizations
export DOCKER_BUILDKIT=1  # Enable BuildKit for faster builds
export COMPOSE_DOCKER_CLI_BUILD=1

# Function to cleanup
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v --remove-orphans
    
    # OrbStack-specific: Clean up dangling volumes more aggressively
    docker volume prune -f 2>/dev/null || true
}

# Function to check OrbStack health
check_orbstack_health() {
    if ! docker system info >/dev/null 2>&1; then
        echo -e "${RED}❌ OrbStack/Docker daemon is not running${NC}"
        exit 1
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Check OrbStack health
check_orbstack_health

# Clean any existing containers from previous runs
echo -e "${YELLOW}🧹 Cleaning up any existing containers...${NC}"
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v --remove-orphans 2>/dev/null || true

# Build the test image with OrbStack optimizations
echo -e "${GREEN}🔨 Building test Docker image with OrbStack optimizations...${NC}"
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME build \
    --parallel \
    --progress=plain \
    test-runner

# Start services with OrbStack-optimized settings
echo -e "${GREEN}🏃 Starting services...${NC}"
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d neo4j

# OrbStack-optimized Neo4j health check
echo -e "${YELLOW}⏳ Waiting for Neo4j to be ready...${NC}"
MAX_ATTEMPTS=20  # Reduced from 30 due to OrbStack's faster startup
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T neo4j \
        wget --no-verbose --tries=1 --spider http://localhost:7474 2>/dev/null; then
        echo -e "${GREEN}✅ Neo4j is ready!${NC}"
        break
    fi
    echo -e "   Waiting for Neo4j... (attempt $((ATTEMPT+1))/$MAX_ATTEMPTS)"
    sleep 1  # Reduced from 2 due to OrbStack's faster networking
    ATTEMPT=$((ATTEMPT+1))
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "${RED}❌ Neo4j failed to start after $MAX_ATTEMPTS attempts${NC}"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs neo4j
    exit 1
fi

# Additional stabilization time (reduced for OrbStack)
sleep 2

# Show resource usage before tests
echo -e "${YELLOW}📊 Resource usage before tests:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Run the tests with OrbStack-optimized settings
echo -e "${GREEN}🧪 Running tests: $TEST_FILE${NC}"
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME run \
    --rm \
    -e PYTEST_TIMEOUT=300 \
    test-runner \
    python -m pytest $TEST_FILE -v -s --tb=short

# Capture exit code
TEST_EXIT_CODE=$?

# Show resource usage after tests
echo -e "${YELLOW}📊 Resource usage after tests:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Show logs if tests failed
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ Tests failed. Showing Neo4j logs:${NC}"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs --tail=50 neo4j
else
    echo -e "${GREEN}✅ All tests passed!${NC}"
fi

# Performance report
echo -e "${YELLOW}📈 Performance Notes:${NC}"
echo "   - OrbStack provides faster container startup"
echo "   - File sync operations are optimized"
echo "   - Lower memory usage compared to Docker Desktop"
echo "   - Native macOS networking performance"

exit $TEST_EXIT_CODE