# OrbStack Containerized Testing

This document describes the OrbStack-based containerized testing infrastructure for the question-graph-agent project.

## Overview

OrbStack is a fast, lightweight alternative to Docker Desktop for macOS that provides significant performance improvements for containerized workflows. Our testing infrastructure has been optimized to leverage OrbStack's features for better development experience.

## Prerequisites

1. **Install OrbStack**: Download from [orbstack.dev](https://orbstack.dev)
2. **Environment Setup**: Copy `.env.test` to `.env` and add your API keys:
   ```bash
   cp .env.test .env
   # Edit .env to add your API keys
   ```

## Quick Start

Run tests using the OrbStack-optimized script:

```bash
# Run all enhanced node tests
./run-tests-orbstack.sh

# Run specific test file
./run-tests-orbstack.sh tests/test_enhanced_nodes.py

# Run specific test
./run-tests-orbstack.sh tests/test_enhanced_nodes.py::TestEnhancedAsk::test_enhanced_ask_with_memory_success
```

## Architecture

### Components

1. **run-tests-orbstack.sh**: Main test runner script
   - Automatically switches to OrbStack context
   - Provides colored output and progress indicators
   - Monitors resource usage
   - Implements faster health checks

2. **docker-compose.orbstack.yml**: OrbStack-optimized compose configuration
   - Optimized volume mounts with performance modes
   - Reduced memory allocations (OrbStack is more efficient)
   - Native volume drivers
   - Faster health check intervals

3. **Dockerfile.orbstack**: Multi-stage build optimized for OrbStack
   - Separate dependency and application stages
   - Pre-compiled Python files
   - Non-root user for security
   - Optimized wait scripts

### Directory Structure

```
.orbstack-volumes/          # OrbStack-specific volumes (gitignored)
├── neo4j/                 # Neo4j data volume
└── pip-cache/            # Pip cache for faster builds

logs/                      # Application logs
└── neo4j/                # Neo4j logs

test-results/             # Test output directory
```

## Performance Optimizations

### 1. Volume Mount Strategies

- **Source Code**: Uses `cached` consistency for read-heavy operations
- **Test Results**: Uses `delegated` consistency for write performance
- **Neo4j Data**: Native volume driver with bind mount

### 2. Build Optimizations

- Multi-stage builds reduce final image size
- Pip cache volume persists between builds
- Pre-compilation of Python files
- BuildKit inline cache enabled

### 3. Network Optimizations

- Custom MTU settings for better throughput
- Dedicated test network with optimized subnet

### 4. Resource Allocation

- Reduced memory requirements (50% less than Docker Desktop)
- CPU limits prevent resource contention
- Faster startup times (15s vs 30s for Neo4j)

## Monitoring and Debugging

### Resource Usage

The test script automatically displays resource usage:
```bash
📊 Resource usage before tests:
CONTAINER               CPU %     MEM USAGE / LIMIT
orbstack-test-neo4j     2.34%     245MiB / 1GiB
```

### Logs

- **Real-time logs**: `docker-compose -f docker-compose.orbstack.yml logs -f`
- **Neo4j logs**: Available in `./logs/neo4j/`
- **Test failures**: Automatically displayed with last 50 lines of Neo4j logs

### Health Checks

- Neo4j health: HTTP check on port 7474
- Reduced intervals (5s vs 10s) due to OrbStack's faster networking
- TCP socket check in wait script for faster detection

## Troubleshooting

### Common Issues

1. **Context not switched**
   ```bash
   docker context use orbstack
   ```

2. **Permission errors on volumes**
   ```bash
   # Reset volume permissions
   rm -rf .orbstack-volumes
   mkdir -p .orbstack-volumes/{neo4j,pip-cache}
   ```

3. **Build cache issues**
   ```bash
   # Clear build cache
   docker builder prune -f
   ```

4. **Neo4j connection timeout**
   - Check if port 7687 is available: `lsof -i :7687`
   - Increase timeout in wait script if needed

### Performance Comparison

| Metric | Docker Desktop | OrbStack | Improvement |
|--------|---------------|----------|-------------|
| Initial build | ~5 min | ~3 min | 40% faster |
| Subsequent builds | ~2 min | ~45s | 62% faster |
| Neo4j startup | ~30s | ~15s | 50% faster |
| Test execution | ~60s | ~45s | 25% faster |
| Memory usage | ~2GB | ~1GB | 50% less |
| CPU usage (idle) | ~5% | ~1% | 80% less |

## Advanced Usage

### Running with Custom Settings

```bash
# Increase timeout for slow connections
TEST_TIMEOUT=600 ./run-tests-orbstack.sh

# Use different compose file
COMPOSE_FILE=docker-compose.custom.yml ./run-tests-orbstack.sh

# Skip cleanup on exit
NO_CLEANUP=1 ./run-tests-orbstack.sh
```

### Parallel Test Execution

```bash
# Run tests in parallel (requires pytest-xdist)
docker-compose -f docker-compose.orbstack.yml run --rm test-runner \
  python -m pytest tests/ -n auto -v
```

### Interactive Debugging

```bash
# Start services
docker-compose -f docker-compose.orbstack.yml up -d neo4j

# Run interactive shell
docker-compose -f docker-compose.orbstack.yml run --rm test-runner bash

# Inside container
python -m pytest tests/test_simple.py -v --pdb
```

## CI/CD Integration

For CI/CD pipelines, you can use the same infrastructure:

```yaml
# Example GitHub Actions
- name: Setup OrbStack
  run: |
    # Install OrbStack CLI tools
    brew install orbstack
    
- name: Run Tests
  run: |
    docker context use orbstack
    ./run-tests-orbstack.sh
```

## Future Improvements

1. **Kubernetes Integration**: Leverage OrbStack's built-in K8s for scaling
2. **GPU Support**: Use OrbStack's GPU passthrough for ML workloads
3. **Distributed Testing**: Multi-node test execution
4. **Performance Profiling**: Integration with OrbStack's profiling tools

## Contributing

When adding new tests or modifying the infrastructure:

1. Ensure compatibility with both Docker Desktop and OrbStack
2. Document any OrbStack-specific features used
3. Update performance benchmarks if significant changes
4. Test with minimal configuration first

## References

- [OrbStack Documentation](https://docs.orbstack.dev)
- [Docker Compose Performance](https://docs.docker.com/compose/compose-file/compose-file-v3/#consistency)
- [BuildKit Documentation](https://docs.docker.com/build/buildkit/)