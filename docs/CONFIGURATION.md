# Configuration Guide

This guide explains how to configure the Graphiti integration with Neo4j for the question-graph agent.

## Overview

The configuration system uses Pydantic settings management with environment variables and `.env` files. Configuration is organized into three main sections:

1. **Neo4j Configuration** - Database connection settings
2. **Graphiti Configuration** - Temporal knowledge graph service settings  
3. **Application Configuration** - Feature flags and runtime settings

## Environment Variables

### Neo4j Settings

Configure your Neo4j database connection:

```bash
# Required
NEO4J_PASSWORD=your-secure-password

# Optional (with defaults)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_DATABASE=neo4j
NEO4J_ENCRYPTED=false
NEO4J_TRUST=TRUST_ALL_CERTIFICATES
NEO4J_MAX_CONNECTION_LIFETIME=3600
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=60
```

### Graphiti Settings

Configure the Graphiti service connection:

```bash
# Optional
GRAPHITI_API_KEY=your-api-key
GRAPHITI_ENDPOINT=http://localhost:8000
GRAPHITI_ENABLE_CACHE=true
GRAPHITI_CACHE_TTL=3600
GRAPHITI_REQUEST_TIMEOUT=30
GRAPHITI_MAX_RETRIES=3
GRAPHITI_RETRY_DELAY=1.0
GRAPHITI_RETRY_BACKOFF=2.0
```

### Application Settings

Configure application behavior:

```bash
# Environment
APP_ENV=development  # development, staging, production, testing
APP_DEBUG=false
APP_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Feature Flags
APP_ENABLE_GRAPHITI_MEMORY=true
APP_ENABLE_ASYNC_PROCESSING=true
APP_ENABLE_NLP_MODELS=false

# Performance
APP_MAX_WORKERS=4
APP_BATCH_SIZE=100
APP_EXTRACTION_TIMEOUT=30

# Paths
APP_DATA_DIR=./data
APP_CACHE_DIR=./cache
APP_LOG_DIR=./logs
```

## Usage Examples

### Basic Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings:
```env
NEO4J_PASSWORD=mysecurepassword
GRAPHITI_API_KEY=myapikey
APP_ENV=development
```

3. Use in your code:
```python
from graphiti_config import get_config

config = get_config()
print(f"Neo4j URI: {config.neo4j.uri}")
print(f"Environment: {config.app.env}")
```

### Checking Feature Flags

```python
from graphiti_config import is_graphiti_enabled, is_async_enabled

if is_graphiti_enabled():
    # Initialize Graphiti integration
    pass

if is_async_enabled():
    # Use async processing
    pass
```

### Getting Configuration for Drivers

```python
from neo4j import GraphDatabase
from graphiti_config import get_config

config = get_config()

# Get Neo4j driver configuration
driver = GraphDatabase.driver(**config.get_neo4j_config())

# Get Graphiti API headers
headers = config.get_graphiti_headers()
```

### Using Path Helpers

```python
from graphiti_config import get_data_path, get_cache_path, get_log_path

# Paths are automatically created if they don't exist
model_file = get_data_path("models/embeddings.pkl")
cache_file = get_cache_path("responses/latest.json")
log_file = get_log_path("errors/critical.log")
```

## Environment-Specific Behavior

The configuration system supports different environments:

### Development
- Debug logging enabled
- Relaxed timeouts
- Local database connections
- Cache enabled for faster development

### Production
- Optimized performance settings
- Secure connections required
- Stricter timeouts
- Enhanced monitoring

### Testing
- In-memory databases
- Disabled caching
- Fast timeouts
- Isolated data directories

Check environment:
```python
config = get_config()

if config.is_production:
    # Production-specific logic
elif config.is_development:
    # Development-specific logic
elif config.is_testing:
    # Testing-specific logic
```

## Validation

The configuration system includes built-in validation:

```python
config = get_config()

# Validate Neo4j settings
if not config.validate_neo4j_connection():
    print("Neo4j configuration incomplete!")

# Validate Graphiti settings  
if not config.validate_graphiti_connection():
    print("Graphiti configuration incomplete!")
```

## Advanced Configuration

### Custom URI Schemes

Supported Neo4j URI schemes:
- `bolt://` - Standard Bolt protocol
- `bolt+s://` - Bolt with TLS
- `bolt+ssc://` - Bolt with self-signed certificates
- `neo4j://` - Neo4j protocol with routing
- `neo4j+s://` - Neo4j with TLS
- `neo4j+ssc://` - Neo4j with self-signed certificates

### Connection Pool Tuning

For high-throughput applications:
```env
NEO4J_MAX_CONNECTION_POOL_SIZE=100
NEO4J_MAX_CONNECTION_LIFETIME=1800
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=30
```

### Retry Configuration

For unreliable networks:
```env
GRAPHITI_MAX_RETRIES=5
GRAPHITI_RETRY_DELAY=2.0
GRAPHITI_RETRY_BACKOFF=2.5
```

## Security Considerations

1. **Never commit `.env` files** - Use `.env.example` as a template
2. **Use strong passwords** - Especially in production
3. **Enable encryption** - Set `NEO4J_ENCRYPTED=true` for production
4. **Rotate API keys** - Regular rotation of `GRAPHITI_API_KEY`
5. **Validate certificates** - Use proper trust settings in production

## Troubleshooting

### Common Issues

1. **Missing password error**
   ```
   ValidationError: password field required
   ```
   Solution: Set `NEO4J_PASSWORD` environment variable

2. **Invalid URI scheme**
   ```
   ValueError: Invalid Neo4j URI scheme
   ```
   Solution: Use a supported URI scheme (bolt, neo4j, etc.)

3. **Connection timeout**
   - Increase `NEO4J_CONNECTION_ACQUISITION_TIMEOUT`
   - Check network connectivity
   - Verify Neo4j is running

4. **Directory permissions**
   - Ensure write permissions for data/cache/log directories
   - Use absolute paths if needed

### Debug Mode

Enable debug mode for detailed logging:
```env
APP_DEBUG=true
APP_LOG_LEVEL=DEBUG
```

### Configuration Export

Export configuration for debugging:
```python
config = get_config()
print(config.to_dict(include_secrets=False))
```

## Best Practices

1. **Use environment-specific files**
   - `.env.development`
   - `.env.production`
   - `.env.testing`

2. **Validate early**
   - Check configuration at startup
   - Fail fast on missing required settings

3. **Monitor configuration**
   - Log configuration changes
   - Track feature flag usage

4. **Test configuration**
   - Unit test configuration classes
   - Integration test with real services

5. **Document changes**
   - Update `.env.example` when adding variables
   - Document new feature flags
   - Note breaking changes