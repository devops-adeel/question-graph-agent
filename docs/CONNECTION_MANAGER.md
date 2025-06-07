# Connection Manager Guide

This guide explains how to use the connection managers for Neo4j and Graphiti with automatic retry logic, connection pooling, and error handling.

## Overview

The connection management system provides:

- **Automatic retry logic** for transient failures
- **Connection pooling** for efficient resource usage
- **State tracking** with callbacks for monitoring
- **Metrics collection** for performance analysis
- **Both sync and async** connection support
- **Context managers** for safe resource handling

## Components

### Neo4jConnectionManager

Manages connections to Neo4j database with features:
- Automatic reconnection on failure
- Session management with retry logic
- Query execution helpers
- Connection state tracking
- Performance metrics

### GraphitiConnectionManager

Manages connections to Graphiti service with features:
- HTTP client with retry logic
- Request helpers with automatic retries
- Health check verification
- Connection state tracking
- Performance metrics

### ConnectionPool

Generic connection pooling for any manager type:
- Configurable pool size
- Connection reuse
- Thread-safe acquisition
- Timeout support

## Basic Usage

### Simple Query Execution

```python
from graphiti_connection import get_neo4j_connection

# Get global connection manager
neo4j = get_neo4j_connection()

# Execute a query with automatic retry
results = neo4j.execute_query(
    "MATCH (n:Person) WHERE n.age > $age RETURN n.name as name",
    {"age": 25}
)

for record in results:
    print(f"Name: {record['name']}")
```

### Using Sessions

```python
from graphiti_connection import Neo4jConnectionManager

manager = Neo4jConnectionManager()

# Session with automatic cleanup
with manager.session() as session:
    # Run queries in transaction
    session.run("CREATE (n:Person {name: $name})", {"name": "Alice"})
    
    result = session.run("MATCH (n:Person) RETURN count(n) as count")
    count = result.single()["count"]
    print(f"Total persons: {count}")
```

### Making HTTP Requests

```python
from graphiti_connection import get_graphiti_connection

# Get global connection manager
graphiti = get_graphiti_connection()

# Make request with automatic retry
response = graphiti.request("POST", "/api/entities", json={
    "type": "Question",
    "content": "What is Python?",
    "metadata": {"difficulty": "medium"}
})

if response.status_code == 201:
    entity = response.json()
    print(f"Created entity: {entity['id']}")
```

## Async Usage

### Async Queries

```python
import asyncio
from graphiti_connection import Neo4jConnectionManager

async def async_example():
    manager = Neo4jConnectionManager()
    
    # Async query execution
    results = await manager.execute_query_async(
        "MATCH (n:Person) RETURN n.name as name LIMIT 10"
    )
    
    for record in results:
        print(f"Name: {record['name']}")
    
    # Async session
    async with manager.async_session() as session:
        result = await session.run("RETURN 'Hello async!' as message")
        async for record in result:
            print(record["message"])
    
    await manager.close_async()

# Run async function
asyncio.run(async_example())
```

### Async HTTP Requests

```python
async def graphiti_async():
    manager = GraphitiConnectionManager()
    
    # Async request
    response = await manager.request_async("GET", "/api/status")
    status = response.json()
    
    await manager.close_async()
```

## Connection Pooling

### Basic Pool Usage

```python
from graphiti_connection import ConnectionPool, Neo4jConnectionManager

# Create pool with 5 connections
pool = ConnectionPool(Neo4jConnectionManager, size=5)

# Acquire connection from pool
with pool.acquire() as conn:
    results = conn.execute_query("RETURN 1")
    # Connection automatically returned to pool

# Pool can handle concurrent requests
import concurrent.futures

def worker(query):
    with pool.acquire() as conn:
        return conn.execute_query(query)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(worker, f"RETURN {i}") for i in range(10)]
    results = [f.result() for f in futures]

# Clean up
pool.close()
```

### Pool with Timeout

```python
try:
    # Acquire with timeout
    with pool.acquire(timeout=5.0) as conn:
        # Use connection
        pass
except TimeoutError:
    print("Failed to acquire connection within 5 seconds")
```

## Error Handling

### Automatic Retries

The connection managers automatically retry on transient failures:

```python
# Configure retry behavior in config
config.graphiti.max_retries = 5
config.graphiti.retry_delay = 2.0
config.graphiti.retry_backoff = 2.0

manager = GraphitiConnectionManager(config)

# This will retry up to 5 times with exponential backoff
try:
    response = manager.request("GET", "/api/data")
except Exception as e:
    print(f"Failed after {manager.metrics.total_retries} retries: {e}")
```

### Manual Retry Control

```python
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
)
def custom_operation(manager):
    with manager.session() as session:
        return session.run("MATCH (n) RETURN n LIMIT 1")

# Use with custom retry logic
try:
    result = custom_operation(neo4j_manager)
except Exception as e:
    print(f"Operation failed: {e}")
```

## Monitoring and Metrics

### Connection State Monitoring

```python
from graphiti_connection import ConnectionState

def on_state_change(state: ConnectionState):
    if state == ConnectionState.CONNECTED:
        print("✅ Connected successfully")
    elif state == ConnectionState.FAILED:
        print("❌ Connection failed")
    elif state == ConnectionState.CONNECTING:
        print("⏳ Connecting...")

manager = Neo4jConnectionManager()
manager.add_state_callback(on_state_change)

# State changes will trigger callback
manager.connect()
```

### Performance Metrics

```python
# After using connection
metrics = manager.metrics

print(f"Total connections: {metrics.total_connections}")
print(f"Successful: {metrics.successful_connections}")
print(f"Failed: {metrics.failed_connections}")
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Total retries: {metrics.total_retries}")
print(f"Last error: {metrics.last_error}")
print(f"Connection time: {metrics.connection_duration:.2f}s")
```

## Configuration

### Connection Settings

Configure connection behavior through environment variables:

```bash
# Neo4j settings
NEO4J_MAX_CONNECTION_LIFETIME=3600
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=60

# Graphiti settings
GRAPHITI_REQUEST_TIMEOUT=30
GRAPHITI_MAX_RETRIES=3
GRAPHITI_RETRY_DELAY=1.0
GRAPHITI_RETRY_BACKOFF=2.0
```

### Custom Configuration

```python
from graphiti_config import RuntimeConfig, Neo4jConfig

# Create custom config
config = RuntimeConfig()
config.neo4j.max_connection_pool_size = 100
config.neo4j.connection_acquisition_timeout = 30

# Use with manager
manager = Neo4jConnectionManager(config)
```

## Best Practices

### 1. Use Global Managers

```python
# ✅ Good - reuses connection
neo4j = get_neo4j_connection()
results1 = neo4j.execute_query("RETURN 1")
results2 = neo4j.execute_query("RETURN 2")

# ❌ Bad - creates multiple connections
manager1 = Neo4jConnectionManager()
results1 = manager1.execute_query("RETURN 1")
manager2 = Neo4jConnectionManager()
results2 = manager2.execute_query("RETURN 2")
```

### 2. Always Use Context Managers

```python
# ✅ Good - automatic cleanup
with manager.session() as session:
    session.run(query)

# ❌ Bad - manual cleanup required
session = manager._driver.session()
session.run(query)
session.close()  # Easy to forget
```

### 3. Handle Connection Failures

```python
def resilient_operation():
    try:
        manager = get_neo4j_connection()
        return manager.execute_query("MATCH (n) RETURN n")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        # Fallback behavior
        return []
```

### 4. Monitor Connection Health

```python
import time

def health_check():
    manager = get_neo4j_connection()
    
    try:
        start = time.time()
        manager.execute_query("RETURN 1")
        duration = time.time() - start
        
        return {
            "status": "healthy",
            "response_time": duration,
            "success_rate": manager.metrics.success_rate
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_error": manager.metrics.last_error
        }
```

### 5. Clean Up Resources

```python
import atexit
from graphiti_connection import close_all_connections

# Register cleanup on exit
atexit.register(close_all_connections)

# Or in async context
async def cleanup():
    await close_all_connections_async()
```

## Troubleshooting

### Connection Timeouts

If connections are timing out:
1. Increase `connection_acquisition_timeout`
2. Check network connectivity
3. Verify service is running
4. Check connection pool size

### Retry Exhaustion

If retries are exhausted:
1. Increase `max_retries` in configuration
2. Adjust `retry_delay` and `retry_backoff`
3. Check for persistent service issues
4. Implement circuit breaker pattern

### Memory Leaks

Prevent connection leaks:
1. Always use context managers
2. Call `close()` in finally blocks
3. Monitor pool size
4. Use global managers

### Performance Issues

For better performance:
1. Increase connection pool size
2. Reuse connections
3. Batch operations
4. Use async for I/O heavy workloads

## Advanced Topics

### Custom Retry Strategies

```python
from tenacity import retry_if_result

def is_retriable_error(exception):
    return isinstance(exception, (ServiceUnavailable, TransientError))

@retry(
    retry=retry_if_exception_type(is_retriable_error),
    stop=stop_after_attempt(5),
)
def custom_query(manager, query):
    return manager.execute_query(query)
```

### Connection Warmup

```python
def warmup_connections(pool_size=5):
    pool = ConnectionPool(Neo4jConnectionManager, size=pool_size)
    
    # Pre-create all connections
    connections = []
    for _ in range(pool_size):
        with pool.acquire() as conn:
            conn.execute_query("RETURN 1")
            connections.append(conn)
    
    logger.info(f"Warmed up {pool_size} connections")
    return pool
```

### Circuit Breaker Pattern

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
    
    def call(self, func, *args, **kwargs):
        if self.is_open:
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.is_open = False
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.error("Circuit breaker opened")
            
            raise e
```