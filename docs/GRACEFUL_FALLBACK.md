# Graceful Fallback Guide

This guide explains the graceful fallback mechanisms that ensure the application continues to function when Graphiti service is unavailable or degraded.

## Overview

The fallback system provides multiple strategies for handling Graphiti unavailability:
- **Local caching** for read operations
- **Local storage** using SQLite for persistence
- **Queue-based writes** for eventual consistency
- **Circuit breaker** pattern to prevent cascading failures
- **Automatic recovery** when service returns

## Fallback Modes

### 1. DISABLED (Default)
- No fallback active
- Operations fail if Graphiti is unavailable
- Used during normal operation

### 2. READ_ONLY
- Only cached data is available
- Write operations are rejected
- Useful during maintenance windows

### 3. MEMORY_ONLY
- Data stored in memory only
- Fast but not persistent
- Suitable for temporary outages

### 4. LOCAL_STORAGE
- Uses SQLite for local persistence
- Full read/write capability
- Data survives application restarts

### 5. QUEUE_WRITES
- Writes are queued for later sync
- Reads from local storage
- Ensures eventual consistency

## Quick Start

### Basic Usage

```python
from graphiti_fallback import FallbackManager, get_fallback_manager

# Get global fallback manager
fallback = get_fallback_manager()

# Check and activate fallback if needed
if await fallback.check_and_activate():
    print(f"Fallback activated: {fallback.state.mode}")

# Use fallback context
async with fallback.fallback_context():
    # Operations here have fallback protection
    entity = QuestionEntity(id="q1", content="Test?")
    fallback.store_entity(entity)
```

### Using Decorators

```python
from graphiti_fallback import with_fallback

class QuestionService:
    @with_fallback
    async def create_question(self, question: QuestionEntity):
        # This method automatically uses fallback on failure
        return await graphiti_client.create_entity(question)
    
    @with_fallback
    async def get_question(self, question_id: str):
        # Read operations also have fallback support
        return await graphiti_client.get_entity(question_id)
```

## Circuit Breaker Integration

The circuit breaker prevents cascading failures and integrates with the fallback system:

```python
from graphiti_circuit_breaker import circuit_breaker, get_circuit_breaker

# Get a circuit breaker
breaker = get_circuit_breaker("graphiti")

# Use as decorator
@circuit_breaker(breaker)
async def call_graphiti_api():
    # Circuit breaker monitors failures
    # Opens circuit after threshold
    # Activates fallback automatically
    pass

# Check circuit status
status = breaker.get_status()
print(f"Circuit State: {status['state']}")
print(f"Failure Rate: {status['stats']['failure_rate']}")
```

## Configuration

### Environment Variables

```env
# Fallback configuration
FALLBACK_MODE=queue_writes  # Default fallback mode
FALLBACK_CACHE_DIR=~/.graphiti_cache  # Cache directory
FALLBACK_DB_PATH=~/.graphiti_fallback.db  # Local database

# Circuit breaker settings
CIRCUIT_FAILURE_THRESHOLD=5  # Failures before opening
CIRCUIT_SUCCESS_THRESHOLD=2  # Successes to close
CIRCUIT_TIMEOUT=60  # Seconds before retry
```

### Programmatic Configuration

```python
from graphiti_fallback import FallbackManager
from graphiti_circuit_breaker import CircuitBreaker

# Custom fallback configuration
fallback = FallbackManager()
fallback.cache = LocalCache(Path("/custom/cache"))
fallback.storage = LocalStorage(Path("/custom/fallback.db"))

# Custom circuit breaker
breaker = CircuitBreaker(
    failure_threshold=3,
    success_threshold=1,
    timeout=30.0,
    fallback_manager=fallback
)
```

## Monitoring and Metrics

### Fallback Metrics

```python
# Get fallback status
status = fallback.get_status()
print(f"Mode: {status['mode']}")
print(f"Active: {status['is_active']}")
print(f"Queued Operations: {status['queued_operations']}")
print(f"Cache Hit Rate: {status['cache_hit_rate']:.1%}")

# Monitor state changes
def on_state_change(state):
    if state.is_active:
        alert_team(f"Fallback activated: {state.reason}")

fallback.add_callback(on_state_change)
```

### Circuit Breaker Metrics

```python
# Get circuit breaker status
status = breaker.get_status()
print(f"State: {status['state']}")
print(f"Total Calls: {status['stats']['total_calls']}")
print(f"Failure Rate: {status['stats']['failure_rate']}")
print(f"Time in State: {status['time_in_current_state']}")

# Monitor state transitions
def on_circuit_change(old_state, new_state):
    logger.info(f"Circuit: {old_state} -> {new_state}")

breaker.add_state_callback(on_circuit_change)
```

## Usage Patterns

### 1. Service Class with Fallback

```python
class ResilientQuestionService:
    def __init__(self):
        self.fallback = get_fallback_manager()
        self.breaker = get_circuit_breaker("questions")
    
    @circuit_breaker()
    @with_fallback
    async def create_question(self, question: QuestionEntity):
        # Try normal operation
        return await self.graphiti.create_entity(question)
    
    async def get_question(self, question_id: str):
        # Manual fallback handling for more control
        try:
            return await self.graphiti.get_entity(question_id)
        except Exception as e:
            if await self.fallback.check_and_activate():
                return self.fallback.get_entity(question_id)
            raise
```

### 2. Batch Operations with Queue

```python
async def batch_import_questions(questions: List[QuestionEntity]):
    fallback = get_fallback_manager()
    
    # Activate queue mode for batch operations
    fallback.state.activate(FallbackMode.QUEUE_WRITES, "Batch import")
    
    try:
        for question in questions:
            # These will be queued if Graphiti is down
            fallback.store_entity(question)
        
        print(f"Queued {fallback.state.queued_operations} operations")
        
    finally:
        # Operations will sync when service recovers
        fallback.state.deactivate()
```

### 3. Read-Through Cache Pattern

```python
class CachedQuestionRepository:
    def __init__(self):
        self.fallback = get_fallback_manager()
    
    async def get_question(self, question_id: str):
        # Try cache first
        cached = self.fallback.cache.get(f"question:{question_id}")
        if cached:
            return cached
        
        # Try Graphiti
        try:
            question = await self.graphiti.get_entity(question_id)
            # Cache for next time
            self.fallback.cache.set(f"question:{question_id}", question)
            return question
        except Exception:
            # Check local storage in fallback
            if self.fallback.state.is_active:
                return self.fallback.get_entity(question_id)
            raise
```

### 4. Adaptive Circuit Breaker

```python
from graphiti_circuit_breaker import AdaptiveCircuitBreaker

# Circuit breaker that adapts based on health
adaptive_breaker = AdaptiveCircuitBreaker(
    health_checker=GraphitiHealthChecker()
)

@circuit_breaker(adaptive_breaker)
async def adaptive_api_call():
    # Thresholds adjust based on service health
    # More conservative when degraded
    pass
```

## Sync and Recovery

### Manual Sync

```python
# Get queued operations
operations = fallback.storage.get_queued_operations()

for op in operations:
    try:
        # Sync to Graphiti
        if op.operation_type == "create":
            await graphiti.create_entity(op.data)
        
        # Remove from queue
        fallback.storage.delete_queued_operation(op.id)
        
    except Exception as e:
        logger.error(f"Sync failed: {e}")
```

### Automatic Recovery

```python
# Fallback monitors health and recovers automatically
async with fallback.fallback_context():
    # Operations here are protected
    # Fallback activates if needed
    # Deactivates when service recovers
    pass
```

## Best Practices

### 1. Choose Appropriate Fallback Mode

- **READ_ONLY**: During planned maintenance
- **MEMORY_ONLY**: For stateless services
- **LOCAL_STORAGE**: When data persistence is critical
- **QUEUE_WRITES**: For eventual consistency requirements

### 2. Monitor Fallback Activation

```python
# Alert on fallback activation
def alert_on_fallback(state):
    if state.is_active:
        send_alert(
            severity="warning",
            message=f"Fallback activated: {state.mode}",
            reason=state.reason
        )

fallback.add_callback(alert_on_fallback)
```

### 3. Implement Health-Based Decisions

```python
async def should_use_graphiti():
    health = await check_system_health()
    
    if health['components']['graphiti']['status'] == 'healthy':
        return True
    elif health['components']['graphiti']['status'] == 'degraded':
        # Use with caution
        return random.random() > 0.5  # 50% load shedding
    else:
        return False
```

### 4. Test Fallback Scenarios

```python
import pytest

@pytest.mark.asyncio
async def test_service_with_fallback():
    service = QuestionService()
    
    # Simulate Graphiti failure
    with patch('graphiti_client.create_entity', side_effect=ConnectionError):
        # Should use fallback
        result = await service.create_question(question)
        assert result is not None
        
        # Check fallback was used
        assert service.fallback_manager.state.is_active
```

## Troubleshooting

### Common Issues

1. **Fallback Not Activating**
   - Check health checker configuration
   - Verify error types trigger fallback
   - Review fallback mode settings

2. **Queue Growing Too Large**
   - Monitor queue size
   - Implement queue size limits
   - Consider dropping old operations

3. **Cache Misses**
   - Review cache size limits
   - Check cache key patterns
   - Monitor eviction rates

### Debug Mode

```python
import logging

# Enable debug logging
logging.getLogger('graphiti_fallback').setLevel(logging.DEBUG)
logging.getLogger('graphiti_circuit_breaker').setLevel(logging.DEBUG)

# Detailed fallback status
fallback = get_fallback_manager()
print(f"Fallback State: {fallback.state}")
print(f"Cache Stats: Hits={fallback.state.cache_hits}, Misses={fallback.state.cache_misses}")
print(f"Queued Ops: {fallback.state.queued_operations}")

# Circuit breaker details
breaker = get_circuit_breaker()
print(f"Circuit State: {breaker.state}")
print(f"Stats: {breaker.stats}")
```

## Performance Considerations

### 1. Cache Strategy

- Use memory cache for hot data
- Persist important data to disk
- Implement cache warming
- Set appropriate TTLs

### 2. Queue Management

- Limit queue size to prevent memory issues
- Implement queue priorities
- Consider operation deduplication
- Monitor sync performance

### 3. Circuit Breaker Tuning

- Balance failure threshold with stability
- Adjust timeout based on recovery time
- Monitor half-open success rate
- Consider adaptive thresholds

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()
fallback = get_fallback_manager()

@app.middleware("http")
async def fallback_middleware(request, call_next):
    # Check fallback status
    if fallback.state.is_active:
        request.state.fallback_active = True
    
    response = await call_next(request)
    
    # Add fallback headers
    if fallback.state.is_active:
        response.headers["X-Fallback-Mode"] = fallback.state.mode.value
    
    return response

@app.get("/health/fallback")
async def fallback_status():
    return fallback.get_status()
```

### With Background Tasks

```python
import asyncio

async def sync_worker():
    """Background worker to sync queued operations."""
    fallback = get_fallback_manager()
    
    while True:
        try:
            if fallback.state.mode == FallbackMode.QUEUE_WRITES:
                # Check if Graphiti is healthy
                health = await check_system_health()
                if health['components']['graphiti']['status'] == 'healthy':
                    # Sync operations
                    await fallback._sync_queued_operations()
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Sync worker error: {e}")
            await asyncio.sleep(300)  # Wait longer on error

# Start worker
asyncio.create_task(sync_worker())
```

## Summary

The graceful fallback system ensures your application remains resilient when Graphiti is unavailable. Key features:

- **Multiple fallback strategies** for different scenarios
- **Automatic activation** based on health checks
- **Circuit breaker** prevents cascading failures
- **Queue-based sync** for eventual consistency
- **Comprehensive monitoring** and metrics
- **Easy integration** with decorators and context managers

Use these patterns to build resilient applications that gracefully handle service disruptions.