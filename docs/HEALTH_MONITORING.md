# Health Monitoring Guide

This guide explains the health monitoring and checking capabilities for the Graphiti integration, including real-time monitoring, alerting, and diagnostics.

## Overview

The health monitoring system provides:
- Real-time health checks for Neo4j and Graphiti
- Continuous monitoring with configurable intervals
- Alert mechanisms for failures
- Metrics collection and reporting
- Connection state tracking
- Performance monitoring (latency)

## Quick Start

### Command Line Health Check

```bash
# Check overall system health
python scripts/health_check.py check

# Check specific component
python scripts/health_check.py check neo4j
python scripts/health_check.py check graphiti

# Start continuous monitoring
python scripts/health_check.py monitor --interval 30 --alert

# Export health report
python scripts/health_check.py export -o health_report.json
```

### Programmatic Health Check

```python
import asyncio
from graphiti_health import check_system_health

async def main():
    health = await check_system_health()
    print(f"System Status: {health['status']}")
    
asyncio.run(main())
```

## Health Status Levels

### 1. **HEALTHY** ✅
- All components responding normally
- Latency within acceptable limits
- No errors detected

### 2. **DEGRADED** ⚠️
- Components responding but with issues
- High latency detected (Neo4j >5s, Graphiti >2s)
- Non-critical errors present

### 3. **UNHEALTHY** ❌
- Component not responding
- Critical errors detected
- Service unavailable

### 4. **UNKNOWN** ❓
- Health status cannot be determined
- Check never performed

## Components

### Neo4j Database Health

Checks performed:
- Database connectivity
- Query execution capability
- Node and relationship counts
- Connection pool status
- Response latency

```python
from graphiti_health import Neo4jHealthChecker

checker = Neo4jHealthChecker()
result = await checker.perform_check()

print(f"Status: {result.status}")
print(f"Latency: {result.latency_ms}ms")
print(f"Node Count: {result.details['node_count']}")
```

### Graphiti Service Health

Checks performed:
- Service availability
- HTTP endpoint responsiveness
- API health endpoint status
- Connection state
- Response latency

```python
from graphiti_health import GraphitiHealthChecker

checker = GraphitiHealthChecker()
result = await checker.perform_check()

print(f"Status: {result.status}")
print(f"Endpoint: {result.details['endpoint']}")
```

### System Health

Aggregates health from all components:
- Combines Neo4j and Graphiti health
- Determines overall system status
- Tracks component dependencies

## Continuous Monitoring

### Basic Monitoring

```python
from graphiti_health import HealthMonitor, SystemHealthChecker

# Create system checker
checker = SystemHealthChecker()

# Create monitor with 60-second interval
monitor = HealthMonitor(
    checker,
    interval_seconds=60,
    failure_threshold=3
)

# Start monitoring
await monitor.start()

# ... application runs ...

# Stop monitoring
await monitor.stop()
```

### Monitoring with Alerts

```python
# Add alert callback
def alert_handler(result, consecutive_failures):
    if consecutive_failures >= 3:
        send_email_alert(
            subject=f"Health Alert: {result.component} is {result.status}",
            body=f"Error: {result.error}"
        )

monitor.add_alert_callback(alert_handler)
```

### Health Service

The `HealthService` provides centralized monitoring:

```python
from graphiti_health import HealthService

# Create and start service
service = HealthService()
await service.start()

# Get comprehensive status
status = await service.get_health_status()

# Stop service
await service.stop()
```

## Metrics and Reporting

### Available Metrics

Each health checker tracks:
- `total_checks`: Total health checks performed
- `successful_checks`: Number of successful checks
- `failed_checks`: Number of failed checks
- `degraded_checks`: Number of degraded checks
- `average_latency_ms`: Average response time
- `uptime_percentage`: Percentage of successful/degraded checks
- `consecutive_failures`: Current failure streak

### Accessing Metrics

```python
# Get metrics from checker
metrics = checker.get_metrics()
print(f"Uptime: {metrics['uptime_percentage']}%")
print(f"Avg Latency: {metrics['average_latency_ms']}ms")

# Get history
history = checker.get_history(limit=10)
for check in history:
    print(f"{check['timestamp']}: {check['status']}")
```

### Exporting Reports

```bash
# Export JSON report
python scripts/health_check.py export -o report.json

# Export text report
python scripts/health_check.py export -o report.txt --format text

# Export specific component
python scripts/health_check.py export --component neo4j
```

## Integration Patterns

### 1. Application Startup Check

```python
async def startup():
    # Check health before starting
    health = await check_system_health()
    
    if health['status'] == 'unhealthy':
        logger.error("System unhealthy, cannot start")
        sys.exit(1)
    elif health['status'] == 'degraded':
        logger.warning("System degraded, starting with warnings")
    
    # Continue startup...
```

### 2. Background Monitoring

```python
class Application:
    def __init__(self):
        self.health_service = HealthService()
    
    async def start(self):
        # Start health monitoring
        await self.health_service.start()
        
        # Start application...
    
    async def stop(self):
        # Stop monitoring
        await self.health_service.stop()
```

### 3. HTTP Health Endpoint

```python
from fastapi import FastAPI
from graphiti_health import check_system_health

app = FastAPI()

@app.get("/health")
async def health_check():
    health = await check_system_health()
    
    status_code = 200 if health['status'] == 'healthy' else 503
    
    return {
        "status": health['status'],
        "timestamp": health['timestamp'],
        "components": health['components']
    }, status_code
```

### 4. Kubernetes Probes

```python
@app.get("/health/liveness")
async def liveness():
    # Basic check - is service running?
    return {"status": "alive"}

@app.get("/health/readiness")
async def readiness():
    # Full health check
    health = await check_system_health()
    
    if health['status'] in ['healthy', 'degraded']:
        return {"status": "ready"}
    else:
        return {"status": "not ready"}, 503
```

## Configuration

Health monitoring uses the same configuration as connections:

```env
# Neo4j settings affect health checks
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Graphiti settings
GRAPHITI_ENDPOINT=http://localhost:8000
GRAPHITI_API_KEY=your-api-key
```

### Custom Thresholds

```python
# Create checker with custom config
from graphiti_config import RuntimeConfig

config = RuntimeConfig()
config.neo4j.connection_timeout = 10  # 10 second timeout

checker = Neo4jHealthChecker(config)
```

## Troubleshooting

### Common Issues

1. **High Latency Warnings**
   - Check network connectivity
   - Verify service load
   - Review database query performance

2. **Connection State Issues**
   - Check if services are running
   - Verify authentication credentials
   - Review firewall/network settings

3. **Consecutive Failures**
   - Check service logs
   - Verify configuration
   - Test manual connections

### Debug Mode

Enable detailed logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Perform health check with details
checker = SystemHealthChecker()
result = await checker.perform_check()

# Inspect full details
print(json.dumps(result.to_dict(), indent=2))
```

## Best Practices

1. **Monitor Continuously**
   - Use HealthService for production
   - Set appropriate check intervals
   - Configure alerts for critical systems

2. **Handle Degradation**
   - Don't ignore degraded status
   - Implement graceful degradation
   - Log all health issues

3. **Set Realistic Thresholds**
   - Adjust latency thresholds for your environment
   - Consider network latency
   - Account for service warm-up

4. **Export Regular Reports**
   - Schedule daily/weekly exports
   - Track trends over time
   - Use for capacity planning

## Advanced Usage

### Custom Health Checks

```python
from graphiti_health import HealthChecker, ComponentType

class CustomHealthChecker(HealthChecker):
    def __init__(self):
        super().__init__(ComponentType.SYSTEM)
    
    async def check_health(self):
        # Perform custom checks
        start_time = time.time()
        
        try:
            # Your health check logic
            result = await check_custom_service()
            
            return HealthCheckResult(
                component=self.component,
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                details={"custom": result}
            )
        except Exception as e:
            return HealthCheckResult(
                component=self.component,
                status=HealthStatus.UNHEALTHY,
                error=str(e)
            )
```

### Composite Health Checks

```python
# Check multiple conditions
async def complex_health_check():
    neo4j_checker = Neo4jHealthChecker()
    graphiti_checker = GraphitiHealthChecker()
    
    # Parallel checks
    neo4j_result, graphiti_result = await asyncio.gather(
        neo4j_checker.perform_check(),
        graphiti_checker.perform_check()
    )
    
    # Custom logic
    if neo4j_result.status == HealthStatus.HEALTHY and \
       graphiti_result.status == HealthStatus.HEALTHY:
        return "All systems operational"
    else:
        return "System degraded"
```

### Health-Based Circuit Breaker

```python
class HealthAwareService:
    def __init__(self):
        self.checker = GraphitiHealthChecker()
        self.circuit_open = False
    
    async def call_service(self):
        # Check health before calling
        if self.circuit_open:
            result = await self.checker.perform_check()
            if result.status == HealthStatus.HEALTHY:
                self.circuit_open = False
            else:
                raise ServiceUnavailableError()
        
        try:
            return await self._make_call()
        except Exception as e:
            # Open circuit on failure
            self.circuit_open = True
            raise
```

## Performance Considerations

1. **Check Frequency**
   - Balance between freshness and load
   - Use longer intervals for stable systems
   - Increase frequency during incidents

2. **Timeout Settings**
   - Set reasonable connection timeouts
   - Account for network latency
   - Fail fast on obvious issues

3. **Resource Usage**
   - Health checks consume connections
   - Monitor checker resource usage
   - Use connection pooling

## Integration with Monitoring Systems

### Prometheus Metrics

```python
from prometheus_client import Gauge, Counter

health_status = Gauge('system_health_status', 'System health status')
health_latency = Gauge('system_health_latency_ms', 'Health check latency')
health_checks_total = Counter('health_checks_total', 'Total health checks')

async def export_metrics():
    result = await check_system_health()
    
    status_value = 1 if result['status'] == 'healthy' else 0
    health_status.set(status_value)
    
    health_checks_total.inc()
```

### Logging Integration

```python
import structlog

logger = structlog.get_logger()

def log_health_check(result):
    logger.info(
        "health_check_performed",
        component=result.component.value,
        status=result.status.value,
        latency_ms=result.latency_ms,
        error=result.error
    )
```

## Summary

The health monitoring system provides comprehensive visibility into the Graphiti integration's operational status. Use it to:

- Ensure system reliability
- Detect issues early
- Track performance trends
- Enable proactive maintenance
- Support operational decisions

Regular health monitoring is essential for maintaining a robust and reliable system.