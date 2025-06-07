"""
Health check and monitoring module for Graphiti integration.

This module provides comprehensive health checks, monitoring capabilities,
and diagnostics for both Neo4j and Graphiti connections. It includes
periodic health checks, metrics collection, and alerting mechanisms.
"""

import asyncio
import time
import logging
import json
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import threading
from collections import deque

from neo4j.exceptions import Neo4jError, ServiceUnavailable
import httpx

from graphiti_config import get_config, RuntimeConfig
from graphiti_connection import (
    Neo4jConnectionManager,
    GraphitiConnectionManager,
    get_neo4j_connection,
    get_graphiti_connection,
    ConnectionState,
)


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Component type enumeration."""
    NEO4J = "neo4j"
    GRAPHITI = "graphiti"
    SYSTEM = "system"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: ComponentType
    status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class HealthMetrics:
    """Health metrics for monitoring."""
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    degraded_checks: int = 0
    average_latency_ms: float = 0.0
    last_check_time: Optional[datetime] = None
    last_healthy_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    
    def record_check(self, result: HealthCheckResult):
        """Record a health check result."""
        self.total_checks += 1
        self.last_check_time = result.timestamp
        
        # Update latency
        if result.latency_ms is not None:
            # Running average
            self.average_latency_ms = (
                (self.average_latency_ms * (self.total_checks - 1) + result.latency_ms) 
                / self.total_checks
            )
        
        # Update status counts
        if result.status == HealthStatus.HEALTHY:
            self.successful_checks += 1
            self.last_healthy_time = result.timestamp
            self.consecutive_failures = 0
        elif result.status == HealthStatus.DEGRADED:
            self.degraded_checks += 1
            self.consecutive_failures = 0
        else:
            self.failed_checks += 1
            self.last_failure_time = result.timestamp
            self.consecutive_failures += 1
        
        # Update uptime percentage
        if self.total_checks > 0:
            self.uptime_percentage = (
                (self.successful_checks + self.degraded_checks) / self.total_checks * 100
            )


class HealthChecker:
    """Base health checker class."""
    
    def __init__(self, component: ComponentType, config: Optional[RuntimeConfig] = None):
        """Initialize health checker."""
        self.component = component
        self.config = config or get_config()
        self.metrics = HealthMetrics()
        self._history: deque[HealthCheckResult] = deque(maxlen=100)
        self._callbacks: List[Callable[[HealthCheckResult], None]] = []
    
    def add_callback(self, callback: Callable[[HealthCheckResult], None]):
        """Add callback for health check results."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, result: HealthCheckResult):
        """Notify all callbacks of health check result."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in health check callback: {e}")
    
    async def check_health(self) -> HealthCheckResult:
        """Check health (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get health metrics."""
        return asdict(self.metrics)
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get health check history."""
        history = list(self._history)
        if limit:
            history = history[-limit:]
        return [result.to_dict() for result in history]
    
    async def perform_check(self) -> HealthCheckResult:
        """Perform health check and record result."""
        result = await self.check_health()
        self.metrics.record_check(result)
        self._history.append(result)
        self._notify_callbacks(result)
        return result


class Neo4jHealthChecker(HealthChecker):
    """Health checker for Neo4j database."""
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """Initialize Neo4j health checker."""
        super().__init__(ComponentType.NEO4J, config)
        self.connection_manager = get_neo4j_connection()
    
    async def check_health(self) -> HealthCheckResult:
        """Check Neo4j health."""
        start_time = time.time()
        details = {}
        
        try:
            # Basic connectivity check
            async with self.connection_manager.async_session() as session:
                # Check database is online
                result = await session.run("RETURN 1 as alive")
                record = await result.single()
                
                if not record or record["alive"] != 1:
                    return HealthCheckResult(
                        component=self.component,
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=(time.time() - start_time) * 1000,
                        error="Database not responding correctly"
                    )
                
                # Get database statistics
                stats_result = await session.run("""
                    CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Store file sizes')
                    YIELD attributes
                    RETURN attributes
                """)
                stats = await stats_result.data()
                
                # Get node/relationship counts
                count_result = await session.run("""
                    MATCH (n)
                    WITH count(n) as nodeCount
                    MATCH ()-[r]->()
                    RETURN nodeCount, count(r) as relCount
                """)
                counts = await count_result.single()
                
                details = {
                    "node_count": counts["nodeCount"] if counts else 0,
                    "relationship_count": counts["relCount"] if counts else 0,
                    "connection_state": self.connection_manager.state.value,
                    "connection_metrics": {
                        "success_rate": self.connection_manager.metrics.success_rate,
                        "total_connections": self.connection_manager.metrics.total_connections,
                        "failed_connections": self.connection_manager.metrics.failed_connections,
                    }
                }
                
                # Check latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Determine status based on latency and connection state
                if latency_ms > 5000:  # 5 seconds
                    status = HealthStatus.DEGRADED
                    details["warning"] = "High latency detected"
                elif self.connection_manager.state != ConnectionState.CONNECTED:
                    status = HealthStatus.DEGRADED
                    details["warning"] = f"Connection state: {self.connection_manager.state.value}"
                else:
                    status = HealthStatus.HEALTHY
                
                return HealthCheckResult(
                    component=self.component,
                    status=status,
                    latency_ms=latency_ms,
                    details=details
                )
                
        except ServiceUnavailable as e:
            return HealthCheckResult(
                component=self.component,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"Service unavailable: {str(e)}",
                details={"connection_state": self.connection_manager.state.value}
            )
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return HealthCheckResult(
                component=self.component,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
                details=details
            )


class GraphitiHealthChecker(HealthChecker):
    """Health checker for Graphiti service."""
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """Initialize Graphiti health checker."""
        super().__init__(ComponentType.GRAPHITI, config)
        self.connection_manager = get_graphiti_connection()
    
    async def check_health(self) -> HealthCheckResult:
        """Check Graphiti health."""
        start_time = time.time()
        details = {}
        
        try:
            # Check health endpoint
            response = await self.connection_manager.request_async("GET", "/health")
            response.raise_for_status()
            
            health_data = response.json() if response.content else {}
            details = {
                "endpoint": self.config.graphiti.endpoint,
                "status_code": response.status_code,
                "connection_state": self.connection_manager.state.value,
                "connection_metrics": {
                    "success_rate": self.connection_manager.metrics.success_rate,
                    "total_connections": self.connection_manager.metrics.total_connections,
                    "failed_connections": self.connection_manager.metrics.failed_connections,
                },
                "service_info": health_data
            }
            
            # Check latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Determine status
            if response.status_code == 200:
                if latency_ms > 2000:  # 2 seconds
                    status = HealthStatus.DEGRADED
                    details["warning"] = "High latency detected"
                else:
                    status = HealthStatus.HEALTHY
            elif 500 <= response.status_code < 600:
                status = HealthStatus.UNHEALTHY
            else:
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                component=self.component,
                status=status,
                latency_ms=latency_ms,
                details=details
            )
            
        except httpx.HTTPStatusError as e:
            return HealthCheckResult(
                component=self.component,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"HTTP error: {e.response.status_code}",
                details={
                    "status_code": e.response.status_code,
                    "connection_state": self.connection_manager.state.value
                }
            )
        except Exception as e:
            logger.error(f"Graphiti health check failed: {e}")
            return HealthCheckResult(
                component=self.component,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
                details=details
            )


class SystemHealthChecker(HealthChecker):
    """Overall system health checker."""
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """Initialize system health checker."""
        super().__init__(ComponentType.SYSTEM, config)
        self.neo4j_checker = Neo4jHealthChecker(config)
        self.graphiti_checker = GraphitiHealthChecker(config)
    
    async def check_health(self) -> HealthCheckResult:
        """Check overall system health."""
        start_time = time.time()
        
        # Check all components
        neo4j_result, graphiti_result = await asyncio.gather(
            self.neo4j_checker.perform_check(),
            self.graphiti_checker.perform_check(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(neo4j_result, Exception):
            neo4j_result = HealthCheckResult(
                component=ComponentType.NEO4J,
                status=HealthStatus.UNHEALTHY,
                error=str(neo4j_result)
            )
        
        if isinstance(graphiti_result, Exception):
            graphiti_result = HealthCheckResult(
                component=ComponentType.GRAPHITI,
                status=HealthStatus.UNHEALTHY,
                error=str(graphiti_result)
            )
        
        # Determine overall status
        statuses = [neo4j_result.status, graphiti_result.status]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED
        
        details = {
            "components": {
                "neo4j": neo4j_result.to_dict(),
                "graphiti": graphiti_result.to_dict(),
            },
            "metrics": {
                "neo4j": self.neo4j_checker.get_metrics(),
                "graphiti": self.graphiti_checker.get_metrics(),
            }
        }
        
        return HealthCheckResult(
            component=self.component,
            status=overall_status,
            latency_ms=(time.time() - start_time) * 1000,
            details=details
        )


class HealthMonitor:
    """Periodic health monitoring service."""
    
    def __init__(self, 
                 checker: HealthChecker,
                 interval_seconds: int = 60,
                 failure_threshold: int = 3):
        """Initialize health monitor.
        
        Args:
            checker: Health checker to use
            interval_seconds: Check interval in seconds
            failure_threshold: Consecutive failures before alerting
        """
        self.checker = checker
        self.interval_seconds = interval_seconds
        self.failure_threshold = failure_threshold
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._alert_callbacks: List[Callable[[HealthCheckResult, int], None]] = []
    
    def add_alert_callback(self, callback: Callable[[HealthCheckResult, int], None]):
        """Add callback for alerts.
        
        Args:
            callback: Function called with (result, consecutive_failures)
        """
        self._alert_callbacks.append(callback)
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                result = await self.checker.perform_check()
                
                # Check for alerts
                if (result.status == HealthStatus.UNHEALTHY and 
                    self.checker.metrics.consecutive_failures >= self.failure_threshold):
                    for callback in self._alert_callbacks:
                        try:
                            callback(result, self.checker.metrics.consecutive_failures)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")
                
                # Wait for next check
                await asyncio.sleep(self.interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(self.interval_seconds)
    
    async def start(self):
        """Start monitoring."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started health monitoring for {self.checker.component.value}")
    
    async def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped health monitoring for {self.checker.component.value}")


class HealthService:
    """Central health monitoring service."""
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """Initialize health service."""
        self.config = config or get_config()
        self.system_checker = SystemHealthChecker(config)
        self.monitors: Dict[ComponentType, HealthMonitor] = {}
        self._started = False
    
    async def start(self):
        """Start health monitoring."""
        if self._started:
            return
        
        # Create monitors for each component
        self.monitors[ComponentType.NEO4J] = HealthMonitor(
            self.system_checker.neo4j_checker,
            interval_seconds=30
        )
        self.monitors[ComponentType.GRAPHITI] = HealthMonitor(
            self.system_checker.graphiti_checker,
            interval_seconds=30
        )
        self.monitors[ComponentType.SYSTEM] = HealthMonitor(
            self.system_checker,
            interval_seconds=60
        )
        
        # Add default alert logging
        for monitor in self.monitors.values():
            monitor.add_alert_callback(self._log_alert)
        
        # Start all monitors
        await asyncio.gather(*[
            monitor.start() for monitor in self.monitors.values()
        ])
        
        self._started = True
        logger.info("Health monitoring service started")
    
    async def stop(self):
        """Stop health monitoring."""
        if not self._started:
            return
        
        # Stop all monitors
        await asyncio.gather(*[
            monitor.stop() for monitor in self.monitors.values()
        ])
        
        self._started = False
        logger.info("Health monitoring service stopped")
    
    def _log_alert(self, result: HealthCheckResult, consecutive_failures: int):
        """Log health alerts."""
        logger.error(
            f"HEALTH ALERT: {result.component.value} is {result.status.value} "
            f"(consecutive failures: {consecutive_failures}). "
            f"Error: {result.error}"
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status for all components."""
        result = await self.system_checker.perform_check()
        
        return {
            "status": result.status.value,
            "timestamp": result.timestamp.isoformat(),
            "components": result.details.get("components", {}),
            "metrics": result.details.get("metrics", {}),
            "history": {
                "neo4j": self.system_checker.neo4j_checker.get_history(10),
                "graphiti": self.system_checker.graphiti_checker.get_history(10),
                "system": self.system_checker.get_history(10),
            }
        }


# Global health service
_health_service: Optional[HealthService] = None


def get_health_service() -> HealthService:
    """Get or create global health service."""
    global _health_service
    if _health_service is None:
        _health_service = HealthService()
    return _health_service


async def check_system_health() -> Dict[str, Any]:
    """Quick system health check."""
    service = get_health_service()
    return await service.get_health_status()