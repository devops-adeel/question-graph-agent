"""
Tests for the health check and monitoring module.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import httpx

from graphiti_health import (
    HealthStatus,
    ComponentType,
    HealthCheckResult,
    HealthMetrics,
    Neo4jHealthChecker,
    GraphitiHealthChecker,
    SystemHealthChecker,
    HealthMonitor,
    HealthService,
    get_health_service,
    check_system_health,
)
from graphiti_connection import ConnectionState


class TestHealthCheckResult:
    """Test HealthCheckResult class."""
    
    def test_health_check_result_creation(self):
        """Test creating health check result."""
        result = HealthCheckResult(
            component=ComponentType.NEO4J,
            status=HealthStatus.HEALTHY,
            latency_ms=15.5,
            details={"node_count": 100},
            error=None
        )
        
        assert result.component == ComponentType.NEO4J
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms == 15.5
        assert result.details["node_count"] == 100
        assert result.error is None
        assert isinstance(result.timestamp, datetime)
    
    def test_health_check_result_to_dict(self):
        """Test converting result to dictionary."""
        result = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.DEGRADED,
            latency_ms=2500.0,
            error="High latency"
        )
        
        data = result.to_dict()
        
        assert data["component"] == "graphiti"
        assert data["status"] == "degraded"
        assert data["latency_ms"] == 2500.0
        assert data["error"] == "High latency"
        assert "timestamp" in data


class TestHealthMetrics:
    """Test HealthMetrics class."""
    
    def test_record_healthy_check(self):
        """Test recording healthy check."""
        metrics = HealthMetrics()
        result = HealthCheckResult(
            component=ComponentType.NEO4J,
            status=HealthStatus.HEALTHY,
            latency_ms=10.0
        )
        
        metrics.record_check(result)
        
        assert metrics.total_checks == 1
        assert metrics.successful_checks == 1
        assert metrics.failed_checks == 0
        assert metrics.consecutive_failures == 0
        assert metrics.average_latency_ms == 10.0
        assert metrics.uptime_percentage == 100.0
    
    def test_record_failed_check(self):
        """Test recording failed check."""
        metrics = HealthMetrics()
        result = HealthCheckResult(
            component=ComponentType.NEO4J,
            status=HealthStatus.UNHEALTHY,
            latency_ms=5000.0
        )
        
        metrics.record_check(result)
        
        assert metrics.total_checks == 1
        assert metrics.successful_checks == 0
        assert metrics.failed_checks == 1
        assert metrics.consecutive_failures == 1
        assert metrics.average_latency_ms == 5000.0
        assert metrics.uptime_percentage == 0.0
    
    def test_consecutive_failures(self):
        """Test consecutive failure tracking."""
        metrics = HealthMetrics()
        
        # Record failures
        for _ in range(3):
            metrics.record_check(HealthCheckResult(
                component=ComponentType.NEO4J,
                status=HealthStatus.UNHEALTHY
            ))
        
        assert metrics.consecutive_failures == 3
        
        # Record success
        metrics.record_check(HealthCheckResult(
            component=ComponentType.NEO4J,
            status=HealthStatus.HEALTHY
        ))
        
        assert metrics.consecutive_failures == 0


class TestNeo4jHealthChecker:
    """Test Neo4j health checker."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create mock connection manager."""
        manager = Mock()
        manager.state = ConnectionState.CONNECTED
        manager.metrics = Mock(
            success_rate=1.0,
            total_connections=10,
            failed_connections=0
        )
        return manager
    
    @pytest.fixture
    def checker(self, mock_connection_manager):
        """Create Neo4j health checker with mocks."""
        with patch('graphiti_health.get_neo4j_connection', return_value=mock_connection_manager):
            return Neo4jHealthChecker()
    
    @pytest.mark.asyncio
    async def test_healthy_check(self, checker, mock_connection_manager):
        """Test healthy Neo4j check."""
        # Mock session and results
        mock_session = AsyncMock()
        mock_connection_manager.async_session.return_value.__aenter__.return_value = mock_session
        
        # Mock query results
        mock_result1 = AsyncMock()
        mock_result1.single.return_value = {"alive": 1}
        
        mock_result2 = AsyncMock()
        mock_result2.data.return_value = []
        
        mock_result3 = AsyncMock()
        mock_result3.single.return_value = {"nodeCount": 100, "relCount": 200}
        
        mock_session.run.side_effect = [mock_result1, mock_result2, mock_result3]
        
        result = await checker.check_health()
        
        assert result.component == ComponentType.NEO4J
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms is not None
        assert result.details["node_count"] == 100
        assert result.details["relationship_count"] == 200
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_unhealthy_check_service_unavailable(self, checker, mock_connection_manager):
        """Test unhealthy check when service unavailable."""
        from neo4j.exceptions import ServiceUnavailable
        
        mock_connection_manager.async_session.side_effect = ServiceUnavailable("Connection refused")
        
        result = await checker.check_health()
        
        assert result.component == ComponentType.NEO4J
        assert result.status == HealthStatus.UNHEALTHY
        assert "Service unavailable" in result.error
    
    @pytest.mark.asyncio
    async def test_degraded_check_high_latency(self, checker, mock_connection_manager):
        """Test degraded status for high latency."""
        # Mock slow response
        async def slow_session():
            await asyncio.sleep(0.1)  # Simulate delay
            return mock_session
        
        mock_session = AsyncMock()
        mock_connection_manager.async_session.return_value.__aenter__.side_effect = slow_session
        
        # Mock query results
        mock_result = AsyncMock()
        mock_result.single.return_value = {"alive": 1}
        mock_session.run.return_value = mock_result
        
        # Patch time to simulate high latency
        with patch('graphiti_health.time.time') as mock_time:
            mock_time.side_effect = [0, 6]  # 6 second latency
            result = await checker.check_health()
        
        assert result.status == HealthStatus.DEGRADED
        assert "High latency" in result.details.get("warning", "")


class TestGraphitiHealthChecker:
    """Test Graphiti health checker."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create mock connection manager."""
        manager = Mock()
        manager.state = ConnectionState.CONNECTED
        manager.metrics = Mock(
            success_rate=0.95,
            total_connections=20,
            failed_connections=1
        )
        return manager
    
    @pytest.fixture
    def checker(self, mock_connection_manager):
        """Create Graphiti health checker with mocks."""
        with patch('graphiti_health.get_graphiti_connection', return_value=mock_connection_manager):
            return GraphitiHealthChecker()
    
    @pytest.mark.asyncio
    async def test_healthy_check(self, checker, mock_connection_manager):
        """Test healthy Graphiti check."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"status": "ok"}'
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = Mock()
        
        mock_connection_manager.request_async = AsyncMock(return_value=mock_response)
        
        result = await checker.check_health()
        
        assert result.component == ComponentType.GRAPHITI
        assert result.status == HealthStatus.HEALTHY
        assert result.details["status_code"] == 200
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_unhealthy_check_server_error(self, checker, mock_connection_manager):
        """Test unhealthy check with server error."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service unavailable",
            request=Mock(),
            response=mock_response
        )
        
        mock_connection_manager.request_async = AsyncMock(return_value=mock_response)
        
        result = await checker.check_health()
        
        assert result.component == ComponentType.GRAPHITI
        assert result.status == HealthStatus.UNHEALTHY
        assert "HTTP error: 503" in result.error
    
    @pytest.mark.asyncio
    async def test_degraded_check_high_latency(self, checker, mock_connection_manager):
        """Test degraded status for high latency."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        
        mock_connection_manager.request_async = AsyncMock(return_value=mock_response)
        
        # Patch time to simulate high latency
        with patch('graphiti_health.time.time') as mock_time:
            mock_time.side_effect = [0, 2.5]  # 2.5 second latency
            result = await checker.check_health()
        
        assert result.status == HealthStatus.DEGRADED
        assert "High latency" in result.details.get("warning", "")


class TestSystemHealthChecker:
    """Test system health checker."""
    
    @pytest.fixture
    def checker(self):
        """Create system health checker."""
        return SystemHealthChecker()
    
    @pytest.mark.asyncio
    async def test_all_healthy(self, checker):
        """Test when all components are healthy."""
        # Mock component checkers
        neo4j_result = HealthCheckResult(
            component=ComponentType.NEO4J,
            status=HealthStatus.HEALTHY
        )
        graphiti_result = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.HEALTHY
        )
        
        checker.neo4j_checker.perform_check = AsyncMock(return_value=neo4j_result)
        checker.graphiti_checker.perform_check = AsyncMock(return_value=graphiti_result)
        
        result = await checker.check_health()
        
        assert result.component == ComponentType.SYSTEM
        assert result.status == HealthStatus.HEALTHY
        assert "neo4j" in result.details["components"]
        assert "graphiti" in result.details["components"]
    
    @pytest.mark.asyncio
    async def test_one_unhealthy(self, checker):
        """Test when one component is unhealthy."""
        # Mock component checkers
        neo4j_result = HealthCheckResult(
            component=ComponentType.NEO4J,
            status=HealthStatus.HEALTHY
        )
        graphiti_result = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.UNHEALTHY,
            error="Connection failed"
        )
        
        checker.neo4j_checker.perform_check = AsyncMock(return_value=neo4j_result)
        checker.graphiti_checker.perform_check = AsyncMock(return_value=graphiti_result)
        
        result = await checker.check_health()
        
        assert result.component == ComponentType.SYSTEM
        assert result.status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_degraded_components(self, checker):
        """Test when components are degraded."""
        # Mock component checkers
        neo4j_result = HealthCheckResult(
            component=ComponentType.NEO4J,
            status=HealthStatus.DEGRADED
        )
        graphiti_result = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.HEALTHY
        )
        
        checker.neo4j_checker.perform_check = AsyncMock(return_value=neo4j_result)
        checker.graphiti_checker.perform_check = AsyncMock(return_value=graphiti_result)
        
        result = await checker.check_health()
        
        assert result.component == ComponentType.SYSTEM
        assert result.status == HealthStatus.DEGRADED


class TestHealthMonitor:
    """Test health monitor."""
    
    @pytest.fixture
    def mock_checker(self):
        """Create mock health checker."""
        checker = Mock()
        checker.component = ComponentType.NEO4J
        checker.metrics = Mock(consecutive_failures=0)
        checker.perform_check = AsyncMock()
        return checker
    
    @pytest.fixture
    def monitor(self, mock_checker):
        """Create health monitor."""
        return HealthMonitor(mock_checker, interval_seconds=1, failure_threshold=3)
    
    @pytest.mark.asyncio
    async def test_monitor_start_stop(self, monitor):
        """Test starting and stopping monitor."""
        await monitor.start()
        assert monitor._running is True
        assert monitor._task is not None
        
        await monitor.stop()
        assert monitor._running is False
    
    @pytest.mark.asyncio
    async def test_monitor_alerts(self, monitor, mock_checker):
        """Test monitor alerts on failures."""
        # Mock consecutive failures
        mock_checker.metrics.consecutive_failures = 3
        unhealthy_result = HealthCheckResult(
            component=ComponentType.NEO4J,
            status=HealthStatus.UNHEALTHY,
            error="Connection failed"
        )
        mock_checker.perform_check.return_value = unhealthy_result
        
        # Add alert callback
        alert_called = False
        alert_result = None
        alert_failures = None
        
        def alert_callback(result, failures):
            nonlocal alert_called, alert_result, alert_failures
            alert_called = True
            alert_result = result
            alert_failures = failures
        
        monitor.add_alert_callback(alert_callback)
        
        # Start monitor and wait briefly
        await monitor.start()
        await asyncio.sleep(0.1)
        await monitor.stop()
        
        assert alert_called is True
        assert alert_result.status == HealthStatus.UNHEALTHY
        assert alert_failures == 3


class TestHealthService:
    """Test health service."""
    
    @pytest.fixture
    def service(self):
        """Create health service."""
        return HealthService()
    
    @pytest.mark.asyncio
    async def test_service_start_stop(self, service):
        """Test starting and stopping service."""
        # Mock monitors
        for component in [ComponentType.NEO4J, ComponentType.GRAPHITI, ComponentType.SYSTEM]:
            service.monitors[component] = Mock()
            service.monitors[component].start = AsyncMock()
            service.monitors[component].stop = AsyncMock()
        
        await service.start()
        assert service._started is True
        
        # Verify all monitors started
        for monitor in service.monitors.values():
            monitor.start.assert_called_once()
        
        await service.stop()
        assert service._started is False
        
        # Verify all monitors stopped
        for monitor in service.monitors.values():
            monitor.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_health_status(self, service):
        """Test getting health status."""
        # Mock system checker
        mock_result = HealthCheckResult(
            component=ComponentType.SYSTEM,
            status=HealthStatus.HEALTHY,
            details={
                "components": {
                    "neo4j": {"status": "healthy"},
                    "graphiti": {"status": "healthy"}
                },
                "metrics": {}
            }
        )
        
        service.system_checker.perform_check = AsyncMock(return_value=mock_result)
        service.system_checker.get_history = Mock(return_value=[])
        service.system_checker.neo4j_checker.get_history = Mock(return_value=[])
        service.system_checker.graphiti_checker.get_history = Mock(return_value=[])
        
        status = await service.get_health_status()
        
        assert status["status"] == "healthy"
        assert "timestamp" in status
        assert "components" in status
        assert "metrics" in status
        assert "history" in status


class TestGlobalFunctions:
    """Test global functions."""
    
    def test_get_health_service(self):
        """Test getting global health service."""
        service1 = get_health_service()
        service2 = get_health_service()
        
        assert service1 is service2  # Singleton
        assert isinstance(service1, HealthService)
    
    @pytest.mark.asyncio
    async def test_check_system_health(self):
        """Test quick system health check."""
        with patch('graphiti_health.get_health_service') as mock_get_service:
            mock_service = Mock()
            mock_service.get_health_status = AsyncMock(return_value={"status": "healthy"})
            mock_get_service.return_value = mock_service
            
            result = await check_system_health()
            
            assert result["status"] == "healthy"
            mock_service.get_health_status.assert_called_once()