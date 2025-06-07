"""
Tests for the circuit breaker module.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import time

from graphiti_circuit_breaker import (
    CircuitState,
    CircuitStats,
    CircuitBreaker,
    CircuitOpenError,
    AdaptiveCircuitBreaker,
    circuit_breaker,
    CircuitBreakerManager,
    get_circuit_breaker,
)
from graphiti_health import HealthStatus, HealthCheckResult, ComponentType


class TestCircuitStats:
    """Test CircuitStats class."""
    
    def test_initial_stats(self):
        """Test initial circuit statistics."""
        stats = CircuitStats()
        
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.consecutive_failures == 0
        assert stats.failure_rate == 0.0
        assert stats.success_rate == 0.0
    
    def test_record_success(self):
        """Test recording successful calls."""
        stats = CircuitStats()
        
        stats.record_success()
        
        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.failed_calls == 0
        assert stats.consecutive_successes == 1
        assert stats.consecutive_failures == 0
        assert stats.success_rate == 1.0
    
    def test_record_failure(self):
        """Test recording failed calls."""
        stats = CircuitStats()
        
        stats.record_failure()
        
        assert stats.total_calls == 1
        assert stats.successful_calls == 0
        assert stats.failed_calls == 1
        assert stats.consecutive_failures == 1
        assert stats.consecutive_successes == 0
        assert stats.failure_rate == 1.0
    
    def test_consecutive_tracking(self):
        """Test consecutive success/failure tracking."""
        stats = CircuitStats()
        
        # Record failures
        stats.record_failure()
        stats.record_failure()
        assert stats.consecutive_failures == 2
        assert stats.consecutive_successes == 0
        
        # Record success - resets consecutive failures
        stats.record_success()
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 1


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    @pytest.fixture
    def breaker(self):
        """Create circuit breaker with test configuration."""
        return CircuitBreaker(
            failure_threshold=2,
            success_threshold=1,
            timeout=1.0,  # 1 second for faster tests
            half_open_max_calls=2
        )
    
    @pytest.mark.asyncio
    async def test_initial_state(self, breaker):
        """Test initial circuit state is closed."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.total_calls == 0
    
    @pytest.mark.asyncio
    async def test_successful_calls(self, breaker):
        """Test circuit remains closed on successful calls."""
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.successful_calls == 1
        assert breaker.stats.failed_calls == 0
    
    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self, breaker):
        """Test circuit opens after failure threshold."""
        async def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.consecutive_failures == 1
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.consecutive_failures == 2
    
    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, breaker):
        """Test open circuit rejects calls immediately."""
        # Force circuit open
        breaker._transition_to(CircuitState.OPEN)
        
        async def test_func():
            return "should not execute"
        
        with pytest.raises(CircuitOpenError):
            await breaker.call(test_func)
        
        # Function should not have been called
        assert breaker.stats.total_calls == 0
    
    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, breaker):
        """Test circuit transitions to half-open after timeout."""
        # Open circuit
        breaker._transition_to(CircuitState.OPEN)
        breaker._stats.last_failure_time = datetime.now() - timedelta(seconds=2)
        
        async def test_func():
            return "success"
        
        # Should transition to half-open and allow call
        result = await breaker.call(test_func)
        
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_to_closed(self, breaker):
        """Test successful calls in half-open close the circuit."""
        breaker._transition_to(CircuitState.HALF_OPEN)
        
        async def success_func():
            return "success"
        
        # Need success_threshold (1) successful calls
        await breaker.call(success_func)
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.consecutive_successes == 1
    
    @pytest.mark.asyncio
    async def test_half_open_to_open(self, breaker):
        """Test failure in half-open reopens the circuit."""
        breaker._transition_to(CircuitState.HALF_OPEN)
        
        async def failing_func():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_call_limit(self, breaker):
        """Test half-open state respects max calls limit."""
        breaker._transition_to(CircuitState.HALF_OPEN)
        
        async def test_func():
            return "success"
        
        # Make max allowed calls
        for _ in range(breaker.half_open_max_calls):
            await breaker.call(test_func)
        
        # Next call should be rejected
        with pytest.raises(CircuitOpenError):
            await breaker.call(test_func)
    
    def test_state_callbacks(self, breaker):
        """Test state change callbacks."""
        callback_calls = []
        
        def callback(old_state, new_state):
            callback_calls.append((old_state, new_state))
        
        breaker.add_state_callback(callback)
        
        # Transition to open
        breaker._transition_to(CircuitState.OPEN)
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == (CircuitState.CLOSED, CircuitState.OPEN)
    
    def test_reset(self, breaker):
        """Test manual circuit reset."""
        # Open circuit and add some stats
        breaker._transition_to(CircuitState.OPEN)
        breaker._stats.total_calls = 10
        breaker._stats.failed_calls = 5
        
        breaker.reset()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.total_calls == 0
        assert breaker.stats.failed_calls == 0
    
    def test_get_status(self, breaker):
        """Test getting circuit status."""
        breaker._stats.total_calls = 10
        breaker._stats.successful_calls = 7
        breaker._stats.failed_calls = 3
        
        status = breaker.get_status()
        
        assert status["state"] == "closed"
        assert status["stats"]["total_calls"] == 10
        assert status["stats"]["failure_rate"] == "30.0%"
        assert status["config"]["failure_threshold"] == 2


class TestAdaptiveCircuitBreaker:
    """Test AdaptiveCircuitBreaker class."""
    
    @pytest.fixture
    def mock_health_checker(self):
        """Create mock health checker."""
        checker = Mock()
        checker.perform_check = AsyncMock()
        return checker
    
    @pytest.fixture
    def adaptive_breaker(self, mock_health_checker):
        """Create adaptive circuit breaker."""
        return AdaptiveCircuitBreaker(
            health_checker=mock_health_checker,
            failure_threshold=5,
            timeout=60.0
        )
    
    @pytest.mark.asyncio
    async def test_adapts_on_healthy(self, adaptive_breaker, mock_health_checker):
        """Test adaptation when service is healthy."""
        mock_health_checker.perform_check.return_value = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.HEALTHY
        )
        
        # Force adaptation
        adaptive_breaker._last_health_check = datetime.now() - timedelta(minutes=5)
        
        async def test_func():
            return "success"
        
        await adaptive_breaker.call(test_func)
        
        # Should restore normal thresholds
        assert adaptive_breaker.failure_threshold == 5
        assert adaptive_breaker.timeout == 60.0
    
    @pytest.mark.asyncio
    async def test_adapts_on_degraded(self, adaptive_breaker, mock_health_checker):
        """Test adaptation when service is degraded."""
        mock_health_checker.perform_check.return_value = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.DEGRADED
        )
        
        # Force adaptation
        adaptive_breaker._last_health_check = datetime.now() - timedelta(minutes=5)
        
        async def test_func():
            return "success"
        
        await adaptive_breaker.call(test_func)
        
        # Should use conservative thresholds
        assert adaptive_breaker.failure_threshold == 3
        assert adaptive_breaker.timeout == 90.0
    
    @pytest.mark.asyncio
    async def test_adapts_on_unhealthy(self, adaptive_breaker, mock_health_checker):
        """Test adaptation when service is unhealthy."""
        mock_health_checker.perform_check.return_value = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.UNHEALTHY
        )
        
        # Force adaptation
        adaptive_breaker._last_health_check = datetime.now() - timedelta(minutes=5)
        
        async def test_func():
            return "success"
        
        await adaptive_breaker.call(test_func)
        
        # Should use very conservative thresholds
        assert adaptive_breaker.failure_threshold == 1
        assert adaptive_breaker.timeout == 120.0


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Test basic decorator functionality."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        @circuit_breaker(breaker)
        async def test_func(value):
            return value * 2
        
        result = await test_func(5)
        
        assert result == 10
        assert breaker.stats.successful_calls == 1
    
    @pytest.mark.asyncio
    async def test_decorator_with_failures(self):
        """Test decorator handles failures."""
        breaker = CircuitBreaker(failure_threshold=1)
        
        @circuit_breaker(breaker)
        async def failing_func():
            raise ValueError("Test error")
        
        # First call fails and opens circuit
        with pytest.raises(ValueError):
            await failing_func()
        
        assert breaker.state == CircuitState.OPEN
        
        # Second call rejected by circuit
        with pytest.raises(CircuitOpenError):
            await failing_func()
    
    @pytest.mark.asyncio
    async def test_decorator_default_breaker(self):
        """Test decorator creates default breaker."""
        @circuit_breaker()
        async def test_func():
            return "success"
        
        result = await test_func()
        
        assert result == "success"
        assert hasattr(test_func, 'circuit_breaker')
        assert isinstance(test_func.circuit_breaker, CircuitBreaker)


class TestCircuitBreakerManager:
    """Test CircuitBreakerManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create circuit breaker manager."""
        return CircuitBreakerManager()
    
    def test_get_breaker(self, manager):
        """Test getting or creating breakers."""
        # First call creates breaker
        breaker1 = manager.get_breaker("test")
        assert isinstance(breaker1, CircuitBreaker)
        
        # Second call returns same breaker
        breaker2 = manager.get_breaker("test")
        assert breaker1 is breaker2
    
    def test_get_breaker_with_config(self, manager):
        """Test creating breaker with configuration."""
        breaker = manager.get_breaker(
            "custom",
            failure_threshold=10,
            timeout=30.0
        )
        
        assert breaker.failure_threshold == 10
        assert breaker.timeout == 30.0
    
    def test_get_all_status(self, manager):
        """Test getting status of all breakers."""
        # Create multiple breakers
        breaker1 = manager.get_breaker("service1")
        breaker2 = manager.get_breaker("service2")
        
        # Add some stats
        breaker1._stats.total_calls = 5
        breaker2._stats.total_calls = 10
        
        all_status = manager.get_all_status()
        
        assert "service1" in all_status
        assert "service2" in all_status
        assert all_status["service1"]["stats"]["total_calls"] == 5
        assert all_status["service2"]["stats"]["total_calls"] == 10
    
    def test_reset_all(self, manager):
        """Test resetting all breakers."""
        # Create breakers and open them
        breaker1 = manager.get_breaker("service1")
        breaker2 = manager.get_breaker("service2")
        
        breaker1._transition_to(CircuitState.OPEN)
        breaker2._transition_to(CircuitState.OPEN)
        
        manager.reset_all()
        
        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED
    
    def test_remove_breaker(self, manager):
        """Test removing breaker."""
        breaker = manager.get_breaker("test")
        
        manager.remove_breaker("test")
        
        # Getting again creates new breaker
        new_breaker = manager.get_breaker("test")
        assert new_breaker is not breaker


class TestGlobalFunctions:
    """Test global functions."""
    
    def test_get_circuit_breaker(self):
        """Test getting global circuit breaker."""
        breaker1 = get_circuit_breaker("test")
        breaker2 = get_circuit_breaker("test")
        
        assert breaker1 is breaker2  # Same instance
        assert isinstance(breaker1, CircuitBreaker)
    
    def test_get_circuit_breaker_different_names(self):
        """Test getting different named breakers."""
        breaker1 = get_circuit_breaker("service1")
        breaker2 = get_circuit_breaker("service2")
        
        assert breaker1 is not breaker2