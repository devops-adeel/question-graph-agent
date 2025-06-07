"""
Circuit breaker implementation for Graphiti integration.

This module provides a circuit breaker pattern to prevent cascading failures
and reduce load on unhealthy services. It works in conjunction with the
fallback system to provide resilient operation.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import threading

from graphiti_health import HealthStatus, GraphitiHealthChecker
from graphiti_fallback import FallbackManager, get_fallback_manager


logger = logging.getLogger(__name__)


T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    time_in_current_state: float = 0.0
    
    def record_success(self):
        """Record successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
    
    def record_failure(self):
        """Record failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now()
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls


class CircuitBreaker:
    """Circuit breaker for service calls."""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 success_threshold: int = 2,
                 timeout: float = 60.0,
                 half_open_max_calls: int = 3,
                 fallback_manager: Optional[FallbackManager] = None):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Consecutive failures before opening
            success_threshold: Consecutive successes to close from half-open
            timeout: Seconds before attempting half-open from open
            half_open_max_calls: Max calls allowed in half-open state
            fallback_manager: Fallback manager for integration
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls
        self.fallback_manager = fallback_manager or get_fallback_manager()
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._state_callbacks: List[Callable[[CircuitState, CircuitState], None]] = []
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats
    
    def add_state_callback(self, callback: Callable[[CircuitState, CircuitState], None]):
        """Add callback for state changes.
        
        Args:
            callback: Function called with (old_state, new_state)
        """
        self._state_callbacks.append(callback)
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self._state
        if old_state == new_state:
            return
        
        with self._lock:
            self._state = new_state
            self._stats.last_state_change = datetime.now()
            self._half_open_calls = 0
            
            # Update time in state
            if self._stats.last_state_change:
                self._stats.time_in_current_state = 0.0
            
            logger.info(f"Circuit breaker state: {old_state.value} -> {new_state.value}")
            
            # Notify callbacks
            for callback in self._state_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state callback: {e}")
            
            # Integrate with fallback manager
            if new_state == CircuitState.OPEN:
                # Activate fallback when circuit opens
                asyncio.create_task(self.fallback_manager.check_and_activate())
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from open state."""
        if self._state != CircuitState.OPEN:
            return False
        
        if not self._stats.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self._stats.last_failure_time).total_seconds()
        return time_since_failure >= self.timeout
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker.
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        # Check if we should attempt reset
        if self._should_attempt_reset():
            self._transition_to(CircuitState.HALF_OPEN)
        
        # Check circuit state
        if self._state == CircuitState.OPEN:
            raise CircuitOpenError("Circuit breaker is open")
        
        if self._state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError("Half-open call limit reached")
                self._half_open_calls += 1
        
        # Execute function
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
        finally:
            duration = time.time() - start_time
            logger.debug(f"Circuit breaker call took {duration:.2f}s")
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self._stats.record_success()
            
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            
            elif self._state == CircuitState.CLOSED:
                # Reset consecutive failures in closed state
                self._stats.consecutive_failures = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._stats.record_failure()
            
            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
            
            elif self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open returns to open
                self._transition_to(CircuitState.OPEN)
    
    def reset(self):
        """Manually reset circuit breaker."""
        self._transition_to(CircuitState.CLOSED)
        self._stats = CircuitStats()
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            status = {
                "state": self._state.value,
                "stats": {
                    "total_calls": self._stats.total_calls,
                    "successful_calls": self._stats.successful_calls,
                    "failed_calls": self._stats.failed_calls,
                    "failure_rate": f"{self._stats.failure_rate:.1%}",
                    "consecutive_failures": self._stats.consecutive_failures,
                    "consecutive_successes": self._stats.consecutive_successes,
                },
                "config": {
                    "failure_threshold": self.failure_threshold,
                    "success_threshold": self.success_threshold,
                    "timeout": self.timeout,
                    "half_open_max_calls": self.half_open_max_calls,
                }
            }
            
            if self._stats.last_state_change:
                time_in_state = (datetime.now() - self._stats.last_state_change).total_seconds()
                status["time_in_current_state"] = f"{time_in_state:.1f}s"
            
            if self._state == CircuitState.OPEN and self._stats.last_failure_time:
                time_until_retry = max(0, self.timeout - (
                    datetime.now() - self._stats.last_failure_time
                ).total_seconds())
                status["time_until_retry"] = f"{time_until_retry:.1f}s"
            
            return status


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class AdaptiveCircuitBreaker(CircuitBreaker):
    """Circuit breaker with adaptive thresholds based on system health."""
    
    def __init__(self, 
                 health_checker: Optional[GraphitiHealthChecker] = None,
                 **kwargs):
        """Initialize adaptive circuit breaker."""
        super().__init__(**kwargs)
        self.health_checker = health_checker or GraphitiHealthChecker()
        self._last_health_check = datetime.now()
        self._health_check_interval = 30  # seconds
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute with adaptive behavior."""
        # Periodically check health and adapt
        if (datetime.now() - self._last_health_check).total_seconds() > self._health_check_interval:
            await self._adapt_thresholds()
        
        return await super().call(func, *args, **kwargs)
    
    async def _adapt_thresholds(self):
        """Adapt thresholds based on system health."""
        try:
            result = await self.health_checker.perform_check()
            self._last_health_check = datetime.now()
            
            if result.status == HealthStatus.HEALTHY:
                # Restore normal thresholds
                self.failure_threshold = 5
                self.timeout = 60.0
                
            elif result.status == HealthStatus.DEGRADED:
                # Be more conservative
                self.failure_threshold = 3
                self.timeout = 90.0
                
            elif result.status == HealthStatus.UNHEALTHY:
                # Very conservative
                self.failure_threshold = 1
                self.timeout = 120.0
                
        except Exception as e:
            logger.error(f"Failed to check health for adaptation: {e}")


def circuit_breaker(breaker: Optional[CircuitBreaker] = None):
    """Decorator to add circuit breaker to async functions.
    
    Args:
        breaker: Circuit breaker instance (creates default if None)
    """
    if breaker is None:
        breaker = CircuitBreaker()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        # Attach breaker for access
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def get_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker.
        
        Args:
            name: Breaker name
            **kwargs: Arguments for new breaker
            
        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(**kwargs)
            return self._breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all breakers."""
        with self._lock:
            return {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def remove_breaker(self, name: str):
        """Remove circuit breaker."""
        with self._lock:
            self._breakers.pop(name, None)


# Global circuit breaker manager
_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker(name: str = "default", **kwargs) -> CircuitBreaker:
    """Get or create a circuit breaker.
    
    Args:
        name: Breaker name
        **kwargs: Arguments for new breaker
        
    Returns:
        Circuit breaker instance
    """
    return _breaker_manager.get_breaker(name, **kwargs)