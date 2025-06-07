"""
Connection manager for Neo4j and Graphiti with retry logic.

This module provides robust connection management with automatic retries,
connection pooling, and graceful error handling for both Neo4j database
and Graphiti service connections.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List, Callable, TypeVar, Union
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
from enum import Enum
import threading

from neo4j import GraphDatabase, Driver, Session, AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import (
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
    AuthError,
)
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)

from graphiti_config import get_config, RuntimeConfig


logger = logging.getLogger(__name__)


T = TypeVar('T')


class ConnectionState(str, Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class ConnectionMetrics:
    """Metrics for connection monitoring."""
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_retries: int = 0
    last_connection_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    last_error: Optional[str] = None
    connection_duration: float = 0.0
    
    def record_success(self, duration: float):
        """Record successful connection."""
        self.total_connections += 1
        self.successful_connections += 1
        self.last_connection_time = time.time()
        self.connection_duration = duration
    
    def record_failure(self, error: str):
        """Record failed connection."""
        self.total_connections += 1
        self.failed_connections += 1
        self.last_failure_time = time.time()
        self.last_error = error
    
    def record_retry(self):
        """Record retry attempt."""
        self.total_retries += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate connection success rate."""
        if self.total_connections == 0:
            return 0.0
        return self.successful_connections / self.total_connections


class Neo4jConnectionManager:
    """Manages Neo4j database connections with retry logic."""
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """Initialize connection manager."""
        self.config = config or get_config()
        self._driver: Optional[Driver] = None
        self._async_driver: Optional[AsyncDriver] = None
        self._state = ConnectionState.DISCONNECTED
        self._metrics = ConnectionMetrics()
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[ConnectionState], None]] = []
    
    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state
    
    @property
    def metrics(self) -> ConnectionMetrics:
        """Get connection metrics."""
        return self._metrics
    
    def add_state_callback(self, callback: Callable[[ConnectionState], None]):
        """Add callback for state changes."""
        self._callbacks.append(callback)
    
    def _set_state(self, state: ConnectionState):
        """Set connection state and notify callbacks."""
        self._state = state
        for callback in self._callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, ConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )
    def connect(self) -> Driver:
        """Connect to Neo4j with retry logic."""
        start_time = time.time()
        self._set_state(ConnectionState.CONNECTING)
        
        try:
            if not self.config.validate_neo4j_connection():
                raise ValueError("Invalid Neo4j configuration")
            
            with self._lock:
                if self._driver is None:
                    logger.info(f"Connecting to Neo4j at {self.config.neo4j.uri}")
                    self._driver = GraphDatabase.driver(
                        **self.config.get_neo4j_config()
                    )
                    
                    # Verify connectivity
                    self._driver.verify_connectivity()
                    
                    duration = time.time() - start_time
                    self._metrics.record_success(duration)
                    self._set_state(ConnectionState.CONNECTED)
                    logger.info(f"Connected to Neo4j in {duration:.2f}s")
                
                return self._driver
                
        except Exception as e:
            self._metrics.record_failure(str(e))
            self._set_state(ConnectionState.FAILED)
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def connect_async(self) -> AsyncDriver:
        """Connect to Neo4j asynchronously with retry logic."""
        start_time = time.time()
        self._set_state(ConnectionState.CONNECTING)
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((ServiceUnavailable, ConnectionError)),
        )
        async def _connect():
            if not self.config.validate_neo4j_connection():
                raise ValueError("Invalid Neo4j configuration")
            
            if self._async_driver is None:
                logger.info(f"Connecting to Neo4j async at {self.config.neo4j.uri}")
                self._async_driver = AsyncGraphDatabase.driver(
                    **self.config.get_neo4j_config()
                )
                
                # Verify connectivity
                await self._async_driver.verify_connectivity()
                
                duration = time.time() - start_time
                self._metrics.record_success(duration)
                self._set_state(ConnectionState.CONNECTED)
                logger.info(f"Connected to Neo4j async in {duration:.2f}s")
            
            return self._async_driver
        
        try:
            return await _connect()
        except Exception as e:
            self._metrics.record_failure(str(e))
            self._set_state(ConnectionState.FAILED)
            logger.error(f"Failed to connect to Neo4j async: {e}")
            raise
    
    @contextmanager
    def session(self, **kwargs) -> Session:
        """Create a Neo4j session with automatic retry."""
        driver = self.connect()
        session = None
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
            retry=retry_if_exception_type((SessionExpired, TransientError)),
        )
        def _create_session():
            nonlocal session
            session = driver.session(**kwargs)
            return session
        
        try:
            yield _create_session()
        finally:
            if session:
                session.close()
    
    @asynccontextmanager
    async def async_session(self, **kwargs) -> AsyncSession:
        """Create an async Neo4j session with automatic retry."""
        driver = await self.connect_async()
        session = None
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
            retry=retry_if_exception_type((SessionExpired, TransientError)),
        )
        async def _create_session():
            nonlocal session
            session = await driver.session(**kwargs)
            return session
        
        try:
            yield await _create_session()
        finally:
            if session:
                await session.close()
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None, **kwargs) -> List[Dict[str, Any]]:
        """Execute a query with automatic retry and connection management."""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
            retry=retry_if_exception_type((SessionExpired, TransientError, ServiceUnavailable)),
            before_sleep=lambda retry_state: self._metrics.record_retry(),
        )
        def _execute():
            with self.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        
        return _execute()
    
    async def execute_query_async(self, query: str, parameters: Optional[Dict[str, Any]] = None, **kwargs) -> List[Dict[str, Any]]:
        """Execute a query asynchronously with automatic retry."""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
            retry=retry_if_exception_type((SessionExpired, TransientError, ServiceUnavailable)),
            before_sleep=lambda retry_state: self._metrics.record_retry(),
        )
        async def _execute():
            async with self.async_session() as session:
                result = await session.run(query, parameters or {})
                records = []
                async for record in result:
                    records.append(dict(record))
                return records
        
        return await _execute()
    
    def close(self):
        """Close the connection."""
        with self._lock:
            if self._driver:
                self._driver.close()
                self._driver = None
            if self._async_driver:
                # Async driver close needs to be handled in async context
                logger.warning("Async driver should be closed in async context")
            self._set_state(ConnectionState.CLOSED)
            logger.info("Neo4j connection closed")
    
    async def close_async(self):
        """Close the connection asynchronously."""
        if self._driver:
            self._driver.close()
            self._driver = None
        if self._async_driver:
            await self._async_driver.close()
            self._async_driver = None
        self._set_state(ConnectionState.CLOSED)
        logger.info("Neo4j connections closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_async()


class GraphitiConnectionManager:
    """Manages Graphiti service connections with retry logic."""
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """Initialize connection manager."""
        self.config = config or get_config()
        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._state = ConnectionState.DISCONNECTED
        self._metrics = ConnectionMetrics()
    
    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state
    
    @property
    def metrics(self) -> ConnectionMetrics:
        """Get connection metrics."""
        return self._metrics
    
    def _create_client_kwargs(self) -> Dict[str, Any]:
        """Create client configuration."""
        return {
            "base_url": self.config.graphiti.endpoint,
            "headers": self.config.get_graphiti_headers(),
            "timeout": httpx.Timeout(
                self.config.graphiti.request_timeout,
                connect=10.0,
            ),
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def connect(self) -> httpx.Client:
        """Connect to Graphiti service with retry logic."""
        start_time = time.time()
        self._state = ConnectionState.CONNECTING
        
        try:
            if not self.config.validate_graphiti_connection():
                raise ValueError("Invalid Graphiti configuration")
            
            if self._client is None:
                logger.info(f"Connecting to Graphiti at {self.config.graphiti.endpoint}")
                self._client = httpx.Client(**self._create_client_kwargs())
                
                # Verify connectivity with health check
                response = self._client.get("/health", follow_redirects=True)
                response.raise_for_status()
                
                duration = time.time() - start_time
                self._metrics.record_success(duration)
                self._state = ConnectionState.CONNECTED
                logger.info(f"Connected to Graphiti in {duration:.2f}s")
            
            return self._client
            
        except Exception as e:
            self._metrics.record_failure(str(e))
            self._state = ConnectionState.FAILED
            logger.error(f"Failed to connect to Graphiti: {e}")
            raise
    
    async def connect_async(self) -> httpx.AsyncClient:
        """Connect to Graphiti service asynchronously."""
        start_time = time.time()
        self._state = ConnectionState.CONNECTING
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        )
        async def _connect():
            if not self.config.validate_graphiti_connection():
                raise ValueError("Invalid Graphiti configuration")
            
            if self._async_client is None:
                logger.info(f"Connecting to Graphiti async at {self.config.graphiti.endpoint}")
                self._async_client = httpx.AsyncClient(**self._create_client_kwargs())
                
                # Verify connectivity
                response = await self._async_client.get("/health", follow_redirects=True)
                response.raise_for_status()
                
                duration = time.time() - start_time
                self._metrics.record_success(duration)
                self._state = ConnectionState.CONNECTED
                logger.info(f"Connected to Graphiti async in {duration:.2f}s")
            
            return self._async_client
        
        try:
            return await _connect()
        except Exception as e:
            self._metrics.record_failure(str(e))
            self._state = ConnectionState.FAILED
            logger.error(f"Failed to connect to Graphiti async: {e}")
            raise
    
    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make HTTP request with retry logic."""
        client = self.connect()
        
        @retry(
            stop=stop_after_attempt(self.config.graphiti.max_retries),
            wait=wait_exponential(
                multiplier=self.config.graphiti.retry_delay,
                max=self.config.graphiti.retry_delay * self.config.graphiti.retry_backoff ** 3
            ),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
            before_sleep=lambda retry_state: self._metrics.record_retry(),
        )
        def _request():
            return client.request(method, path, **kwargs)
        
        return _request()
    
    async def request_async(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make async HTTP request with retry logic."""
        client = await self.connect_async()
        
        @retry(
            stop=stop_after_attempt(self.config.graphiti.max_retries),
            wait=wait_exponential(
                multiplier=self.config.graphiti.retry_delay,
                max=self.config.graphiti.retry_delay * self.config.graphiti.retry_backoff ** 3
            ),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
            before_sleep=lambda retry_state: self._metrics.record_retry(),
        )
        async def _request():
            return await client.request(method, path, **kwargs)
        
        return await _request()
    
    def close(self):
        """Close the connection."""
        if self._client:
            self._client.close()
            self._client = None
        self._state = ConnectionState.CLOSED
        logger.info("Graphiti connection closed")
    
    async def close_async(self):
        """Close connections asynchronously."""
        if self._client:
            self._client.close()
            self._client = None
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
        self._state = ConnectionState.CLOSED
        logger.info("Graphiti connections closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_async()


class ConnectionPool:
    """Connection pool for managing multiple connections."""
    
    def __init__(self, 
                 manager_class: type,
                 size: int = 5,
                 config: Optional[RuntimeConfig] = None):
        """Initialize connection pool."""
        self.manager_class = manager_class
        self.size = size
        self.config = config or get_config()
        self._pool: List[Any] = []
        self._available: List[Any] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
    
    def _create_connection(self):
        """Create a new connection."""
        manager = self.manager_class(self.config)
        if hasattr(manager, 'connect'):
            manager.connect()
        return manager
    
    @contextmanager
    def acquire(self, timeout: Optional[float] = None):
        """Acquire a connection from the pool."""
        start_time = time.time()
        
        with self._condition:
            while not self._available and not self._closed:
                if len(self._pool) < self.size:
                    # Create new connection
                    try:
                        conn = self._create_connection()
                        self._pool.append(conn)
                        self._available.append(conn)
                        break
                    except Exception as e:
                        logger.error(f"Failed to create connection: {e}")
                        raise
                else:
                    # Wait for available connection
                    if timeout:
                        remaining = timeout - (time.time() - start_time)
                        if remaining <= 0:
                            raise TimeoutError("Failed to acquire connection from pool")
                        if not self._condition.wait(timeout=remaining):
                            raise TimeoutError("Failed to acquire connection from pool")
                    else:
                        self._condition.wait()
            
            if self._closed:
                raise RuntimeError("Connection pool is closed")
            
            conn = self._available.pop()
        
        try:
            yield conn
        finally:
            with self._condition:
                if not self._closed:
                    self._available.append(conn)
                    self._condition.notify()
    
    def close(self):
        """Close all connections in the pool."""
        with self._lock:
            self._closed = True
            for conn in self._pool:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            self._pool.clear()
            self._available.clear()
            self._condition.notify_all()


# Global connection managers
_neo4j_manager: Optional[Neo4jConnectionManager] = None
_graphiti_manager: Optional[GraphitiConnectionManager] = None
_lock = threading.Lock()


def get_neo4j_connection() -> Neo4jConnectionManager:
    """Get or create global Neo4j connection manager."""
    global _neo4j_manager
    
    with _lock:
        if _neo4j_manager is None:
            _neo4j_manager = Neo4jConnectionManager()
        return _neo4j_manager


def get_graphiti_connection() -> GraphitiConnectionManager:
    """Get or create global Graphiti connection manager."""
    global _graphiti_manager
    
    with _lock:
        if _graphiti_manager is None:
            _graphiti_manager = GraphitiConnectionManager()
        return _graphiti_manager


def close_all_connections():
    """Close all global connections."""
    global _neo4j_manager, _graphiti_manager
    
    with _lock:
        if _neo4j_manager:
            _neo4j_manager.close()
            _neo4j_manager = None
        if _graphiti_manager:
            _graphiti_manager.close()
            _graphiti_manager = None


async def close_all_connections_async():
    """Close all global connections asynchronously."""
    global _neo4j_manager, _graphiti_manager
    
    if _neo4j_manager:
        await _neo4j_manager.close_async()
        _neo4j_manager = None
    if _graphiti_manager:
        await _graphiti_manager.close_async()
        _graphiti_manager = None