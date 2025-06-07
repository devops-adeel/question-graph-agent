"""
Tests for the Graphiti connection manager module.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from contextlib import contextmanager

import httpx
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from graphiti_connection import (
    ConnectionState,
    ConnectionMetrics,
    Neo4jConnectionManager,
    GraphitiConnectionManager,
    ConnectionPool,
    get_neo4j_connection,
    get_graphiti_connection,
    close_all_connections,
)
from graphiti_config import RuntimeConfig, Neo4jConfig, GraphitiConfig


class TestConnectionMetrics:
    """Test connection metrics tracking."""
    
    def test_record_success(self):
        """Test recording successful connection."""
        metrics = ConnectionMetrics()
        
        metrics.record_success(1.5)
        
        assert metrics.total_connections == 1
        assert metrics.successful_connections == 1
        assert metrics.failed_connections == 0
        assert metrics.connection_duration == 1.5
        assert metrics.last_connection_time is not None
    
    def test_record_failure(self):
        """Test recording failed connection."""
        metrics = ConnectionMetrics()
        
        metrics.record_failure("Connection refused")
        
        assert metrics.total_connections == 1
        assert metrics.successful_connections == 0
        assert metrics.failed_connections == 1
        assert metrics.last_error == "Connection refused"
        assert metrics.last_failure_time is not None
    
    def test_success_rate(self):
        """Test success rate calculation."""
        metrics = ConnectionMetrics()
        
        # No connections
        assert metrics.success_rate == 0.0
        
        # Mixed results
        metrics.record_success(1.0)
        metrics.record_success(1.0)
        metrics.record_failure("Error")
        
        assert metrics.success_rate == pytest.approx(2/3)


class TestNeo4jConnectionManager:
    """Test Neo4j connection manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=RuntimeConfig)
        config.neo4j = Mock(spec=Neo4jConfig)
        config.neo4j.uri = "bolt://localhost:7687"
        config.neo4j.password.get_secret_value.return_value = "password"
        config.validate_neo4j_connection.return_value = True
        config.get_neo4j_config.return_value = {
            "uri": "bolt://localhost:7687",
            "auth": ("neo4j", "password"),
        }
        return config
    
    def test_initialization(self, mock_config):
        """Test connection manager initialization."""
        manager = Neo4jConnectionManager(mock_config)
        
        assert manager.state == ConnectionState.DISCONNECTED
        assert manager.metrics.total_connections == 0
        assert manager._driver is None
    
    @patch('graphiti_connection.GraphDatabase')
    def test_connect_success(self, mock_graph_db, mock_config):
        """Test successful connection."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jConnectionManager(mock_config)
        driver = manager.connect()
        
        assert driver == mock_driver
        assert manager.state == ConnectionState.CONNECTED
        assert manager.metrics.successful_connections == 1
        mock_driver.verify_connectivity.assert_called_once()
    
    @patch('graphiti_connection.GraphDatabase')
    def test_connect_retry_on_service_unavailable(self, mock_graph_db, mock_config):
        """Test retry logic on service unavailable."""
        mock_driver = Mock()
        mock_driver.verify_connectivity.side_effect = [
            ServiceUnavailable("Service unavailable"),
            None  # Success on second attempt
        ]
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jConnectionManager(mock_config)
        driver = manager.connect()
        
        assert driver == mock_driver
        assert mock_driver.verify_connectivity.call_count == 2
    
    @patch('graphiti_connection.GraphDatabase')
    def test_session_context_manager(self, mock_graph_db, mock_config):
        """Test session context manager."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jConnectionManager(mock_config)
        
        with manager.session() as session:
            assert session == mock_session
        
        mock_session.close.assert_called_once()
    
    @patch('graphiti_connection.GraphDatabase')
    def test_execute_query(self, mock_graph_db, mock_config):
        """Test query execution with retry."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]))
        
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jConnectionManager(mock_config)
        results = manager.execute_query("MATCH (n) RETURN n")
        
        assert len(results) == 2
        assert results[0]["name"] == "Alice"
        mock_session.close.assert_called_once()
    
    def test_state_callbacks(self, mock_config):
        """Test state change callbacks."""
        states = []
        
        manager = Neo4jConnectionManager(mock_config)
        manager.add_state_callback(lambda state: states.append(state))
        
        manager._set_state(ConnectionState.CONNECTING)
        manager._set_state(ConnectionState.CONNECTED)
        
        assert states == [ConnectionState.CONNECTING, ConnectionState.CONNECTED]
    
    @patch('graphiti_connection.AsyncGraphDatabase')
    @pytest.mark.asyncio
    async def test_async_connect(self, mock_async_graph_db, mock_config):
        """Test async connection."""
        mock_driver = AsyncMock()
        mock_async_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jConnectionManager(mock_config)
        driver = await manager.connect_async()
        
        assert driver == mock_driver
        assert manager.state == ConnectionState.CONNECTED
        mock_driver.verify_connectivity.assert_awaited_once()
    
    @patch('graphiti_connection.AsyncGraphDatabase')
    @pytest.mark.asyncio
    async def test_async_session_context_manager(self, mock_async_graph_db, mock_config):
        """Test async session context manager."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value = mock_session
        mock_async_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jConnectionManager(mock_config)
        
        async with manager.async_session() as session:
            assert session == mock_session
        
        mock_session.close.assert_awaited_once()


class TestGraphitiConnectionManager:
    """Test Graphiti connection manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=RuntimeConfig)
        config.graphiti = Mock(spec=GraphitiConfig)
        config.graphiti.endpoint = "http://localhost:8000"
        config.graphiti.request_timeout = 30
        config.graphiti.max_retries = 3
        config.graphiti.retry_delay = 1.0
        config.graphiti.retry_backoff = 2.0
        config.validate_graphiti_connection.return_value = True
        config.get_graphiti_headers.return_value = {
            "Content-Type": "application/json"
        }
        return config
    
    def test_initialization(self, mock_config):
        """Test connection manager initialization."""
        manager = GraphitiConnectionManager(mock_config)
        
        assert manager.state == ConnectionState.DISCONNECTED
        assert manager.metrics.total_connections == 0
        assert manager._client is None
    
    @patch('httpx.Client')
    def test_connect_success(self, mock_client_class, mock_config):
        """Test successful connection."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        manager = GraphitiConnectionManager(mock_config)
        client = manager.connect()
        
        assert client == mock_client
        assert manager.state == ConnectionState.CONNECTED
        assert manager.metrics.successful_connections == 1
        mock_client.get.assert_called_with("/health", follow_redirects=True)
    
    @patch('httpx.Client')
    def test_connect_retry_on_error(self, mock_client_class, mock_config):
        """Test retry logic on connection error."""
        mock_client = Mock()
        mock_client.get.side_effect = [
            httpx.ConnectError("Connection failed"),
            Mock(raise_for_status=Mock())  # Success on second attempt
        ]
        mock_client_class.return_value = mock_client
        
        manager = GraphitiConnectionManager(mock_config)
        client = manager.connect()
        
        assert client == mock_client
        assert mock_client.get.call_count == 2
    
    @patch('httpx.Client')
    def test_request_with_retry(self, mock_client_class, mock_config):
        """Test HTTP request with retry."""
        mock_client = Mock()
        mock_response = Mock()
        mock_client.request.side_effect = [
            httpx.TimeoutException("Request timeout"),
            mock_response  # Success on second attempt
        ]
        mock_client.get.return_value = Mock(raise_for_status=Mock())
        mock_client_class.return_value = mock_client
        
        manager = GraphitiConnectionManager(mock_config)
        response = manager.request("POST", "/api/entity")
        
        assert response == mock_response
        assert mock_client.request.call_count == 2
        assert manager.metrics.total_retries == 1
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_connect(self, mock_async_client_class, mock_config):
        """Test async connection."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_async_client_class.return_value = mock_client
        
        manager = GraphitiConnectionManager(mock_config)
        client = await manager.connect_async()
        
        assert client == mock_client
        assert manager.state == ConnectionState.CONNECTED
        mock_client.get.assert_awaited_with("/health", follow_redirects=True)


class TestConnectionPool:
    """Test connection pool functionality."""
    
    @pytest.fixture
    def mock_manager_class(self):
        """Create mock connection manager class."""
        class MockManager:
            def __init__(self, config):
                self.config = config
                self.connected = False
            
            def connect(self):
                self.connected = True
            
            def close(self):
                self.connected = False
        
        return MockManager
    
    def test_pool_creation(self, mock_manager_class):
        """Test connection pool creation."""
        pool = ConnectionPool(mock_manager_class, size=3)
        
        assert pool.size == 3
        assert len(pool._pool) == 0
        assert len(pool._available) == 0
    
    def test_acquire_connection(self, mock_manager_class):
        """Test acquiring connection from pool."""
        pool = ConnectionPool(mock_manager_class, size=2)
        
        with pool.acquire() as conn1:
            assert conn1.connected
            assert len(pool._pool) == 1
            
            with pool.acquire() as conn2:
                assert conn2.connected
                assert len(pool._pool) == 2
                assert conn1 is not conn2
    
    def test_connection_reuse(self, mock_manager_class):
        """Test connection reuse in pool."""
        pool = ConnectionPool(mock_manager_class, size=1)
        
        with pool.acquire() as conn1:
            conn1.data = "test"
        
        with pool.acquire() as conn2:
            assert conn2 is conn1
            assert conn2.data == "test"
    
    def test_pool_exhaustion_timeout(self, mock_manager_class):
        """Test timeout when pool is exhausted."""
        pool = ConnectionPool(mock_manager_class, size=1)
        
        with pool.acquire() as conn1:
            # Try to acquire another connection with timeout
            with pytest.raises(TimeoutError):
                with pool.acquire(timeout=0.1):
                    pass
    
    def test_pool_close(self, mock_manager_class):
        """Test closing connection pool."""
        pool = ConnectionPool(mock_manager_class, size=2)
        
        with pool.acquire():
            pass
        
        pool.close()
        
        assert pool._closed
        assert len(pool._pool) == 0
        
        # Cannot acquire after close
        with pytest.raises(RuntimeError):
            with pool.acquire():
                pass


class TestGlobalConnectionManagers:
    """Test global connection manager functions."""
    
    @patch('graphiti_connection._neo4j_manager', None)
    @patch('graphiti_connection._graphiti_manager', None)
    def test_get_neo4j_connection(self):
        """Test getting global Neo4j connection."""
        manager1 = get_neo4j_connection()
        manager2 = get_neo4j_connection()
        
        assert manager1 is manager2  # Same instance
        assert isinstance(manager1, Neo4jConnectionManager)
    
    @patch('graphiti_connection._neo4j_manager', None)
    @patch('graphiti_connection._graphiti_manager', None)
    def test_get_graphiti_connection(self):
        """Test getting global Graphiti connection."""
        manager1 = get_graphiti_connection()
        manager2 = get_graphiti_connection()
        
        assert manager1 is manager2  # Same instance
        assert isinstance(manager1, GraphitiConnectionManager)
    
    @patch('graphiti_connection._neo4j_manager')
    @patch('graphiti_connection._graphiti_manager')
    def test_close_all_connections(self, mock_graphiti, mock_neo4j):
        """Test closing all global connections."""
        mock_neo4j_instance = Mock()
        mock_graphiti_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        mock_graphiti.return_value = mock_graphiti_instance
        
        close_all_connections()
        
        mock_neo4j_instance.close.assert_called_once()
        mock_graphiti_instance.close.assert_called_once()