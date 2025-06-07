"""
Tests for the graceful fallback module.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import tempfile
import json
import sqlite3

from graphiti_fallback import (
    FallbackMode,
    FallbackState,
    QueuedOperation,
    LocalCache,
    LocalStorage,
    FallbackManager,
    FallbackDecorator,
    get_fallback_manager,
    with_fallback,
)
from graphiti_health import HealthStatus, HealthCheckResult, ComponentType
from graphiti_entities import QuestionEntity, DifficultyLevel
from graphiti_config import RuntimeConfig


class TestFallbackState:
    """Test FallbackState class."""
    
    def test_initial_state(self):
        """Test initial fallback state."""
        state = FallbackState()
        
        assert state.mode == FallbackMode.DISABLED
        assert state.is_active is False
        assert state.activated_at is None
        assert state.reason is None
        assert state.queued_operations == 0
    
    def test_activate_fallback(self):
        """Test activating fallback mode."""
        state = FallbackState()
        
        state.activate(FallbackMode.LOCAL_STORAGE, "Service unavailable")
        
        assert state.mode == FallbackMode.LOCAL_STORAGE
        assert state.is_active is True
        assert state.activated_at is not None
        assert state.reason == "Service unavailable"
    
    def test_deactivate_fallback(self):
        """Test deactivating fallback mode."""
        state = FallbackState()
        state.activate(FallbackMode.MEMORY_ONLY, "Test")
        
        state.deactivate()
        
        assert state.is_active is False
        assert state.reason is None


class TestLocalCache:
    """Test LocalCache class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create cache with temporary directory."""
        return LocalCache(temp_cache_dir)
    
    def test_cache_operations(self, cache):
        """Test basic cache operations."""
        # Test set and get
        cache.set("test_key", {"data": "test_value"})
        result = cache.get("test_key")
        
        assert result == {"data": "test_value"}
    
    def test_cache_persistence(self, cache, temp_cache_dir):
        """Test cache persistence to disk."""
        # Store with persistence
        cache.set("persist_key", {"data": "persistent"}, persist=True)
        
        # Check file exists
        cache_file = temp_cache_dir / "persist_key.pkl"
        assert cache_file.exists()
        
        # Create new cache instance and check data
        new_cache = LocalCache(temp_cache_dir)
        result = new_cache.get("persist_key")
        
        assert result == {"data": "persistent"}
    
    def test_cache_delete(self, cache):
        """Test cache deletion."""
        cache.set("delete_key", "value")
        assert cache.get("delete_key") == "value"
        
        cache.delete("delete_key")
        assert cache.get("delete_key") is None
    
    def test_cache_clear(self, cache, temp_cache_dir):
        """Test clearing cache."""
        # Add multiple items
        cache.set("key1", "value1", persist=True)
        cache.set("key2", "value2", persist=True)
        
        # Clear cache
        cache.clear()
        
        # Check memory cache is empty
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        
        # Check files are deleted
        assert len(list(temp_cache_dir.glob("*.pkl"))) == 0


class TestLocalStorage:
    """Test LocalStorage class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = Path(f.name)
        yield path
        path.unlink(missing_ok=True)
    
    @pytest.fixture
    def storage(self, temp_db_path):
        """Create storage with temporary database."""
        return LocalStorage(temp_db_path)
    
    def test_store_and_get_entity(self, storage):
        """Test storing and retrieving entity."""
        entity = QuestionEntity(
            id="q1",
            content="Test question?",
            difficulty=DifficultyLevel.EASY
        )
        
        storage.store_entity(entity)
        result = storage.get_entity("q1")
        
        assert result is not None
        assert result["id"] == "q1"
        assert result["content"] == "Test question?"
    
    def test_store_relationship(self, storage):
        """Test storing relationship."""
        from graphiti_relationships import AnsweredRelationship
        
        rel = AnsweredRelationship(
            id="r1",
            source_id="user1",
            target_id="question1",
            timestamp=datetime.now()
        )
        
        # Should not raise exception
        storage.store_relationship(rel)
    
    def test_queue_operations(self, storage):
        """Test queueing operations."""
        op = QueuedOperation(
            operation_type="create",
            entity_type="question",
            entity_id="q1",
            data={"content": "Test question?"}
        )
        
        storage.queue_operation(op)
        
        # Get queued operations
        operations = storage.get_queued_operations()
        
        assert len(operations) == 1
        assert operations[0].operation_type == "create"
        assert operations[0].entity_type == "question"
        assert operations[0].data["content"] == "Test question?"
    
    def test_database_initialization(self, temp_db_path):
        """Test database schema creation."""
        storage = LocalStorage(temp_db_path)
        
        # Check tables exist
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor}
        
        assert "entities" in tables
        assert "relationships" in tables
        assert "queued_operations" in tables
        
        conn.close()
        storage.close()


class TestFallbackManager:
    """Test FallbackManager class."""
    
    @pytest.fixture
    def mock_health_checker(self):
        """Create mock health checker."""
        checker = Mock()
        checker.perform_check = AsyncMock()
        return checker
    
    @pytest.fixture
    def manager(self, mock_health_checker):
        """Create fallback manager with mocks."""
        with patch('graphiti_fallback.GraphitiHealthChecker', return_value=mock_health_checker):
            manager = FallbackManager()
            # Use in-memory storage for tests
            with tempfile.TemporaryDirectory() as tmpdir:
                manager.cache = LocalCache(Path(tmpdir))
                with tempfile.NamedTemporaryFile(suffix=".db") as f:
                    manager.storage = LocalStorage(Path(f.name))
                    yield manager
    
    @pytest.mark.asyncio
    async def test_check_and_activate_unhealthy(self, manager, mock_health_checker):
        """Test activating fallback when service unhealthy."""
        # Mock unhealthy result
        mock_health_checker.perform_check.return_value = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.UNHEALTHY,
            error="Connection refused"
        )
        
        activated = await manager.check_and_activate()
        
        assert activated is True
        assert manager.state.is_active is True
        assert manager.state.mode in [FallbackMode.LOCAL_STORAGE, FallbackMode.QUEUE_WRITES]
        assert manager.state.reason == "Connection refused"
    
    @pytest.mark.asyncio
    async def test_check_and_activate_healthy(self, manager, mock_health_checker):
        """Test no activation when service healthy."""
        # Mock healthy result
        mock_health_checker.perform_check.return_value = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.HEALTHY
        )
        
        activated = await manager.check_and_activate()
        
        assert activated is False
        assert manager.state.is_active is False
    
    @pytest.mark.asyncio
    async def test_deactivate_fallback(self, manager, mock_health_checker):
        """Test deactivating fallback when service recovers."""
        # First activate fallback
        manager.state.activate(FallbackMode.LOCAL_STORAGE, "Test")
        
        # Mock healthy result
        mock_health_checker.perform_check.return_value = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.HEALTHY
        )
        
        await manager.check_and_activate()
        
        assert manager.state.is_active is False
    
    def test_store_entity_in_fallback(self, manager):
        """Test storing entity in fallback mode."""
        manager.state.activate(FallbackMode.LOCAL_STORAGE, "Test")
        
        entity = QuestionEntity(
            id="q1",
            content="Test question?",
            difficulty=DifficultyLevel.MEDIUM
        )
        
        result = manager.store_entity(entity)
        
        assert result is True
        # Check entity was stored
        stored = manager.get_entity("q1")
        assert stored is not None
    
    def test_store_entity_read_only_mode(self, manager):
        """Test storing entity fails in read-only mode."""
        manager.state.activate(FallbackMode.READ_ONLY, "Test")
        
        entity = QuestionEntity(
            id="q1",
            content="Test question?",
            difficulty=DifficultyLevel.MEDIUM
        )
        
        result = manager.store_entity(entity)
        
        assert result is False
    
    def test_queue_writes_mode(self, manager):
        """Test queueing writes for later sync."""
        manager.state.activate(FallbackMode.QUEUE_WRITES, "Test")
        
        entity = QuestionEntity(
            id="q1",
            content="Test question?",
            difficulty=DifficultyLevel.HARD
        )
        
        result = manager.store_entity(entity)
        
        assert result is True
        assert manager.state.queued_operations == 1
        
        # Check operation was queued
        operations = manager.storage.get_queued_operations()
        assert len(operations) == 1
        assert operations[0].entity_id == "q1"
    
    def test_cache_hit_tracking(self, manager):
        """Test cache hit/miss tracking."""
        manager.state.activate(FallbackMode.LOCAL_STORAGE, "Test")
        
        # Store entity
        entity = QuestionEntity(id="q1", content="Test?")
        manager.store_entity(entity)
        
        # First get - should be cache hit
        manager.get_entity("q1")
        assert manager.state.cache_hits == 1
        assert manager.state.cache_misses == 0
        
        # Get non-existent - should be cache miss
        manager.get_entity("q2")
        assert manager.state.cache_hits == 1
        assert manager.state.cache_misses == 1
    
    @pytest.mark.asyncio
    async def test_fallback_context(self, manager, mock_health_checker):
        """Test fallback context manager."""
        # Mock unhealthy then healthy
        mock_health_checker.perform_check.side_effect = [
            HealthCheckResult(
                component=ComponentType.GRAPHITI,
                status=HealthStatus.UNHEALTHY,
                error="Test error"
            ),
            HealthCheckResult(
                component=ComponentType.GRAPHITI,
                status=HealthStatus.HEALTHY
            )
        ]
        
        async with manager.fallback_context():
            assert manager.state.is_active is True
        
        # Should attempt to deactivate after context
        assert mock_health_checker.perform_check.call_count == 2
    
    def test_get_status(self, manager):
        """Test getting fallback status."""
        manager.state.activate(FallbackMode.QUEUE_WRITES, "Test")
        manager.state.queued_operations = 5
        manager.state.cache_hits = 10
        manager.state.cache_misses = 2
        
        status = manager.get_status()
        
        assert status["mode"] == "queue_writes"
        assert status["is_active"] is True
        assert status["queued_operations"] == 5
        assert status["cache_hit_rate"] == 10 / 12  # 10 hits / (10 hits + 2 misses)


class TestFallbackDecorator:
    """Test FallbackDecorator class."""
    
    @pytest.fixture
    def mock_manager(self):
        """Create mock fallback manager."""
        manager = Mock()
        manager.check_and_activate = AsyncMock(return_value=True)
        manager.store_entity = Mock(return_value=True)
        manager.get_entity = Mock(return_value={"id": "test"})
        return manager
    
    @pytest.mark.asyncio
    async def test_decorator_with_exception(self, mock_manager):
        """Test decorator handles exceptions with fallback."""
        decorator = FallbackDecorator(mock_manager)
        
        # Create function that raises exception
        @decorator
        async def failing_function(self, entity):
            raise ConnectionError("Service unavailable")
        
        # Create mock entity
        entity = Mock()
        entity.id = "test"
        
        # Should not raise exception
        result = await failing_function(None, entity)
        
        # Should have activated fallback
        mock_manager.check_and_activate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_decorator_without_exception(self, mock_manager):
        """Test decorator passes through when no exception."""
        decorator = FallbackDecorator(mock_manager)
        
        @decorator
        async def working_function(self, value):
            return value * 2
        
        result = await working_function(None, 5)
        
        assert result == 10
        # Should not have checked fallback
        mock_manager.check_and_activate.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_decorator_write_operation(self, mock_manager):
        """Test decorator handles write operations."""
        decorator = FallbackDecorator(mock_manager)
        
        @decorator
        async def create_entity(self, entity):
            raise ConnectionError("Service unavailable")
        
        entity = QuestionEntity(id="q1", content="Test?")
        result = await create_entity(None, entity)
        
        # Should have called store_entity
        mock_manager.store_entity.assert_called_once_with(entity)
    
    @pytest.mark.asyncio
    async def test_decorator_read_operation(self, mock_manager):
        """Test decorator handles read operations."""
        decorator = FallbackDecorator(mock_manager)
        
        @decorator
        async def get_entity(self, entity_id):
            raise ConnectionError("Service unavailable")
        
        result = await get_entity(None, "test_id")
        
        # Should have called get_entity
        mock_manager.get_entity.assert_called_once_with("test_id")


class TestGlobalFunctions:
    """Test global functions."""
    
    def test_get_fallback_manager(self):
        """Test getting global fallback manager."""
        manager1 = get_fallback_manager()
        manager2 = get_fallback_manager()
        
        assert manager1 is manager2  # Singleton
        assert isinstance(manager1, FallbackManager)
    
    @pytest.mark.asyncio
    async def test_with_fallback_decorator(self):
        """Test with_fallback decorator function."""
        @with_fallback
        async def test_function(value):
            return value * 2
        
        result = await test_function(5)
        assert result == 10