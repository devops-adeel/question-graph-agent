"""
Tests for the database initialization module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from neo4j.exceptions import ConstraintError, DatabaseError

from graphiti_init import (
    DatabaseInitializer,
    initialize_database,
    reset_database,
    get_database_status,
)
from graphiti_config import RuntimeConfig


class TestDatabaseInitializer:
    """Test database initializer functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=RuntimeConfig)
        config.neo4j = Mock()
        config.neo4j.uri = "bolt://localhost:7687"
        return config
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create mock connection manager."""
        manager = Mock()
        manager.async_session = MagicMock()
        return manager
    
    @pytest.fixture
    def mock_session(self):
        """Create mock async session."""
        session = AsyncMock()
        session.run = AsyncMock()
        return session
    
    @pytest.fixture
    def initializer(self, mock_config, mock_connection_manager):
        """Create database initializer with mocks."""
        with patch('graphiti_init.Neo4jConnectionManager', return_value=mock_connection_manager):
            return DatabaseInitializer(mock_config)
    
    @pytest.mark.asyncio
    async def test_initialize_database_success(self, initializer, mock_session):
        """Test successful database initialization."""
        # Mock session context manager
        initializer.connection_manager.async_session.return_value.__aenter__.return_value = mock_session
        initializer.connection_manager.async_session.return_value.__aexit__.return_value = None
        
        # Mock verification to return True
        with patch.object(initializer, '_verify_initialization', return_value=True):
            result = await initializer.initialize_database()
        
        assert result is True
        assert initializer._initialized is True
        
        # Verify methods were called
        assert mock_session.run.called
    
    @pytest.mark.asyncio
    async def test_initialize_database_already_initialized(self, initializer):
        """Test initialization when already initialized."""
        initializer._initialized = True
        
        result = await initializer.initialize_database(force=False)
        
        assert result is True
        # Should not create indexes/constraints
        assert not initializer.connection_manager.async_session.called
    
    @pytest.mark.asyncio
    async def test_initialize_database_force_reinit(self, initializer, mock_session):
        """Test forced re-initialization."""
        initializer._initialized = True
        
        # Mock session
        initializer.connection_manager.async_session.return_value.__aenter__.return_value = mock_session
        initializer.connection_manager.async_session.return_value.__aexit__.return_value = None
        
        with patch.object(initializer, '_verify_initialization', return_value=True):
            result = await initializer.initialize_database(force=True)
        
        assert result is True
        assert mock_session.run.called
    
    @pytest.mark.asyncio
    async def test_create_indexes(self, initializer, mock_session):
        """Test index creation."""
        initializer.connection_manager.async_session.return_value.__aenter__.return_value = mock_session
        initializer.connection_manager.async_session.return_value.__aexit__.return_value = None
        
        await initializer._create_indexes()
        
        # Check that index creation queries were run
        calls = mock_session.run.call_args_list
        assert len(calls) > 0
        
        # Verify some expected indexes
        index_queries = [call[0][0] for call in calls]
        assert any("Entity" in q and "id" in q for q in index_queries)
        assert any("Question" in q and "difficulty" in q for q in index_queries)
    
    @pytest.mark.asyncio
    async def test_create_constraints(self, initializer, mock_session):
        """Test constraint creation."""
        initializer.connection_manager.async_session.return_value.__aenter__.return_value = mock_session
        initializer.connection_manager.async_session.return_value.__aexit__.return_value = None
        
        await initializer._create_constraints()
        
        # Check constraint creation queries
        calls = mock_session.run.call_args_list
        assert len(calls) > 0
        
        constraint_queries = [call[0][0] for call in calls]
        assert any("Entity" in q and "UNIQUE" in q for q in constraint_queries)
    
    @pytest.mark.asyncio
    async def test_create_initial_topics(self, initializer, mock_session):
        """Test initial topic creation."""
        initializer.connection_manager.async_session.return_value.__aenter__.return_value = mock_session
        initializer.connection_manager.async_session.return_value.__aexit__.return_value = None
        
        # Mock query results
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"topic_id": "test-id"})
        mock_session.run.return_value = mock_result
        
        await initializer._create_initial_topics()
        
        # Verify topics were created
        calls = mock_session.run.call_args_list
        assert len(calls) > 0
        
        # Check for parent topics
        topic_queries = [str(call) for call in calls]
        assert any("General Knowledge" in str(q) for q in topic_queries)
        assert any("Mathematics" in str(q) for q in topic_queries)
    
    @pytest.mark.asyncio
    async def test_verify_initialization_success(self, initializer, mock_session):
        """Test successful initialization verification."""
        initializer.connection_manager.async_session.return_value.__aenter__.return_value = mock_session
        initializer.connection_manager.async_session.return_value.__aexit__.return_value = None
        
        # Mock all verification queries to return positive counts
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"count": 1})
        mock_session.run.return_value = mock_result
        
        result = await initializer._verify_initialization()
        
        assert result is True
        assert mock_session.run.call_count == 4  # 4 verification checks
    
    @pytest.mark.asyncio
    async def test_verify_initialization_failure(self, initializer, mock_session):
        """Test failed initialization verification."""
        initializer.connection_manager.async_session.return_value.__aenter__.return_value = mock_session
        initializer.connection_manager.async_session.return_value.__aexit__.return_value = None
        
        # Mock first query to return 0
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"count": 0})
        mock_session.run.return_value = mock_result
        
        result = await initializer._verify_initialization()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_reset_database_without_confirm(self, initializer):
        """Test database reset without confirmation."""
        result = await initializer.reset_database(confirm=False)
        
        assert result is False
        # Should not run any queries
        assert not initializer.connection_manager.async_session.called
    
    @pytest.mark.asyncio
    async def test_reset_database_with_confirm(self, initializer, mock_session):
        """Test database reset with confirmation."""
        initializer.connection_manager.async_session.return_value.__aenter__.return_value = mock_session
        initializer.connection_manager.async_session.return_value.__aexit__.return_value = None
        
        # Mock constraint and index results
        mock_constraints = AsyncMock()
        mock_constraints.data = AsyncMock(return_value=[
            {"name": "constraint1"},
            {"name": "constraint2"}
        ])
        
        mock_indexes = AsyncMock()
        mock_indexes.data = AsyncMock(return_value=[
            {"name": "index1"},
            {"name": "index2"}
        ])
        
        # Set up return values for different queries
        mock_session.run.side_effect = [
            None,  # DELETE query
            mock_constraints,  # SHOW CONSTRAINTS
            None,  # DROP CONSTRAINT 1
            None,  # DROP CONSTRAINT 2
            mock_indexes,  # SHOW INDEXES
            None,  # DROP INDEX 1
            None,  # DROP INDEX 2
        ]
        
        result = await initializer.reset_database(confirm=True)
        
        assert result is True
        assert initializer._initialized is False
        
        # Verify delete query was run
        delete_call = mock_session.run.call_args_list[0]
        assert "DELETE" in delete_call[0][0]
    
    @pytest.mark.asyncio
    async def test_get_initialization_status(self, initializer, mock_session):
        """Test getting initialization status."""
        initializer.connection_manager.async_session.return_value.__aenter__.return_value = mock_session
        initializer.connection_manager.async_session.return_value.__aexit__.return_value = None
        
        # Mock query results
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(side_effect=[
            {"count": 5},  # indexes
            {"count": 3},  # constraints
            {"count": 10},  # topics
            {"count": 2},  # users
            {"count": 100},  # episodes
        ])
        mock_session.run.return_value = mock_result
        
        # Mock verification
        with patch.object(initializer, '_verify_initialization', return_value=True):
            status = await initializer.get_initialization_status()
        
        assert status["indexes"] == 5
        assert status["constraints"] == 3
        assert status["topics"] == 10
        assert status["users"] == 2
        assert status["episodes"] == 100
        assert status["verified"] is True


class TestModuleFunctions:
    """Test module-level functions."""
    
    @pytest.mark.asyncio
    async def test_initialize_database_function(self):
        """Test initialize_database function."""
        mock_initializer = Mock(spec=DatabaseInitializer)
        mock_initializer.initialize_database = AsyncMock(return_value=True)
        
        with patch('graphiti_init.DatabaseInitializer', return_value=mock_initializer):
            result = await initialize_database()
        
        assert result is True
        mock_initializer.initialize_database.assert_called_once_with(force=False)
    
    @pytest.mark.asyncio
    async def test_reset_database_function(self):
        """Test reset_database function."""
        mock_initializer = Mock(spec=DatabaseInitializer)
        mock_initializer.reset_database = AsyncMock(return_value=True)
        
        with patch('graphiti_init.DatabaseInitializer', return_value=mock_initializer):
            result = await reset_database(confirm=True)
        
        assert result is True
        mock_initializer.reset_database.assert_called_once_with(confirm=True)
    
    @pytest.mark.asyncio
    async def test_get_database_status_function(self):
        """Test get_database_status function."""
        expected_status = {
            "initialized": True,
            "indexes": 5,
            "constraints": 3,
        }
        
        mock_initializer = Mock(spec=DatabaseInitializer)
        mock_initializer.get_initialization_status = AsyncMock(return_value=expected_status)
        
        with patch('graphiti_init.DatabaseInitializer', return_value=mock_initializer):
            status = await get_database_status()
        
        assert status == expected_status
        mock_initializer.get_initialization_status.assert_called_once()