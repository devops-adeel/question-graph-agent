"""
Tests for the GraphitiClient module.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import uuid

from graphiti_client import (
    GraphitiClient,
    GraphitiSession,
    initialize_graphiti_state,
)
from graphiti_entities import (
    QuestionEntity,
    AnswerEntity,
    UserEntity,
    TopicEntity,
    DifficultyLevel,
    AnswerStatus,
)
from graphiti_relationships import AnsweredRelationship
from question_graph import QuestionState


class TestGraphitiSession:
    """Test GraphitiSession class."""
    
    def test_session_creation(self):
        """Test creating a new session."""
        session = GraphitiSession()
        
        assert session.session_id is not None
        assert session.user_id == "default_user"
        assert session.episode_count == 0
        assert session.entity_count == 0
        assert session.relationship_count == 0
        assert isinstance(session.start_time, datetime)
    
    def test_session_counters(self):
        """Test incrementing session counters."""
        session = GraphitiSession()
        
        session.increment_episode()
        session.increment_entity()
        session.increment_entity()
        session.increment_relationship()
        
        assert session.episode_count == 1
        assert session.entity_count == 2
        assert session.relationship_count == 1


class TestGraphitiClient:
    """Test GraphitiClient class."""
    
    @pytest.fixture
    def mock_neo4j_manager(self):
        """Create mock Neo4j connection manager."""
        manager = Mock()
        manager.connect_async = AsyncMock(return_value=Mock())
        manager.close_async = AsyncMock()
        manager.execute_query_async = AsyncMock(return_value=[])
        return manager
    
    @pytest.fixture
    def mock_graphiti_manager(self):
        """Create mock Graphiti connection manager."""
        manager = Mock()
        manager.close_async = AsyncMock()
        return manager
    
    @pytest.fixture
    def mock_fallback_manager(self):
        """Create mock fallback manager."""
        manager = Mock()
        manager.check_and_activate = AsyncMock(return_value=False)
        manager.state = Mock(is_active=False)
        return manager
    
    @pytest.fixture
    def client(self, mock_neo4j_manager, mock_graphiti_manager, mock_fallback_manager):
        """Create GraphitiClient with mocks."""
        with patch('graphiti_client.Neo4jConnectionManager', return_value=mock_neo4j_manager):
            with patch('graphiti_client.GraphitiConnectionManager', return_value=mock_graphiti_manager):
                with patch('graphiti_client.get_fallback_manager', return_value=mock_fallback_manager):
                    client = GraphitiClient()
                    # Mock the Graphiti instance
                    client._graphiti = Mock()
                    client._graphiti.add_entity = AsyncMock()
                    client._graphiti.add_relationship = AsyncMock()
                    client._graphiti.add_episode = AsyncMock()
                    return client
    
    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Test successful connection."""
        with patch('graphiti_client.check_system_health', return_value={'status': 'healthy'}):
            result = await client.connect()
        
        assert result is True
        client._neo4j_manager.connect_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_unhealthy_with_fallback(self, client):
        """Test connection with unhealthy system and fallback."""
        with patch('graphiti_client.check_system_health', return_value={'status': 'unhealthy'}):
            result = await client.connect()
        
        assert result is False
        client.fallback_manager.check_and_activate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test disconnection."""
        await client.disconnect()
        
        client._neo4j_manager.close_async.assert_called_once()
        client._graphiti_manager.close_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_question(self, client):
        """Test storing a question entity."""
        question = QuestionEntity(
            id="q1",
            content="Test question?",
            difficulty=DifficultyLevel.MEDIUM,
            topics=["test"]
        )
        
        result = await client.store_question(question)
        
        assert result is True
        assert client._session.entity_count == 1
        client._graphiti.add_entity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_answer(self, client):
        """Test storing an answer with relationships."""
        question = QuestionEntity(
            id="q1",
            content="Test question?",
            difficulty=DifficultyLevel.MEDIUM
        )
        
        answer = AnswerEntity(
            id="a1",
            question_id="q1",
            user_id="u1",
            content="Test answer",
            status=AnswerStatus.CORRECT,
            timestamp=datetime.now()
        )
        
        user = UserEntity(
            id="u1",
            session_id="session1"
        )
        
        result = await client.store_answer(answer, question, user)
        
        assert result is True
        assert client._session.entity_count == 1
        assert client._session.relationship_count == 1
        
        # Verify both entity and relationship were added
        assert client._graphiti.add_entity.call_count == 1
        assert client._graphiti.add_relationship.call_count == 1
    
    @pytest.mark.asyncio
    async def test_create_qa_episode(self, client):
        """Test creating a Q&A episode."""
        question = QuestionEntity(id="q1", content="Test?")
        answer = AnswerEntity(
            id="a1",
            question_id="q1",
            user_id="u1",
            content="Answer",
            status=AnswerStatus.CORRECT
        )
        user = UserEntity(id="u1")
        
        result = await client.create_qa_episode(
            question=question,
            answer=answer,
            user=user,
            evaluation_correct=True
        )
        
        assert result is True
        assert client._session.episode_count == 1
        client._graphiti.add_episode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_history(self, client):
        """Test getting user history."""
        # Mock query results
        mock_results = [
            {
                "q": {"id": "q1", "content": "Question 1?"},
                "a": {"id": "a1", "content": "Answer 1"},
                "r": {"timestamp": datetime.now().isoformat()}
            }
        ]
        client._neo4j_manager.execute_query_async.return_value = mock_results
        
        history = await client.get_user_history("user1", limit=5)
        
        assert len(history) == 1
        assert history[0]["question"]["id"] == "q1"
        client._neo4j_manager.execute_query_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_related_questions(self, client):
        """Test getting related questions."""
        # Mock query results
        mock_results = [
            {"q": {
                "id": "q1",
                "content": "Related question?",
                "difficulty": "MEDIUM",
                "topics": ["test"],
                "asked_count": 0,
                "correct_rate": 0.0
            }}
        ]
        client._neo4j_manager.execute_query_async.return_value = mock_results
        
        questions = await client.get_related_questions("test", difficulty="MEDIUM")
        
        assert len(questions) == 1
        assert questions[0].id == "q1"
        assert questions[0].difficulty == DifficultyLevel.MEDIUM
    
    @pytest.mark.asyncio
    async def test_update_user_mastery(self, client):
        """Test updating user mastery."""
        user = UserEntity(id="u1")
        topic = TopicEntity(id="t1", name="Test Topic")
        
        # Mock query results
        mock_mastery = [{"m": {
            "mastery_score": 0.5,
            "learning_rate": 0.1,
            "total_attempts": 5,
            "correct_attempts": 3
        }}]
        client._neo4j_manager.execute_query_async.side_effect = [
            mock_mastery,  # First query to get/create mastery
            []  # Second query to update score
        ]
        
        result = await client.update_user_mastery(
            user=user,
            topic=topic,
            correct=True,
            time_taken=10.5
        )
        
        assert result is True
        assert client._neo4j_manager.execute_query_async.call_count == 2
    
    def test_get_session_stats(self, client):
        """Test getting session statistics."""
        # Set some counts
        client._session.episode_count = 5
        client._session.entity_count = 10
        client._session.relationship_count = 8
        
        stats = client.get_session_stats()
        
        assert stats["episode_count"] == 5
        assert stats["entity_count"] == 10
        assert stats["relationship_count"] == 8
        assert stats["session_id"] == client._session.session_id
        assert "duration_seconds" in stats
    
    @pytest.mark.asyncio
    async def test_session_context(self, client):
        """Test session context manager."""
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch.object(client, 'disconnect', new_callable=AsyncMock) as mock_disconnect:
                mock_connect.return_value = True
                
                async with client.session_context() as session:
                    assert session is client
                
                mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()


class TestQuestionStateIntegration:
    """Test QuestionState integration with GraphitiClient."""
    
    @pytest.mark.asyncio
    async def test_initialize_graphiti_state_new(self):
        """Test initializing new state with Graphiti."""
        with patch('question_graph.GraphitiClient') as mock_client_class:
            mock_client = Mock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client_class.return_value = mock_client
            
            state = await initialize_graphiti_state(enable_graphiti=True)
            
            assert state is not None
            assert state.session_id is not None
            assert state.graphiti_client is mock_client
            assert state.current_user is not None
            mock_client.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_graphiti_state_existing(self):
        """Test enhancing existing state with Graphiti."""
        existing_state = QuestionState()
        existing_state.session_id = "existing_session"
        
        with patch('question_graph.GraphitiClient') as mock_client_class:
            mock_client = Mock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client_class.return_value = mock_client
            
            state = await initialize_graphiti_state(
                state=existing_state,
                session_id="existing_session",
                enable_graphiti=True
            )
            
            assert state is existing_state
            assert state.session_id == "existing_session"
            assert state.graphiti_client is mock_client
    
    @pytest.mark.asyncio
    async def test_initialize_graphiti_state_disabled(self):
        """Test initializing state with Graphiti disabled."""
        state = await initialize_graphiti_state(enable_graphiti=False)
        
        assert state is not None
        assert state.session_id is not None
        assert state.graphiti_client is None
        assert state.current_user is None
    
    @pytest.mark.asyncio
    async def test_initialize_graphiti_state_failure(self):
        """Test initialization when Graphiti connection fails."""
        with patch('question_graph.GraphitiClient') as mock_client_class:
            mock_client = Mock()
            mock_client.connect = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client_class.return_value = mock_client
            
            state = await initialize_graphiti_state(enable_graphiti=True)
            
            assert state is not None
            assert state.session_id is not None
            assert state.graphiti_client is None  # Should be None after failure
            assert state.current_user is None
    
    def test_question_state_with_graphiti(self):
        """Test QuestionState can hold GraphitiClient."""
        client = GraphitiClient(enable_fallback=False, enable_circuit_breaker=False)
        user = UserEntity(id="test_user")
        
        state = QuestionState(
            graphiti_client=client,
            current_user=user,
            session_id="test_session"
        )
        
        assert state.graphiti_client is client
        assert state.current_user is user
        assert state.session_id == "test_session"
        
        # Test serialization excludes these fields
        state_dict = state.model_dump()
        assert "graphiti_client" not in state_dict
        assert "current_user" not in state_dict
        assert "session_id" in state_dict  # This should be included