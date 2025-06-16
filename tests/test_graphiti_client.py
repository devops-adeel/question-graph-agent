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
    SessionStats,
    EntityAdapter,
)
from graphiti_entities import (
    QuestionEntity,
    AnswerEntity,
    UserEntity,
    TopicEntity,
)
from graphiti_relationships import AnsweredRelationship, RequiresKnowledgeRelationship
from question_graph import QuestionState


class TestEntityAdapter:
    """Test EntityAdapter class."""
    
    def test_adapt_question_entity(self):
        """Test adapting QuestionEntity for Graphiti."""
        question = QuestionEntity(
            content="What is the capital of France?",
            difficulty="easy",
            topics=["geography", "france"]
        )
        
        adapted = EntityAdapter.adapt_entity(question)
        
        assert adapted["name"] == "What is the capital of France?"
        assert adapted["entity_type"] == "question"
        assert adapted["properties"]["difficulty"] == "easy"
        assert adapted["properties"]["topics"] == ["geography", "france"]
    
    def test_adapt_user_entity(self):
        """Test adapting UserEntity for Graphiti."""
        user = UserEntity(
            user_id="user123",
            name="Test User"
        )
        
        adapted = EntityAdapter.adapt_entity(user)
        
        assert adapted["name"] == "Test User"
        assert adapted["entity_type"] == "user"
        assert adapted["properties"]["user_id"] == "user123"


class TestGraphitiClient:
    """Test GraphitiClient class."""
    
    @pytest.fixture
    def mock_graphiti(self):
        """Create mock Graphiti instance."""
        mock = Mock()
        mock.add_entity = AsyncMock()
        mock.add_relationship = AsyncMock()
        mock.add_episode = AsyncMock()
        mock.search = AsyncMock(return_value=[])
        mock.get_entity = AsyncMock()
        mock.get_entities = AsyncMock(return_value=[])
        return mock
    
    @pytest.fixture
    def client(self, mock_graphiti):
        """Create GraphitiClient with mocked Graphiti."""
        with patch('graphiti_client.Graphiti', return_value=mock_graphiti):
            return GraphitiClient()
    
    @pytest.mark.asyncio
    async def test_store_entity(self, client, mock_graphiti):
        """Test storing an entity."""
        question = QuestionEntity(
            content="What is Python?",
            difficulty="easy",
            topics=["programming"]
        )
        
        # Mock the response
        mock_graphiti.add_entity.return_value = {
            "uuid": "test-uuid",
            "name": "What is Python?",
            "entity_type": "question"
        }
        
        result = await client.store_entity(question)
        
        assert result == "test-uuid"
        mock_graphiti.add_entity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_relationship(self, client, mock_graphiti):
        """Test storing a relationship."""
        relationship = AnsweredRelationship(
            user_id="user123",
            question_id="q456",
            answer_id="a789",
            score=1.0,
            response_time=2.5
        )
        
        # Mock the response
        mock_graphiti.add_relationship.return_value = {
            "uuid": "rel-uuid",
            "relationship_type": "answered"
        }
        
        result = await client.store_relationship(
            relationship,
            source_id="user123",
            target_id="q456"
        )
        
        assert result == "rel-uuid"
        mock_graphiti.add_relationship.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_question(self, client, mock_graphiti):
        """Test storing a question through convenience method."""
        # Mock entity storage
        mock_graphiti.add_entity.return_value = {
            "uuid": "q-uuid",
            "name": "Test question?",
            "entity_type": "question"
        }
        
        entity = await client.store_question(
            question="Test question?",
            difficulty="medium",
            topics=["test"]
        )
        
        assert isinstance(entity, QuestionEntity)
        assert entity.id == "q-uuid"
        assert entity.content == "Test question?"
    
    @pytest.mark.asyncio
    async def test_store_qa_pair(self, client, mock_graphiti):
        """Test storing a complete Q&A pair."""
        # Mock all the storage calls
        mock_graphiti.add_entity.side_effect = [
            {"uuid": "answer-uuid"},  # Answer entity
            {"uuid": "user-uuid"}     # User entity  
        ]
        mock_graphiti.add_relationship.return_value = {"uuid": "rel-uuid"}
        
        success = await client.store_qa_pair(
            question_id="q123",
            question="What is 2+2?",
            answer="4",
            is_correct=True,
            user_id="user456",
            response_time=1.5
        )
        
        assert success is True
        assert mock_graphiti.add_entity.call_count == 2
        mock_graphiti.add_relationship.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_recent_questions(self, client, mock_graphiti):
        """Test retrieving recent questions."""
        # Mock search results
        mock_graphiti.search.return_value = [
            {
                "uuid": "q1",
                "name": "Question 1",
                "entity_type": "question",
                "created_at": datetime.now().isoformat()
            },
            {
                "uuid": "q2", 
                "name": "Question 2",
                "entity_type": "question",
                "created_at": datetime.now().isoformat()
            }
        ]
        
        questions = await client.get_recent_questions(limit=2)
        
        assert len(questions) == 2
        assert all(isinstance(q, QuestionEntity) for q in questions)
        mock_graphiti.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_qa_episode(self, client, mock_graphiti):
        """Test creating a Q&A episode."""
        # Create mock entities
        question = QuestionEntity(
            id="q123",
            content="What is AI?",
            difficulty="medium",
            topics=["technology", "ai"]
        )
        answer = AnswerEntity(
            id="a456",
            content="AI is artificial intelligence",
            status="correct",
            response_time=2.5
        )
        user = UserEntity(
            id="u789",
            user_id="user123",
            name="Test User",
            session_id="session456"
        )
        
        # Mock episode creation
        mock_graphiti.add_episode.return_value = {
            "uuid": "episode-uuid",
            "name": "Q&A Session Episode"
        }
        
        # Mock episode builder
        with patch.object(client, 'episode_builder') as mock_builder:
            mock_builder.build_qa_episode.return_value = {
                "name": "Q&A Episode",
                "episode_body": "User answered: AI is artificial intelligence",
                "source": "message",
                "reference_time": datetime.now()
            }
            
            result = await client.create_qa_episode(
                question=question,
                answer=answer,
                user=user,
                evaluation_correct=True
            )
        
        assert result is True
        mock_builder.build_qa_episode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client, mock_graphiti):
        """Test error handling in client methods."""
        # Make add_entity raise an exception
        mock_graphiti.add_entity.side_effect = Exception("Storage failed")
        
        question = QuestionEntity(
            content="Test question?",
            difficulty="easy",
            topics=["test"]
        )
        
        # Should handle error gracefully
        result = await client.store_entity(question)
        assert result is None  # Returns None on error


class TestSessionStats:
    """Test SessionStats class."""
    
    def test_stats_initialization(self):
        """Test SessionStats initialization."""
        stats = SessionStats()
        
        assert stats.questions_asked == 0
        assert stats.correct_answers == 0
        assert stats.total_time == 0.0
        assert stats.topics == set()
    
    def test_update_stats(self):
        """Test updating session statistics."""
        stats = SessionStats()
        
        stats.questions_asked = 10
        stats.correct_answers = 7
        stats.total_time = 120.5
        stats.topics = {"math", "science", "history"}
        
        summary = stats.get_summary()
        
        assert summary["questions"] == 10
        assert summary["correct"] == 7
        assert summary["accuracy"] == 0.7
        assert summary["avg_time"] == 12.05
        assert summary["topics_count"] == 3


# Note: The following tests were for non-existent functionality
# and have been removed:
# - TestGraphitiSession (GraphitiSession class doesn't exist)
# - Tests for initialize_graphiti_state (function doesn't exist)