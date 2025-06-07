"""
Tests for the memory storage functions.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import hashlib

from graphiti_memory import (
    QAPair,
    MemoryStorage,
    MemoryAnalytics,
    DifficultyLevel,
)
from graphiti_client import GraphitiClient
from graphiti_entities import (
    QuestionEntity,
    AnswerEntity,
    UserEntity,
    TopicEntity,
    AnswerStatus,
)


class TestQAPair:
    """Test QAPair class."""
    
    def test_qa_pair_creation(self):
        """Test creating a QA pair."""
        qa_pair = QAPair(
            question="What is 2+2?",
            answer="4",
            user_id="user123",
            session_id="session456",
            correct=True,
            response_time=2.5
        )
        
        assert qa_pair.question == "What is 2+2?"
        assert qa_pair.answer == "4"
        assert qa_pair.user_id == "user123"
        assert qa_pair.correct is True
        assert qa_pair.response_time == 2.5
        assert qa_pair.question_id is not None
        assert qa_pair.answer_id is not None
    
    def test_qa_pair_id_generation(self):
        """Test automatic ID generation."""
        qa_pair = QAPair(
            question="Test question?",
            answer="Test answer",
            user_id="user1",
            session_id="session1"
        )
        
        # Question ID should be deterministic based on content
        expected_q_hash = hashlib.md5("Test question?".encode()).hexdigest()[:8]
        assert qa_pair.question_id.startswith(f"q_{expected_q_hash}_")
        
        # Answer ID should be unique
        assert qa_pair.answer_id.startswith("a_")
        assert len(qa_pair.answer_id) > 10
    
    def test_qa_pair_with_custom_ids(self):
        """Test QA pair with custom IDs."""
        qa_pair = QAPair(
            question="Test?",
            answer="Yes",
            user_id="user1",
            session_id="session1",
            question_id="custom_q_id",
            answer_id="custom_a_id"
        )
        
        assert qa_pair.question_id == "custom_q_id"
        assert qa_pair.answer_id == "custom_a_id"


class TestMemoryStorage:
    """Test MemoryStorage class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock GraphitiClient."""
        client = Mock(spec=GraphitiClient)
        client.store_question = AsyncMock(return_value=True)
        client.store_answer = AsyncMock(return_value=True)
        client.create_qa_episode = AsyncMock(return_value=True)
        client.update_user_mastery = AsyncMock(return_value=True)
        client._graphiti = Mock()
        client._graphiti.add_entity = AsyncMock()
        client._graphiti.add_relationship = AsyncMock()
        client.entity_adapter = Mock()
        client.entity_adapter.to_graphiti_entity = Mock(return_value={})
        client.entity_adapter.to_graphiti_relationship = Mock(return_value={})
        client._neo4j_manager = Mock()
        client._neo4j_manager.execute_query_async = AsyncMock(return_value=[])
        return client
    
    @pytest.fixture
    def storage(self, mock_client):
        """Create MemoryStorage with mock client."""
        return MemoryStorage(client=mock_client)
    
    @pytest.mark.asyncio
    async def test_store_qa_pair(self, storage, mock_client):
        """Test storing a complete Q&A pair."""
        qa_pair = QAPair(
            question="What is Python?",
            answer="A programming language",
            user_id="user1",
            session_id="session1",
            topics=["programming", "python"],
            correct=True,
            response_time=5.0
        )
        
        # Mock entity extraction
        with patch.object(storage.entity_extractor, 'extract_from_text', 
                         return_value={'topics': [{'name': 'programming'}, {'name': 'python'}]}):
            result = await storage.store_qa_pair(qa_pair)
        
        assert result is True
        
        # Verify methods were called
        mock_client.store_question.assert_called_once()
        mock_client.store_answer.assert_called_once()
        mock_client.create_qa_episode.assert_called_once()
        
        # Verify mastery update was called for each topic
        assert mock_client.update_user_mastery.call_count == 2
    
    @pytest.mark.asyncio
    async def test_store_qa_pair_without_client(self):
        """Test storing Q&A pair without client."""
        storage = MemoryStorage(client=None)
        qa_pair = QAPair(
            question="Test?",
            answer="Yes",
            user_id="user1",
            session_id="session1"
        )
        
        result = await storage.store_qa_pair(qa_pair)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_store_question_only(self, storage, mock_client):
        """Test storing just a question."""
        with patch.object(storage.entity_extractor, 'extract_from_text',
                         return_value={'topics': [{'name': 'math'}]}):
            question_entity = await storage.store_question_only(
                question="What is 2+2?",
                difficulty=DifficultyLevel.EASY
            )
        
        assert question_entity is not None
        assert question_entity.content == "What is 2+2?"
        assert question_entity.difficulty == DifficultyLevel.EASY
        assert "math" in question_entity.topics
        
        mock_client.store_question.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_batch_qa_pairs(self, storage, mock_client):
        """Test storing multiple Q&A pairs."""
        qa_pairs = [
            QAPair(
                question=f"Question {i}?",
                answer=f"Answer {i}",
                user_id="user1",
                session_id="session1",
                topics=["test"]
            )
            for i in range(3)
        ]
        
        with patch.object(storage, 'store_qa_pair', side_effect=[True, True, False]):
            successful, failed = await storage.store_batch_qa_pairs(qa_pairs)
        
        assert successful == 2
        assert failed == 1
    
    @pytest.mark.asyncio
    async def test_update_answer_evaluation(self, storage, mock_client):
        """Test updating answer evaluation."""
        mock_client._neo4j_manager.execute_query_async.return_value = [{"a": {"id": "answer1"}}]
        
        result = await storage.update_answer_evaluation(
            answer_id="answer1",
            correct=True,
            comment="Good job!"
        )
        
        assert result is True
        
        # Verify query was executed
        query_call = mock_client._neo4j_manager.execute_query_async.call_args
        assert "answer1" in str(query_call)
        assert "CORRECT" in str(query_call)
        assert "Good job!" in str(query_call)
    
    @pytest.mark.asyncio
    async def test_create_topic_relationships(self, storage, mock_client):
        """Test creating topic relationships."""
        question = QuestionEntity(
            id="q1",
            content="Test question",
            topics=["topic1", "topic2"]
        )
        
        await storage._create_topic_relationships(question, ["topic1", "topic2"])
        
        # Should create entity and relationship for each topic
        assert mock_client._graphiti.add_entity.call_count == 2
        assert mock_client._graphiti.add_relationship.call_count == 2
    
    def test_get_or_create_topic(self):
        """Test topic caching."""
        storage = MemoryStorage(client=None)
        
        # First call creates topic
        topic1 = asyncio.run(storage._get_or_create_topic("Python"))
        assert topic1.name == "Python"
        assert topic1.id.startswith("topic_")
        
        # Second call returns cached topic
        topic2 = asyncio.run(storage._get_or_create_topic("Python"))
        assert topic1 is topic2  # Same object


class TestMemoryAnalytics:
    """Test MemoryAnalytics class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock GraphitiClient."""
        client = Mock(spec=GraphitiClient)
        client._neo4j_manager = Mock()
        client._neo4j_manager.execute_query_async = AsyncMock()
        return client
    
    @pytest.fixture
    def analytics(self, mock_client):
        """Create MemoryAnalytics with mock client."""
        return MemoryAnalytics(client=mock_client)
    
    @pytest.mark.asyncio
    async def test_get_user_stats(self, analytics, mock_client):
        """Test getting user statistics."""
        mock_result = [{
            "stats": {
                "user_id": "user1",
                "total_questions": 10,
                "correct_answers": 7,
                "accuracy": 0.7,
                "avg_response_time": 5.5,
                "topics_encountered": 5
            }
        }]
        mock_client._neo4j_manager.execute_query_async.return_value = mock_result
        
        stats = await analytics.get_user_stats("user1")
        
        assert stats["user_id"] == "user1"
        assert stats["total_questions"] == 10
        assert stats["correct_answers"] == 7
        assert stats["accuracy"] == 0.7
        
        # Verify query was called
        mock_client._neo4j_manager.execute_query_async.assert_called_once()
        query_call = mock_client._neo4j_manager.execute_query_async.call_args
        assert "user1" in str(query_call)
    
    @pytest.mark.asyncio
    async def test_get_topic_performance(self, analytics, mock_client):
        """Test getting topic performance."""
        mock_result = [{
            "performance": {
                "topic": "mathematics",
                "total_questions": 5,
                "correct_questions": 3,
                "accuracy": 0.6,
                "avg_response_time": 8.2,
                "difficulty_distribution": ["EASY", "MEDIUM", "MEDIUM", "HARD", "EASY"]
            }
        }]
        mock_client._neo4j_manager.execute_query_async.return_value = mock_result
        
        performance = await analytics.get_topic_performance("user1", "mathematics")
        
        assert performance["topic"] == "mathematics"
        assert performance["total_questions"] == 5
        assert performance["correct_questions"] == 3
        assert performance["accuracy"] == 0.6
    
    @pytest.mark.asyncio
    async def test_get_session_summary(self, analytics, mock_client):
        """Test getting session summary."""
        mock_result = [{
            "summary": {
                "session_id": "session1",
                "user_id": "user1",
                "questions_asked": 8,
                "correct_answers": 6,
                "accuracy": 0.75,
                "duration_seconds": 300,
                "avg_response_time": 6.5,
                "topics_covered": ["math", "science", "history"]
            }
        }]
        mock_client._neo4j_manager.execute_query_async.return_value = mock_result
        
        summary = await analytics.get_session_summary("session1")
        
        assert summary["session_id"] == "session1"
        assert summary["questions_asked"] == 8
        assert summary["correct_answers"] == 6
        assert summary["accuracy"] == 0.75
        assert len(summary["topics_covered"]) == 3
    
    @pytest.mark.asyncio
    async def test_analytics_without_client(self):
        """Test analytics without client."""
        analytics = MemoryAnalytics(client=None)
        
        stats = await analytics.get_user_stats("user1")
        assert stats == {}
        
        performance = await analytics.get_topic_performance("user1", "topic1")
        assert performance == {}
        
        summary = await analytics.get_session_summary("session1")
        assert summary == {}