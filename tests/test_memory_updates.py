"""
Test suite for memory update functionality.

Tests cover post-evaluation memory updates including user statistics,
topic mastery tracking, and evaluation event handling.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from memory_updates import (
    EvaluationResult,
    UserPerformanceUpdate,
    MemoryUpdateService,
    EvaluationEventHandler,
)
from graphiti_entities import DifficultyLevel, AnswerStatus
from graphiti_memory import MemoryStorage
from memory_retrieval import MemoryRetrieval


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""
    
    def test_evaluation_result_creation(self):
        """Test creating evaluation result with all fields."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            evaluation_comment="Well done!",
            confidence_score=0.95,
            response_time=5.2,
            topics=["mathematics", "algebra"],
            difficulty=DifficultyLevel.MEDIUM
        )
        
        assert result.question_id == "q_123"
        assert result.answer_id == "a_456"
        assert result.user_id == "user_789"
        assert result.session_id == "session_abc"
        assert result.correct is True
        assert result.evaluation_comment == "Well done!"
        assert result.confidence_score == 0.95
        assert result.response_time == 5.2
        assert result.topics == ["mathematics", "algebra"]
        assert result.difficulty == DifficultyLevel.MEDIUM
        assert isinstance(result.timestamp, datetime)
    
    def test_evaluation_result_defaults(self):
        """Test evaluation result with default values."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=False,
            evaluation_comment="Try again"
        )
        
        assert result.confidence_score == 0.0
        assert result.response_time is None
        assert result.topics == []
        assert result.difficulty == DifficultyLevel.MEDIUM


class TestUserPerformanceUpdate:
    """Test UserPerformanceUpdate dataclass."""
    
    def test_performance_update_creation(self):
        """Test creating performance update."""
        update = UserPerformanceUpdate(
            total_questions_delta=5,
            correct_answers_delta=3,
            response_time_sum=25.5,
            response_time_count=5,
            topics_attempted=["math", "science"],
            difficulty_attempts={"easy": 2, "medium": 3}
        )
        
        assert update.total_questions_delta == 5
        assert update.correct_answers_delta == 3
        assert update.response_time_sum == 25.5
        assert update.response_time_count == 5
        assert update.topics_attempted == ["math", "science"]
        assert update.difficulty_attempts == {"easy": 2, "medium": 3}
    
    def test_performance_update_defaults(self):
        """Test performance update with defaults."""
        update = UserPerformanceUpdate()
        
        assert update.total_questions_delta == 0
        assert update.correct_answers_delta == 0
        assert update.response_time_sum == 0.0
        assert update.response_time_count == 0
        assert update.topics_attempted == []
        assert update.difficulty_attempts == {}


class TestMemoryUpdateService:
    """Test MemoryUpdateService class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock GraphitiClient."""
        client = MagicMock()
        client._neo4j_manager = MagicMock()
        client._neo4j_manager.execute_query_async = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock MemoryStorage."""
        storage = MagicMock(spec=MemoryStorage)
        storage.update_answer_evaluation = AsyncMock(return_value=True)
        return storage
    
    @pytest.fixture
    def mock_retrieval(self):
        """Create mock MemoryRetrieval."""
        retrieval = MagicMock(spec=MemoryRetrieval)
        return retrieval
    
    @pytest.fixture
    def service(self, mock_client, mock_storage, mock_retrieval):
        """Create MemoryUpdateService with mocks."""
        service = MemoryUpdateService(client=mock_client)
        service.storage = mock_storage
        service.retrieval = mock_retrieval
        return service
    
    @pytest.mark.asyncio
    async def test_record_evaluation_immediate(self, service):
        """Test recording evaluation immediately."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            evaluation_comment="Correct!",
            topics=["math"]
        )
        
        # Mock _process_evaluation
        service._process_evaluation = AsyncMock(return_value=True)
        
        success = await service.record_evaluation(result, immediate=True)
        
        assert success is True
        service._process_evaluation.assert_called_once_with(result)
        assert len(service._pending_updates) == 0
    
    @pytest.mark.asyncio
    async def test_record_evaluation_batch(self, service):
        """Test recording evaluation for batch processing."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            evaluation_comment="Correct!"
        )
        
        success = await service.record_evaluation(result, immediate=False)
        
        assert success is True
        assert len(service._pending_updates) == 1
        assert service._pending_updates[0] == result
    
    @pytest.mark.asyncio
    async def test_record_evaluation_no_client(self):
        """Test recording evaluation without client."""
        service = MemoryUpdateService(client=None)
        
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            evaluation_comment="Correct!"
        )
        
        success = await service.record_evaluation(result)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_process_evaluation_success(self, service, mock_storage, mock_client):
        """Test successful evaluation processing."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            evaluation_comment="Well done!",
            topics=["mathematics", "algebra"],
            difficulty=DifficultyLevel.HARD
        )
        
        # Mock internal methods
        service._create_evaluation_relationship = AsyncMock(return_value=True)
        service._update_user_statistics = AsyncMock(return_value=True)
        service._update_topic_mastery = AsyncMock(return_value=True)
        service._track_performance_trends = AsyncMock(return_value=True)
        
        success = await service._process_evaluation(result)
        
        assert success is True
        
        # Verify all methods were called
        mock_storage.update_answer_evaluation.assert_called_once_with(
            answer_id="a_456",
            correct=True,
            comment="Well done!"
        )
        service._create_evaluation_relationship.assert_called_once_with(result)
        service._update_user_statistics.assert_called_once_with(result)
        service._update_topic_mastery.assert_called_once_with(result)
        service._track_performance_trends.assert_called_once_with(result)
    
    @pytest.mark.asyncio
    async def test_process_evaluation_answer_update_failure(self, service, mock_storage):
        """Test evaluation processing when answer update fails."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=False,
            evaluation_comment="Incorrect"
        )
        
        # Mock answer update failure
        mock_storage.update_answer_evaluation.return_value = False
        
        success = await service._process_evaluation(result)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_create_evaluation_relationship(self, service, mock_client):
        """Test creating evaluation relationship."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            evaluation_comment="Great job!",
            confidence_score=0.9
        )
        
        success = await service._create_evaluation_relationship(result)
        
        assert success is True
        
        # Verify query was executed
        mock_client._neo4j_manager.execute_query_async.assert_called_once()
        query, params = mock_client._neo4j_manager.execute_query_async.call_args[0]
        
        assert "MATCH (q:Question {id: $question_id})" in query
        assert "MATCH (a:Answer {id: $answer_id})" in query
        assert "MERGE (a)-[r:EVALUATED" in query
        
        assert params["question_id"] == "q_123"
        assert params["answer_id"] == "a_456"
        assert params["correct"] is True
        assert params["confidence"] == 0.9
        assert params["comment"] == "Great job!"
    
    @pytest.mark.asyncio
    async def test_update_user_statistics(self, service, mock_client):
        """Test updating user statistics."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            evaluation_comment="Correct!"
        )
        
        # Mock session stats update
        service._update_session_stats = AsyncMock(return_value=True)
        
        success = await service._update_user_statistics(result)
        
        assert success is True
        
        # Verify user stats query
        mock_client._neo4j_manager.execute_query_async.assert_called_once()
        query, params = mock_client._neo4j_manager.execute_query_async.call_args[0]
        
        assert "MATCH (u:User {id: $user_id})" in query
        assert "SET u.total_questions" in query
        assert "SET u.correct_answers" in query
        assert params["user_id"] == "user_789"
        assert params["correct"] is True
        
        # Verify session stats were updated
        service._update_session_stats.assert_called_once_with(result)
    
    @pytest.mark.asyncio
    async def test_update_session_stats(self, service, mock_client):
        """Test updating session statistics."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=False,
            evaluation_comment="Try again"
        )
        
        success = await service._update_session_stats(result)
        
        assert success is True
        
        # Verify session stats query
        mock_client._neo4j_manager.execute_query_async.assert_called_once()
        query, params = mock_client._neo4j_manager.execute_query_async.call_args[0]
        
        assert "MERGE (s:Session {id: $session_id})" in query
        assert "SET s.total_questions" in query
        assert params["session_id"] == "session_abc"
        assert params["user_id"] == "user_789"
        assert params["correct"] is False
    
    @pytest.mark.asyncio
    async def test_update_topic_mastery(self, service, mock_client):
        """Test updating topic mastery."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            evaluation_comment="Excellent!",
            topics=["mathematics", "algebra"],
            difficulty=DifficultyLevel.HARD
        )
        
        # Mock mastery calculation
        service._calculate_topic_mastery = AsyncMock(return_value={
            "level": 0.8,
            "confidence": 0.9
        })
        
        success = await service._update_topic_mastery(result)
        
        assert success is True
        
        # Should be called for each topic
        assert service._calculate_topic_mastery.call_count == 2
        assert mock_client._neo4j_manager.execute_query_async.call_count == 2
        
        # Verify mastery update query
        query, params = mock_client._neo4j_manager.execute_query_async.call_args[0]
        
        assert "MATCH (u:User {id: $user_id})" in query
        assert "MATCH (t:Topic {name: $topic_name})" in query
        assert "MERGE (u)-[m:HAS_MASTERY]->(t)" in query
        assert params["user_id"] == "user_789"
        assert params["mastery_level"] == 0.8
        assert params["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_calculate_topic_mastery_new_topic(self, service, mock_client):
        """Test calculating mastery for new topic."""
        # Mock no existing mastery
        mock_client._neo4j_manager.execute_query_async.return_value = []
        
        mastery_data = await service._calculate_topic_mastery(
            user_id="user_123",
            topic_name="physics",
            latest_correct=True,
            latest_difficulty=DifficultyLevel.MEDIUM
        )
        
        # For new topic with one correct answer
        assert mastery_data["level"] > 0.5  # Should be above neutral
        assert mastery_data["confidence"] == 0.1  # Low confidence (1/10)
    
    @pytest.mark.asyncio
    async def test_calculate_topic_mastery_existing(self, service, mock_client):
        """Test calculating mastery for existing topic."""
        # Mock existing mastery data
        mock_client._neo4j_manager.execute_query_async.return_value = [{
            "current_level": 0.7,
            "current_confidence": 0.5,
            "total": 5,
            "correct": 4
        }]
        
        mastery_data = await service._calculate_topic_mastery(
            user_id="user_123",
            topic_name="mathematics",
            latest_correct=True,
            latest_difficulty=DifficultyLevel.HARD
        )
        
        # Should increase for correct answer on hard question
        assert mastery_data["level"] > 0.7
        assert mastery_data["confidence"] == 0.6  # 6/10 attempts
    
    @pytest.mark.asyncio
    async def test_track_performance_trends(self, service, mock_client):
        """Test tracking performance trends."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            evaluation_comment="Perfect!",
            confidence_score=1.0,
            response_time=3.5,
            topics=["math", "algebra"],
            difficulty=DifficultyLevel.EXPERT
        )
        
        # Mock streak update
        service._update_streaks = AsyncMock()
        
        success = await service._track_performance_trends(result)
        
        assert success is True
        
        # Verify performance event creation
        mock_client._neo4j_manager.execute_query_async.assert_called_once()
        query, params = mock_client._neo4j_manager.execute_query_async.call_args[0]
        
        assert "CREATE (e:PerformanceEvent {" in query
        assert params["user_id"] == "user_789"
        assert params["correct"] is True
        assert params["difficulty"] == "expert"
        assert params["response_time"] == 3.5
        assert params["confidence"] == 1.0
        assert params["topics"] == ["math", "algebra"]
        
        # Verify streaks were updated
        service._update_streaks.assert_called_once_with(result)
    
    @pytest.mark.asyncio
    async def test_update_streaks_correct(self, service, mock_client):
        """Test updating streaks for correct answer."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            evaluation_comment="Correct!"
        )
        
        await service._update_streaks(result)
        
        # Verify streak increment query
        mock_client._neo4j_manager.execute_query_async.assert_called_once()
        query, params = mock_client._neo4j_manager.execute_query_async.call_args[0]
        
        assert "SET u.current_streak = COALESCE(u.current_streak, 0) + 1" in query
        assert "u.best_streak" in query
        assert params["user_id"] == "user_789"
    
    @pytest.mark.asyncio
    async def test_update_streaks_incorrect(self, service, mock_client):
        """Test resetting streak for incorrect answer."""
        result = EvaluationResult(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=False,
            evaluation_comment="Incorrect"
        )
        
        await service._update_streaks(result)
        
        # Verify streak reset query
        mock_client._neo4j_manager.execute_query_async.assert_called_once()
        query, params = mock_client._neo4j_manager.execute_query_async.call_args[0]
        
        assert "SET u.current_streak = 0" in query
        assert params["user_id"] == "user_789"
    
    @pytest.mark.asyncio
    async def test_flush_pending_updates(self, service):
        """Test flushing pending updates."""
        # Add pending updates
        result1 = EvaluationResult(
            question_id="q_1",
            answer_id="a_1",
            user_id="user_1",
            session_id="session_1",
            correct=True,
            evaluation_comment="Good!"
        )
        result2 = EvaluationResult(
            question_id="q_2",
            answer_id="a_2",
            user_id="user_1",
            session_id="session_1",
            correct=False,
            evaluation_comment="Try again"
        )
        
        service._pending_updates = [result1, result2]
        
        # Mock process evaluation
        service._process_evaluation = AsyncMock(side_effect=[True, False])
        
        successful, failed = await service.flush_pending_updates()
        
        assert successful == 1
        assert failed == 1
        assert len(service._pending_updates) == 0
        assert service._process_evaluation.call_count == 2
    
    @pytest.mark.asyncio
    async def test_flush_pending_updates_empty(self, service):
        """Test flushing with no pending updates."""
        successful, failed = await service.flush_pending_updates()
        
        assert successful == 0
        assert failed == 0
    
    @pytest.mark.asyncio
    async def test_get_user_progress_summary(self, service, mock_client):
        """Test getting user progress summary."""
        # Mock overall stats query
        mock_client._neo4j_manager.execute_query_async.side_effect = [
            # Overall stats
            [{
                "total_questions": 100,
                "correct_answers": 75,
                "avg_response_time": 4.5,
                "current_streak": 5,
                "best_streak": 12,
                "topics_practiced": 8,
                "avg_mastery_level": 0.7
            }],
            # Recent stats
            [{
                "recent_total": 20,
                "recent_correct": 18,
                "recent_avg_time": 3.8
            }],
            # Topic breakdown
            [
                {
                    "topic": "mathematics",
                    "mastery_level": 0.85,
                    "attempts": 30,
                    "correct": 28,
                    "last_practiced": datetime.now()
                },
                {
                    "topic": "physics",
                    "mastery_level": 0.65,
                    "attempts": 25,
                    "correct": 18,
                    "last_practiced": datetime.now() - timedelta(days=2)
                }
            ]
        ]
        
        summary = await service.get_user_progress_summary("user_123")
        
        assert summary["overall"]["total_questions"] == 100
        assert summary["overall"]["correct_answers"] == 75
        assert summary["overall"]["accuracy"] == 0.75
        assert summary["overall"]["current_streak"] == 5
        assert summary["overall"]["best_streak"] == 12
        
        assert summary["recent_performance"]["questions_last_7_days"] == 20
        assert summary["recent_performance"]["correct_last_7_days"] == 18
        assert summary["recent_performance"]["recent_accuracy"] == 0.9
        
        assert len(summary["topics"]) == 2
        assert summary["topics"][0]["name"] == "mathematics"
        assert summary["topics"][0]["mastery"] == 0.85
        assert summary["topics"][0]["accuracy"] == 28/30
    
    @pytest.mark.asyncio
    async def test_get_user_progress_summary_no_client(self):
        """Test getting progress summary without client."""
        service = MemoryUpdateService(client=None)
        summary = await service.get_user_progress_summary("user_123")
        
        assert summary == {}
    
    @pytest.mark.asyncio
    async def test_get_user_progress_summary_error(self, service, mock_client):
        """Test handling error in progress summary."""
        mock_client._neo4j_manager.execute_query_async.side_effect = Exception("DB error")
        
        summary = await service.get_user_progress_summary("user_123")
        
        assert summary == {}


class TestEvaluationEventHandler:
    """Test EvaluationEventHandler class."""
    
    @pytest.fixture
    def mock_update_service(self):
        """Create mock MemoryUpdateService."""
        service = MagicMock(spec=MemoryUpdateService)
        service.record_evaluation = AsyncMock(return_value=True)
        service.flush_pending_updates = AsyncMock(return_value=(0, 0))
        return service
    
    @pytest.fixture
    def handler(self, mock_update_service):
        """Create EvaluationEventHandler with mock service."""
        return EvaluationEventHandler(mock_update_service)
    
    @pytest.mark.asyncio
    async def test_handle_evaluation_event(self, handler, mock_update_service):
        """Test handling evaluation event."""
        await handler.handle_evaluation_event(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=True,
            comment="Well done!",
            confidence=0.95,
            response_time=4.2,
            topics=["math", "algebra"],
            difficulty=DifficultyLevel.HARD
        )
        
        assert len(handler._event_queue) == 1
        
        event = handler._event_queue[0]
        assert event.question_id == "q_123"
        assert event.answer_id == "a_456"
        assert event.user_id == "user_789"
        assert event.correct is True
        assert event.confidence_score == 0.95
        assert event.response_time == 4.2
        assert event.topics == ["math", "algebra"]
        assert event.difficulty == DifficultyLevel.HARD
    
    @pytest.mark.asyncio
    async def test_handle_evaluation_event_defaults(self, handler):
        """Test handling event with default values."""
        await handler.handle_evaluation_event(
            question_id="q_123",
            answer_id="a_456",
            user_id="user_789",
            session_id="session_abc",
            correct=False,
            comment="Try again"
        )
        
        event = handler._event_queue[0]
        assert event.confidence_score == 0.0
        assert event.response_time is None
        assert event.topics == []
        assert event.difficulty == DifficultyLevel.MEDIUM
    
    @pytest.mark.asyncio
    async def test_process_batch_triggers(self, handler, mock_update_service):
        """Test batch processing triggers when batch size reached."""
        handler._batch_size = 2
        
        # Add first event
        await handler.handle_evaluation_event(
            question_id="q_1",
            answer_id="a_1",
            user_id="user_1",
            session_id="session_1",
            correct=True,
            comment="Good!"
        )
        
        # Should not trigger batch yet
        assert mock_update_service.record_evaluation.call_count == 0
        
        # Add second event - should trigger batch
        await handler.handle_evaluation_event(
            question_id="q_2",
            answer_id="a_2",
            user_id="user_1",
            session_id="session_1",
            correct=False,
            comment="Wrong"
        )
        
        # Should have processed batch
        assert mock_update_service.record_evaluation.call_count == 2
        assert len(handler._event_queue) == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_empty(self, handler, mock_update_service):
        """Test processing empty batch."""
        await handler._process_batch()
        
        assert mock_update_service.record_evaluation.call_count == 0
    
    @pytest.mark.asyncio
    async def test_flush(self, handler, mock_update_service):
        """Test flushing remaining events."""
        # Add some events
        await handler.handle_evaluation_event(
            question_id="q_1",
            answer_id="a_1",
            user_id="user_1",
            session_id="session_1",
            correct=True,
            comment="Good!"
        )
        
        # Flush
        await handler.flush()
        
        # Should process remaining events
        assert mock_update_service.record_evaluation.call_count == 1
        assert mock_update_service.flush_pending_updates.call_count == 1
        assert len(handler._event_queue) == 0


# Integration tests
class TestMemoryUpdateIntegration:
    """Integration tests for memory update system."""
    
    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self):
        """Test complete evaluation flow with mocked client."""
        # Create mock client
        mock_client = MagicMock()
        mock_client._neo4j_manager = MagicMock()
        mock_client._neo4j_manager.execute_query_async = AsyncMock()
        
        # Create service
        service = MemoryUpdateService(client=mock_client)
        
        # Mock storage update
        service.storage.update_answer_evaluation = AsyncMock(return_value=True)
        
        # Create evaluation result
        result = EvaluationResult(
            question_id="q_integration_123",
            answer_id="a_integration_456",
            user_id="user_integration",
            session_id="session_integration",
            correct=True,
            evaluation_comment="Excellent work!",
            confidence_score=0.98,
            response_time=2.5,
            topics=["computer_science", "algorithms"],
            difficulty=DifficultyLevel.EXPERT
        )
        
        # Record evaluation
        success = await service.record_evaluation(result)
        
        assert success is True
        
        # Verify all updates were made
        assert service.storage.update_answer_evaluation.called
        assert mock_client._neo4j_manager.execute_query_async.call_count >= 4  # Multiple queries
    
    @pytest.mark.asyncio
    async def test_event_handler_integration(self):
        """Test event handler with update service."""
        # Create mock client
        mock_client = MagicMock()
        mock_client._neo4j_manager = MagicMock()
        mock_client._neo4j_manager.execute_query_async = AsyncMock()
        
        # Create service and handler
        service = MemoryUpdateService(client=mock_client)
        service.storage.update_answer_evaluation = AsyncMock(return_value=True)
        
        handler = EvaluationEventHandler(service)
        handler._batch_size = 2
        
        # Handle multiple events
        await handler.handle_evaluation_event(
            question_id="q_1",
            answer_id="a_1",
            user_id="user_test",
            session_id="session_test",
            correct=True,
            comment="Correct!",
            topics=["math"]
        )
        
        await handler.handle_evaluation_event(
            question_id="q_2",
            answer_id="a_2",
            user_id="user_test",
            session_id="session_test",
            correct=False,
            comment="Try again",
            topics=["science"]
        )
        
        # Verify batch was processed
        assert service.storage.update_answer_evaluation.call_count == 2
        assert mock_client._neo4j_manager.execute_query_async.call_count >= 8  # Multiple queries per evaluation
        
        # Flush any remaining
        await handler.flush()
        
        assert len(handler._event_queue) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])