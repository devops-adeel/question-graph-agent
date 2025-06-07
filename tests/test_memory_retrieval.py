"""
Tests for memory retrieval functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from memory_retrieval import (
    MemoryRetrieval,
    QuestionSelector,
)
from enhanced_question_agent import (
    EnhancedQuestionAgent,
    QuestionGenerationContext,
    MemoryAwareQuestionNode,
)
from memory_integration import (
    MemoryEnhancedAsk,
    MemoryContextBuilder,
    MemoryStateEnhancer,
)
from graphiti_client import GraphitiClient
from graphiti_entities import (
    QuestionEntity,
    DifficultyLevel,
    AnswerStatus,
    UserEntity,
)
from question_graph import QuestionState, Answer
from pydantic_graph import GraphRunContext


class TestMemoryRetrieval:
    """Test MemoryRetrieval class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock GraphitiClient."""
        client = Mock(spec=GraphitiClient)
        client._neo4j_manager = Mock()
        client._neo4j_manager.execute_query_async = AsyncMock()
        return client
    
    @pytest.fixture
    def retrieval(self, mock_client):
        """Create MemoryRetrieval with mock client."""
        return MemoryRetrieval(client=mock_client)
    
    @pytest.mark.asyncio
    async def test_get_asked_questions(self, retrieval, mock_client):
        """Test retrieving previously asked questions."""
        mock_result = [
            {
                "q": {
                    "id": "q1",
                    "content": "What is Python?",
                    "difficulty": "medium",
                    "topics": ["programming", "python"],
                    "asked_count": 5,
                    "correct_rate": 0.8
                }
            },
            {
                "q": {
                    "id": "q2",
                    "content": "What is 2+2?",
                    "difficulty": "easy",
                    "topics": ["math"],
                    "asked_count": 10,
                    "correct_rate": 0.95
                }
            }
        ]
        mock_client._neo4j_manager.execute_query_async.return_value = mock_result
        
        questions = await retrieval.get_asked_questions("user1", limit=10)
        
        assert len(questions) == 2
        assert questions[0].content == "What is Python?"
        assert questions[0].difficulty == DifficultyLevel.MEDIUM
        assert questions[1].content == "What is 2+2?"
        assert "math" in questions[1].topics
        
        # Verify query was called
        mock_client._neo4j_manager.execute_query_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_performance(self, retrieval, mock_client):
        """Test getting user performance metrics."""
        mock_result = [{
            "performance": {
                "total_questions": 20,
                "correct_answers": 15,
                "accuracy": 0.75,
                "avg_response_time": 5.5,
                "difficulty_performance": {
                    "easy": {"total": 10, "correct": 9},
                    "medium": {"total": 8, "correct": 5},
                    "hard": {"total": 2, "correct": 1}
                },
                "recent_performance": {"last_5_correct": True}
            }
        }]
        mock_client._neo4j_manager.execute_query_async.return_value = mock_result
        
        performance = await retrieval.get_user_performance("user1")
        
        assert performance["total_questions"] == 20
        assert performance["accuracy"] == 0.75
        assert performance["recommended_difficulty"] == DifficultyLevel.MEDIUM
        assert performance["difficulty_performance"]["easy"]["correct"] == 9
    
    @pytest.mark.asyncio
    async def test_get_weak_topics(self, retrieval, mock_client):
        """Test getting weak topics for a user."""
        mock_result = [
            {"topic": "calculus", "accuracy": 0.3},
            {"topic": "algebra", "accuracy": 0.4},
            {"topic": "geometry", "accuracy": 0.45}
        ]
        mock_client._neo4j_manager.execute_query_async.return_value = mock_result
        
        weak_topics = await retrieval.get_weak_topics("user1", threshold=0.5)
        
        assert len(weak_topics) == 3
        assert weak_topics[0] == ("calculus", 0.3)
        assert weak_topics[1] == ("algebra", 0.4)
        assert weak_topics[2] == ("geometry", 0.45)
    
    @pytest.mark.asyncio
    async def test_get_recommended_questions(self, retrieval, mock_client):
        """Test getting personalized question recommendations."""
        # Mock performance data
        performance_result = [{
            "performance": {
                "total_questions": 10,
                "correct_answers": 6,
                "accuracy": 0.6,
                "avg_response_time": 5.0,
                "difficulty_performance": {
                    "easy": {"total": 5, "correct": 4},
                    "medium": {"total": 5, "correct": 2},
                    "hard": {"total": 0, "correct": 0}
                },
                "recent_performance": {"last_5_correct": False}
            }
        }]
        
        # Mock asked questions
        asked_result = [{
            "q": {
                "id": "q_old",
                "content": "Old question",
                "difficulty": "easy",
                "topics": ["basic"]
            }
        }]
        
        # Mock weak topics
        weak_topics_result = [
            {"topic": "algebra", "accuracy": 0.4}
        ]
        
        # Mock topic questions
        topic_questions_result = [{
            "q": {
                "id": "q_algebra",
                "content": "Solve for x: 2x + 5 = 13",
                "difficulty": "medium",
                "topics": ["algebra"],
                "asked_count": 0,
                "correct_rate": 0.0
            }
        }]
        
        # Set up mock returns
        mock_client._neo4j_manager.execute_query_async.side_effect = [
            performance_result,
            asked_result,
            weak_topics_result,
            topic_questions_result,
            []  # General questions
        ]
        
        recommendations = await retrieval.get_recommended_questions("user1", count=3)
        
        assert len(recommendations) > 0
        assert recommendations[0].content == "Solve for x: 2x + 5 = 13"
        assert recommendations[0].difficulty == DifficultyLevel.MEDIUM
    
    @pytest.mark.asyncio
    async def test_calculate_recommended_difficulty(self, retrieval):
        """Test difficulty recommendation calculation."""
        # Test case 1: Struggling with easy questions
        perf1 = {
            "easy": {"total": 10, "correct": 5},
            "medium": {"total": 5, "correct": 1},
            "hard": {"total": 0, "correct": 0}
        }
        assert retrieval._calculate_recommended_difficulty(perf1) == DifficultyLevel.EASY
        
        # Test case 2: Doing well with easy, struggling with medium
        perf2 = {
            "easy": {"total": 10, "correct": 9},
            "medium": {"total": 10, "correct": 4},
            "hard": {"total": 2, "correct": 0}
        }
        assert retrieval._calculate_recommended_difficulty(perf2) == DifficultyLevel.MEDIUM
        
        # Test case 3: Doing well overall
        perf3 = {
            "easy": {"total": 10, "correct": 9},
            "medium": {"total": 10, "correct": 8},
            "hard": {"total": 5, "correct": 2}
        }
        assert retrieval._calculate_recommended_difficulty(perf3) == DifficultyLevel.HARD


class TestEnhancedQuestionAgent:
    """Test EnhancedQuestionAgent class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock GraphitiClient."""
        client = Mock(spec=GraphitiClient)
        client._neo4j_manager = Mock()
        client._neo4j_manager.execute_query_async = AsyncMock(return_value=[])
        return client
    
    @pytest.fixture
    def agent(self, mock_client):
        """Create EnhancedQuestionAgent with mock client."""
        return EnhancedQuestionAgent(graphiti_client=mock_client)
    
    @pytest.mark.asyncio
    async def test_generate_question_with_memory(self, agent, mock_client):
        """Test question generation with memory context."""
        context = QuestionGenerationContext(
            user_id="user1",
            session_id="session1",
            prefer_weak_topics=True
        )
        
        # Mock selector to return None (force generation)
        with patch.object(agent.selector, 'select_next_question', return_value=None):
            # Mock retrieval methods
            with patch.object(agent.retrieval, 'get_user_performance', return_value={
                "accuracy": 0.7,
                "recommended_difficulty": DifficultyLevel.MEDIUM
            }):
                with patch.object(agent.retrieval, 'get_asked_questions', return_value=[]):
                    with patch.object(agent.retrieval, 'get_weak_topics', return_value=[
                        ("algebra", 0.4),
                        ("geometry", 0.45)
                    ]):
                        # Mock the agent run
                        with patch.object(agent.agent, 'run') as mock_run:
                            mock_run.return_value = Mock(data="What is x if 3x = 9?")
                            
                            question = await agent.generate_question(context)
                            
                            assert question == "What is x if 3x = 9?"
                            mock_run.assert_called_once()
                            
                            # Verify prompt includes weak topics
                            prompt = mock_run.call_args[0][0]
                            assert "algebra" in prompt
                            assert "geometry" in prompt
    
    @pytest.mark.asyncio
    async def test_generate_question_fallback(self):
        """Test fallback generation without memory."""
        agent = EnhancedQuestionAgent(graphiti_client=None)
        context = QuestionGenerationContext(user_id="user1")
        
        with patch.object(agent.agent, 'run') as mock_run:
            mock_run.return_value = Mock(data="What is 2 + 2?")
            
            question = await agent.generate_question(context)
            
            assert question == "What is 2 + 2?"


class TestMemoryIntegration:
    """Test memory integration components."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock GraphitiClient."""
        client = Mock(spec=GraphitiClient)
        client._neo4j_manager = Mock()
        client._neo4j_manager.execute_query_async = AsyncMock(return_value=[])
        return client
    
    @pytest.fixture
    def mock_state(self, mock_client):
        """Create mock QuestionState with GraphitiClient."""
        state = QuestionState()
        state.graphiti_client = mock_client
        state.current_user = UserEntity(
            id="user1",
            session_id="session1",
            total_questions=0,
            correct_answers=0,
            average_response_time=0.0
        )
        state.session_id = "session1"
        state.consecutive_incorrect = 0
        return state
    
    @pytest.fixture
    def mock_context(self, mock_state):
        """Create mock GraphRunContext."""
        ctx = Mock(spec=GraphRunContext)
        ctx.state = mock_state
        return ctx
    
    @pytest.mark.asyncio
    async def test_memory_enhanced_ask(self, mock_context, mock_client):
        """Test MemoryEnhancedAsk node."""
        node = MemoryEnhancedAsk()
        
        # Mock the enhanced agent
        with patch('memory_integration.EnhancedQuestionAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.generate_question = AsyncMock(return_value="What is Python?")
            mock_agent_class.return_value = mock_agent
            
            # Mock memory storage
            with patch('memory_integration.MemoryStorage') as mock_storage_class:
                mock_storage = Mock()
                mock_storage.store_question_only = AsyncMock(return_value=Mock(id="q123"))
                mock_storage_class.return_value = mock_storage
                
                result = await node.run(mock_context)
                
                assert isinstance(result, Answer)
                assert result.question == "What is Python?"
                assert mock_context.state.question == "What is Python?"
                
                # Verify question was stored
                mock_storage.store_question_only.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_context_builder(self, mock_client):
        """Test MemoryContextBuilder."""
        builder = MemoryContextBuilder(graphiti_client=mock_client)
        
        # Mock retrieval methods
        with patch.object(builder.retrieval, 'get_user_performance', return_value={
            "accuracy": 0.75,
            "total_questions": 20
        }):
            with patch.object(builder.retrieval, 'get_weak_topics', return_value=[
                ("algebra", 0.4)
            ]):
                with patch.object(builder.retrieval, 'get_asked_questions', return_value=[
                    Mock(id="q1"), Mock(id="q2")
                ]):
                    context = await builder.build_question_context("user1", "session1")
                    
                    assert context["user_id"] == "user1"
                    assert context["session_id"] == "session1"
                    assert context["user_performance"]["accuracy"] == 0.75
                    assert len(context["weak_topics"]) == 1
                    assert context["recent_question_count"] == 2
    
    def test_memory_state_enhancer(self):
        """Test MemoryStateEnhancer."""
        state = QuestionState()
        mock_client = Mock(spec=GraphitiClient)
        
        enhanced_state = MemoryStateEnhancer.enhance_state(
            state,
            graphiti_client=mock_client,
            user_id="user1",
            session_id="session1"
        )
        
        assert hasattr(enhanced_state, 'graphiti_client')
        assert enhanced_state.graphiti_client == mock_client
        assert hasattr(enhanced_state, 'current_user')
        assert enhanced_state.current_user.id == "user1"
        assert hasattr(enhanced_state, 'session_id')
        assert enhanced_state.session_id == "session1"
        assert hasattr(enhanced_state, 'consecutive_incorrect')
        assert enhanced_state.consecutive_incorrect == 0