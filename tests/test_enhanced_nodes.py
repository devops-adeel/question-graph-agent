"""
Tests for enhanced graph nodes with memory storage integration.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from datetime import datetime

import logfire
from pydantic_graph import GraphRunContext

from enhanced_nodes import (
    EnhancedAsk,
    EnhancedAnswer,
    EnhancedEvaluate,
    EnhancedReprimand,
    create_enhanced_state,
)
from question_graph import QuestionState, Answer, Evaluate, EvaluationOutput, format_as_xml
from graphiti_memory import MemoryStorage, QAPair, DifficultyLevel
from graphiti_entities import QuestionEntity, AnswerStatus


@pytest.fixture
def mock_graphiti_client():
    """Create mock GraphitiClient."""
    client = Mock()
    client.store_question = AsyncMock(return_value=True)
    client.store_answer = AsyncMock(return_value=True)
    client.create_qa_episode = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_ask_agent():
    """Mock ask_agent."""
    with patch('enhanced_nodes.ask_agent') as mock:
        result = Mock()
        result.output = "What is the capital of France?"
        result.all_messages.return_value = []
        mock.run = AsyncMock(return_value=result)
        yield mock


@pytest.fixture
def mock_evaluate_agent():
    """Mock evaluate_agent."""
    with patch('enhanced_nodes.evaluate_agent') as mock:
        result = Mock()
        result.output = Mock(correct=True, comment="Correct!")
        result.all_messages.return_value = []
        mock.run = AsyncMock(return_value=result)
        yield mock


@pytest.fixture
def question_state_with_memory(mock_graphiti_client):
    """Create QuestionState with GraphitiClient."""
    state = QuestionState()
    state.graphiti_client = mock_graphiti_client
    state.current_user = Mock(id="user123")
    state.session_id = "session456"
    return state


@pytest.fixture
def mock_logfire_span():
    """Mock logfire span context."""
    with patch('enhanced_nodes.logfire.span') as mock_span:
        span = MagicMock()
        span.__enter__ = Mock(return_value=span)
        span.__exit__ = Mock(return_value=None)
        span.set_attribute = Mock()
        mock_span.return_value = span
        yield span


class TestEnhancedAsk:
    """Test EnhancedAsk node."""
    
    @pytest.mark.asyncio
    async def test_enhanced_ask_with_memory_success(
        self, mock_ask_agent, mock_graphiti_client, question_state_with_memory, mock_logfire_span
    ):
        """Test question generation and successful storage in memory."""
        # Setup
        mock_question_entity = QuestionEntity(
            id="q_123",
            content="What is the capital of France?",
            difficulty=DifficultyLevel.MEDIUM,
            topics=["geography", "france"]
        )
        
        with patch('enhanced_nodes.MemoryStorage') as MockMemoryStorage:
            mock_storage = Mock()
            mock_storage.store_question_only = AsyncMock(return_value=mock_question_entity)
            MockMemoryStorage.return_value = mock_storage
            
            # Create context
            ctx = GraphRunContext(state=question_state_with_memory, deps=None)
            
            # Run node
            node = EnhancedAsk()
            result = await node.run(ctx)
            
            # Verify ask_agent was called
            mock_ask_agent.run.assert_called_once_with(
                'Ask a simple question with a single correct answer.',
                message_history=ctx.state.ask_agent_messages,
            )
            
            # Verify memory storage was attempted
            MockMemoryStorage.assert_called_once_with(client=mock_graphiti_client)
            mock_storage.store_question_only.assert_called_once_with(
                question="What is the capital of France?",
                difficulty=DifficultyLevel.MEDIUM
            )
            
            # Verify state updated
            assert ctx.state.question == "What is the capital of France?"
            assert ctx.state.current_question_id == "q_123"
            
            # Verify span attributes
            mock_logfire_span.set_attribute.assert_any_call('memory_enabled', True)
            mock_logfire_span.set_attribute.assert_any_call('question_stored', True)
            mock_logfire_span.set_attribute.assert_any_call('question_id', 'q_123')
            
            # Verify result
            assert isinstance(result, Answer)
            assert result.question == "What is the capital of France?"
    
    @pytest.mark.asyncio
    async def test_enhanced_ask_without_memory(self, mock_ask_agent, mock_logfire_span):
        """Test graceful operation without GraphitiClient."""
        # Setup state without graphiti_client
        state = QuestionState()
        ctx = GraphRunContext(state=state, deps=None)
        
        # Run node
        node = EnhancedAsk()
        result = await node.run(ctx)
        
        # Verify question generation still works
        mock_ask_agent.run.assert_called_once()
        assert ctx.state.question == "What is the capital of France?"
        
        # Verify no memory storage attempted
        mock_logfire_span.set_attribute.assert_any_call('memory_enabled', False)
        
        # Verify result
        assert isinstance(result, Answer)
        assert result.question == "What is the capital of France?"
    
    @pytest.mark.asyncio
    async def test_enhanced_ask_memory_storage_failure(
        self, mock_ask_agent, mock_graphiti_client, question_state_with_memory, mock_logfire_span
    ):
        """Test error handling when memory storage fails."""
        with patch('enhanced_nodes.MemoryStorage') as MockMemoryStorage:
            mock_storage = Mock()
            mock_storage.store_question_only = AsyncMock(side_effect=Exception("Storage failed"))
            MockMemoryStorage.return_value = mock_storage
            
            with patch('enhanced_nodes.logger') as mock_logger:
                ctx = GraphRunContext(state=question_state_with_memory, deps=None)
                
                # Run node
                node = EnhancedAsk()
                result = await node.run(ctx)
                
                # Verify question still generated
                assert ctx.state.question == "What is the capital of France?"
                
                # Verify error logged
                mock_logger.error.assert_called_once()
                assert "Failed to store question in memory" in mock_logger.error.call_args[0][0]
                
                # Verify error in span
                mock_logfire_span.set_attribute.assert_any_call('memory_error', 'Storage failed')
                
                # Verify result still returned
                assert isinstance(result, Answer)


class TestEnhancedAnswer:
    """Test EnhancedAnswer node."""
    
    @pytest.mark.asyncio
    async def test_enhanced_answer_tracks_response_time(self, mock_logfire_span):
        """Test response time tracking."""
        state = QuestionState()
        ctx = GraphRunContext(state=state, deps=None)
        
        # Mock input and time
        with patch('builtins.input', return_value="Paris"):
            with patch('time.time', side_effect=[100.0, 102.5]):  # 2.5 second response
                node = EnhancedAnswer(question="What is the capital of France?")
                result = await node.run(ctx)
                
                # Verify response time calculated
                assert result.response_time == 2.5
                
                # Verify state updated
                assert ctx.state.last_response_time == 2.5
                
                # Verify span attributes
                mock_logfire_span.set_attribute.assert_any_call('response_time', 2.5)
                mock_logfire_span.set_attribute.assert_any_call('user_answer', 'Paris')
    
    @pytest.mark.asyncio
    async def test_enhanced_answer_returns_enhanced_evaluate(self, mock_logfire_span):
        """Test correct node transition with response time."""
        state = QuestionState()
        ctx = GraphRunContext(state=state, deps=None)
        
        with patch('builtins.input', return_value="Paris"):
            with patch('time.time', side_effect=[100.0, 101.0]):
                node = EnhancedAnswer(question="What is the capital of France?")
                result = await node.run(ctx)
                
                # Verify correct return type
                assert isinstance(result, EnhancedEvaluate)
                assert result.answer == "Paris"
                assert result.response_time == 1.0


class TestEnhancedEvaluate:
    """Test EnhancedEvaluate node."""
    
    @pytest.mark.asyncio
    async def test_enhanced_evaluate_correct_answer_with_memory(
        self, mock_evaluate_agent, mock_graphiti_client, question_state_with_memory, mock_logfire_span
    ):
        """Test Q&A pair storage for correct answers."""
        # Setup state
        question_state_with_memory.question = "What is the capital of France?"
        question_state_with_memory.current_question_id = "q_123"
        ctx = GraphRunContext(state=question_state_with_memory, deps=None)
        
        # Mock evaluate agent for correct answer
        mock_evaluate_agent.run.return_value.output.correct = True
        mock_evaluate_agent.run.return_value.output.comment = "Correct! Paris is the capital."
        
        with patch('enhanced_nodes.MemoryStorage') as MockMemoryStorage:
            mock_storage = Mock()
            mock_storage.store_qa_pair = AsyncMock(return_value=True)
            MockMemoryStorage.return_value = mock_storage
            
            with patch('enhanced_nodes.format_as_xml', return_value="<xml>formatted</xml>"):
                # Run node
                node = EnhancedEvaluate(answer="Paris", response_time=2.5)
                result = await node.run(ctx)
                
                # Verify Q&A pair created correctly
                mock_storage.store_qa_pair.assert_called_once()
                qa_pair = mock_storage.store_qa_pair.call_args[0][0]
                assert qa_pair.question == "What is the capital of France?"
                assert qa_pair.answer == "Paris"
                assert qa_pair.user_id == "user123"
                assert qa_pair.session_id == "session456"
                assert qa_pair.correct is True
                assert qa_pair.response_time == 2.5
                assert qa_pair.question_id == "q_123"
                assert qa_pair.confidence_score == 0.8
                
                # Verify End returned
                from pydantic_graph import End
                assert isinstance(result, End)
                assert result.output == "Correct! Paris is the capital."
    
    @pytest.mark.asyncio
    async def test_enhanced_evaluate_incorrect_answer_with_memory(
        self, mock_evaluate_agent, mock_graphiti_client, question_state_with_memory, mock_logfire_span
    ):
        """Test Q&A pair storage for incorrect answers."""
        question_state_with_memory.question = "What is the capital of France?"
        ctx = GraphRunContext(state=question_state_with_memory, deps=None)
        
        # Mock evaluate agent for incorrect answer
        mock_evaluate_agent.run.return_value.output.correct = False
        mock_evaluate_agent.run.return_value.output.comment = "Incorrect. The capital is Paris."
        
        with patch('enhanced_nodes.MemoryStorage') as MockMemoryStorage:
            mock_storage = Mock()
            mock_storage.store_qa_pair = AsyncMock(return_value=True)
            MockMemoryStorage.return_value = mock_storage
            
            with patch('enhanced_nodes.format_as_xml', return_value="<xml>formatted</xml>"):
                # Run node
                node = EnhancedEvaluate(answer="London", response_time=3.0)
                result = await node.run(ctx)
                
                # Verify Q&A pair has correct=False and lower confidence
                qa_pair = mock_storage.store_qa_pair.call_args[0][0]
                assert qa_pair.correct is False
                assert qa_pair.confidence_score == 0.3
                
                # Verify EnhancedReprimand returned
                assert isinstance(result, EnhancedReprimand)
                assert result.comment == "Incorrect. The capital is Paris."
    
    @pytest.mark.asyncio
    async def test_enhanced_evaluate_uses_stored_question_id(
        self, mock_evaluate_agent, mock_graphiti_client, question_state_with_memory, mock_logfire_span
    ):
        """Test reuse of question_id from EnhancedAsk."""
        question_state_with_memory.question = "Test question?"
        question_state_with_memory.current_question_id = "existing_q_id"
        ctx = GraphRunContext(state=question_state_with_memory, deps=None)
        
        with patch('enhanced_nodes.MemoryStorage') as MockMemoryStorage:
            mock_storage = Mock()
            mock_storage.store_qa_pair = AsyncMock(return_value=True)
            MockMemoryStorage.return_value = mock_storage
            
            with patch('enhanced_nodes.format_as_xml', return_value="<xml>formatted</xml>"):
                node = EnhancedEvaluate(answer="Test answer")
                await node.run(ctx)
                
                # Verify existing question_id used
                qa_pair = mock_storage.store_qa_pair.call_args[0][0]
                assert qa_pair.question_id == "existing_q_id"
    
    @pytest.mark.asyncio
    async def test_enhanced_evaluate_no_question_error(self, mock_logfire_span):
        """Test error handling for missing question."""
        state = QuestionState()
        state.question = None
        ctx = GraphRunContext(state=state, deps=None)
        
        node = EnhancedEvaluate(answer="Some answer")
        
        with pytest.raises(ValueError, match="No question available to evaluate against"):
            await node.run(ctx)
    
    @pytest.mark.asyncio
    async def test_enhanced_evaluate_memory_failure_continues(
        self, mock_evaluate_agent, mock_graphiti_client, question_state_with_memory, mock_logfire_span
    ):
        """Test evaluation continues despite storage failure."""
        question_state_with_memory.question = "Test question?"
        ctx = GraphRunContext(state=question_state_with_memory, deps=None)
        
        mock_evaluate_agent.run.return_value.output.correct = True
        mock_evaluate_agent.run.return_value.output.comment = "Correct!"
        
        with patch('enhanced_nodes.MemoryStorage') as MockMemoryStorage:
            mock_storage = Mock()
            mock_storage.store_qa_pair = AsyncMock(side_effect=Exception("Storage error"))
            MockMemoryStorage.return_value = mock_storage
            
            with patch('enhanced_nodes.logger') as mock_logger:
                with patch('enhanced_nodes.format_as_xml', return_value="<xml>formatted</xml>"):
                    node = EnhancedEvaluate(answer="Test answer")
                    result = await node.run(ctx)
                    
                    # Verify error logged
                    mock_logger.error.assert_called_once()
                    assert "Failed to store Q&A pair" in mock_logger.error.call_args[0][0]
                    
                    # Verify error in span
                    mock_logfire_span.set_attribute.assert_any_call('memory_error', 'Storage error')
                    
                    # Verify End still returned
                    from pydantic_graph import End
                    assert isinstance(result, End)


class TestEnhancedReprimand:
    """Test EnhancedReprimand node."""
    
    @pytest.mark.asyncio
    async def test_enhanced_reprimand_tracks_consecutive_incorrect(self, mock_logfire_span):
        """Test consecutive incorrect answer tracking."""
        state = QuestionState()
        state.consecutive_incorrect = 2
        ctx = GraphRunContext(state=state, deps=None)
        
        with patch('enhanced_nodes.logfire.warning') as mock_warning:
            node = EnhancedReprimand(comment="Try again!")
            result = await node.run(ctx)
            
            # Verify count incremented
            assert ctx.state.consecutive_incorrect == 3
            
            # Verify warning logged at threshold
            mock_warning.assert_called_once_with(
                'User struggling with questions',
                consecutive_incorrect=3
            )
            
            # Verify span attribute
            mock_logfire_span.set_attribute.assert_any_call('consecutive_incorrect', 3)
    
    @pytest.mark.asyncio
    async def test_enhanced_reprimand_initializes_tracking(self, mock_logfire_span):
        """Test initialization of tracking when not present."""
        state = QuestionState()
        # Don't set consecutive_incorrect
        ctx = GraphRunContext(state=state, deps=None)
        
        node = EnhancedReprimand(comment="Try again!")
        result = await node.run(ctx)
        
        # Verify initialized to 1
        assert ctx.state.consecutive_incorrect == 1
        
        # Verify no warning (below threshold)
        mock_logfire_span.set_attribute.assert_any_call('consecutive_incorrect', 1)
    
    @pytest.mark.asyncio
    async def test_enhanced_reprimand_resets_question(self, mock_logfire_span):
        """Test question reset for new round."""
        state = QuestionState()
        state.question = "Old question?"
        ctx = GraphRunContext(state=state, deps=None)
        
        node = EnhancedReprimand(comment="Wrong answer")
        result = await node.run(ctx)
        
        # Verify question cleared
        assert ctx.state.question is None
        
        # Verify EnhancedAsk returned
        assert isinstance(result, EnhancedAsk)


class TestCreateEnhancedState:
    """Test create_enhanced_state function."""
    
    def test_create_enhanced_state_from_scratch(self):
        """Test state enhancement from None."""
        state = create_enhanced_state(None)
        
        # Verify QuestionState created
        assert isinstance(state, QuestionState)
        
        # Verify all enhanced fields added
        assert hasattr(state, 'current_question_id')
        assert state.current_question_id is None
        
        assert hasattr(state, 'last_response_time')
        assert state.last_response_time is None
        
        assert hasattr(state, 'consecutive_incorrect')
        assert state.consecutive_incorrect == 0
    
    def test_create_enhanced_state_preserves_existing(self):
        """Test existing state preserved."""
        # Create state with existing data
        original_state = QuestionState()
        original_state.question = "Existing question?"
        original_state.ask_agent_messages = ["msg1", "msg2"]
        
        # Enhance it
        enhanced = create_enhanced_state(original_state)
        
        # Verify original data preserved
        assert enhanced.question == "Existing question?"
        assert enhanced.ask_agent_messages == ["msg1", "msg2"]
        
        # Verify enhanced fields added
        assert hasattr(enhanced, 'current_question_id')
        assert hasattr(enhanced, 'last_response_time')
        assert hasattr(enhanced, 'consecutive_incorrect')
    
    def test_create_enhanced_state_no_duplicate_fields(self):
        """Test no duplicate field creation."""
        # Create state with some enhanced fields already
        state = QuestionState()
        state.current_question_id = "existing_id"
        state.consecutive_incorrect = 5
        
        # Enhance it
        enhanced = create_enhanced_state(state)
        
        # Verify existing values preserved
        assert enhanced.current_question_id == "existing_id"
        assert enhanced.consecutive_incorrect == 5
        
        # Verify missing field added
        assert hasattr(enhanced, 'last_response_time')
        assert enhanced.last_response_time is None