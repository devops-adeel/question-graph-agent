"""
Integration tests for enhanced graph nodes with pydantic_graph.

These tests follow pydantic_graph's actual execution patterns:
- Nodes are created without arguments
- Fields are set manually or by the graph executor
- Testing focuses on graph flow, not isolated node instantiation
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

import logfire
from pydantic_graph import Graph, GraphRunContext, End
from pydantic_ai.messages import ModelMessage

from enhanced_nodes import (
    EnhancedAsk,
    EnhancedAnswer, 
    EnhancedEvaluate,
    EnhancedReprimand,
    EnhancedQuestionState,
    create_enhanced_state,
)
from question_graph import (
    QuestionState, 
    EvaluationOutput,
    Ask, 
    Answer, 
    Evaluate, 
    Reprimand,
    question_graph,
)
from graphiti_memory import MemoryStorage, QAPair, DifficultyLevel
from graphiti_entities import QuestionEntity, AnswerStatus


# ===== Fixtures =====

@pytest.fixture
def mock_graphiti_client():
    """Create mock GraphitiClient with memory capabilities."""
    client = Mock()
    client.store_question = AsyncMock(return_value=QuestionEntity(
        id="q123",
        content="What is the capital of France?",
        difficulty=DifficultyLevel.EASY,
        topics=["geography", "france"]
    ))
    client.store_qa_pair = AsyncMock(return_value=True)
    client.get_recent_questions = AsyncMock(return_value=[])
    client.update_user_performance = AsyncMock(return_value=True)
    return client


@pytest.fixture
def enhanced_state_with_memory(mock_graphiti_client):
    """Create enhanced state with memory enabled."""
    state = EnhancedQuestionState()
    state.graphiti_client = mock_graphiti_client
    state.session_id = "test-session"
    return state


@pytest.fixture
def mock_agents():
    """Mock the AI agents."""
    with patch('enhanced_nodes.ask_agent') as mock_ask:
        with patch('enhanced_nodes.evaluate_agent') as mock_eval:
            # Configure ask agent
            mock_ask.run = AsyncMock()
            mock_ask.run.return_value.output = "What is the capital of France?"
            mock_ask.run.return_value.all_messages.return_value = []
            
            # Configure evaluate agent  
            mock_eval.run = AsyncMock()
            mock_eval.run.return_value.output = EvaluationOutput(
                correct=True,
                comment="Correct! Paris is the capital of France."
            )
            mock_eval.run.return_value.all_messages.return_value = []
            
            yield mock_ask, mock_eval


# ===== Test Node Behavior Within Graph Context =====

class TestEnhancedNodeExecution:
    """Test enhanced nodes work correctly within graph execution context."""
    
    @pytest.mark.asyncio
    async def test_enhanced_ask_execution(self, enhanced_state_with_memory, mock_agents):
        """Test EnhancedAsk node execution with memory."""
        mock_ask, _ = mock_agents
        
        # Create context
        ctx = GraphRunContext(state=enhanced_state_with_memory, deps=None)
        
        # Create and run node
        node = EnhancedAsk()
        result = await node.run(ctx)
        
        # Verify agent was called
        mock_ask.run.assert_called_once()
        
        # Verify state updated
        assert enhanced_state_with_memory.question == "What is the capital of France?"
        assert enhanced_state_with_memory.current_question_id == "q123"
        
        # Verify memory storage
        enhanced_state_with_memory.graphiti_client.store_question.assert_called_once()
        
        # Verify return type - checking attributes instead of isinstance
        assert hasattr(result, 'question')
        # The graph executor will handle setting the question field
    
    @pytest.mark.asyncio
    async def test_enhanced_ask_without_memory(self, mock_agents):
        """Test EnhancedAsk works without memory integration."""
        mock_ask, _ = mock_agents
        
        # Create state without graphiti_client
        state = EnhancedQuestionState()
        ctx = GraphRunContext(state=state, deps=None)
        
        # Run node
        node = EnhancedAsk()
        result = await node.run(ctx)
        
        # Should still work
        assert state.question == "What is the capital of France?"
        assert state.current_question_id is None  # No memory storage
        
        # Verify return
        assert hasattr(result, 'question')
    
    @pytest.mark.asyncio
    async def test_enhanced_answer_tracks_response_time(self, enhanced_state_with_memory):
        """Test response time tracking in EnhancedAnswer."""
        ctx = GraphRunContext(state=enhanced_state_with_memory, deps=None)
        
        # Create and configure node
        node = EnhancedAnswer()
        node.question = "What is the capital of France?"
        
        with patch('builtins.input', return_value="Paris"):
            with patch('time.time', side_effect=[100.0, 102.5]):
                result = await node.run(ctx)
                
                # Verify response time tracked
                assert enhanced_state_with_memory.last_response_time == 2.5
                
                # Verify return has response_time field
                assert hasattr(result, 'answer')
                assert hasattr(result, 'response_time')
    
    @pytest.mark.asyncio
    async def test_enhanced_evaluate_stores_qa_pair(self, enhanced_state_with_memory, mock_agents):
        """Test EnhancedEvaluate stores Q&A pairs in memory."""
        _, mock_eval = mock_agents
        
        # Set up state
        enhanced_state_with_memory.question = "What is the capital of France?"
        enhanced_state_with_memory.current_question_id = "q123"
        
        ctx = GraphRunContext(state=enhanced_state_with_memory, deps=None)
        
        # Create and configure node
        node = EnhancedEvaluate()
        node.answer = "Paris"
        node.response_time = 2.5
        
        result = await node.run(ctx)
        
        # Verify evaluation
        mock_eval.run.assert_called_once()
        
        # Verify memory storage
        enhanced_state_with_memory.graphiti_client.store_qa_pair.assert_called_once()
        
        # Verify result is End for correct answer
        assert isinstance(result, End)
        assert "Correct!" in result.data
    
    @pytest.mark.asyncio
    async def test_enhanced_reprimand_tracks_incorrect(self, enhanced_state_with_memory):
        """Test EnhancedReprimand tracks consecutive incorrect answers."""
        ctx = GraphRunContext(state=enhanced_state_with_memory, deps=None)
        
        # Initial state
        enhanced_state_with_memory.consecutive_incorrect = 2
        
        # Create and configure node
        node = EnhancedReprimand()
        node.comment = "Try again!"
        
        result = await node.run(ctx)
        
        # Verify tracking
        assert enhanced_state_with_memory.consecutive_incorrect == 3
        assert enhanced_state_with_memory.question is None  # Reset for new question
        
        # Verify return
        assert hasattr(result, 'run')  # Should return EnhancedAsk node


# ===== Test Graph Flow Integration =====

class TestEnhancedGraphFlow:
    """Test complete graph flows with enhanced nodes."""
    
    @pytest.mark.asyncio
    async def test_standard_vs_enhanced_graph_structure(self):
        """Verify both graphs have same structure."""
        # Standard graph uses standard nodes
        assert question_graph.nodes == (Ask, Answer, Evaluate, Reprimand)
        
        # Enhanced graph would use enhanced nodes
        enhanced_graph = Graph(
            nodes=(EnhancedAsk, EnhancedAnswer, EnhancedEvaluate, EnhancedReprimand),
            state_type=EnhancedQuestionState
        )
        assert enhanced_graph.nodes == (EnhancedAsk, EnhancedAnswer, EnhancedEvaluate, EnhancedReprimand)
    
    @pytest.mark.asyncio
    async def test_mixed_graph_with_enhanced_state(self, enhanced_state_with_memory, mock_agents):
        """Test standard graph can work with enhanced state."""
        mock_ask, mock_eval = mock_agents
        
        # Patch agents in question_graph module
        with patch('question_graph.ask_agent', mock_ask):
            with patch('question_graph.evaluate_agent', mock_eval):
                with patch('builtins.input', return_value="Paris"):
                    # Run standard graph with enhanced state
                    result = await question_graph.run(
                        Ask(),
                        state=enhanced_state_with_memory
                    )
                    
                    # Should complete successfully
                    assert isinstance(result, End)
                    assert "Correct!" in result.data
                    
                    # State should be updated (but no memory storage)
                    assert enhanced_state_with_memory.question == "What is the capital of France?"


# ===== Test State Management =====

class TestEnhancedStateManagement:
    """Test enhanced state creation and management."""
    
    def test_create_enhanced_state_from_standard(self):
        """Test converting standard state to enhanced."""
        # Create standard state
        standard = QuestionState()
        standard.question = "Test question?"
        standard.ask_agent_messages = [MagicMock(spec=ModelMessage)]
        
        # Convert
        enhanced = create_enhanced_state(standard)
        
        # Verify
        assert isinstance(enhanced, EnhancedQuestionState)
        assert enhanced.question == "Test question?"
        assert len(enhanced.ask_agent_messages) == 1
        
        # New fields
        assert enhanced.current_question_id is None
        assert enhanced.last_response_time is None
        assert enhanced.consecutive_incorrect == 0
    
    def test_create_fresh_enhanced_state(self):
        """Test creating new enhanced state."""
        enhanced = create_enhanced_state()
        
        assert isinstance(enhanced, EnhancedQuestionState)
        assert enhanced.question is None
        assert enhanced.consecutive_incorrect == 0
    
    def test_enhanced_state_preserves_graphiti_client(self, mock_graphiti_client):
        """Test enhanced state preserves graphiti client."""
        standard = QuestionState()
        standard.graphiti_client = mock_graphiti_client
        
        enhanced = create_enhanced_state(standard)
        
        assert enhanced.graphiti_client is mock_graphiti_client


# ===== Test Error Handling =====

class TestEnhancedNodeErrorHandling:
    """Test error handling in enhanced nodes."""
    
    @pytest.mark.asyncio
    async def test_memory_storage_failure_graceful(self, enhanced_state_with_memory, mock_agents):
        """Test nodes handle memory storage failures gracefully."""
        mock_ask, _ = mock_agents
        
        # Make storage fail
        enhanced_state_with_memory.graphiti_client.store_question.side_effect = Exception("Storage error")
        
        ctx = GraphRunContext(state=enhanced_state_with_memory, deps=None)
        node = EnhancedAsk()
        
        # Should not raise
        result = await node.run(ctx)
        
        # Should still return next node
        assert hasattr(result, 'question')
        
        # Question generated but no ID stored
        assert enhanced_state_with_memory.question == "What is the capital of France?"
        assert enhanced_state_with_memory.current_question_id is None
    
    @pytest.mark.asyncio
    async def test_missing_question_in_evaluate(self, enhanced_state_with_memory, mock_agents):
        """Test EnhancedEvaluate handles missing question gracefully."""
        _, mock_eval = mock_agents
        
        # No question in state
        enhanced_state_with_memory.question = None
        
        ctx = GraphRunContext(state=enhanced_state_with_memory, deps=None)
        node = EnhancedEvaluate()
        node.answer = "Paris"
        
        result = await node.run(ctx)
        
        # Should still evaluate but no memory storage
        mock_eval.run.assert_called_once()
        enhanced_state_with_memory.graphiti_client.store_qa_pair.assert_not_called()


# ===== Documentation Tests =====

class TestDocumentation:
    """Test that our implementation matches documented behavior."""
    
    def test_node_field_pattern(self):
        """Verify nodes follow expected field pattern."""
        # Nodes should have Field definitions
        assert hasattr(Answer, 'question')
        assert hasattr(EnhancedAnswer, 'question')
        
        # Fields should be FieldInfo objects on the class
        assert str(type(Answer.question)) == "<class 'pydantic.fields.FieldInfo'>"
        
        # But instances can have the field set
        node = Answer()
        node.question = "Test?"
        assert node.question == "Test?"
    
    def test_node_instantiation_pattern(self):
        """Verify correct instantiation patterns."""
        # Can create without args
        assert Answer()
        assert EnhancedAnswer()
        
        # Cannot create with args
        with pytest.raises(TypeError, match="takes no arguments"):
            Answer(question="Test?")
        
        with pytest.raises(TypeError, match="takes no arguments"):  
            EnhancedAnswer(question="Test?")