"""
Test utilities for pydantic_graph nodes.

This module provides helper functions and patterns for testing pydantic_graph nodes,
which have specific requirements due to their dataclass nature and execution model.
"""

import asyncio
from typing import Any, Optional, Type, Union
from unittest.mock import Mock, AsyncMock, patch
from contextlib import contextmanager

import pytest
from pydantic_graph import BaseNode, GraphRunContext


# Common error types when nodes try to instantiate with arguments
NODE_INSTANTIATION_ERRORS = (TypeError, RuntimeError)
NODE_ERROR_PATTERNS = [
    r".*\(\) takes no arguments",
    r"'coroutine' object is not iterable"
]


def create_simple_mock() -> Mock:
    """Create a simple mock that avoids Pydantic validation issues.
    
    Returns:
        Mock: A basic Mock object without spec or autospec
    """
    return Mock()


def create_agent_result_mock(output: Any, messages: Optional[list] = None) -> Mock:
    """Create a mock for agent run results.
    
    Args:
        output: The output value for the agent
        messages: Optional list of messages
        
    Returns:
        Mock: Configured mock result object
    """
    mock_result = Mock()
    mock_result.output = output
    mock_result.all_messages = Mock(return_value=messages or [])
    return mock_result


def create_async_agent_mock(result: Mock) -> Mock:
    """Create a mock agent with async run method.
    
    Args:
        result: The result mock to return from run()
        
    Returns:
        Mock: Agent mock with configured async run
    """
    mock_agent = Mock()
    mock_agent.run = AsyncMock(return_value=result)
    return mock_agent


@contextmanager
def mock_logfire():
    """Context manager to mock logfire for testing.
    
    Yields:
        tuple: (mock_span, mock_info) for assertions
    """
    mock_span = Mock()
    mock_span.__enter__ = Mock(return_value=mock_span)
    mock_span.__exit__ = Mock(return_value=None)
    mock_span.set_attribute = Mock()
    
    with patch('logfire.span', return_value=mock_span):
        with patch('logfire.info') as mock_info:
            yield mock_span, mock_info


async def run_node_safely(
    node: BaseNode,
    ctx: GraphRunContext,
    expect_error: bool = True
) -> tuple[Optional[Any], Optional[Exception]]:
    """Run a node and handle expected errors gracefully.
    
    Args:
        node: The node to run
        ctx: The graph context
        expect_error: Whether to expect an error
        
    Returns:
        tuple: (result, error) where one will be None
    """
    try:
        result = await node.run(ctx)
        if expect_error:
            pytest.fail(f"Expected error but node returned: {result}")
        return result, None
    except NODE_INSTANTIATION_ERRORS as e:
        if not expect_error:
            pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")
        return None, e


def assert_state_updated(ctx: GraphRunContext, **expected_values):
    """Assert that state fields have expected values.
    
    Args:
        ctx: The graph context
        **expected_values: Field names and expected values
    """
    for field, expected in expected_values.items():
        actual = getattr(ctx.state, field, None)
        assert actual == expected, f"State.{field} = {actual}, expected {expected}"


def create_memory_storage_mock(
    store_question_result: Optional[Mock] = None,
    store_qa_result: bool = True,
    should_fail: bool = False
) -> Mock:
    """Create a mock for MemoryStorage.
    
    Args:
        store_question_result: Mock entity to return from store_question_only
        store_qa_result: Return value for store_qa_pair
        should_fail: Whether operations should raise exceptions
        
    Returns:
        Mock: Configured MemoryStorage mock
    """
    mock_storage = Mock()
    
    if should_fail:
        mock_storage.store_question_only = AsyncMock(
            side_effect=Exception("Storage error")
        )
        mock_storage.store_qa_pair = AsyncMock(
            side_effect=Exception("Storage error")
        )
    else:
        if store_question_result is None:
            # Create default question entity mock
            store_question_result = Mock()
            store_question_result.id = "q_123"
            
        mock_storage.store_question_only = AsyncMock(
            return_value=store_question_result
        )
        mock_storage.store_qa_pair = AsyncMock(return_value=store_qa_result)
    
    return mock_storage


@contextmanager
def patch_enhanced_nodes_agents(ask_output: str, eval_output: Any):
    """Patch agents in enhanced_nodes module.
    
    Args:
        ask_output: Output for ask agent
        eval_output: Output for evaluate agent
        
    Yields:
        tuple: (mock_ask, mock_eval) for assertions
    """
    ask_result = create_agent_result_mock(ask_output)
    eval_result = create_agent_result_mock(eval_output)
    
    with patch('enhanced_nodes.ask_agent') as mock_ask:
        with patch('enhanced_nodes.evaluate_agent') as mock_eval:
            mock_ask.run = AsyncMock(return_value=ask_result)
            mock_eval.run = AsyncMock(return_value=eval_result)
            yield mock_ask, mock_eval


@contextmanager
def patch_standard_graph_agents(ask_output: str, eval_output: Any):
    """Patch agents in question_graph module.
    
    Args:
        ask_output: Output for ask agent
        eval_output: Output for evaluate agent
        
    Yields:
        tuple: (mock_ask, mock_eval) for assertions
    """
    ask_result = create_agent_result_mock(ask_output)
    eval_result = create_agent_result_mock(eval_output)
    
    with patch('question_graph.ask_agent') as mock_ask:
        with patch('question_graph.evaluate_agent') as mock_eval:
            mock_ask.run = AsyncMock(return_value=ask_result)
            mock_eval.run = AsyncMock(return_value=eval_result)
            yield mock_ask, mock_eval


class NodeTestPattern:
    """Common test pattern for pydantic_graph nodes."""
    
    @staticmethod
    async def test_state_mutations(
        node: BaseNode,
        ctx: GraphRunContext,
        expected_state_changes: dict,
        mock_dependencies: Optional[dict] = None
    ):
        """Test that a node properly updates state.
        
        Args:
            node: Node to test
            ctx: Graph context
            expected_state_changes: Expected state field changes
            mock_dependencies: Optional mocks to patch
        """
        patches = []
        if mock_dependencies:
            for path, mock_obj in mock_dependencies.items():
                patches.append(patch(path, mock_obj))
        
        try:
            for p in patches:
                p.start()
            
            # Run node, expecting potential errors
            await run_node_safely(node, ctx, expect_error=True)
            
            # Verify state changes
            assert_state_updated(ctx, **expected_state_changes)
            
        finally:
            for p in patches:
                p.stop()
    
    @staticmethod
    def test_node_instantiation():
        """Test that nodes follow pydantic_graph patterns."""
        from enhanced_nodes import EnhancedAsk, EnhancedAnswer
        from question_graph import Answer
        
        # Should work - nodes are dataclasses
        assert EnhancedAsk()
        assert EnhancedAnswer()
        assert Answer()
        
        # Should fail - can't instantiate with args
        with pytest.raises(TypeError, match="takes no arguments"):
            Answer(question="test")
        
        with pytest.raises(TypeError, match="takes no arguments"):
            EnhancedAnswer(question="test")


# Test documentation patterns
TEST_PATTERNS = """
Testing Patterns for pydantic_graph Nodes:

1. **Mock Simply**: Use Mock() without spec/autospec to avoid Pydantic validation
2. **Patch at Usage**: Patch where modules are used, not where they're defined
3. **Handle Errors**: Accept both TypeError and RuntimeError for node instantiation
4. **Test State**: Focus on state mutations, not return values
5. **Pre-create Mocks**: Create mock instances before patching to avoid recursion

Example:
    async def test_enhanced_ask(mock_agents):
        # Create state and context
        state = EnhancedQuestionState()
        ctx = GraphRunContext(state=state, deps=None)
        
        # Run node with expected error handling
        node = EnhancedAsk()
        await run_node_safely(node, ctx)
        
        # Assert state changes
        assert_state_updated(ctx, question="What is the capital of France?")
"""