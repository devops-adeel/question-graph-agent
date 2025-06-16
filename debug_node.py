#!/usr/bin/env python3
"""Debug script to test node instantiation."""

import asyncio
from pydantic_graph import BaseNode, GraphRunContext
from pydantic import Field
from question_graph import QuestionState, Answer
from enhanced_nodes import EnhancedAnswer


async def test_node_instantiation():
    """Test if nodes can be instantiated with arguments."""
    print("Testing node instantiation...")
    
    # Test Answer instantiation
    try:
        answer_node = Answer(question="Test question")
        print(f"✓ Answer instantiation successful: {answer_node}")
        print(f"  Type: {type(answer_node)}")
        print(f"  Question: {answer_node.question}")
    except Exception as e:
        print(f"✗ Answer instantiation failed: {type(e).__name__}: {e}")
    
    # Test EnhancedAnswer instantiation
    try:
        enhanced_answer_node = EnhancedAnswer(question="Test question")
        print(f"✓ EnhancedAnswer instantiation successful: {enhanced_answer_node}")
        print(f"  Type: {type(enhanced_answer_node)}")
        print(f"  Question: {enhanced_answer_node.question}")
    except Exception as e:
        print(f"✗ EnhancedAnswer instantiation failed: {type(e).__name__}: {e}")
    
    # Test if they're both subclasses of BaseNode
    print(f"\nAnswer is BaseNode subclass: {issubclass(Answer, BaseNode)}")
    print(f"EnhancedAnswer is BaseNode subclass: {issubclass(EnhancedAnswer, BaseNode)}")
    
    # Check their base classes
    print(f"\nAnswer.__bases__: {Answer.__bases__}")
    print(f"EnhancedAnswer.__bases__: {EnhancedAnswer.__bases__}")
    
    # Check if they have __init__ methods
    print(f"\nAnswer has __init__: {hasattr(Answer, '__init__')}")
    print(f"EnhancedAnswer has __init__: {hasattr(EnhancedAnswer, '__init__')}")


if __name__ == "__main__":
    asyncio.run(test_node_instantiation())