#!/usr/bin/env python3
"""
Example script demonstrating GraphitiClient usage.

This script shows how to use the GraphitiClient for storing and retrieving
Q&A data in the temporal knowledge graph.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphiti_client import GraphitiClient
from graphiti_entities import (
    QuestionEntity,
    AnswerEntity,
    UserEntity,
    TopicEntity,
    DifficultyLevel,
    AnswerStatus,
)
from graphiti_config import get_config
from question_graph import QuestionState, initialize_graphiti_state


async def demonstrate_basic_usage():
    """Demonstrate basic GraphitiClient operations."""
    print("=== Basic GraphitiClient Usage ===\n")
    
    # Create client
    client = GraphitiClient()
    
    # Connect to Graphiti
    print("Connecting to Graphiti...")
    connected = await client.connect()
    
    if connected:
        print("✅ Connected successfully")
    else:
        print("⚠️  Connection failed - using fallback mode")
        if client.fallback_manager and client.fallback_manager.state.is_active:
            print(f"   Fallback mode: {client.fallback_manager.state.mode.value}")
    
    try:
        # Create entities
        print("\n1. Creating entities...")
        
        user = UserEntity(
            id="demo_user",
            session_id=client._session.session_id,
            total_questions=0,
            correct_answers=0
        )
        
        question = QuestionEntity(
            id="demo_q1",
            content="What is the capital of France?",
            difficulty=DifficultyLevel.EASY,
            topics=["geography", "europe", "capitals"]
        )
        
        # Store question
        await client.store_question(question)
        print(f"   Stored question: {question.id}")
        
        # Create and store answer
        answer = AnswerEntity(
            id="demo_a1",
            question_id=question.id,
            user_id=user.id,
            content="Paris",
            status=AnswerStatus.CORRECT,
            response_time=5.2,
            confidence_score=0.9
        )
        
        await client.store_answer(answer, question, user)
        print(f"   Stored answer: {answer.id}")
        
        # Create episode
        await client.create_qa_episode(
            question=question,
            answer=answer,
            user=user,
            evaluation_correct=True
        )
        print("   Created Q&A episode")
        
        # Show session stats
        stats = client.get_session_stats()
        print(f"\n2. Session Statistics:")
        print(f"   Session ID: {stats['session_id']}")
        print(f"   Entities: {stats['entity_count']}")
        print(f"   Relationships: {stats['relationship_count']}")
        print(f"   Episodes: {stats['episode_count']}")
        print(f"   Duration: {stats['duration_seconds']:.1f}s")
        
    finally:
        await client.disconnect()
        print("\n✅ Disconnected")


async def demonstrate_user_history():
    """Demonstrate retrieving user history."""
    print("\n=== User History Demo ===\n")
    
    async with GraphitiClient().session_context() as client:
        # Create some Q&A history
        user = UserEntity(id="history_user", session_id=client._session.session_id)
        
        questions_and_answers = [
            ("What is 2+2?", "4", True, DifficultyLevel.EASY, ["math"]),
            ("Name the largest planet", "Jupiter", True, DifficultyLevel.MEDIUM, ["astronomy"]),
            ("What is the speed of light?", "300,000 km/s", False, DifficultyLevel.HARD, ["physics"]),
        ]
        
        print("Creating Q&A history...")
        for i, (q_text, a_text, correct, difficulty, topics) in enumerate(questions_and_answers):
            question = QuestionEntity(
                id=f"hist_q{i}",
                content=q_text,
                difficulty=difficulty,
                topics=topics
            )
            
            answer = AnswerEntity(
                id=f"hist_a{i}",
                question_id=question.id,
                user_id=user.id,
                content=a_text,
                status=AnswerStatus.CORRECT if correct else AnswerStatus.INCORRECT,
                response_time=10.0 + i * 2
            )
            
            await client.store_question(question)
            await client.store_answer(answer, question, user)
            await client.create_qa_episode(question, answer, user, correct)
            print(f"   Created Q&A {i+1}: {q_text[:30]}... -> {a_text}")
        
        # Retrieve history
        print("\nRetrieving user history...")
        history = await client.get_user_history(user.id, limit=5)
        
        print(f"\nFound {len(history)} Q&A pairs:")
        for i, item in enumerate(history):
            q = item.get("question", {})
            a = item.get("answer", {})
            print(f"\n{i+1}. Question: {q.get('content', 'N/A')}")
            print(f"   Answer: {a.get('content', 'N/A')}")
            print(f"   Status: {a.get('status', 'N/A')}")


async def demonstrate_related_questions():
    """Demonstrate finding related questions."""
    print("\n=== Related Questions Demo ===\n")
    
    async with GraphitiClient().session_context() as client:
        # Create questions with topics
        math_questions = [
            ("What is 5 x 6?", DifficultyLevel.EASY),
            ("Solve: 2x + 5 = 13", DifficultyLevel.MEDIUM),
            ("Find the derivative of x^3 + 2x", DifficultyLevel.HARD),
            ("What is the area of a circle with radius 5?", DifficultyLevel.MEDIUM),
        ]
        
        print("Creating math questions...")
        for i, (content, difficulty) in enumerate(math_questions):
            question = QuestionEntity(
                id=f"math_q{i}",
                content=content,
                difficulty=difficulty,
                topics=["mathematics"]
            )
            await client.store_question(question)
            print(f"   Created: {content}")
        
        # Find related questions
        print("\nFinding related questions...")
        
        # By topic
        related = await client.get_related_questions(
            topic="mathematics",
            limit=3
        )
        print(f"\nFound {len(related)} mathematics questions:")
        for q in related:
            print(f"   - {q.content} ({q.difficulty.value})")
        
        # By topic and difficulty
        medium_math = await client.get_related_questions(
            topic="mathematics",
            difficulty="MEDIUM",
            limit=5
        )
        print(f"\nFound {len(medium_math)} medium difficulty math questions:")
        for q in medium_math:
            print(f"   - {q.content}")


async def demonstrate_mastery_tracking():
    """Demonstrate user mastery tracking."""
    print("\n=== Mastery Tracking Demo ===\n")
    
    async with GraphitiClient().session_context() as client:
        user = UserEntity(id="mastery_user", session_id=client._session.session_id)
        topic = TopicEntity(id="math_topic", name="Mathematics", complexity_score=0.7)
        
        print("Simulating user answering questions...")
        
        # Simulate answering questions
        attempts = [
            (True, 10.5),   # Correct, 10.5 seconds
            (True, 8.2),    # Correct, 8.2 seconds
            (False, 15.0),  # Incorrect, 15 seconds
            (True, 7.1),    # Correct, 7.1 seconds
            (True, 6.5),    # Correct, 6.5 seconds
        ]
        
        for i, (correct, time_taken) in enumerate(attempts):
            await client.update_user_mastery(
                user=user,
                topic=topic,
                correct=correct,
                time_taken=time_taken
            )
            print(f"   Attempt {i+1}: {'✅' if correct else '❌'} ({time_taken}s)")
        
        print("\nMastery progression tracked in Neo4j")
        print("(Check Neo4j browser to see HAS_MASTERY relationships)")


async def demonstrate_state_integration():
    """Demonstrate QuestionState integration."""
    print("\n=== QuestionState Integration Demo ===\n")
    
    # Initialize state with Graphiti
    print("1. Initializing QuestionState with Graphiti...")
    state = await initialize_graphiti_state(
        user_id="integration_user",
        enable_graphiti=True
    )
    
    print(f"   Session ID: {state.session_id}")
    print(f"   User ID: {state.current_user.id if state.current_user else 'None'}")
    print(f"   Graphiti enabled: {state.graphiti_client is not None}")
    
    if state.graphiti_client:
        # Use the client from state
        client = state.graphiti_client
        
        # Store a question through state
        question = QuestionEntity(
            id="state_q1",
            content="What programming language is this written in?",
            difficulty=DifficultyLevel.EASY,
            topics=["programming", "python"]
        )
        
        await client.store_question(question)
        print("\n2. Stored question through state's GraphitiClient")
        
        # Get stats
        stats = client.get_session_stats()
        print(f"\n3. Session stats:")
        print(f"   Entities: {stats['entity_count']}")
        print(f"   Circuit state: {stats['circuit_state']}")
        
        # Clean up
        await client.disconnect()
        print("\n✅ Cleaned up state's GraphitiClient")


async def demonstrate_error_handling():
    """Demonstrate error handling and fallback."""
    print("\n=== Error Handling Demo ===\n")
    
    # Create client with fallback enabled
    client = GraphitiClient(enable_fallback=True, enable_circuit_breaker=True)
    
    print("1. Testing connection with fallback...")
    connected = await client.connect()
    
    if not connected and client.fallback_manager.state.is_active:
        print(f"   ⚠️  Using fallback mode: {client.fallback_manager.state.mode.value}")
        
        # Operations will use fallback
        question = QuestionEntity(
            id="fallback_q1",
            content="This will be stored in fallback",
            difficulty=DifficultyLevel.MEDIUM
        )
        
        result = await client.store_question(question)
        print(f"   Stored in fallback: {result}")
        
        # Check fallback stats
        fallback_status = client.fallback_manager.get_status()
        print(f"\n2. Fallback status:")
        print(f"   Mode: {fallback_status['mode']}")
        print(f"   Queued operations: {fallback_status['queued_operations']}")
        print(f"   Cache hit rate: {fallback_status['cache_hit_rate']:.1%}")
    
    else:
        print("   ✅ Connected normally (no fallback needed)")
    
    # Check circuit breaker
    if client.circuit_breaker:
        cb_status = client.circuit_breaker.get_status()
        print(f"\n3. Circuit breaker status:")
        print(f"   State: {cb_status['state']}")
        print(f"   Failure rate: {cb_status['stats']['failure_rate']}")
    
    await client.disconnect()


async def main():
    """Run all demonstrations."""
    print("GraphitiClient Demo")
    print("=" * 50)
    
    # Get configuration
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  Neo4j URI: {config.neo4j.uri}")
    print(f"  Graphiti Endpoint: {config.graphiti.endpoint}")
    
    try:
        # Run demonstrations
        await demonstrate_basic_usage()
        await demonstrate_user_history()
        await demonstrate_related_questions()
        await demonstrate_mastery_tracking()
        await demonstrate_state_integration()
        await demonstrate_error_handling()
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("\nMake sure Neo4j and Graphiti services are running!")
        return 1
    
    print("\n✅ All demonstrations completed!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)