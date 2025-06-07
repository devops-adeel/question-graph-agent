"""
Example usage of memory update functionality.

This example demonstrates how to use the MemoryUpdateService and
EvaluationEventHandler to track user performance after answer evaluation.
"""

import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_updates import (
    EvaluationResult,
    MemoryUpdateService,
    EvaluationEventHandler,
)
from graphiti_entities import DifficultyLevel, AnswerStatus
from graphiti_client import GraphitiClient
from graphiti_config import get_config


async def basic_evaluation_recording():
    """Basic example of recording a single evaluation."""
    print("\n=== Basic Evaluation Recording ===")
    
    # Initialize client and service
    config = get_config()
    
    # For this example, we'll use mock mode
    # In production, you'd connect to real Neo4j
    print("Note: Using mock mode for this example")
    
    # Create service (without real client for demo)
    service = MemoryUpdateService(client=None, config=config)
    
    # Create evaluation result
    result = EvaluationResult(
        question_id="q_example_001",
        answer_id="a_example_001",
        user_id="demo_user_123",
        session_id="demo_session_abc",
        correct=True,
        evaluation_comment="Great job! Your understanding of recursion is excellent.",
        confidence_score=0.95,
        response_time=4.2,
        topics=["computer_science", "algorithms", "recursion"],
        difficulty=DifficultyLevel.MEDIUM
    )
    
    print(f"\nRecording evaluation for user {result.user_id}:")
    print(f"  Question: {result.question_id}")
    print(f"  Answer: {result.answer_id}")
    print(f"  Correct: {result.correct}")
    print(f"  Topics: {', '.join(result.topics)}")
    print(f"  Difficulty: {result.difficulty.value}")
    print(f"  Response time: {result.response_time}s")
    
    # Record evaluation (would normally update graph)
    success = await service.record_evaluation(result, immediate=True)
    
    if success:
        print("\n✓ Evaluation recorded successfully (mock mode)")
    else:
        print("\n✗ Failed to record evaluation (no client connected)")


async def batch_evaluation_processing():
    """Example of batch processing multiple evaluations."""
    print("\n\n=== Batch Evaluation Processing ===")
    
    # Create service
    service = MemoryUpdateService(client=None)
    
    # Create multiple evaluation results
    evaluations = [
        EvaluationResult(
            question_id=f"q_batch_{i:03d}",
            answer_id=f"a_batch_{i:03d}",
            user_id="demo_user_456",
            session_id="demo_session_xyz",
            correct=(i % 3 != 0),  # Every 3rd answer is incorrect
            evaluation_comment="Correct!" if i % 3 != 0 else "Not quite right.",
            confidence_score=0.9 if i % 3 != 0 else 0.3,
            response_time=3.0 + i * 0.5,
            topics=["mathematics"] if i < 5 else ["physics"],
            difficulty=DifficultyLevel.EASY if i < 3 else DifficultyLevel.MEDIUM
        )
        for i in range(10)
    ]
    
    print(f"Recording {len(evaluations)} evaluations in batch mode...")
    
    # Record all evaluations for batch processing
    for eval_result in evaluations:
        await service.record_evaluation(eval_result, immediate=False)
    
    print(f"Pending updates: {len(service._pending_updates)}")
    
    # Process batch
    successful, failed = await service.flush_pending_updates()
    
    print(f"\nBatch processing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed} (expected in mock mode)")


async def event_handler_example():
    """Example using the EvaluationEventHandler for automatic batching."""
    print("\n\n=== Event Handler Example ===")
    
    # Create service and handler
    service = MemoryUpdateService(client=None)
    handler = EvaluationEventHandler(service)
    handler._batch_size = 3  # Process every 3 events
    
    print(f"Event handler configured with batch size: {handler._batch_size}")
    
    # Simulate evaluation events
    events = [
        {
            "question_id": "q_event_001",
            "answer_id": "a_event_001",
            "user_id": "student_789",
            "session_id": "session_morning",
            "correct": True,
            "comment": "Excellent work!",
            "confidence": 0.98,
            "response_time": 2.5,
            "topics": ["biology", "cells"],
            "difficulty": DifficultyLevel.EASY
        },
        {
            "question_id": "q_event_002",
            "answer_id": "a_event_002",
            "user_id": "student_789",
            "session_id": "session_morning",
            "correct": False,
            "comment": "Review the concept of mitosis.",
            "confidence": 0.4,
            "response_time": 5.2,
            "topics": ["biology", "cell_division"],
            "difficulty": DifficultyLevel.MEDIUM
        },
        {
            "question_id": "q_event_003",
            "answer_id": "a_event_003",
            "user_id": "student_789",
            "session_id": "session_morning",
            "correct": True,
            "comment": "Good understanding of DNA structure.",
            "confidence": 0.85,
            "response_time": 3.8,
            "topics": ["biology", "genetics"],
            "difficulty": DifficultyLevel.MEDIUM
        },
        {
            "question_id": "q_event_004",
            "answer_id": "a_event_004",
            "user_id": "student_789",
            "session_id": "session_morning",
            "correct": True,
            "comment": "Perfect!",
            "confidence": 1.0,
            "response_time": 1.5,
            "topics": ["biology", "ecology"],
            "difficulty": DifficultyLevel.EASY
        }
    ]
    
    print("\nProcessing evaluation events...")
    
    for i, event in enumerate(events):
        print(f"\nEvent {i+1}: {event['question_id']} - {'✓' if event['correct'] else '✗'}")
        
        await handler.handle_evaluation_event(**event)
        
        # Check if batch was triggered
        if len(handler._event_queue) == 0:
            print("  → Batch processed automatically!")
    
    # Flush remaining events
    print("\nFlushing remaining events...")
    await handler.flush()
    
    print("All events processed!")


async def progress_summary_simulation():
    """Simulate getting a user's progress summary."""
    print("\n\n=== Progress Summary Simulation ===")
    
    # This is a simulation of what the summary would look like
    # In production, this would query the actual graph database
    
    mock_summary = {
        "overall": {
            "total_questions": 150,
            "correct_answers": 112,
            "accuracy": 0.747,
            "avg_response_time": 4.2,
            "current_streak": 7,
            "best_streak": 15,
            "topics_practiced": 12,
            "avg_mastery": 0.68
        },
        "recent_performance": {
            "questions_last_7_days": 35,
            "correct_last_7_days": 30,
            "recent_accuracy": 0.857,
            "recent_avg_time": 3.5
        },
        "topics": [
            {
                "name": "mathematics",
                "mastery": 0.82,
                "accuracy": 0.85,
                "attempts": 45,
                "last_practiced": datetime.now().isoformat()
            },
            {
                "name": "physics",
                "mastery": 0.65,
                "accuracy": 0.70,
                "attempts": 38,
                "last_practiced": datetime.now().isoformat()
            },
            {
                "name": "chemistry",
                "mastery": 0.55,
                "accuracy": 0.62,
                "attempts": 32,
                "last_practiced": datetime.now().isoformat()
            }
        ]
    }
    
    print("\nUser Progress Summary (Simulated):")
    print("\nOverall Performance:")
    print(f"  Total Questions: {mock_summary['overall']['total_questions']}")
    print(f"  Correct Answers: {mock_summary['overall']['correct_answers']}")
    print(f"  Accuracy: {mock_summary['overall']['accuracy']:.1%}")
    print(f"  Current Streak: {mock_summary['overall']['current_streak']}")
    print(f"  Best Streak: {mock_summary['overall']['best_streak']}")
    
    print("\nRecent Performance (Last 7 Days):")
    print(f"  Questions: {mock_summary['recent_performance']['questions_last_7_days']}")
    print(f"  Accuracy: {mock_summary['recent_performance']['recent_accuracy']:.1%}")
    print(f"  Improvement: +{(mock_summary['recent_performance']['recent_accuracy'] - mock_summary['overall']['accuracy']):.1%}")
    
    print("\nTopic Mastery:")
    for topic in mock_summary['topics']:
        print(f"\n  {topic['name'].capitalize()}:")
        print(f"    Mastery Level: {topic['mastery']:.0%}")
        print(f"    Accuracy: {topic['accuracy']:.0%}")
        print(f"    Practice Sessions: {topic['attempts']}")


async def mastery_calculation_demo():
    """Demonstrate mastery level calculation."""
    print("\n\n=== Mastery Calculation Demo ===")
    
    service = MemoryUpdateService(client=None)
    
    # Simulate mastery calculation for different scenarios
    scenarios = [
        {
            "name": "Beginner Success",
            "current_level": 0.3,
            "total_attempts": 2,
            "correct_attempts": 1,
            "latest_correct": True,
            "latest_difficulty": DifficultyLevel.EASY
        },
        {
            "name": "Expert Challenge",
            "current_level": 0.8,
            "total_attempts": 20,
            "correct_attempts": 18,
            "latest_correct": True,
            "latest_difficulty": DifficultyLevel.EXPERT
        },
        {
            "name": "Struggling Student",
            "current_level": 0.4,
            "total_attempts": 10,
            "correct_attempts": 3,
            "latest_correct": False,
            "latest_difficulty": DifficultyLevel.MEDIUM
        }
    ]
    
    print("\nMastery level calculations for different scenarios:")
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Current mastery: {scenario['current_level']:.0%}")
        print(f"  Total attempts: {scenario['total_attempts']}")
        print(f"  Success rate: {scenario['correct_attempts']}/{scenario['total_attempts']}")
        print(f"  Latest answer: {'✓' if scenario['latest_correct'] else '✗'} ({scenario['latest_difficulty'].value})")
        
        # Calculate new mastery (simplified simulation)
        historical_accuracy = scenario['correct_attempts'] / scenario['total_attempts']
        
        # Weight based on difficulty
        difficulty_weights = {
            DifficultyLevel.EASY: 0.2,
            DifficultyLevel.MEDIUM: 0.3,
            DifficultyLevel.HARD: 0.4,
            DifficultyLevel.EXPERT: 0.5
        }
        
        weight = difficulty_weights[scenario['latest_difficulty']]
        recent_score = 1.0 if scenario['latest_correct'] else 0.0
        
        new_level = (scenario['current_level'] * (1 - weight)) + (recent_score * weight)
        
        # Apply performance adjustments
        if historical_accuracy > 0.8 and scenario['total_attempts'] >= 5:
            new_level = min(1.0, new_level * 1.1)
        elif historical_accuracy < 0.3 and scenario['total_attempts'] >= 5:
            new_level = max(0.0, new_level * 0.9)
        
        confidence = min(1.0, scenario['total_attempts'] / 10)
        
        print(f"  → New mastery: {new_level:.0%} (confidence: {confidence:.0%})")
        
        # Show trend
        if new_level > scenario['current_level']:
            print(f"  → Trend: ↑ Improving (+{(new_level - scenario['current_level']):.0%})")
        elif new_level < scenario['current_level']:
            print(f"  → Trend: ↓ Declining ({(new_level - scenario['current_level']):.0%})")
        else:
            print(f"  → Trend: → Stable")


async def main():
    """Run all examples."""
    print("Memory Updates Example")
    print("=" * 50)
    
    # Run examples
    await basic_evaluation_recording()
    await batch_evaluation_processing()
    await event_handler_example()
    await progress_summary_simulation()
    await mastery_calculation_demo()
    
    print("\n\nAll examples completed!")
    print("\nKey Takeaways:")
    print("- MemoryUpdateService handles post-evaluation updates")
    print("- Supports both immediate and batch processing")
    print("- Tracks user statistics, topic mastery, and streaks")
    print("- EvaluationEventHandler provides automatic batching")
    print("- Progress summaries give comprehensive performance view")


if __name__ == "__main__":
    asyncio.run(main())