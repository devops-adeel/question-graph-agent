#!/usr/bin/env python3
"""
Example script demonstrating graceful fallback functionality.

This script shows how the system handles Graphiti unavailability
with various fallback modes and recovery strategies.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphiti_fallback import (
    FallbackMode,
    FallbackManager,
    get_fallback_manager,
    with_fallback,
)
from graphiti_health import HealthStatus
from graphiti_entities import QuestionEntity, AnswerEntity, UserEntity, DifficultyLevel
from graphiti_relationships import AnsweredRelationship
from graphiti_config import get_config


# Simulate a service that uses Graphiti
class QuestionService:
    """Example service with fallback support."""
    
    def __init__(self):
        self.fallback_manager = get_fallback_manager()
    
    @with_fallback
    async def create_question(self, question: QuestionEntity):
        """Create a question (with fallback support)."""
        # This would normally call Graphiti API
        # For demo, we'll simulate a failure
        if self.should_fail():
            raise ConnectionError("Graphiti service unavailable")
        
        print(f"✅ Created question: {question.id}")
        return question
    
    @with_fallback
    async def get_question(self, question_id: str):
        """Get a question (with fallback support)."""
        # Simulate failure
        if self.should_fail():
            raise ConnectionError("Graphiti service unavailable")
        
        # Normal operation would fetch from Graphiti
        return None
    
    def should_fail(self) -> bool:
        """Simulate service availability."""
        # For demo, check if fallback is already active
        return not self.fallback_manager.state.is_active


async def demonstrate_fallback_modes():
    """Demonstrate different fallback modes."""
    print("=== Fallback Modes Demonstration ===\n")
    
    manager = get_fallback_manager()
    
    # 1. Read-Only Mode
    print("1. Read-Only Mode:")
    manager.state.activate(FallbackMode.READ_ONLY, "Maintenance mode")
    
    entity = QuestionEntity(
        id="q1",
        content="What is fallback mode?",
        difficulty=DifficultyLevel.EASY
    )
    
    result = manager.store_entity(entity)
    print(f"   Store attempt: {'Failed (expected)' if not result else 'Success'}")
    print(f"   Mode: {manager.state.mode.value}")
    
    manager.state.deactivate()
    
    # 2. Memory-Only Mode
    print("\n2. Memory-Only Mode:")
    manager.state.activate(FallbackMode.MEMORY_ONLY, "Temporary storage")
    
    result = manager.store_entity(entity)
    print(f"   Store attempt: {'Success' if result else 'Failed'}")
    
    retrieved = manager.get_entity("q1")
    print(f"   Retrieve attempt: {'Success' if retrieved else 'Failed'}")
    
    manager.state.deactivate()
    
    # 3. Local Storage Mode
    print("\n3. Local Storage Mode:")
    manager.state.activate(FallbackMode.LOCAL_STORAGE, "Persistent fallback")
    
    result = manager.store_entity(entity)
    print(f"   Store attempt: {'Success' if result else 'Failed'}")
    print(f"   Data persisted to: {manager.storage.db_path}")
    
    manager.state.deactivate()
    
    # 4. Queue Writes Mode
    print("\n4. Queue Writes Mode:")
    manager.state.activate(FallbackMode.QUEUE_WRITES, "Degraded service")
    
    result = manager.store_entity(entity)
    print(f"   Store attempt: {'Success' if result else 'Failed'}")
    print(f"   Queued operations: {manager.state.queued_operations}")
    
    manager.state.deactivate()


async def simulate_service_outage():
    """Simulate a service outage and recovery."""
    print("\n=== Service Outage Simulation ===\n")
    
    service = QuestionService()
    manager = service.fallback_manager
    
    # Add callback to monitor state changes
    def state_callback(state):
        if state.is_active:
            print(f"📡 Fallback activated: {state.mode.value} - {state.reason}")
        else:
            print("✅ Fallback deactivated - Service recovered")
    
    manager.add_callback(state_callback)
    
    # Create questions
    questions = [
        QuestionEntity(
            id=f"q{i}",
            content=f"Question {i}?",
            difficulty=DifficultyLevel.MEDIUM
        )
        for i in range(5)
    ]
    
    print("Attempting to create questions...")
    
    for i, question in enumerate(questions):
        try:
            # First attempt will fail and activate fallback
            await service.create_question(question)
            print(f"  Question {i+1}: Created successfully")
        except Exception as e:
            print(f"  Question {i+1}: Failed - {e}")
    
    # Show fallback status
    status = manager.get_status()
    print(f"\nFallback Status:")
    print(f"  Mode: {status['mode']}")
    print(f"  Active: {status['is_active']}")
    print(f"  Queued Operations: {status['queued_operations']}")
    print(f"  Cache Hit Rate: {status['cache_hit_rate']:.1%}")


async def demonstrate_automatic_recovery():
    """Demonstrate automatic recovery when service comes back."""
    print("\n=== Automatic Recovery Demonstration ===\n")
    
    manager = get_fallback_manager()
    
    # Mock health check to simulate recovery
    from unittest.mock import AsyncMock, patch
    from graphiti_health import HealthCheckResult, ComponentType
    
    print("1. Simulating service failure...")
    
    with patch.object(manager.health_checker, 'perform_check') as mock_check:
        # First check - service down
        mock_check.return_value = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.UNHEALTHY,
            error="Connection refused"
        )
        
        activated = await manager.check_and_activate()
        print(f"   Fallback activated: {activated}")
        print(f"   Mode: {manager.state.mode.value}")
        
        # Store some data while in fallback
        entity = QuestionEntity(
            id="recovery_test",
            content="Will this sync?",
            difficulty=DifficultyLevel.HARD
        )
        manager.store_entity(entity)
        print(f"   Stored entity in fallback mode")
        
        # Simulate service recovery
        print("\n2. Simulating service recovery...")
        mock_check.return_value = HealthCheckResult(
            component=ComponentType.GRAPHITI,
            status=HealthStatus.HEALTHY
        )
        
        # Use fallback context to trigger recovery
        async with manager.fallback_context():
            print("   Service health check performed")
        
        print(f"   Fallback active: {manager.state.is_active}")
        print("   ✅ Service recovered and fallback deactivated")


async def demonstrate_fallback_metrics():
    """Demonstrate fallback metrics and monitoring."""
    print("\n=== Fallback Metrics Demonstration ===\n")
    
    manager = get_fallback_manager()
    manager.state.activate(FallbackMode.LOCAL_STORAGE, "Metrics demo")
    
    # Perform various operations
    entities = []
    for i in range(10):
        entity = QuestionEntity(
            id=f"metric_q{i}",
            content=f"Metric question {i}?",
            difficulty=DifficultyLevel.EASY
        )
        manager.store_entity(entity)
        entities.append(entity)
    
    # Access entities to generate cache hits/misses
    print("Accessing entities:")
    for i in range(15):
        entity_id = f"metric_q{i % 10}"
        result = manager.get_entity(entity_id)
        print(f"  Get {entity_id}: {'Hit' if result else 'Miss'}")
    
    # Access non-existent entities for cache misses
    for i in range(5):
        manager.get_entity(f"nonexistent_{i}")
    
    # Display metrics
    status = manager.get_status()
    print(f"\nFallback Metrics:")
    print(f"  Cache Hits: {manager.state.cache_hits}")
    print(f"  Cache Misses: {manager.state.cache_misses}")
    print(f"  Cache Hit Rate: {status['cache_hit_rate']:.1%}")
    print(f"  Mode: {status['mode']}")
    
    manager.state.deactivate()


async def demonstrate_queue_sync():
    """Demonstrate queued operation syncing."""
    print("\n=== Queue Sync Demonstration ===\n")
    
    manager = get_fallback_manager()
    
    # Activate queue writes mode
    manager.state.activate(FallbackMode.QUEUE_WRITES, "Degraded performance")
    
    print("1. Queueing operations...")
    
    # Queue multiple operations
    for i in range(5):
        entity = QuestionEntity(
            id=f"sync_q{i}",
            content=f"Question for sync {i}?",
            difficulty=DifficultyLevel.MEDIUM
        )
        manager.store_entity(entity)
        print(f"   Queued: {entity.id}")
    
    print(f"\n2. Queued operations: {manager.state.queued_operations}")
    
    # Show queued operations
    operations = manager.storage.get_queued_operations()
    print("\n3. Operation details:")
    for op in operations:
        print(f"   - {op.operation_type} {op.entity_type} {op.entity_id}")
        print(f"     Timestamp: {op.timestamp.strftime('%H:%M:%S')}")
    
    # Simulate sync process
    print("\n4. Simulating sync process...")
    print("   (In production, this would sync to Graphiti when available)")
    
    manager.state.deactivate()


async def main():
    """Run all demonstrations."""
    print("Graceful Fallback Examples")
    print("=" * 50)
    
    # Get configuration
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  Neo4j URI: {config.neo4j.uri}")
    print(f"  Graphiti Endpoint: {config.graphiti.endpoint}")
    
    try:
        # Run demonstrations
        await demonstrate_fallback_modes()
        await simulate_service_outage()
        await demonstrate_automatic_recovery()
        await demonstrate_fallback_metrics()
        await demonstrate_queue_sync()
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        return 1
    
    print("\n✅ All fallback demonstrations completed!")
    print("\nKey Takeaways:")
    print("- Multiple fallback modes for different scenarios")
    print("- Automatic activation when service unavailable")
    print("- Queue writes for eventual consistency")
    print("- Local storage for data persistence")
    print("- Metrics tracking for monitoring")
    print("- Automatic recovery when service returns")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)