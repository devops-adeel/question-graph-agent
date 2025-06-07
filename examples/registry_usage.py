"""
Example usage of the Graphiti entity registration module.

This script demonstrates how to register entities with Graphiti
and create episodes from Q&A interactions.
"""

import asyncio
import logging
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from graphiti_registry import (
    EntityTypeRegistry,
    EntityAdapter,
    RelationshipAdapter,
    EpisodeBuilder,
    EntityRegistrar,
)
from graphiti_entities import (
    QuestionEntity,
    AnswerEntity,
    UserEntity,
    TopicEntity,
    DifficultyLevel,
    AnswerStatus,
    EntityFactory,
)
from graphiti_relationships import (
    AnsweredRelationship,
    RequiresKnowledgeRelationship,
    MasteryRelationship,
    RelationshipBuilder,
)
from graphiti_config import get_config
from graphiti_connection import get_neo4j_connection


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_graphiti_client():
    """Set up Graphiti client with Neo4j connection."""
    config = get_config()
    
    # Initialize Graphiti client
    client = Graphiti(
        neo4j_uri=config.neo4j.uri,
        neo4j_user=config.neo4j.user,
        neo4j_password=config.neo4j.password.get_secret_value()
    )
    
    # Initialize client
    await client.initialize()
    
    return client


async def register_initial_entities(registrar: EntityRegistrar):
    """Register initial entities with Graphiti."""
    logger.info("Registering initial entities...")
    
    # Register entity types
    await registrar.register_entity_types()
    
    # Create some topics
    topics = [
        EntityFactory.create_topic(
            name="Mathematics",
            complexity_score=0.7,
            prerequisites=["Basic Arithmetic"]
        ),
        EntityFactory.create_topic(
            name="Geography",
            complexity_score=0.5,
            prerequisites=["Reading Comprehension"]
        ),
        EntityFactory.create_topic(
            name="History",
            complexity_score=0.6,
            prerequisites=["Reading Comprehension", "Critical Thinking"]
        ),
    ]
    
    # Register topics
    for topic in topics:
        await registrar.upsert_entity(topic, EntityTypeRegistry.TOPIC)
    
    # Create a user
    user = EntityFactory.create_user(
        id="student_001",
        session_id="session_2024_01"
    )
    await registrar.upsert_entity(user, EntityTypeRegistry.USER)
    
    logger.info(f"Registered {len(topics)} topics and 1 user")
    return user, topics


async def simulate_qa_interaction(
    registrar: EntityRegistrar,
    user: UserEntity,
    topics: list[TopicEntity]
):
    """Simulate a Q&A interaction."""
    logger.info("\n=== Simulating Q&A Interaction ===")
    
    # Create a question
    question = EntityFactory.create_question(
        content="What is the capital of France?",
        difficulty=DifficultyLevel.EASY,
        topics=["Geography", "Europe"]
    )
    
    # Register question
    await registrar.upsert_entity(question, EntityTypeRegistry.QUESTION)
    logger.info(f"Created question: {question.content}")
    
    # Simulate user answering
    answer = EntityFactory.create_answer(
        content="Paris",
        question_id=question.id,
        user_id=user.id,
        status=AnswerStatus.CORRECT,
        response_time_seconds=3.5,
        confidence_score=0.95
    )
    
    # Register answer
    await registrar.upsert_entity(answer, EntityTypeRegistry.ANSWER)
    logger.info(f"User answered: {answer.content} ({answer.status.value})")
    
    # Create relationships
    relationships = []
    
    # User answered question
    answered_rel = RelationshipBuilder.create_answered(
        user_id=user.id,
        question_id=question.id,
        answer_id=answer.id,
        status=answer.status,
        time_taken=answer.response_time_seconds,
        confidence=answer.confidence_score
    )
    relationships.append(answered_rel)
    
    # Question requires geography knowledge
    requires_rel = RelationshipBuilder.create_requires_knowledge(
        question_id=question.id,
        topic_id="Geography",
        relevance=0.9,
        is_prerequisite=False
    )
    relationships.append(requires_rel)
    
    # Update user mastery
    mastery_rel = RelationshipBuilder.create_mastery(
        user_id=user.id,
        topic_id="Geography",
        score=0.75,
        learning_rate=0.1
    )
    relationships.append(mastery_rel)
    
    # Create Q&A episode
    evaluation = {
        "correct": True,
        "feedback": "Excellent! Paris is indeed the capital of France.",
        "score": 1.0
    }
    
    episode_data = EpisodeBuilder.build_qa_episode(
        question=question,
        answer=answer,
        user=user,
        evaluation=evaluation
    )
    
    # Register episode
    episode_id = await registrar.upsert_episode(episode_data)
    logger.info(f"Created Q&A episode: {episode_id}")
    
    return question, answer, relationships


async def create_session_summary(
    registrar: EntityRegistrar,
    user: UserEntity,
    questions_answered: int = 10
):
    """Create a session summary episode."""
    logger.info("\n=== Creating Session Summary ===")
    
    # Simulate session statistics
    session_stats = {
        "total_questions": questions_answered,
        "correct_answers": 8,
        "success_rate": 0.8,
        "average_response_time": 4.2,
        "improved_topics": ["Geography", "History"],
        "struggling_topics": ["Mathematics"],
        "session_duration_minutes": 15.5
    }
    
    topics_covered = ["Geography", "History", "Mathematics"]
    
    # Create session summary episode
    summary_episode = EpisodeBuilder.build_session_summary_episode(
        user=user,
        session_stats=session_stats,
        topics_covered=topics_covered
    )
    
    # Register episode
    episode_id = await registrar.upsert_episode(summary_episode)
    logger.info(f"Created session summary episode: {episode_id}")
    
    # Update user entity with session results
    user.total_questions += questions_answered
    user.correct_answers += session_stats["correct_answers"]
    user.average_response_time = session_stats["average_response_time"]
    
    # Re-register updated user
    await registrar.upsert_entity(user, EntityTypeRegistry.USER)
    logger.info("Updated user statistics")
    
    return session_stats


async def demonstrate_entity_conversion():
    """Demonstrate entity conversion to Graphiti format."""
    logger.info("\n=== Entity Conversion Examples ===")
    
    # Question entity conversion
    question = QuestionEntity(
        content="What is the Pythagorean theorem?",
        difficulty=DifficultyLevel.MEDIUM,
        topics=["Mathematics", "Geometry"],
        asked_count=15,
        correct_rate=0.67
    )
    
    graphiti_question = EntityAdapter.to_graphiti_entity(
        question,
        EntityTypeRegistry.QUESTION
    )
    
    logger.info(f"Question Entity:")
    logger.info(f"  Name: {graphiti_question.name}")
    logger.info(f"  Type: {graphiti_question.entity_type}")
    logger.info(f"  Facts: {len(graphiti_question.observations)} observations")
    for fact in graphiti_question.observations[:3]:
        logger.info(f"    - {fact}")
    
    # Relationship to facts conversion
    mastery_rel = MasteryRelationship(
        source_id="user123",
        target_id="Mathematics",
        mastery_score=0.82,
        learning_rate=0.15,
        forgetting_rate=0.02,
        last_practice_date=datetime.utcnow()
    )
    
    rel_facts = RelationshipAdapter.to_graphiti_facts(mastery_rel)
    logger.info(f"\nMastery Relationship Facts:")
    for fact in rel_facts:
        logger.info(f"  - {fact}")


async def main():
    """Run the example demonstration."""
    try:
        # Set up Graphiti client
        logger.info("Setting up Graphiti client...")
        client = await setup_graphiti_client()
        
        # Create registrar
        registrar = EntityRegistrar(client)
        
        # Register initial entities
        user, topics = await register_initial_entities(registrar)
        
        # Simulate Q&A interaction
        question, answer, relationships = await simulate_qa_interaction(
            registrar, user, topics
        )
        
        # Create session summary
        session_stats = await create_session_summary(registrar, user)
        
        # Demonstrate entity conversion
        await demonstrate_entity_conversion()
        
        # Check cache
        logger.info(f"\n=== Cache Status ===")
        logger.info(f"Cached entities: {len(registrar._entity_cache)}")
        
        # Close client
        await client.close()
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())