"""
Entity registration module for Graphiti integration.

This module handles the registration of custom entity types with Graphiti,
mapping our Pydantic models to Graphiti's entity system and ensuring proper
serialization and deserialization.
"""

import logging
from typing import Dict, Type, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from graphiti_core import Entity as GraphitiEntity, EntityType
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils import parse_json_datetime

from graphiti_entities import (
    BaseEntity,
    QuestionEntity,
    AnswerEntity,
    UserEntity,
    TopicEntity,
    DifficultyLevel,
    AnswerStatus,
)
from graphiti_relationships import (
    BaseRelationship,
    AnsweredRelationship,
    RequiresKnowledgeRelationship,
    MasteryRelationship,
)


logger = logging.getLogger(__name__)


class EntityTypeRegistry:
    """Registry for custom entity types in Graphiti."""
    
    # Define custom entity types
    QUESTION = "question"
    ANSWER = "answer"
    USER = "user"
    TOPIC = "topic"
    
    # Map to GraphitiEntity types
    _type_mapping: Dict[str, EntityType] = {
        QUESTION: EntityType.generic,  # Using generic until custom types supported
        ANSWER: EntityType.generic,
        USER: EntityType.person,  # User maps to person type
        TOPIC: EntityType.topic,  # Topic is natively supported
    }
    
    @classmethod
    def get_entity_type(cls, custom_type: str) -> EntityType:
        """Get Graphiti entity type for custom type."""
        return cls._type_mapping.get(custom_type, EntityType.generic)


class EntityAdapter:
    """Adapts Pydantic entities to Graphiti entities."""
    
    @staticmethod
    def to_graphiti_entity(entity: BaseEntity, entity_type: str) -> GraphitiEntity:
        """Convert Pydantic entity to Graphiti entity.
        
        Args:
            entity: Pydantic entity instance
            entity_type: Type identifier (question, answer, etc.)
            
        Returns:
            Graphiti entity instance
        """
        # Get appropriate Graphiti entity type
        graphiti_type = EntityTypeRegistry.get_entity_type(entity_type)
        
        # Build facts list based on entity type
        facts = EntityAdapter._build_facts(entity, entity_type)
        
        # Create Graphiti entity
        return GraphitiEntity(
            name=EntityAdapter._get_entity_name(entity, entity_type),
            entity_type=graphiti_type,
            observations=facts,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )
    
    @staticmethod
    def _get_entity_name(entity: BaseEntity, entity_type: str) -> str:
        """Generate appropriate name for entity."""
        if entity_type == EntityTypeRegistry.QUESTION:
            # Truncate question for name
            content = entity.content[:50]
            return f"Question: {content}..." if len(entity.content) > 50 else f"Question: {content}"
        elif entity_type == EntityTypeRegistry.ANSWER:
            return f"Answer by {entity.user_id} to {entity.question_id}"
        elif entity_type == EntityTypeRegistry.USER:
            return f"User {entity.id}"
        elif entity_type == EntityTypeRegistry.TOPIC:
            return entity.name
        else:
            return f"{entity_type}_{entity.id}"
    
    @staticmethod
    def _build_facts(entity: BaseEntity, entity_type: str) -> List[str]:
        """Build facts list for entity."""
        facts = []
        
        if entity_type == EntityTypeRegistry.QUESTION:
            question: QuestionEntity = entity
            facts.append(f"Content: {question.content}")
            facts.append(f"Difficulty: {question.difficulty.value}")
            facts.append(f"Topics: {', '.join(question.topics)}")
            facts.append(f"Asked {question.asked_count} times")
            facts.append(f"Success rate: {question.correct_rate:.2%}")
            
        elif entity_type == EntityTypeRegistry.ANSWER:
            answer: AnswerEntity = entity
            facts.append(f"Response: {answer.content}")
            facts.append(f"Status: {answer.status.value}")
            facts.append(f"Response time: {answer.response_time_seconds}s")
            if answer.confidence_score:
                facts.append(f"Confidence: {answer.confidence_score:.2f}")
            if answer.feedback:
                facts.append(f"Feedback: {answer.feedback}")
                
        elif entity_type == EntityTypeRegistry.USER:
            user: UserEntity = entity
            facts.append(f"Total questions: {user.total_questions}")
            facts.append(f"Correct answers: {user.correct_answers}")
            facts.append(f"Average response time: {user.average_response_time:.2f}s")
            if user.topics_mastered:
                facts.append(f"Mastered topics: {', '.join(user.topics_mastered)}")
                
        elif entity_type == EntityTypeRegistry.TOPIC:
            topic: TopicEntity = entity
            facts.append(f"Topic: {topic.name}")
            facts.append(f"Complexity: {topic.complexity_score:.2f}")
            if topic.parent_topic:
                facts.append(f"Parent: {topic.parent_topic}")
            if topic.prerequisites:
                facts.append(f"Prerequisites: {', '.join(topic.prerequisites)}")
        
        return facts


class RelationshipAdapter:
    """Adapts Pydantic relationships for Graphiti."""
    
    @staticmethod
    def to_graphiti_facts(relationship: BaseRelationship) -> List[str]:
        """Convert relationship to facts for Graphiti episodes.
        
        Args:
            relationship: Pydantic relationship instance
            
        Returns:
            List of facts describing the relationship
        """
        facts = []
        
        if isinstance(relationship, AnsweredRelationship):
            facts.append(f"{relationship.source_id} answered {relationship.target_id}")
            facts.append(f"Answer was {relationship.answer_status.value}")
            facts.append(f"Time taken: {relationship.time_taken_seconds}s")
            if relationship.confidence_score:
                facts.append(f"Confidence: {relationship.confidence_score:.2f}")
                
        elif isinstance(relationship, RequiresKnowledgeRelationship):
            facts.append(f"{relationship.source_id} requires knowledge of {relationship.target_id}")
            facts.append(f"Relevance: {relationship.relevance_score:.2f}")
            if relationship.is_prerequisite:
                facts.append("This is a prerequisite")
                
        elif isinstance(relationship, MasteryRelationship):
            facts.append(f"{relationship.source_id} has mastery of {relationship.target_id}")
            facts.append(f"Mastery score: {relationship.mastery_score:.2f}")
            facts.append(f"Learning rate: {relationship.learning_rate:.2f}")
            facts.append(f"Last practice: {relationship.last_practice_date}")
            
        return facts


class EpisodeBuilder:
    """Builds Graphiti episodes from Q&A interactions."""
    
    @staticmethod
    def build_qa_episode(
        question: QuestionEntity,
        answer: AnswerEntity,
        user: UserEntity,
        evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build episode for a Q&A interaction.
        
        Args:
            question: Question entity
            answer: Answer entity
            user: User entity
            evaluation: Evaluation results
            
        Returns:
            Episode data for Graphiti
        """
        # Build episode name
        episode_name = f"Q&A: {question.content[:50]}..."
        if len(question.content) <= 50:
            episode_name = f"Q&A: {question.content}"
        
        # Build episode content
        content_facts = [
            f"{user.id} was asked: {question.content}",
            f"They responded: {answer.content}",
            f"The answer was {answer.status.value}",
        ]
        
        if evaluation.get("feedback"):
            content_facts.append(f"Feedback: {evaluation['feedback']}")
        
        if evaluation.get("correct"):
            content_facts.append("This improved their mastery of related topics")
        else:
            content_facts.append("This identified a knowledge gap")
        
        # Build entity references
        entity_references = [
            {"entity_id": user.id, "entity_type": EntityTypeRegistry.USER},
            {"entity_id": question.id, "entity_type": EntityTypeRegistry.QUESTION},
            {"entity_id": answer.id, "entity_type": EntityTypeRegistry.ANSWER},
        ]
        
        # Add topic references
        for topic in question.topics:
            entity_references.append({
                "entity_id": topic,
                "entity_type": EntityTypeRegistry.TOPIC
            })
        
        return {
            "name": episode_name,
            "content": " ".join(content_facts),
            "episode_type": EpisodeType.interaction,
            "entity_references": entity_references,
            "timestamp": answer.created_at,
            "metadata": {
                "question_id": question.id,
                "answer_id": answer.id,
                "user_id": user.id,
                "correct": evaluation.get("correct", False),
                "difficulty": question.difficulty.value,
                "response_time": answer.response_time_seconds,
            }
        }
    
    @staticmethod
    def build_session_summary_episode(
        user: UserEntity,
        session_stats: Dict[str, Any],
        topics_covered: List[str]
    ) -> Dict[str, Any]:
        """Build episode for session summary.
        
        Args:
            user: User entity
            session_stats: Session statistics
            topics_covered: List of topics covered
            
        Returns:
            Episode data for Graphiti
        """
        # Build summary facts
        content_facts = [
            f"{user.id} completed a Q&A session",
            f"Questions answered: {session_stats['total_questions']}",
            f"Correct answers: {session_stats['correct_answers']}",
            f"Success rate: {session_stats['success_rate']:.2%}",
            f"Topics covered: {', '.join(topics_covered)}",
        ]
        
        if session_stats.get("improved_topics"):
            content_facts.append(f"Improved in: {', '.join(session_stats['improved_topics'])}")
        
        if session_stats.get("struggling_topics"):
            content_facts.append(f"Needs practice in: {', '.join(session_stats['struggling_topics'])}")
        
        # Build entity references
        entity_references = [
            {"entity_id": user.id, "entity_type": EntityTypeRegistry.USER}
        ]
        
        for topic in topics_covered:
            entity_references.append({
                "entity_id": topic,
                "entity_type": EntityTypeRegistry.TOPIC
            })
        
        return {
            "name": f"Session Summary for {user.id}",
            "content": " ".join(content_facts),
            "episode_type": EpisodeType.summary,
            "entity_references": entity_references,
            "timestamp": datetime.utcnow(),
            "metadata": session_stats
        }


class EntityRegistrar:
    """Handles registration of entities with Graphiti."""
    
    def __init__(self, graphiti_client):
        """Initialize registrar with Graphiti client.
        
        Args:
            graphiti_client: Initialized Graphiti client
        """
        self.client = graphiti_client
        self._registered_types = set()
        self._entity_cache: Dict[str, GraphitiEntity] = {}
    
    async def register_entity_types(self):
        """Register all custom entity types with Graphiti."""
        try:
            # Note: Graphiti may not support custom entity type registration
            # This is a placeholder for when the API supports it
            logger.info("Registering custom entity types with Graphiti")
            
            # For now, we use the mapping in EntityTypeRegistry
            for custom_type in [
                EntityTypeRegistry.QUESTION,
                EntityTypeRegistry.ANSWER,
                EntityTypeRegistry.USER,
                EntityTypeRegistry.TOPIC,
            ]:
                self._registered_types.add(custom_type)
                logger.info(f"Registered entity type: {custom_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register entity types: {e}")
            return False
    
    async def upsert_entity(self, entity: BaseEntity, entity_type: str) -> Optional[str]:
        """Insert or update an entity in Graphiti.
        
        Args:
            entity: Pydantic entity instance
            entity_type: Entity type identifier
            
        Returns:
            Entity ID if successful, None otherwise
        """
        try:
            # Convert to Graphiti entity
            graphiti_entity = EntityAdapter.to_graphiti_entity(entity, entity_type)
            
            # Add to Graphiti
            result = await self.client.add_entity(graphiti_entity)
            
            # Cache the entity
            self._entity_cache[entity.id] = graphiti_entity
            
            logger.info(f"Upserted {entity_type} entity: {entity.id}")
            return entity.id
            
        except Exception as e:
            logger.error(f"Failed to upsert entity {entity.id}: {e}")
            return None
    
    async def upsert_episode(self, episode_data: Dict[str, Any]) -> Optional[str]:
        """Insert an episode in Graphiti.
        
        Args:
            episode_data: Episode data dictionary
            
        Returns:
            Episode ID if successful, None otherwise
        """
        try:
            # Add episode to Graphiti
            result = await self.client.add_episode(
                name=episode_data["name"],
                content=episode_data["content"],
                episode_type=episode_data["episode_type"],
                entity_references=episode_data["entity_references"],
                timestamp=episode_data["timestamp"],
                metadata=episode_data.get("metadata", {})
            )
            
            logger.info(f"Created episode: {episode_data['name']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create episode: {e}")
            return None
    
    async def batch_upsert_entities(
        self,
        entities: List[tuple[BaseEntity, str]]
    ) -> Dict[str, bool]:
        """Batch upsert multiple entities.
        
        Args:
            entities: List of (entity, entity_type) tuples
            
        Returns:
            Dictionary mapping entity IDs to success status
        """
        results = {}
        
        for entity, entity_type in entities:
            success = await self.upsert_entity(entity, entity_type)
            results[entity.id] = success is not None
        
        return results
    
    def get_cached_entity(self, entity_id: str) -> Optional[GraphitiEntity]:
        """Get cached entity if available.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Cached Graphiti entity or None
        """
        return self._entity_cache.get(entity_id)