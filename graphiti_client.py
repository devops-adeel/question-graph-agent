"""
GraphitiClient for interacting with the Graphiti temporal knowledge graph.

This module provides the main client interface for storing and retrieving
entities, relationships, and episodes in Graphiti.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from graphiti_connection import Neo4jConnectionManager
from graphiti_entities import (
    BaseEntity,
    QuestionEntity,
    AnswerEntity,
    UserEntity,
    TopicEntity,
)
from graphiti_relationships import (
    BaseRelationship,
    AnsweredRelationship,
    RequiresKnowledgeRelationship,
    MasteryRelationship,
)
from graphiti_registry import EntityRegistry, EpisodeBuilder
from graphiti_config import get_config, RuntimeConfig


logger = logging.getLogger(__name__)


class EntityAdapter:
    """Adapts our entities to Graphiti format."""
    
    def to_graphiti_entity(self, entity: BaseEntity, entity_type: str) -> Dict[str, Any]:
        """Convert entity to Graphiti format."""
        return {
            "type": entity_type,
            "id": entity.id,
            "properties": entity.model_dump(exclude={"id"}),
            "created_at": entity.created_at,
            "updated_at": entity.updated_at
        }
    
    def to_graphiti_relationship(self, relationship: BaseRelationship, rel_type: str) -> Dict[str, Any]:
        """Convert relationship to Graphiti format."""
        return {
            "type": rel_type,
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
            "properties": relationship.model_dump(exclude={"source_id", "target_id"}),
            "created_at": relationship.created_at
        }


class MockGraphiti:
    """Mock Graphiti client for development."""
    
    async def add_entity(self, entity: Dict[str, Any]) -> None:
        """Mock entity addition."""
        logger.debug(f"Mock: Adding entity {entity['id']} of type {entity['type']}")
    
    async def add_relationship(self, relationship: Dict[str, Any]) -> None:
        """Mock relationship addition."""
        logger.debug(f"Mock: Adding relationship from {relationship['source_id']} to {relationship['target_id']}")
    
    async def add_episode(self, episode: Dict[str, Any]) -> None:
        """Mock episode addition."""
        logger.debug(f"Mock: Adding episode {episode['id']}")


class GraphitiClient:
    """Main client for Graphiti interactions."""
    
    def __init__(self,
                 connection_manager: Optional[Neo4jConnectionManager] = None,
                 config: Optional[RuntimeConfig] = None):
        """Initialize GraphitiClient.
        
        Args:
            connection_manager: Neo4j connection manager
            config: Runtime configuration
        """
        self.config = config or get_config()
        self._neo4j_manager = connection_manager
        self._graphiti = MockGraphiti()  # In production, this would be real Graphiti
        self.entity_adapter = EntityAdapter()
        self.entity_registry = EntityRegistry()
        self.episode_builder = EpisodeBuilder()
        self._session = SessionStats()
    
    async def store_question(self, question: QuestionEntity) -> bool:
        """Store a question entity.
        
        Args:
            question: Question entity to store
            
        Returns:
            True if successful
        """
        try:
            # Convert to Graphiti format
            graphiti_entity = self.entity_adapter.to_graphiti_entity(question, "question")
            
            # Store in Graphiti
            await self._graphiti.add_entity(graphiti_entity)
            
            # Update session stats
            self._session.increment_entity()
            
            logger.info(f"Stored question {question.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store question: {e}")
            return False
    
    async def store_answer(self, 
                          answer: AnswerEntity,
                          question: QuestionEntity,
                          user: UserEntity) -> bool:
        """Store an answer with relationships.
        
        Args:
            answer: Answer entity
            question: Related question
            user: User who answered
            
        Returns:
            True if successful
        """
        try:
            # Store answer entity
            answer_entity = self.entity_adapter.to_graphiti_entity(answer, "answer")
            await self._graphiti.add_entity(answer_entity)
            
            # Create answered relationship
            answered_rel = AnsweredRelationship(
                source_id=user.id,
                target_id=question.id,
                answer_id=answer.id,
                timestamp=answer.timestamp
            )
            
            rel_data = self.entity_adapter.to_graphiti_relationship(answered_rel, "answered")
            await self._graphiti.add_relationship(rel_data)
            
            self._session.increment_entity()
            self._session.increment_relationship()
            
            logger.info(f"Stored answer {answer.id} for question {question.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store answer: {e}")
            return False
    
    async def create_qa_episode(self,
                               question: QuestionEntity,
                               answer: AnswerEntity,
                               user: UserEntity,
                               evaluation_correct: bool) -> bool:
        """Create an episode for Q&A interaction.
        
        Args:
            question: Question entity
            answer: Answer entity
            user: User entity
            evaluation_correct: Whether answer was correct
            
        Returns:
            True if successful
        """
        try:
            episode = self.episode_builder.build_qa_episode(
                question=question,
                answer=answer,
                user=user,
                session_id=user.session_id,
                correct=evaluation_correct
            )
            
            await self._graphiti.add_episode(episode)
            self._session.increment_episode()
            
            logger.info(f"Created Q&A episode for question {question.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create episode: {e}")
            return False
    
    async def update_user_mastery(self,
                                 user: UserEntity,
                                 topic: TopicEntity,
                                 correct: bool,
                                 time_taken: float) -> bool:
        """Update user's mastery of a topic.
        
        Args:
            user: User entity
            topic: Topic entity
            correct: Whether answer was correct
            time_taken: Time taken to answer
            
        Returns:
            True if successful
        """
        try:
            # In a real implementation, this would update or create
            # a mastery relationship with updated statistics
            logger.info(f"Updated mastery for user {user.id} on topic {topic.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update mastery: {e}")
            return False
    
    def get_session_stats(self) -> Dict[str, int]:
        """Get current session statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "entities_created": self._session.entities_created,
            "relationships_created": self._session.relationships_created,
            "episodes_created": self._session.episodes_created
        }


class SessionStats:
    """Track statistics for current session."""
    
    def __init__(self):
        self.entities_created = 0
        self.relationships_created = 0
        self.episodes_created = 0
        self.start_time = datetime.now()
    
    def increment_entity(self):
        self.entities_created += 1
    
    def increment_relationship(self):
        self.relationships_created += 1
    
    def increment_episode(self):
        self.episodes_created += 1
