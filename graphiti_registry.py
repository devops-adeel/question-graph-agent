"""
Entity registry and episode builder for Graphiti integration.

This module provides registration and episode building functionality
for the question-graph agent's Graphiti integration.
"""

import logging
from typing import Dict, Any, Type, Optional, List
from datetime import datetime
import hashlib

from graphiti_entities import (
    BaseEntity,
    QuestionEntity,
    AnswerEntity,
    UserEntity,
    TopicEntity,
)
from graphiti_relationships import BaseRelationship


logger = logging.getLogger(__name__)


class EntityRegistry:
    """Registry for entity types and their handlers."""
    
    def __init__(self):
        """Initialize entity registry."""
        self._entity_types: Dict[str, Type[BaseEntity]] = {
            "question": QuestionEntity,
            "answer": AnswerEntity,
            "user": UserEntity,
            "topic": TopicEntity,
        }
        self._relationship_types: Dict[str, Type[BaseRelationship]] = {}
        self._handlers: Dict[str, Any] = {}
    
    def register_entity_type(self, name: str, entity_class: Type[BaseEntity]) -> None:
        """Register a new entity type.
        
        Args:
            name: Entity type name
            entity_class: Entity class
        """
        self._entity_types[name] = entity_class
        logger.info(f"Registered entity type: {name}")
    
    def register_relationship_type(self, name: str, rel_class: Type[BaseRelationship]) -> None:
        """Register a new relationship type.
        
        Args:
            name: Relationship type name
            rel_class: Relationship class
        """
        self._relationship_types[name] = rel_class
        logger.info(f"Registered relationship type: {name}")
    
    def get_entity_type(self, name: str) -> Optional[Type[BaseEntity]]:
        """Get entity type by name.
        
        Args:
            name: Entity type name
            
        Returns:
            Entity class or None
        """
        return self._entity_types.get(name)
    
    def get_relationship_type(self, name: str) -> Optional[Type[BaseRelationship]]:
        """Get relationship type by name.
        
        Args:
            name: Relationship type name
            
        Returns:
            Relationship class or None
        """
        return self._relationship_types.get(name)
    
    def list_entity_types(self) -> List[str]:
        """List all registered entity types."""
        return list(self._entity_types.keys())
    
    def list_relationship_types(self) -> List[str]:
        """List all registered relationship types."""
        return list(self._relationship_types.keys())


class EpisodeBuilder:
    """Builds episodes for temporal tracking in Graphiti."""
    
    def __init__(self):
        """Initialize episode builder."""
        self._episode_count = 0
    
    def build_qa_episode(self,
                        question: QuestionEntity,
                        answer: AnswerEntity,
                        user: UserEntity,
                        session_id: str,
                        correct: bool,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a Q&A episode.
        
        Args:
            question: Question entity
            answer: Answer entity
            user: User entity
            session_id: Session ID
            correct: Whether answer was correct
            metadata: Additional metadata
            
        Returns:
            Episode dictionary
        """
        self._episode_count += 1
        
        # Generate episode ID
        episode_id = self._generate_episode_id(
            user.id,
            question.id,
            answer.id,
            session_id
        )
        
        episode = {
            "id": episode_id,
            "type": "qa_interaction",
            "timestamp": answer.timestamp,
            "session_id": session_id,
            "entities": [
                {"id": user.id, "type": "user", "role": "answerer"},
                {"id": question.id, "type": "question", "role": "asked"},
                {"id": answer.id, "type": "answer", "role": "response"}
            ],
            "relationships": [
                {
                    "type": "answered",
                    "source": user.id,
                    "target": question.id,
                    "properties": {"answer_id": answer.id}
                }
            ],
            "metrics": {
                "correct": correct,
                "response_time": answer.response_time,
                "confidence_score": answer.confidence_score
            },
            "metadata": metadata or {}
        }
        
        # Add topic entities if present
        if question.topics:
            for topic_name in question.topics:
                topic_id = f"topic_{hashlib.md5(topic_name.encode()).hexdigest()[:8]}"
                episode["entities"].append({
                    "id": topic_id,
                    "type": "topic",
                    "role": "subject"
                })
                episode["relationships"].append({
                    "type": "requires_knowledge",
                    "source": question.id,
                    "target": topic_id,
                    "properties": {"relevance": 1.0}
                })
        
        return episode
    
    def build_session_episode(self,
                             user: UserEntity,
                             session_id: str,
                             questions: List[QuestionEntity],
                             summary_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Build a session summary episode.
        
        Args:
            user: User entity
            session_id: Session ID
            questions: List of questions in session
            summary_stats: Session statistics
            
        Returns:
            Episode dictionary
        """
        episode_id = f"episode_session_{session_id}_{int(datetime.now().timestamp())}"
        
        episode = {
            "id": episode_id,
            "type": "session_summary",
            "timestamp": datetime.now(),
            "session_id": session_id,
            "entities": [
                {"id": user.id, "type": "user", "role": "participant"}
            ],
            "metrics": summary_stats,
            "metadata": {
                "question_count": len(questions),
                "question_ids": [q.id for q in questions]
            }
        }
        
        return episode
    
    def build_mastery_episode(self,
                             user: UserEntity,
                             topic: TopicEntity,
                             mastery_level: float,
                             assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a mastery update episode.
        
        Args:
            user: User entity
            topic: Topic entity
            mastery_level: Current mastery level (0-1)
            assessment_data: Assessment details
            
        Returns:
            Episode dictionary
        """
        episode_id = self._generate_episode_id(
            user.id,
            topic.id,
            "mastery",
            str(int(datetime.now().timestamp()))
        )
        
        episode = {
            "id": episode_id,
            "type": "mastery_update",
            "timestamp": datetime.now(),
            "entities": [
                {"id": user.id, "type": "user", "role": "learner"},
                {"id": topic.id, "type": "topic", "role": "subject"}
            ],
            "relationships": [
                {
                    "type": "has_mastery",
                    "source": user.id,
                    "target": topic.id,
                    "properties": {
                        "level": mastery_level,
                        "confidence": assessment_data.get("confidence", 0.5)
                    }
                }
            ],
            "metrics": {
                "mastery_level": mastery_level,
                **assessment_data
            }
        }
        
        return episode
    
    def _generate_episode_id(self, *components: str) -> str:
        """Generate unique episode ID from components.
        
        Args:
            components: String components to hash
            
        Returns:
            Episode ID
        """
        combined = "_".join(str(c) for c in components)
        hash_value = hashlib.md5(combined.encode()).hexdigest()[:12]
        return f"episode_{hash_value}_{self._episode_count}"
    
    def get_episode_count(self) -> int:
        """Get total number of episodes created."""
        return self._episode_count
