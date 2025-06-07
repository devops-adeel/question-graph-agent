"""
Graphiti client for managing temporal knowledge graph interactions.

This module provides the main client interface for interacting with Graphiti,
handling entity and relationship management, episode creation, and memory
retrieval for the question-graph agent.

Note: The actual graphiti_python package integration is pending. Currently using
mock classes for development. When graphiti_python is available, remove the mock
imports and use the actual package.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Tuple, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager
import uuid

# Mock imports for now - will be replaced with actual graphiti_python when available
try:
    from graphiti_python import Graphiti, Entity, Relationship, Episode
except ImportError:
    # Mock classes for development
    class Graphiti:
        def __init__(self, driver, config):
            self.driver = driver
            self.config = config
        async def add_entity(self, entity): pass
        async def add_relationship(self, rel): pass
        async def add_episode(self, episode): pass
    
    class Entity: pass
    class Relationship: pass
    class Episode: pass

from neo4j import AsyncDriver

from graphiti_config import get_config, RuntimeConfig
from graphiti_connection import Neo4jConnectionManager, GraphitiConnectionManager
from graphiti_entities import (
    BaseEntity, 
    QuestionEntity, 
    AnswerEntity, 
    UserEntity, 
    TopicEntity,
    EntityFactory
)
from graphiti_relationships import (
    BaseRelationship,
    AnsweredRelationship,
    RequiresKnowledgeRelationship,
    MasteryRelationship,
)
from graphiti_registry import EntityAdapter, EpisodeBuilder
from graphiti_health import HealthStatus, check_system_health
from graphiti_fallback import FallbackManager, get_fallback_manager, with_fallback
from graphiti_circuit_breaker import get_circuit_breaker, circuit_breaker


logger = logging.getLogger(__name__)


T = TypeVar('T')


@dataclass
class GraphitiSession:
    """Represents an active Graphiti session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default_user"
    start_time: datetime = field(default_factory=datetime.now)
    episode_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    
    def increment_episode(self):
        """Increment episode counter."""
        self.episode_count += 1
    
    def increment_entity(self):
        """Increment entity counter."""
        self.entity_count += 1
    
    def increment_relationship(self):
        """Increment relationship counter."""
        self.relationship_count += 1


class GraphitiClient:
    """Main client for Graphiti interactions with fallback support."""
    
    def __init__(self, 
                 config: Optional[RuntimeConfig] = None,
                 enable_fallback: bool = True,
                 enable_circuit_breaker: bool = True):
        """Initialize Graphiti client.
        
        Args:
            config: Runtime configuration
            enable_fallback: Enable fallback mechanisms
            enable_circuit_breaker: Enable circuit breaker
        """
        self.config = config or get_config()
        self._graphiti: Optional[Graphiti] = None
        self._neo4j_manager = Neo4jConnectionManager(self.config)
        self._graphiti_manager = GraphitiConnectionManager(self.config)
        self._session = GraphitiSession()
        
        # Fallback and resilience
        self.enable_fallback = enable_fallback
        self.enable_circuit_breaker = enable_circuit_breaker
        self.fallback_manager = get_fallback_manager() if enable_fallback else None
        self.circuit_breaker = get_circuit_breaker("graphiti") if enable_circuit_breaker else None
        
        # Adapters
        self.entity_adapter = EntityAdapter()
        self.episode_builder = EpisodeBuilder()
        
        logger.info(f"Initialized GraphitiClient with session {self._session.session_id}")
    
    async def connect(self) -> bool:
        """Connect to Graphiti service.
        
        Returns:
            True if connection successful
        """
        try:
            # Check system health first
            health = await check_system_health()
            if health['status'] == 'unhealthy':
                logger.warning("System unhealthy, using fallback if enabled")
                if self.enable_fallback:
                    await self.fallback_manager.check_and_activate()
                return False
            
            # Connect to Neo4j
            driver = await self._neo4j_manager.connect_async()
            
            # Initialize Graphiti
            self._graphiti = Graphiti(
                driver=driver,
                config={
                    "llm_model": self.config.graphiti.llm_model,
                    "embedder_model": self.config.graphiti.embedder_model,
                }
            )
            
            logger.info("Successfully connected to Graphiti")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Graphiti: {e}")
            if self.enable_fallback:
                await self.fallback_manager.check_and_activate()
            return False
    
    async def disconnect(self):
        """Disconnect from Graphiti service."""
        await self._neo4j_manager.close_async()
        if self._graphiti_manager:
            await self._graphiti_manager.close_async()
        logger.info(f"Disconnected GraphitiClient session {self._session.session_id}")
    
    @asynccontextmanager
    async def session_context(self):
        """Context manager for Graphiti session."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()
    
    @with_fallback
    @circuit_breaker()
    async def store_question(self, question: QuestionEntity) -> bool:
        """Store a question entity in Graphiti.
        
        Args:
            question: Question entity to store
            
        Returns:
            True if successful
        """
        try:
            # Convert to Graphiti entity
            graphiti_entity = self.entity_adapter.to_graphiti_entity(
                question, 
                "question"
            )
            
            # Store in Graphiti
            await self._graphiti.add_entity(graphiti_entity)
            
            self._session.increment_entity()
            logger.info(f"Stored question entity: {question.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store question: {e}")
            raise
    
    @with_fallback
    @circuit_breaker()
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
            answer_entity = self.entity_adapter.to_graphiti_entity(
                answer,
                "answer"
            )
            await self._graphiti.add_entity(answer_entity)
            
            # Create answered relationship
            answered_rel = AnsweredRelationship(
                source_id=user.id,
                target_id=question.id,
                timestamp=answer.timestamp,
                answer_id=answer.id,
                time_taken=answer.response_time or 0.0,
                confidence=answer.confidence_score or 0.5,
            )
            
            graphiti_rel = self.entity_adapter.to_graphiti_relationship(
                answered_rel,
                "answered"
            )
            await self._graphiti.add_relationship(graphiti_rel)
            
            self._session.increment_entity()
            self._session.increment_relationship()
            
            logger.info(f"Stored answer {answer.id} with relationships")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store answer: {e}")
            raise
    
    @with_fallback
    @circuit_breaker()
    async def create_qa_episode(self,
                               question: QuestionEntity,
                               answer: AnswerEntity,
                               user: UserEntity,
                               evaluation_correct: bool) -> bool:
        """Create an episode for a Q&A interaction.
        
        Args:
            question: Question asked
            answer: Answer given
            user: User who answered
            evaluation_correct: Whether answer was correct
            
        Returns:
            True if successful
        """
        try:
            episode = self.episode_builder.build_qa_episode(
                question=question,
                answer=answer,
                user=user,
                correct=evaluation_correct,
                session_id=self._session.session_id
            )
            
            await self._graphiti.add_episode(episode)
            
            self._session.increment_episode()
            logger.info(f"Created Q&A episode for question {question.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create episode: {e}")
            raise
    
    @with_fallback
    @circuit_breaker()
    async def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's question history.
        
        Args:
            user_id: User ID
            limit: Maximum number of results
            
        Returns:
            List of question-answer pairs
        """
        try:
            query = """
            MATCH (u:User {id: $user_id})-[r:ANSWERED]->(q:Question)
            OPTIONAL MATCH (a:Answer {id: r.answer_id})
            RETURN q, a, r
            ORDER BY r.timestamp DESC
            LIMIT $limit
            """
            
            results = await self._neo4j_manager.execute_query_async(
                query,
                {"user_id": user_id, "limit": limit}
            )
            
            history = []
            for record in results:
                history.append({
                    "question": record["q"],
                    "answer": record["a"],
                    "relationship": record["r"],
                    "timestamp": record["r"]["timestamp"] if record["r"] else None
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get user history: {e}")
            raise
    
    @with_fallback
    @circuit_breaker()
    async def get_related_questions(self, 
                                   topic: str, 
                                   difficulty: Optional[str] = None,
                                   limit: int = 5) -> List[QuestionEntity]:
        """Get questions related to a topic.
        
        Args:
            topic: Topic name
            difficulty: Optional difficulty filter
            limit: Maximum number of results
            
        Returns:
            List of related questions
        """
        try:
            query = """
            MATCH (q:Question)-[:REQUIRES_KNOWLEDGE]->(t:Topic {name: $topic})
            """
            
            if difficulty:
                query += " WHERE q.difficulty = $difficulty"
            
            query += """
            RETURN q
            ORDER BY q.asked_count ASC, q.created_at DESC
            LIMIT $limit
            """
            
            params = {"topic": topic, "limit": limit}
            if difficulty:
                params["difficulty"] = difficulty
            
            results = await self._neo4j_manager.execute_query_async(query, params)
            
            questions = []
            for record in results:
                q_data = dict(record["q"])
                questions.append(EntityFactory.create_entity("question", q_data))
            
            return questions
            
        except Exception as e:
            logger.error(f"Failed to get related questions: {e}")
            raise
    
    @with_fallback
    @circuit_breaker()
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
            # Get or create mastery relationship
            query = """
            MATCH (u:User {id: $user_id}), (t:Topic {id: $topic_id})
            MERGE (u)-[m:HAS_MASTERY]->(t)
            ON CREATE SET 
                m.mastery_score = $initial_score,
                m.learning_rate = 0.1,
                m.forgetting_rate = 0.01,
                m.last_reviewed = datetime(),
                m.total_attempts = 0,
                m.correct_attempts = 0
            SET
                m.total_attempts = m.total_attempts + 1,
                m.correct_attempts = CASE WHEN $correct THEN m.correct_attempts + 1 ELSE m.correct_attempts END,
                m.last_reviewed = datetime()
            RETURN m
            """
            
            initial_score = 0.5 if correct else 0.2
            
            result = await self._neo4j_manager.execute_query_async(
                query,
                {
                    "user_id": user.id,
                    "topic_id": topic.id,
                    "initial_score": initial_score,
                    "correct": correct
                }
            )
            
            if result:
                # Update mastery score based on performance
                mastery = result[0]["m"]
                current_score = mastery.get("mastery_score", 0.5)
                learning_rate = mastery.get("learning_rate", 0.1)
                
                # Simple learning algorithm
                if correct:
                    new_score = min(1.0, current_score + learning_rate * (1 - current_score))
                else:
                    new_score = max(0.0, current_score - learning_rate * current_score)
                
                # Update score
                update_query = """
                MATCH (u:User {id: $user_id})-[m:HAS_MASTERY]->(t:Topic {id: $topic_id})
                SET m.mastery_score = $new_score
                """
                
                await self._neo4j_manager.execute_query_async(
                    update_query,
                    {
                        "user_id": user.id,
                        "topic_id": topic.id,
                        "new_score": new_score
                    }
                )
                
                logger.info(f"Updated mastery for user {user.id} on topic {topic.id}: {new_score:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update mastery: {e}")
            raise
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics.
        
        Returns:
            Session statistics
        """
        duration = (datetime.now() - self._session.start_time).total_seconds()
        
        return {
            "session_id": self._session.session_id,
            "user_id": self._session.user_id,
            "duration_seconds": duration,
            "episode_count": self._session.episode_count,
            "entity_count": self._session.entity_count,
            "relationship_count": self._session.relationship_count,
            "fallback_active": self.fallback_manager.state.is_active if self.fallback_manager else False,
            "circuit_state": self.circuit_breaker.state.value if self.circuit_breaker else "disabled"
        }