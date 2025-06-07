"""
Memory storage functions for Q&A pairs in Graphiti.

This module provides specialized functions for storing question-answer pairs
in the temporal knowledge graph, including metadata extraction, relationship
creation, and episode generation.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import json

from graphiti_client import GraphitiClient
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
)
from graphiti_registry import EpisodeBuilder
from entity_extraction import EntityExtractor
from graphiti_config import get_config, RuntimeConfig


logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Represents a question-answer pair with metadata."""
    question: str
    answer: str
    user_id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    question_id: Optional[str] = None
    answer_id: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    correct: Optional[bool] = None
    evaluation_comment: Optional[str] = None
    response_time: Optional[float] = None
    confidence_score: Optional[float] = None
    
    def __post_init__(self):
        """Generate IDs if not provided."""
        if not self.question_id:
            # Generate deterministic ID based on question content
            question_hash = hashlib.md5(self.question.encode()).hexdigest()[:8]
            self.question_id = f"q_{question_hash}_{int(self.timestamp.timestamp())}"
        
        if not self.answer_id:
            # Generate unique ID for answer
            answer_hash = hashlib.md5(
                f"{self.question_id}_{self.user_id}_{self.answer}".encode()
            ).hexdigest()[:8]
            self.answer_id = f"a_{answer_hash}_{int(self.timestamp.timestamp())}"


class MemoryStorage:
    """Handles storage of Q&A pairs in Graphiti."""
    
    def __init__(self, 
                 client: Optional[GraphitiClient] = None,
                 config: Optional[RuntimeConfig] = None):
        """Initialize memory storage.
        
        Args:
            client: GraphitiClient instance
            config: Runtime configuration
        """
        self.client = client
        self.config = config or get_config()
        self.entity_extractor = EntityExtractor()
        self.episode_builder = EpisodeBuilder()
        self._topic_cache: Dict[str, TopicEntity] = {}
    
    async def store_qa_pair(self, qa_pair: QAPair) -> bool:
        """Store a complete Q&A pair with all relationships.
        
        Args:
            qa_pair: Question-answer pair to store
            
        Returns:
            True if successful
        """
        if not self.client:
            logger.warning("No GraphitiClient available for storage")
            return False
        
        try:
            logger.info(f"Storing Q&A pair: {qa_pair.question_id}")
            
            # 1. Create entities
            question_entity = await self._create_question_entity(qa_pair)
            answer_entity = await self._create_answer_entity(qa_pair)
            user_entity = await self._get_or_create_user(qa_pair.user_id, qa_pair.session_id)
            
            # 2. Store question
            await self.client.store_question(question_entity)
            
            # 3. Store answer with relationship
            await self.client.store_answer(answer_entity, question_entity, user_entity)
            
            # 4. Create topic relationships
            await self._create_topic_relationships(question_entity, qa_pair.topics)
            
            # 5. Update user mastery if evaluated
            if qa_pair.correct is not None and qa_pair.topics:
                for topic_name in qa_pair.topics:
                    topic = await self._get_or_create_topic(topic_name)
                    await self.client.update_user_mastery(
                        user=user_entity,
                        topic=topic,
                        correct=qa_pair.correct,
                        time_taken=qa_pair.response_time or 0.0
                    )
            
            # 6. Create episode
            await self.client.create_qa_episode(
                question=question_entity,
                answer=answer_entity,
                user=user_entity,
                evaluation_correct=qa_pair.correct or False
            )
            
            logger.info(f"Successfully stored Q&A pair: {qa_pair.question_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store Q&A pair: {e}")
            return False
    
    async def store_question_only(self, 
                                 question: str,
                                 topics: Optional[List[str]] = None,
                                 difficulty: Optional[DifficultyLevel] = None) -> Optional[QuestionEntity]:
        """Store just a question without an answer.
        
        Args:
            question: Question text
            topics: Optional topic list
            difficulty: Optional difficulty level
            
        Returns:
            QuestionEntity if successful
        """
        if not self.client:
            return None
        
        try:
            # Extract topics if not provided
            if topics is None:
                extracted = await self.entity_extractor.extract_from_text(question)
                topics = [e["name"] for e in extracted.get("topics", [])]
            
            # Create question entity
            question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
            question_entity = QuestionEntity(
                id=f"q_{question_hash}_{int(datetime.now().timestamp())}",
                content=question,
                difficulty=difficulty or DifficultyLevel.MEDIUM,
                topics=topics
            )
            
            # Store question
            await self.client.store_question(question_entity)
            
            # Create topic relationships
            await self._create_topic_relationships(question_entity, topics)
            
            return question_entity
            
        except Exception as e:
            logger.error(f"Failed to store question: {e}")
            return None
    
    async def store_batch_qa_pairs(self, qa_pairs: List[QAPair]) -> Tuple[int, int]:
        """Store multiple Q&A pairs in batch.
        
        Args:
            qa_pairs: List of Q&A pairs to store
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        for qa_pair in qa_pairs:
            try:
                if await self.store_qa_pair(qa_pair):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to store Q&A pair {qa_pair.question_id}: {e}")
                failed += 1
        
        logger.info(f"Batch storage complete: {successful} successful, {failed} failed")
        return successful, failed
    
    async def update_answer_evaluation(self,
                                     answer_id: str,
                                     correct: bool,
                                     comment: Optional[str] = None) -> bool:
        """Update an answer's evaluation status.
        
        Args:
            answer_id: Answer ID to update
            correct: Whether answer was correct
            comment: Optional evaluation comment
            
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            # Update answer entity
            query = """
            MATCH (a:Answer {id: $answer_id})
            SET a.status = $status,
                a.evaluation_comment = $comment,
                a.evaluated_at = datetime()
            RETURN a
            """
            
            status = AnswerStatus.CORRECT if correct else AnswerStatus.INCORRECT
            
            result = await self.client._neo4j_manager.execute_query_async(
                query,
                {
                    "answer_id": answer_id,
                    "status": status.value,
                    "comment": comment
                }
            )
            
            return len(result) > 0
            
        except Exception as e:
            logger.error(f"Failed to update answer evaluation: {e}")
            return False
    
    async def _create_question_entity(self, qa_pair: QAPair) -> QuestionEntity:
        """Create question entity from Q&A pair."""
        # Extract additional metadata if needed
        if not qa_pair.topics:
            extracted = await self.entity_extractor.extract_from_text(qa_pair.question)
            qa_pair.topics = [e["name"] for e in extracted.get("topics", [])]
        
        return QuestionEntity(
            id=qa_pair.question_id,
            content=qa_pair.question,
            difficulty=qa_pair.difficulty,
            topics=qa_pair.topics,
            asked_count=1,
            correct_rate=1.0 if qa_pair.correct else 0.0
        )
    
    async def _create_answer_entity(self, qa_pair: QAPair) -> AnswerEntity:
        """Create answer entity from Q&A pair."""
        status = AnswerStatus.PENDING
        if qa_pair.correct is not None:
            status = AnswerStatus.CORRECT if qa_pair.correct else AnswerStatus.INCORRECT
        
        return AnswerEntity(
            id=qa_pair.answer_id,
            question_id=qa_pair.question_id,
            user_id=qa_pair.user_id,
            content=qa_pair.answer,
            status=status,
            timestamp=qa_pair.timestamp,
            response_time=qa_pair.response_time,
            confidence_score=qa_pair.confidence_score,
            evaluation_comment=qa_pair.evaluation_comment
        )
    
    async def _get_or_create_user(self, user_id: str, session_id: str) -> UserEntity:
        """Get existing user or create new one."""
        # For now, create a new user entity
        # In production, this would check if user exists
        return UserEntity(
            id=user_id,
            session_id=session_id,
            total_questions=0,
            correct_answers=0,
            average_response_time=0.0
        )
    
    async def _get_or_create_topic(self, topic_name: str) -> TopicEntity:
        """Get existing topic or create new one."""
        # Check cache first
        if topic_name in self._topic_cache:
            return self._topic_cache[topic_name]
        
        # Create new topic
        topic_id = f"topic_{hashlib.md5(topic_name.encode()).hexdigest()[:8]}"
        topic = TopicEntity(
            id=topic_id,
            name=topic_name,
            complexity_score=0.5  # Default complexity
        )
        
        # Cache for future use
        self._topic_cache[topic_name] = topic
        
        # Store in Graphiti
        if self.client:
            try:
                graphiti_entity = self.client.entity_adapter.to_graphiti_entity(
                    topic,
                    "topic"
                )
                await self.client._graphiti.add_entity(graphiti_entity)
            except Exception as e:
                logger.error(f"Failed to store topic {topic_name}: {e}")
        
        return topic
    
    async def _create_topic_relationships(self, 
                                        question: QuestionEntity,
                                        topics: List[str]) -> None:
        """Create relationships between question and topics."""
        if not self.client or not topics:
            return
        
        for topic_name in topics:
            try:
                topic = await self._get_or_create_topic(topic_name)
                
                # Create requires_knowledge relationship
                relationship = RequiresKnowledgeRelationship(
                    source_id=question.id,
                    target_id=topic.id,
                    relevance_score=1.0  # Could be calculated based on topic extraction confidence
                )
                
                graphiti_rel = self.client.entity_adapter.to_graphiti_relationship(
                    relationship,
                    "requires_knowledge"
                )
                await self.client._graphiti.add_relationship(graphiti_rel)
                
            except Exception as e:
                logger.error(f"Failed to create topic relationship for {topic_name}: {e}")


class MemoryAnalytics:
    """Analytics functions for stored Q&A memories."""
    
    def __init__(self, client: Optional[GraphitiClient] = None):
        """Initialize memory analytics."""
        self.client = client
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary of user statistics
        """
        if not self.client:
            return {}
        
        try:
            query = """
            MATCH (u:User {id: $user_id})
            OPTIONAL MATCH (u)-[r:ANSWERED]->(q:Question)
            OPTIONAL MATCH (a:Answer {user_id: $user_id})
            WITH u, 
                 count(DISTINCT q) as total_questions,
                 count(DISTINCT CASE WHEN a.status = 'CORRECT' THEN a END) as correct_answers,
                 avg(a.response_time) as avg_response_time,
                 collect(DISTINCT q.topics) as all_topics
            RETURN {
                user_id: u.id,
                total_questions: total_questions,
                correct_answers: correct_answers,
                accuracy: CASE WHEN total_questions > 0 
                          THEN toFloat(correct_answers) / total_questions 
                          ELSE 0.0 END,
                avg_response_time: avg_response_time,
                topics_encountered: size(all_topics)
            } as stats
            """
            
            result = await self.client._neo4j_manager.execute_query_async(
                query,
                {"user_id": user_id}
            )
            
            if result:
                return result[0]["stats"]
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {}
    
    async def get_topic_performance(self, 
                                  user_id: str,
                                  topic_name: str) -> Dict[str, Any]:
        """Get user's performance on a specific topic.
        
        Args:
            user_id: User ID
            topic_name: Topic name
            
        Returns:
            Performance statistics for the topic
        """
        if not self.client:
            return {}
        
        try:
            query = """
            MATCH (u:User {id: $user_id})-[r:ANSWERED]->(q:Question)-[:REQUIRES_KNOWLEDGE]->(t:Topic {name: $topic_name})
            OPTIONAL MATCH (a:Answer {user_id: $user_id, question_id: q.id})
            WITH count(DISTINCT q) as total_questions,
                 count(DISTINCT CASE WHEN a.status = 'CORRECT' THEN q END) as correct_questions,
                 avg(a.response_time) as avg_response_time,
                 collect(DISTINCT q.difficulty) as difficulties
            RETURN {
                topic: $topic_name,
                total_questions: total_questions,
                correct_questions: correct_questions,
                accuracy: CASE WHEN total_questions > 0 
                          THEN toFloat(correct_questions) / total_questions 
                          ELSE 0.0 END,
                avg_response_time: avg_response_time,
                difficulty_distribution: difficulties
            } as performance
            """
            
            result = await self.client._neo4j_manager.execute_query_async(
                query,
                {"user_id": user_id, "topic_name": topic_name}
            )
            
            if result:
                return result[0]["performance"]
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get topic performance: {e}")
            return {}
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a Q&A session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session summary statistics
        """
        if not self.client:
            return {}
        
        try:
            query = """
            MATCH (u:User {session_id: $session_id})-[r:ANSWERED]->(q:Question)
            OPTIONAL MATCH (a:Answer {user_id: u.id})
            WHERE a.timestamp >= u.created_at
            WITH u,
                 count(DISTINCT q) as questions_asked,
                 count(DISTINCT CASE WHEN a.status = 'CORRECT' THEN a END) as correct_answers,
                 min(a.timestamp) as start_time,
                 max(a.timestamp) as end_time,
                 avg(a.response_time) as avg_response_time,
                 collect(DISTINCT q.topics) as topics
            RETURN {
                session_id: $session_id,
                user_id: u.id,
                questions_asked: questions_asked,
                correct_answers: correct_answers,
                accuracy: CASE WHEN questions_asked > 0 
                          THEN toFloat(correct_answers) / questions_asked 
                          ELSE 0.0 END,
                duration_seconds: duration.between(start_time, end_time).seconds,
                avg_response_time: avg_response_time,
                topics_covered: topics
            } as summary
            """
            
            result = await self.client._neo4j_manager.execute_query_async(
                query,
                {"session_id": session_id}
            )
            
            if result:
                return result[0]["summary"]
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return {}