"""
Memory update system for post-evaluation processing.

This module handles updating the knowledge graph after answer evaluation,
including user statistics, topic mastery, and evaluation records.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from graphiti_client import GraphitiClient
from graphiti_entities import (
    QuestionEntity,
    AnswerEntity,
    UserEntity,
    TopicEntity,
    AnswerStatus,
    DifficultyLevel,
)
from graphiti_relationships import (
    AnsweredRelationship,
    MasteryRelationship,
    EvaluatedRelationship,
)
from graphiti_memory import MemoryStorage, QAPair
from memory_retrieval import MemoryRetrieval
from graphiti_config import get_config, RuntimeConfig


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of answer evaluation with metadata."""
    question_id: str
    answer_id: str
    user_id: str
    session_id: str
    correct: bool
    evaluation_comment: str
    confidence_score: float = 0.0
    response_time: Optional[float] = None
    topics: List[str] = field(default_factory=list)
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UserPerformanceUpdate:
    """Update information for user performance metrics."""
    total_questions_delta: int = 0
    correct_answers_delta: int = 0
    response_time_sum: float = 0.0
    response_time_count: int = 0
    topics_attempted: List[str] = field(default_factory=list)
    difficulty_attempts: Dict[str, int] = field(default_factory=dict)


class MemoryUpdateService:
    """Service for updating memory after answer evaluation."""
    
    def __init__(self,
                 client: Optional[GraphitiClient] = None,
                 config: Optional[RuntimeConfig] = None):
        """Initialize memory update service.
        
        Args:
            client: GraphitiClient instance
            config: Runtime configuration
        """
        self.client = client
        self.config = config or get_config()
        self.storage = MemoryStorage(client=client, config=config)
        self.retrieval = MemoryRetrieval(client=client, config=config)
        self._pending_updates: List[EvaluationResult] = []
    
    async def record_evaluation(self,
                               evaluation_result: EvaluationResult,
                               immediate: bool = True) -> bool:
        """Record an evaluation result.
        
        Args:
            evaluation_result: Evaluation result to record
            immediate: Whether to update immediately or batch
            
        Returns:
            True if successful
        """
        if not self.client:
            logger.warning("No GraphitiClient available for recording evaluation")
            return False
        
        try:
            if immediate:
                return await self._process_evaluation(evaluation_result)
            else:
                self._pending_updates.append(evaluation_result)
                logger.info(f"Evaluation queued for batch update: {evaluation_result.answer_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to record evaluation: {e}")
            return False
    
    async def _process_evaluation(self, result: EvaluationResult) -> bool:
        """Process a single evaluation result.
        
        Args:
            result: Evaluation result to process
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Processing evaluation for answer {result.answer_id}")
            
            # 1. Update answer entity with evaluation
            success = await self.storage.update_answer_evaluation(
                answer_id=result.answer_id,
                correct=result.correct,
                comment=result.evaluation_comment
            )
            
            if not success:
                logger.error(f"Failed to update answer entity: {result.answer_id}")
                return False
            
            # 2. Create evaluated relationship
            await self._create_evaluation_relationship(result)
            
            # 3. Update user statistics
            await self._update_user_statistics(result)
            
            # 4. Update topic mastery
            if result.topics:
                await self._update_topic_mastery(result)
            
            # 5. Track performance trends
            await self._track_performance_trends(result)
            
            logger.info(f"Successfully processed evaluation for answer {result.answer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing evaluation: {e}")
            return False
    
    async def _create_evaluation_relationship(self, result: EvaluationResult) -> bool:
        """Create evaluation relationship in the graph.
        
        Args:
            result: Evaluation result
            
        Returns:
            True if successful
        """
        try:
            # Create evaluated relationship
            query = """
            MATCH (q:Question {id: $question_id})
            MATCH (a:Answer {id: $answer_id})
            MERGE (a)-[r:EVALUATED {
                timestamp: datetime(),
                correct: $correct,
                confidence: $confidence,
                comment: $comment
            }]->(q)
            RETURN r
            """
            
            params = {
                "question_id": result.question_id,
                "answer_id": result.answer_id,
                "correct": result.correct,
                "confidence": result.confidence_score,
                "comment": result.evaluation_comment
            }
            
            await self.client._neo4j_manager.execute_query_async(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to create evaluation relationship: {e}")
            return False
    
    async def _update_user_statistics(self, result: EvaluationResult) -> bool:
        """Update user statistics based on evaluation.
        
        Args:
            result: Evaluation result
            
        Returns:
            True if successful
        """
        try:
            # Update user node statistics
            query = """
            MATCH (u:User {id: $user_id})
            SET u.total_questions = COALESCE(u.total_questions, 0) + 1,
                u.correct_answers = COALESCE(u.correct_answers, 0) + CASE WHEN $correct THEN 1 ELSE 0 END,
                u.last_active = datetime()
            WITH u
            MATCH (a:Answer {user_id: $user_id})
            WITH u, avg(a.response_time) as avg_response
            SET u.average_response_time = avg_response
            RETURN u
            """
            
            params = {
                "user_id": result.user_id,
                "correct": result.correct
            }
            
            await self.client._neo4j_manager.execute_query_async(query, params)
            
            # Update session statistics
            await self._update_session_stats(result)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user statistics: {e}")
            return False
    
    async def _update_session_stats(self, result: EvaluationResult) -> bool:
        """Update session-level statistics.
        
        Args:
            result: Evaluation result
            
        Returns:
            True if successful
        """
        try:
            query = """
            MERGE (s:Session {id: $session_id})
            ON CREATE SET s.created_at = datetime(),
                         s.user_id = $user_id
            SET s.total_questions = COALESCE(s.total_questions, 0) + 1,
                s.correct_answers = COALESCE(s.correct_answers, 0) + CASE WHEN $correct THEN 1 ELSE 0 END,
                s.last_activity = datetime()
            RETURN s
            """
            
            params = {
                "session_id": result.session_id,
                "user_id": result.user_id,
                "correct": result.correct
            }
            
            await self.client._neo4j_manager.execute_query_async(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session statistics: {e}")
            return False
    
    async def _update_topic_mastery(self, result: EvaluationResult) -> bool:
        """Update user's mastery of topics.
        
        Args:
            result: Evaluation result
            
        Returns:
            True if successful
        """
        try:
            for topic_name in result.topics:
                # Calculate new mastery level
                mastery_data = await self._calculate_topic_mastery(
                    result.user_id,
                    topic_name,
                    result.correct,
                    result.difficulty
                )
                
                # Update or create mastery relationship
                query = """
                MATCH (u:User {id: $user_id})
                MATCH (t:Topic {name: $topic_name})
                MERGE (u)-[m:HAS_MASTERY]->(t)
                SET m.level = $mastery_level,
                    m.confidence = $confidence,
                    m.total_attempts = COALESCE(m.total_attempts, 0) + 1,
                    m.correct_attempts = COALESCE(m.correct_attempts, 0) + CASE WHEN $correct THEN 1 ELSE 0 END,
                    m.last_attempt = datetime(),
                    m.difficulty_distribution = COALESCE(m.difficulty_distribution, {}) + {$difficulty: 1}
                RETURN m
                """
                
                params = {
                    "user_id": result.user_id,
                    "topic_name": topic_name,
                    "mastery_level": mastery_data["level"],
                    "confidence": mastery_data["confidence"],
                    "correct": result.correct,
                    "difficulty": result.difficulty.value
                }
                
                await self.client._neo4j_manager.execute_query_async(query, params)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update topic mastery: {e}")
            return False
    
    async def _calculate_topic_mastery(self,
                                      user_id: str,
                                      topic_name: str,
                                      latest_correct: bool,
                                      latest_difficulty: DifficultyLevel) -> Dict[str, float]:
        """Calculate updated mastery level for a topic.
        
        Args:
            user_id: User ID
            topic_name: Topic name
            latest_correct: Whether latest answer was correct
            latest_difficulty: Difficulty of latest question
            
        Returns:
            Dictionary with mastery level and confidence
        """
        # Get current mastery data
        query = """
        MATCH (u:User {id: $user_id})-[m:HAS_MASTERY]->(t:Topic {name: $topic_name})
        RETURN m.level as current_level,
               m.confidence as current_confidence,
               m.total_attempts as total,
               m.correct_attempts as correct
        """
        
        result = await self.client._neo4j_manager.execute_query_async(
            query,
            {"user_id": user_id, "topic_name": topic_name}
        )
        
        if result:
            current = result[0]
            current_level = current.get("current_level", 0.5)
            current_confidence = current.get("current_confidence", 0.5)
            total_attempts = current.get("total", 0) + 1
            correct_attempts = current.get("correct", 0) + (1 if latest_correct else 0)
        else:
            current_level = 0.5
            current_confidence = 0.5
            total_attempts = 1
            correct_attempts = 1 if latest_correct else 0
        
        # Calculate new mastery level
        # Basic formula: weighted average of historical accuracy and recent performance
        historical_accuracy = correct_attempts / total_attempts if total_attempts > 0 else 0.5
        
        # Weight recent performance based on difficulty
        difficulty_weights = {
            DifficultyLevel.EASY: 0.2,
            DifficultyLevel.MEDIUM: 0.3,
            DifficultyLevel.HARD: 0.4,
            DifficultyLevel.EXPERT: 0.5
        }
        
        recent_weight = difficulty_weights.get(latest_difficulty, 0.3)
        recent_score = 1.0 if latest_correct else 0.0
        
        # Calculate new level
        new_level = (current_level * (1 - recent_weight)) + (recent_score * recent_weight)
        
        # Adjust based on overall accuracy
        if historical_accuracy > 0.8 and total_attempts >= 5:
            new_level = min(1.0, new_level * 1.1)  # Boost for consistent performance
        elif historical_accuracy < 0.3 and total_attempts >= 5:
            new_level = max(0.0, new_level * 0.9)  # Reduce for poor performance
        
        # Calculate confidence based on number of attempts
        confidence = min(1.0, total_attempts / 10)  # Full confidence after 10 attempts
        
        return {
            "level": new_level,
            "confidence": confidence
        }
    
    async def _track_performance_trends(self, result: EvaluationResult) -> bool:
        """Track performance trends over time.
        
        Args:
            result: Evaluation result
            
        Returns:
            True if successful
        """
        try:
            # Create performance event
            query = """
            CREATE (e:PerformanceEvent {
                id: $event_id,
                user_id: $user_id,
                session_id: $session_id,
                timestamp: datetime(),
                correct: $correct,
                difficulty: $difficulty,
                response_time: $response_time,
                confidence: $confidence,
                topics: $topics
            })
            WITH e
            MATCH (u:User {id: $user_id})
            MERGE (u)-[:HAS_PERFORMANCE_EVENT]->(e)
            RETURN e
            """
            
            params = {
                "event_id": f"perf_{result.answer_id}_{int(result.timestamp.timestamp())}",
                "user_id": result.user_id,
                "session_id": result.session_id,
                "correct": result.correct,
                "difficulty": result.difficulty.value,
                "response_time": result.response_time,
                "confidence": result.confidence_score,
                "topics": result.topics
            }
            
            await self.client._neo4j_manager.execute_query_async(query, params)
            
            # Check for streaks
            await self._update_streaks(result)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to track performance trends: {e}")
            return False
    
    async def _update_streaks(self, result: EvaluationResult) -> None:
        """Update user's answer streaks.
        
        Args:
            result: Evaluation result
        """
        try:
            if result.correct:
                # Update correct streak
                query = """
                MATCH (u:User {id: $user_id})
                SET u.current_streak = COALESCE(u.current_streak, 0) + 1,
                    u.best_streak = CASE 
                        WHEN COALESCE(u.current_streak, 0) + 1 > COALESCE(u.best_streak, 0)
                        THEN COALESCE(u.current_streak, 0) + 1
                        ELSE COALESCE(u.best_streak, 0)
                    END
                """
            else:
                # Reset streak
                query = """
                MATCH (u:User {id: $user_id})
                SET u.current_streak = 0
                """
            
            await self.client._neo4j_manager.execute_query_async(
                query,
                {"user_id": result.user_id}
            )
            
        except Exception as e:
            logger.error(f"Failed to update streaks: {e}")
    
    async def flush_pending_updates(self) -> Tuple[int, int]:
        """Process all pending evaluation updates.
        
        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not self._pending_updates:
            return 0, 0
        
        successful = 0
        failed = 0
        
        logger.info(f"Processing {len(self._pending_updates)} pending updates")
        
        for result in self._pending_updates:
            try:
                if await self._process_evaluation(result):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to process pending update: {e}")
                failed += 1
        
        self._pending_updates.clear()
        
        logger.info(f"Batch update complete: {successful} successful, {failed} failed")
        return successful, failed
    
    async def get_user_progress_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive progress summary for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Progress summary dictionary
        """
        if not self.client:
            return {}
        
        try:
            # Get overall statistics
            overall_query = """
            MATCH (u:User {id: $user_id})
            OPTIONAL MATCH (u)-[:HAS_MASTERY]->(t:Topic)
            RETURN u.total_questions as total_questions,
                   u.correct_answers as correct_answers,
                   u.average_response_time as avg_response_time,
                   u.current_streak as current_streak,
                   u.best_streak as best_streak,
                   count(DISTINCT t) as topics_practiced,
                   avg(t.level) as avg_mastery_level
            """
            
            result = await self.client._neo4j_manager.execute_query_async(
                overall_query,
                {"user_id": user_id}
            )
            
            if not result:
                return {}
            
            stats = result[0]
            
            # Get recent performance
            recent_query = """
            MATCH (u:User {id: $user_id})-[:HAS_PERFORMANCE_EVENT]->(e:PerformanceEvent)
            WHERE e.timestamp > datetime() - duration('P7D')
            WITH e
            ORDER BY e.timestamp DESC
            LIMIT 20
            RETURN count(e) as recent_total,
                   sum(CASE WHEN e.correct THEN 1 ELSE 0 END) as recent_correct,
                   avg(e.response_time) as recent_avg_time
            """
            
            recent_result = await self.client._neo4j_manager.execute_query_async(
                recent_query,
                {"user_id": user_id}
            )
            
            recent_stats = recent_result[0] if recent_result else {}
            
            # Get topic breakdown
            topic_query = """
            MATCH (u:User {id: $user_id})-[m:HAS_MASTERY]->(t:Topic)
            RETURN t.name as topic,
                   m.level as mastery_level,
                   m.total_attempts as attempts,
                   m.correct_attempts as correct,
                   m.last_attempt as last_practiced
            ORDER BY m.level DESC
            """
            
            topic_results = await self.client._neo4j_manager.execute_query_async(
                topic_query,
                {"user_id": user_id}
            )
            
            topics = [
                {
                    "name": t["topic"],
                    "mastery": t["mastery_level"],
                    "accuracy": t["correct"] / t["attempts"] if t["attempts"] > 0 else 0,
                    "attempts": t["attempts"],
                    "last_practiced": t["last_practiced"]
                }
                for t in topic_results
            ]
            
            return {
                "overall": {
                    "total_questions": stats.get("total_questions", 0),
                    "correct_answers": stats.get("correct_answers", 0),
                    "accuracy": stats.get("correct_answers", 0) / stats.get("total_questions", 1) if stats.get("total_questions", 0) > 0 else 0,
                    "avg_response_time": stats.get("avg_response_time", 0),
                    "current_streak": stats.get("current_streak", 0),
                    "best_streak": stats.get("best_streak", 0),
                    "topics_practiced": stats.get("topics_practiced", 0),
                    "avg_mastery": stats.get("avg_mastery_level", 0)
                },
                "recent_performance": {
                    "questions_last_7_days": recent_stats.get("recent_total", 0),
                    "correct_last_7_days": recent_stats.get("recent_correct", 0),
                    "recent_accuracy": recent_stats.get("recent_correct", 0) / recent_stats.get("recent_total", 1) if recent_stats.get("recent_total", 0) > 0 else 0,
                    "recent_avg_time": recent_stats.get("recent_avg_time", 0)
                },
                "topics": topics
            }
            
        except Exception as e:
            logger.error(f"Failed to get user progress summary: {e}")
            return {}


class EvaluationEventHandler:
    """Event handler for processing evaluation events."""
    
    def __init__(self, update_service: MemoryUpdateService):
        """Initialize event handler.
        
        Args:
            update_service: Memory update service instance
        """
        self.update_service = update_service
        self._event_queue: List[EvaluationResult] = []
        self._batch_size = 10
        self._batch_timeout = 5.0  # seconds
    
    async def handle_evaluation_event(self,
                                     question_id: str,
                                     answer_id: str,
                                     user_id: str,
                                     session_id: str,
                                     correct: bool,
                                     comment: str,
                                     confidence: float = 0.0,
                                     response_time: Optional[float] = None,
                                     topics: Optional[List[str]] = None,
                                     difficulty: Optional[DifficultyLevel] = None) -> None:
        """Handle an evaluation event.
        
        Args:
            question_id: Question ID
            answer_id: Answer ID
            user_id: User ID
            session_id: Session ID
            correct: Whether answer was correct
            comment: Evaluation comment
            confidence: Confidence score
            response_time: Response time in seconds
            topics: Question topics
            difficulty: Question difficulty
        """
        result = EvaluationResult(
            question_id=question_id,
            answer_id=answer_id,
            user_id=user_id,
            session_id=session_id,
            correct=correct,
            evaluation_comment=comment,
            confidence_score=confidence,
            response_time=response_time,
            topics=topics or [],
            difficulty=difficulty or DifficultyLevel.MEDIUM
        )
        
        # Add to queue
        self._event_queue.append(result)
        
        # Process immediately if batch size reached
        if len(self._event_queue) >= self._batch_size:
            await self._process_batch()
    
    async def _process_batch(self) -> None:
        """Process the current batch of events."""
        if not self._event_queue:
            return
        
        batch = self._event_queue.copy()
        self._event_queue.clear()
        
        logger.info(f"Processing batch of {len(batch)} evaluation events")
        
        for result in batch:
            await self.update_service.record_evaluation(result, immediate=True)
    
    async def flush(self) -> None:
        """Flush any remaining events."""
        await self._process_batch()
        await self.update_service.flush_pending_updates()