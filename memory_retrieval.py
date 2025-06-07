"""
Memory retrieval functions for intelligent question generation.

This module provides functions to retrieve relevant information from the
Graphiti knowledge graph to enable context-aware and personalized question
generation based on user history and performance.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import random

from graphiti_client import GraphitiClient
from graphiti_entities import (
    QuestionEntity,
    DifficultyLevel,
    AnswerStatus,
    UserEntity,
    TopicEntity,
)
from graphiti_config import get_config, RuntimeConfig


logger = logging.getLogger(__name__)


class MemoryRetrieval:
    """Handles retrieval of memories for question generation."""
    
    def __init__(self,
                 client: Optional[GraphitiClient] = None,
                 config: Optional[RuntimeConfig] = None):
        """Initialize memory retrieval.
        
        Args:
            client: GraphitiClient instance
            config: Runtime configuration
        """
        self.client = client
        self.config = config or get_config()
        self._question_cache: Dict[str, QuestionEntity] = {}
    
    async def get_asked_questions(self, 
                                  user_id: str,
                                  session_id: Optional[str] = None,
                                  limit: int = 50) -> List[QuestionEntity]:
        """Get questions previously asked to a user.
        
        Args:
            user_id: User ID
            session_id: Optional session ID to limit scope
            limit: Maximum number of questions to retrieve
            
        Returns:
            List of previously asked questions
        """
        if not self.client:
            return []
        
        try:
            query = """
            MATCH (u:User {id: $user_id})-[r:ANSWERED]->(q:Question)
            WHERE $session_id IS NULL OR u.session_id = $session_id
            RETURN q
            ORDER BY r.timestamp DESC
            LIMIT $limit
            """
            
            result = await self.client._neo4j_manager.execute_query_async(
                query,
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "limit": limit
                }
            )
            
            questions = []
            for record in result:
                q_data = record["q"]
                question = QuestionEntity(
                    id=q_data["id"],
                    content=q_data["content"],
                    difficulty=DifficultyLevel(q_data.get("difficulty", "medium")),
                    topics=q_data.get("topics", []),
                    asked_count=q_data.get("asked_count", 0),
                    correct_rate=q_data.get("correct_rate", 0.0)
                )
                questions.append(question)
                self._question_cache[question.id] = question
            
            logger.info(f"Retrieved {len(questions)} asked questions for user {user_id}")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to get asked questions: {e}")
            return []
    
    async def get_user_performance(self, user_id: str) -> Dict[str, Any]:
        """Get user's overall performance metrics.
        
        Args:
            user_id: User ID
            
        Returns:
            Performance metrics including accuracy, topics, difficulty preferences
        """
        if not self.client:
            return self._default_performance()
        
        try:
            query = """
            MATCH (u:User {id: $user_id})-[r:ANSWERED]->(q:Question)
            OPTIONAL MATCH (a:Answer {user_id: $user_id, question_id: q.id})
            WITH u, q, a
            RETURN {
                total_questions: count(DISTINCT q),
                correct_answers: count(DISTINCT CASE WHEN a.status = 'CORRECT' THEN a END),
                accuracy: CASE WHEN count(DISTINCT q) > 0 
                          THEN toFloat(count(DISTINCT CASE WHEN a.status = 'CORRECT' THEN a END)) / count(DISTINCT q)
                          ELSE 0.0 END,
                avg_response_time: avg(a.response_time),
                difficulty_performance: {
                    easy: {
                        total: count(DISTINCT CASE WHEN q.difficulty = 'EASY' THEN q END),
                        correct: count(DISTINCT CASE WHEN q.difficulty = 'EASY' AND a.status = 'CORRECT' THEN q END)
                    },
                    medium: {
                        total: count(DISTINCT CASE WHEN q.difficulty = 'MEDIUM' THEN q END),
                        correct: count(DISTINCT CASE WHEN q.difficulty = 'MEDIUM' AND a.status = 'CORRECT' THEN q END)
                    },
                    hard: {
                        total: count(DISTINCT CASE WHEN q.difficulty = 'HARD' THEN q END),
                        correct: count(DISTINCT CASE WHEN q.difficulty = 'HARD' AND a.status = 'CORRECT' THEN q END)
                    }
                },
                recent_performance: {
                    last_5_correct: count(DISTINCT CASE WHEN a.status = 'CORRECT' THEN a END) >= 3
                }
            } as performance
            """
            
            result = await self.client._neo4j_manager.execute_query_async(
                query,
                {"user_id": user_id}
            )
            
            if result:
                performance = result[0]["performance"]
                
                # Calculate recommended difficulty
                performance["recommended_difficulty"] = self._calculate_recommended_difficulty(
                    performance["difficulty_performance"]
                )
                
                return performance
            
            return self._default_performance()
            
        except Exception as e:
            logger.error(f"Failed to get user performance: {e}")
            return self._default_performance()
    
    async def get_topic_questions(self,
                                  topic_name: str,
                                  exclude_ids: Optional[Set[str]] = None,
                                  difficulty: Optional[DifficultyLevel] = None,
                                  limit: int = 10) -> List[QuestionEntity]:
        """Get questions for a specific topic.
        
        Args:
            topic_name: Topic to get questions for
            exclude_ids: Question IDs to exclude
            difficulty: Optional difficulty filter
            limit: Maximum questions to return
            
        Returns:
            List of questions for the topic
        """
        if not self.client:
            return []
        
        exclude_ids = exclude_ids or set()
        
        try:
            query = """
            MATCH (q:Question)-[:REQUIRES_KNOWLEDGE]->(t:Topic {name: $topic_name})
            WHERE NOT q.id IN $exclude_ids
            AND ($difficulty IS NULL OR q.difficulty = $difficulty)
            RETURN q
            ORDER BY q.asked_count ASC, q.correct_rate DESC
            LIMIT $limit
            """
            
            result = await self.client._neo4j_manager.execute_query_async(
                query,
                {
                    "topic_name": topic_name,
                    "exclude_ids": list(exclude_ids),
                    "difficulty": difficulty.value if difficulty else None,
                    "limit": limit
                }
            )
            
            questions = []
            for record in result:
                q_data = record["q"]
                question = QuestionEntity(
                    id=q_data["id"],
                    content=q_data["content"],
                    difficulty=DifficultyLevel(q_data.get("difficulty", "MEDIUM")),
                    topics=q_data.get("topics", []),
                    asked_count=q_data.get("asked_count", 0),
                    correct_rate=q_data.get("correct_rate", 0.0)
                )
                questions.append(question)
            
            return questions
            
        except Exception as e:
            logger.error(f"Failed to get topic questions: {e}")
            return []
    
    async def get_weak_topics(self, 
                              user_id: str,
                              threshold: float = 0.5,
                              min_attempts: int = 3) -> List[Tuple[str, float]]:
        """Get topics where user is struggling.
        
        Args:
            user_id: User ID
            threshold: Accuracy threshold below which topic is considered weak
            min_attempts: Minimum attempts to consider topic
            
        Returns:
            List of (topic_name, accuracy) tuples sorted by weakness
        """
        if not self.client:
            return []
        
        try:
            query = """
            MATCH (u:User {id: $user_id})-[r:ANSWERED]->(q:Question)-[:REQUIRES_KNOWLEDGE]->(t:Topic)
            OPTIONAL MATCH (a:Answer {user_id: $user_id, question_id: q.id})
            WITH t.name as topic, 
                 count(DISTINCT q) as attempts,
                 count(DISTINCT CASE WHEN a.status = 'CORRECT' THEN q END) as correct
            WHERE attempts >= $min_attempts
            WITH topic, 
                 attempts, 
                 correct,
                 toFloat(correct) / attempts as accuracy
            WHERE accuracy < $threshold
            RETURN topic, accuracy
            ORDER BY accuracy ASC
            """
            
            result = await self.client._neo4j_manager.execute_query_async(
                query,
                {
                    "user_id": user_id,
                    "threshold": threshold,
                    "min_attempts": min_attempts
                }
            )
            
            weak_topics = [(record["topic"], record["accuracy"]) for record in result]
            
            logger.info(f"Found {len(weak_topics)} weak topics for user {user_id}")
            return weak_topics
            
        except Exception as e:
            logger.error(f"Failed to get weak topics: {e}")
            return []
    
    async def get_recommended_questions(self,
                                        user_id: str,
                                        session_id: Optional[str] = None,
                                        count: int = 5) -> List[QuestionEntity]:
        """Get personalized question recommendations for a user.
        
        This method combines multiple factors:
        - User's performance history
        - Weak topics that need practice
        - Appropriate difficulty level
        - Avoiding recently asked questions
        
        Args:
            user_id: User ID
            session_id: Optional session ID
            count: Number of questions to recommend
            
        Returns:
            List of recommended questions
        """
        if not self.client:
            return []
        
        try:
            # Get user's performance data
            performance = await self.get_user_performance(user_id)
            recommended_difficulty = performance["recommended_difficulty"]
            
            # Get previously asked questions to avoid
            asked_questions = await self.get_asked_questions(user_id, session_id, limit=100)
            asked_ids = {q.id for q in asked_questions}
            
            # Get weak topics
            weak_topics = await self.get_weak_topics(user_id)
            
            recommendations = []
            
            # First, get questions from weak topics
            if weak_topics:
                for topic, accuracy in weak_topics[:3]:  # Focus on top 3 weakest topics
                    topic_questions = await self.get_topic_questions(
                        topic,
                        exclude_ids=asked_ids,
                        difficulty=recommended_difficulty,
                        limit=2
                    )
                    recommendations.extend(topic_questions)
            
            # If not enough questions, get general questions at appropriate difficulty
            if len(recommendations) < count:
                general_questions = await self._get_general_questions(
                    exclude_ids=asked_ids | {q.id for q in recommendations},
                    difficulty=recommended_difficulty,
                    limit=count - len(recommendations)
                )
                recommendations.extend(general_questions)
            
            # Shuffle to avoid predictable patterns
            random.shuffle(recommendations)
            
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations[:count]
            
        except Exception as e:
            logger.error(f"Failed to get recommended questions: {e}")
            return []
    
    async def get_question_context(self, question_id: str) -> Dict[str, Any]:
        """Get context information about a question.
        
        Args:
            question_id: Question ID
            
        Returns:
            Context including topics, difficulty, success rate, etc.
        """
        if not self.client:
            return {}
        
        try:
            query = """
            MATCH (q:Question {id: $question_id})
            OPTIONAL MATCH (q)-[:REQUIRES_KNOWLEDGE]->(t:Topic)
            OPTIONAL MATCH (a:Answer {question_id: q.id})
            WITH q, 
                 collect(DISTINCT t.name) as topics,
                 count(DISTINCT a) as total_attempts,
                 count(DISTINCT CASE WHEN a.status = 'CORRECT' THEN a END) as correct_attempts
            RETURN {
                question_id: q.id,
                content: q.content,
                difficulty: q.difficulty,
                topics: topics,
                asked_count: q.asked_count,
                total_attempts: total_attempts,
                correct_attempts: correct_attempts,
                success_rate: CASE WHEN total_attempts > 0 
                              THEN toFloat(correct_attempts) / total_attempts 
                              ELSE q.correct_rate END,
                last_asked: q.last_asked_at
            } as context
            """
            
            result = await self.client._neo4j_manager.execute_query_async(
                query,
                {"question_id": question_id}
            )
            
            if result:
                return result[0]["context"]
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get question context: {e}")
            return {}
    
    def _calculate_recommended_difficulty(self, 
                                          difficulty_performance: Dict[str, Dict[str, int]]) -> DifficultyLevel:
        """Calculate recommended difficulty based on performance.
        
        Args:
            difficulty_performance: Performance data by difficulty level
            
        Returns:
            Recommended difficulty level
        """
        # Calculate accuracy for each difficulty
        accuracies = {}
        for level in ["easy", "medium", "hard"]:
            perf = difficulty_performance.get(level, {})
            total = perf.get("total", 0)
            correct = perf.get("correct", 0)
            
            if total > 0:
                accuracies[level] = correct / total
            else:
                accuracies[level] = 0.0
        
        # Recommendation logic
        if accuracies["easy"] < 0.7:
            return DifficultyLevel.EASY
        elif accuracies["medium"] < 0.6 or accuracies["easy"] > 0.8:
            return DifficultyLevel.MEDIUM
        elif accuracies["medium"] > 0.7 and accuracies["hard"] < 0.5:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.MEDIUM
    
    def _default_performance(self) -> Dict[str, Any]:
        """Get default performance metrics for new users."""
        return {
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "avg_response_time": 0.0,
            "difficulty_performance": {
                "easy": {"total": 0, "correct": 0},
                "medium": {"total": 0, "correct": 0},
                "hard": {"total": 0, "correct": 0}
            },
            "recent_performance": {"last_5_correct": False},
            "recommended_difficulty": DifficultyLevel.EASY
        }
    
    async def _get_general_questions(self,
                                     exclude_ids: Set[str],
                                     difficulty: DifficultyLevel,
                                     limit: int) -> List[QuestionEntity]:
        """Get general questions not tied to specific topics.
        
        Args:
            exclude_ids: Question IDs to exclude
            difficulty: Difficulty level
            limit: Maximum questions to return
            
        Returns:
            List of general questions
        """
        if not self.client:
            return []
        
        try:
            query = """
            MATCH (q:Question)
            WHERE NOT q.id IN $exclude_ids
            AND q.difficulty = $difficulty
            RETURN q
            ORDER BY q.asked_count ASC, rand()
            LIMIT $limit
            """
            
            result = await self.client._neo4j_manager.execute_query_async(
                query,
                {
                    "exclude_ids": list(exclude_ids),
                    "difficulty": difficulty.value,
                    "limit": limit
                }
            )
            
            questions = []
            for record in result:
                q_data = record["q"]
                question = QuestionEntity(
                    id=q_data["id"],
                    content=q_data["content"],
                    difficulty=DifficultyLevel(q_data.get("difficulty", "MEDIUM")),
                    topics=q_data.get("topics", []),
                    asked_count=q_data.get("asked_count", 0),
                    correct_rate=q_data.get("correct_rate", 0.0)
                )
                questions.append(question)
            
            return questions
            
        except Exception as e:
            logger.error(f"Failed to get general questions: {e}")
            return []


class QuestionSelector:
    """Intelligent question selection based on memory retrieval."""
    
    def __init__(self, retrieval: MemoryRetrieval):
        """Initialize question selector.
        
        Args:
            retrieval: MemoryRetrieval instance
        """
        self.retrieval = retrieval
    
    async def select_next_question(self,
                                   user_id: str,
                                   session_id: Optional[str] = None,
                                   context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Select the next question to ask.
        
        Args:
            user_id: User ID
            session_id: Optional session ID
            context: Optional context from previous interactions
            
        Returns:
            Question content or None if no suitable question found
        """
        # Get recommendations
        recommendations = await self.retrieval.get_recommended_questions(
            user_id=user_id,
            session_id=session_id,
            count=5
        )
        
        if not recommendations:
            logger.warning(f"No recommendations found for user {user_id}")
            return None
        
        # Apply context-based filtering if provided
        if context:
            recommendations = self._apply_context_filter(recommendations, context)
        
        # Select the top recommendation
        if recommendations:
            selected = recommendations[0]
            logger.info(f"Selected question {selected.id} for user {user_id}")
            return selected.content
        
        return None
    
    def _apply_context_filter(self,
                              questions: List[QuestionEntity],
                              context: Dict[str, Any]) -> List[QuestionEntity]:
        """Apply context-based filtering to questions.
        
        Args:
            questions: List of candidate questions
            context: Context information
            
        Returns:
            Filtered list of questions
        """
        # Example context filtering
        if "avoid_topics" in context:
            avoid_topics = set(context["avoid_topics"])
            questions = [q for q in questions 
                        if not any(topic in avoid_topics for topic in q.topics)]
        
        if "prefer_topics" in context:
            prefer_topics = set(context["prefer_topics"])
            # Sort to prioritize preferred topics
            questions.sort(
                key=lambda q: sum(1 for topic in q.topics if topic in prefer_topics),
                reverse=True
            )
        
        return questions