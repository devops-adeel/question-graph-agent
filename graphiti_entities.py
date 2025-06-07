"""
Pydantic entity models for Graphiti knowledge graph integration.

These models represent the core entities in the learning system and are designed
to be compatible with Graphiti's entity extraction and graph storage capabilities.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# Task 1.1: Base entity model with common fields
class BaseEntity(BaseModel):
    """Base class for all Graphiti entities with common fields."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique entity identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Entity creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration for all entities."""
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to current time."""
        self.updated_at = datetime.now()


# Task 1.2: Question entity with difficulty enum
class DifficultyLevel(str, Enum):
    """Question difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
    
    @classmethod
    def from_score(cls, score: float) -> DifficultyLevel:
        """Convert a numeric score (0-1) to difficulty level."""
        if score < 0.25:
            return cls.EASY
        elif score < 0.5:
            return cls.MEDIUM
        elif score < 0.75:
            return cls.HARD
        else:
            return cls.EXPERT


class QuestionEntity(BaseEntity):
    """Represents a question in the knowledge graph."""
    
    content: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="The question text"
    )
    difficulty: DifficultyLevel = Field(
        DifficultyLevel.MEDIUM,
        description="Question difficulty level"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Related topic keywords for the question"
    )
    asked_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this question has been asked"
    )
    correct_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of correct answers (0-1)"
    )
    avg_response_time: Optional[float] = Field(
        default=None,
        gt=0,
        description="Average time in seconds to answer this question"
    )
    
    @field_validator('content')
    @classmethod
    def clean_question_content(cls, v: str) -> str:
        """Normalize question content."""
        v = v.strip()
        # Ensure question ends with a question mark
        if v and not v.endswith('?'):
            v += '?'
        return v
    
    @field_validator('topics')
    @classmethod
    def normalize_topics(cls, v: List[str]) -> List[str]:
        """Lowercase, strip, and deduplicate topics."""
        if not v:
            return []
        normalized = [topic.lower().strip() for topic in v if topic.strip()]
        return list(dict.fromkeys(normalized))  # Remove duplicates while preserving order
    
    @model_validator(mode='after')
    def validate_question_metrics(self) -> 'QuestionEntity':
        """Ensure question metrics are consistent."""
        # If asked_count is 0, correct_rate should also be 0
        if self.asked_count == 0:
            self.correct_rate = 0.0
            self.avg_response_time = None
        return self


# Task 1.3: Answer entity with status enum
class AnswerStatus(str, Enum):
    """Answer evaluation status."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    UNEVALUATED = "unevaluated"


class AnswerEntity(BaseEntity):
    """Represents an answer attempt by a user."""
    
    question_id: str = Field(
        ...,
        description="ID of the question being answered"
    )
    user_id: str = Field(
        ...,
        description="ID of the user who provided the answer"
    )
    session_id: str = Field(
        ...,
        description="Session ID when the answer was provided"
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The answer text"
    )
    status: AnswerStatus = Field(
        AnswerStatus.UNEVALUATED,
        description="Evaluation result of the answer"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="AI confidence in the evaluation (0-1)"
    )
    response_time_seconds: float = Field(
        ...,
        gt=0,
        description="Time taken to provide the answer"
    )
    feedback: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Evaluation feedback or explanation"
    )
    
    @field_validator('content')
    @classmethod
    def clean_answer_content(cls, v: str) -> str:
        """Normalize answer content."""
        return v.strip()
    
    @field_validator('feedback')
    @classmethod
    def clean_feedback(cls, v: Optional[str]) -> Optional[str]:
        """Clean and validate feedback."""
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None
    
    @model_validator(mode='after')
    def validate_answer_consistency(self) -> 'AnswerEntity':
        """Ensure answer data is consistent."""
        # If status is unevaluated, there should be no feedback or confidence
        if self.status == AnswerStatus.UNEVALUATED:
            self.feedback = None
            self.confidence_score = None
        # If evaluated, ensure we have confidence score
        elif self.status != AnswerStatus.UNEVALUATED and self.confidence_score is None:
            self.confidence_score = 0.95  # Default high confidence
        return self


# Task 1.4: User entity with performance tracking
class UserEntity(BaseEntity):
    """Represents a learner in the system."""
    
    username: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique username or identifier"
    )
    session_ids: List[str] = Field(
        default_factory=list,
        description="List of session IDs for this user"
    )
    total_questions_attempted: int = Field(
        default=0,
        ge=0,
        description="Total number of questions attempted"
    )
    correct_answers: int = Field(
        default=0,
        ge=0,
        description="Number of correct answers"
    )
    partial_answers: int = Field(
        default=0,
        ge=0,
        description="Number of partially correct answers"
    )
    topics_seen: List[str] = Field(
        default_factory=list,
        description="List of topics the user has encountered"
    )
    avg_response_time: Optional[float] = Field(
        default=None,
        gt=0,
        description="Average response time in seconds"
    )
    last_active: Optional[datetime] = Field(
        default=None,
        description="Last activity timestamp"
    )
    learning_streak: int = Field(
        default=0,
        ge=0,
        description="Current learning streak in days"
    )
    
    @property
    def accuracy_rate(self) -> float:
        """Calculate user's overall accuracy rate."""
        if self.total_questions_attempted == 0:
            return 0.0
        return self.correct_answers / self.total_questions_attempted
    
    @property
    def partial_accuracy_rate(self) -> float:
        """Calculate accuracy including partial credit."""
        if self.total_questions_attempted == 0:
            return 0.0
        return (self.correct_answers + 0.5 * self.partial_answers) / self.total_questions_attempted
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Ensure username is valid."""
        v = v.strip()
        if not v:
            raise ValueError("Username cannot be empty")
        return v
    
    @field_validator('topics_seen')
    @classmethod
    def normalize_topics_seen(cls, v: List[str]) -> List[str]:
        """Normalize and deduplicate topics."""
        if not v:
            return []
        normalized = [topic.lower().strip() for topic in v if topic.strip()]
        return list(dict.fromkeys(normalized))
    
    @model_validator(mode='after')
    def validate_user_metrics(self) -> 'UserEntity':
        """Ensure user metrics are consistent."""
        # Correct answers cannot exceed total attempted
        if self.correct_answers > self.total_questions_attempted:
            self.correct_answers = self.total_questions_attempted
        
        # Partial answers cannot exceed questions minus correct
        max_partial = self.total_questions_attempted - self.correct_answers
        if self.partial_answers > max_partial:
            self.partial_answers = max_partial
        
        # Update last active if any activity
        if self.total_questions_attempted > 0 and self.last_active is None:
            self.last_active = datetime.now()
        
        return self


# Task 1.5: Topic entity with hierarchy
class ComplexityLevel(str, Enum):
    """Topic complexity levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TopicEntity(BaseEntity):
    """Represents a knowledge topic or concept."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Topic name"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Brief description of the topic"
    )
    parent_topic_id: Optional[str] = Field(
        default=None,
        description="ID of the parent topic for hierarchy"
    )
    child_topic_ids: List[str] = Field(
        default_factory=list,
        description="IDs of child topics"
    )
    complexity: ComplexityLevel = Field(
        ComplexityLevel.INTERMEDIATE,
        description="Topic complexity level"
    )
    complexity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Numeric complexity score (0-1)"
    )
    prerequisite_topic_ids: List[str] = Field(
        default_factory=list,
        description="Topics that should be mastered before this one"
    )
    question_count: int = Field(
        default=0,
        ge=0,
        description="Number of questions associated with this topic"
    )
    avg_mastery_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average mastery score across all users"
    )
    
    @field_validator('name')
    @classmethod
    def normalize_topic_name(cls, v: str) -> str:
        """Normalize topic name."""
        return v.strip().lower()
    
    @field_validator('complexity_score')
    @classmethod
    def validate_complexity_alignment(cls, v: float, info) -> float:
        """Ensure complexity score aligns with complexity level."""
        if 'complexity' in info.data:
            complexity = info.data['complexity']
            expected_ranges = {
                ComplexityLevel.BEGINNER: (0.0, 0.25),
                ComplexityLevel.INTERMEDIATE: (0.25, 0.5),
                ComplexityLevel.ADVANCED: (0.5, 0.75),
                ComplexityLevel.EXPERT: (0.75, 1.0)
            }
            min_val, max_val = expected_ranges.get(complexity, (0.0, 1.0))
            if not min_val <= v <= max_val:
                # Adjust to fit the expected range
                return (min_val + max_val) / 2
        return v


# Task 1.7: Entity factory functions
class EntityFactory:
    """Factory class for creating entities with common patterns."""
    
    @staticmethod
    def create_question_from_text(
        content: str,
        topics: Optional[List[str]] = None,
        difficulty: Optional[DifficultyLevel] = None
    ) -> QuestionEntity:
        """Create a question entity from raw text."""
        # Auto-detect difficulty if not provided
        if difficulty is None:
            word_count = len(content.split())
            if word_count < 10:
                difficulty = DifficultyLevel.EASY
            elif word_count < 20:
                difficulty = DifficultyLevel.MEDIUM
            elif word_count < 30:
                difficulty = DifficultyLevel.HARD
            else:
                difficulty = DifficultyLevel.EXPERT
        
        # Extract topics if not provided
        if topics is None:
            topics = EntityFactory._extract_basic_topics(content)
        
        return QuestionEntity(
            content=content,
            difficulty=difficulty,
            topics=topics
        )
    
    @staticmethod
    def create_answer_from_interaction(
        question_id: str,
        user_id: str,
        session_id: str,
        answer_text: str,
        response_time: float,
        is_correct: Optional[bool] = None
    ) -> AnswerEntity:
        """Create an answer entity from a Q&A interaction."""
        # Determine status based on correctness
        if is_correct is None:
            status = AnswerStatus.UNEVALUATED
        elif is_correct:
            status = AnswerStatus.CORRECT
        else:
            status = AnswerStatus.INCORRECT
        
        return AnswerEntity(
            question_id=question_id,
            user_id=user_id,
            session_id=session_id,
            content=answer_text,
            status=status,
            response_time_seconds=response_time
        )
    
    @staticmethod
    def create_new_user(username: str, session_id: str) -> UserEntity:
        """Create a new user entity."""
        return UserEntity(
            username=username,
            session_ids=[session_id],
            last_active=datetime.now()
        )
    
    @staticmethod
    def create_topic_from_name(
        name: str,
        complexity: Optional[ComplexityLevel] = None,
        parent_id: Optional[str] = None
    ) -> TopicEntity:
        """Create a topic entity from a name."""
        if complexity is None:
            complexity = ComplexityLevel.INTERMEDIATE
        
        # Calculate complexity score based on level
        complexity_scores = {
            ComplexityLevel.BEGINNER: 0.125,
            ComplexityLevel.INTERMEDIATE: 0.375,
            ComplexityLevel.ADVANCED: 0.625,
            ComplexityLevel.EXPERT: 0.875
        }
        
        return TopicEntity(
            name=name,
            complexity=complexity,
            complexity_score=complexity_scores[complexity],
            parent_topic_id=parent_id
        )
    
    @staticmethod
    def _extract_basic_topics(text: str) -> List[str]:
        """Extract basic topics from text using keyword matching."""
        text_lower = text.lower()
        topics = []
        
        # Common topic patterns
        topic_keywords = {
            'mathematics': ['math', 'calculate', 'equation', 'algebra', 'geometry'],
            'science': ['scientific', 'experiment', 'hypothesis', 'theory'],
            'history': ['historical', 'century', 'era', 'civilization'],
            'geography': ['country', 'continent', 'capital', 'location'],
            'literature': ['author', 'novel', 'poem', 'literary'],
            'technology': ['computer', 'software', 'algorithm', 'programming'],
            'biology': ['cell', 'organism', 'species', 'evolution'],
            'physics': ['force', 'energy', 'motion', 'quantum'],
            'chemistry': ['element', 'compound', 'reaction', 'molecule']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        # If no topics found, use general
        if not topics:
            topics = ['general']
        
        return topics
    
    @staticmethod
    def update_user_performance(
        user: UserEntity,
        was_correct: bool,
        was_partial: bool = False,
        response_time: float = None,
        topics: List[str] = None
    ) -> UserEntity:
        """Update user entity with new performance data."""
        user.total_questions_attempted += 1
        
        if was_correct:
            user.correct_answers += 1
        elif was_partial:
            user.partial_answers += 1
        
        # Update average response time
        if response_time and user.avg_response_time:
            # Calculate new average
            total_time = user.avg_response_time * (user.total_questions_attempted - 1)
            user.avg_response_time = (total_time + response_time) / user.total_questions_attempted
        elif response_time:
            user.avg_response_time = response_time
        
        # Add new topics
        if topics:
            current_topics = set(user.topics_seen)
            current_topics.update(topics)
            user.topics_seen = list(current_topics)
        
        # Update last active
        user.last_active = datetime.now()
        user.update_timestamp()
        
        return user