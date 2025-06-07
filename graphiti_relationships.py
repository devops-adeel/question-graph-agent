"""
Pydantic relationship models for Graphiti knowledge graph.

These models represent the relationships (edges) between entities in the learning system.
They are designed to capture temporal aspects, metadata, and maintain data integrity.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError


# Sub-task 2.1: Base relationship model
class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    ANSWERED = "answered"
    REQUIRES_KNOWLEDGE = "requires_knowledge"
    MASTERS = "masters"
    FOLLOWS_UP = "follows_up"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"
    TEACHES = "teaches"


class RelationshipStrength(str, Enum):
    """Strength/weight categories for relationships."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    
    @classmethod
    def from_score(cls, score: float) -> RelationshipStrength:
        """Convert numeric score (0-1) to strength category."""
        if score < 0.2:
            return cls.VERY_WEAK
        elif score < 0.4:
            return cls.WEAK
        elif score < 0.6:
            return cls.MODERATE
        elif score < 0.8:
            return cls.STRONG
        else:
            return cls.VERY_STRONG


class BaseRelationship(BaseModel):
    """Base class for all relationships in the knowledge graph."""
    
    # Core relationship identifiers
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique relationship identifier"
    )
    relationship_type: RelationshipType = Field(
        ...,
        description="Type of relationship"
    )
    source_id: str = Field(
        ...,
        description="ID of the source entity"
    )
    source_type: str = Field(
        ...,
        description="Type of the source entity (e.g., 'User', 'Question')"
    )
    target_id: str = Field(
        ...,
        description="ID of the target entity"
    )
    target_type: str = Field(
        ...,
        description="Type of the target entity"
    )
    
    # Temporal fields for tracking validity
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the relationship was created"
    )
    valid_from: datetime = Field(
        default_factory=datetime.now,
        description="Start of validity period (for temporal graphs)"
    )
    valid_until: Optional[datetime] = Field(
        default=None,
        description="End of validity period (None means currently valid)"
    )
    
    # Relationship metadata
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Numeric strength/weight of the relationship (0-1)"
    )
    strength_category: Optional[RelationshipStrength] = Field(
        default=None,
        description="Categorical strength (auto-computed from numeric strength)"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the relationship (0-1)"
    )
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional relationship-specific metadata"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing relationships"
    )
    
    # Tracking fields
    interaction_count: int = Field(
        default=1,
        ge=1,
        description="Number of interactions that reinforce this relationship"
    )
    last_interaction: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of last interaction"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @field_validator('source_id', 'target_id')
    @classmethod
    def validate_entity_ids(cls, v: str) -> str:
        """Ensure entity IDs are not empty."""
        if not v or not v.strip():
            raise ValueError("Entity ID cannot be empty")
        return v.strip()
    
    @field_validator('source_type', 'target_type')
    @classmethod
    def validate_entity_types(cls, v: str) -> str:
        """Ensure entity types are valid."""
        valid_types = {'User', 'Question', 'Answer', 'Topic', 'Session'}
        v = v.strip()
        if v not in valid_types:
            raise ValueError(f"Entity type must be one of {valid_types}, got '{v}'")
        return v
    
    @field_validator('tags')
    @classmethod
    def normalize_tags(cls, v: list[str]) -> list[str]:
        """Normalize and deduplicate tags."""
        if not v:
            return []
        normalized = [tag.lower().strip() for tag in v if tag.strip()]
        return list(dict.fromkeys(normalized))  # Remove duplicates
    
    @model_validator(mode='after')
    def compute_strength_category(self) -> 'BaseRelationship':
        """Auto-compute strength category from numeric strength."""
        if self.strength_category is None:
            self.strength_category = RelationshipStrength.from_score(self.strength)
        return self
    
    @model_validator(mode='after')
    def validate_temporal_consistency(self) -> 'BaseRelationship':
        """Ensure temporal fields are consistent."""
        if self.valid_until and self.valid_from > self.valid_until:
            raise ValueError("valid_from cannot be after valid_until")
        
        if self.last_interaction < self.created_at:
            self.last_interaction = self.created_at
        
        return self
    
    def is_currently_valid(self) -> bool:
        """Check if the relationship is currently valid."""
        now = datetime.now()
        return (
            self.valid_from <= now and 
            (self.valid_until is None or self.valid_until > now)
        )
    
    def invalidate(self, as_of: Optional[datetime] = None) -> None:
        """Mark the relationship as no longer valid."""
        self.valid_until = as_of or datetime.now()
    
    def strengthen(self, amount: float = 0.1) -> None:
        """Increase the strength of the relationship."""
        self.strength = min(1.0, self.strength + amount)
        self.strength_category = RelationshipStrength.from_score(self.strength)
        self.interaction_count += 1
        self.last_interaction = datetime.now()
    
    def weaken(self, amount: float = 0.1) -> None:
        """Decrease the strength of the relationship."""
        self.strength = max(0.0, self.strength - amount)
        self.strength_category = RelationshipStrength.from_score(self.strength)
        self.last_interaction = datetime.now()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update metadata."""
        self.metadata[key] = value
    
    def to_graph_format(self) -> Dict[str, Any]:
        """Convert to format suitable for graph database."""
        return {
            'id': self.id,
            'type': self.relationship_type,
            'source': self.source_id,
            'target': self.target_id,
            'properties': {
                'strength': self.strength,
                'confidence': self.confidence,
                'created_at': self.created_at.isoformat(),
                'valid_from': self.valid_from.isoformat(),
                'valid_until': self.valid_until.isoformat() if self.valid_until else None,
                'interaction_count': self.interaction_count,
                'metadata': self.metadata
            }
        }


# Sub-task 2.2: AnsweredRelationship
class AnsweredRelationship(BaseRelationship):
    """
    Represents a user answering a question.
    
    This is a ternary relationship: User -> Answer -> Question
    The relationship connects users to questions through their answer attempts.
    """
    
    def __init__(self, **data):
        # Set relationship type
        data['relationship_type'] = RelationshipType.ANSWERED
        super().__init__(**data)
    
    # Answer-specific fields
    answer_id: str = Field(
        ...,
        description="ID of the answer entity that connects user to question"
    )
    session_id: str = Field(
        ...,
        description="Session ID when the answer was provided"
    )
    
    # Performance metrics
    is_correct: bool = Field(
        ...,
        description="Whether the answer was correct"
    )
    is_partial: bool = Field(
        default=False,
        description="Whether the answer was partially correct"
    )
    response_time_seconds: float = Field(
        ...,
        gt=0,
        description="Time taken to answer the question"
    )
    
    # Answer quality metrics
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="User's confidence in their answer (if collected)"
    )
    evaluation_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="AI's evaluation score of the answer quality"
    )
    
    # Context
    attempt_number: int = Field(
        default=1,
        ge=1,
        description="Which attempt this is for this user-question pair"
    )
    hint_used: bool = Field(
        default=False,
        description="Whether the user used a hint"
    )
    
    @field_validator('answer_id', 'session_id')
    @classmethod
    def validate_ids_not_empty(cls, v: str) -> str:
        """Ensure IDs are not empty."""
        if not v or not v.strip():
            raise ValueError("ID cannot be empty")
        return v.strip()
    
    @model_validator(mode='after')
    def set_relationship_properties(self) -> 'AnsweredRelationship':
        """Set relationship properties based on answer correctness."""
        # Source is always User, target is always Question
        if self.source_type != 'User':
            raise ValueError("AnsweredRelationship source must be a User")
        if self.target_type != 'Question':
            raise ValueError("AnsweredRelationship target must be a Question")
        
        # Calculate strength based on correctness and response time
        if self.is_correct:
            # Correct answers strengthen the relationship
            base_strength = 0.8
            # Bonus for quick responses (under 30 seconds)
            if self.response_time_seconds < 30:
                base_strength += 0.1
            # Bonus for first attempt
            if self.attempt_number == 1:
                base_strength += 0.1
            self.strength = min(1.0, base_strength)
        elif self.is_partial:
            # Partial credit
            self.strength = 0.5
        else:
            # Incorrect answers still create a relationship, but weak
            self.strength = 0.2
        
        # Set confidence based on evaluation
        if self.evaluation_score is not None:
            self.confidence = self.evaluation_score
        
        # Add performance data to metadata
        self.metadata.update({
            'answer_id': self.answer_id,
            'response_time': self.response_time_seconds,
            'attempt': self.attempt_number,
            'hint_used': self.hint_used,
            'correctness': 'correct' if self.is_correct else ('partial' if self.is_partial else 'incorrect')
        })
        
        # Add performance tags
        if self.response_time_seconds < 10:
            self.tags.append('quick_response')
        elif self.response_time_seconds > 60:
            self.tags.append('slow_response')
        
        if self.is_correct and self.attempt_number == 1:
            self.tags.append('first_try_success')
        
        return self
    
    def to_cypher_query(self) -> str:
        """
        Generate Cypher query for creating this relationship in Neo4j.
        
        Returns a query that creates the ternary relationship:
        (User)-[:ANSWERED]->(Answer)-[:ANSWERS]->(Question)
        """
        return f"""
        MATCH (u:User {{id: '{self.source_id}'}})
        MATCH (q:Question {{id: '{self.target_id}'}})
        MATCH (a:Answer {{id: '{self.answer_id}'}})
        CREATE (u)-[r1:ANSWERED {{
            id: '{self.id}',
            created_at: datetime('{self.created_at.isoformat()}'),
            session_id: '{self.session_id}',
            attempt_number: {self.attempt_number},
            strength: {self.strength}
        }}]->(a)
        CREATE (a)-[r2:ANSWERS {{
            is_correct: {str(self.is_correct).lower()},
            is_partial: {str(self.is_partial).lower()},
            response_time: {self.response_time_seconds},
            evaluation_score: {self.evaluation_score if self.evaluation_score else 'null'}
        }}]->(q)
        """
    
    @classmethod
    def from_answer_entity(
        cls,
        user_id: str,
        question_id: str,
        answer_entity: Any,  # Would be AnswerEntity in practice
        attempt_number: int = 1
    ) -> 'AnsweredRelationship':
        """Create AnsweredRelationship from an Answer entity."""
        return cls(
            source_id=user_id,
            source_type='User',
            target_id=question_id,
            target_type='Question',
            answer_id=answer_entity.id,
            session_id=answer_entity.session_id,
            is_correct=answer_entity.status == 'correct',
            is_partial=answer_entity.status == 'partial',
            response_time_seconds=answer_entity.response_time_seconds,
            confidence_score=answer_entity.confidence_score,
            evaluation_score=answer_entity.confidence_score,
            attempt_number=attempt_number
        )


# Sub-task 2.3: RequiresKnowledgeRelationship
class RequiresKnowledgeRelationship(BaseRelationship):
    """
    Represents the knowledge requirement relationship between questions and topics.
    
    This relationship indicates which topics/concepts are needed to answer a question.
    The strength indicates how essential the topic is for answering the question.
    """
    
    def __init__(self, **data):
        # Set relationship type and ensure correct source/target types
        data['relationship_type'] = RelationshipType.REQUIRES_KNOWLEDGE
        data['source_type'] = 'Question'
        data['target_type'] = 'Topic'
        super().__init__(**data)
    
    # Knowledge requirement specific fields
    requirement_level: str = Field(
        default='required',
        description="Level of requirement: prerequisite, required, helpful, optional"
    )
    
    # Weight/importance of this topic for the question
    importance_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How important this topic is for answering the question (0-1)"
    )
    
    # Coverage metrics
    topic_coverage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="What percentage of the topic is covered by this question (0-1)"
    )
    
    # Question characteristics related to the topic
    complexity_contribution: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How much this topic contributes to question complexity (0-1)"
    )
    
    # Learning path metadata
    is_primary_topic: bool = Field(
        default=False,
        description="Whether this is the main topic being tested"
    )
    
    # Skill assessment
    skill_components: List[str] = Field(
        default_factory=list,
        description="Specific skills within the topic that are tested"
    )
    
    @field_validator('requirement_level')
    @classmethod
    def validate_requirement_level(cls, v: str) -> str:
        """Ensure requirement level is valid."""
        valid_levels = {'prerequisite', 'required', 'helpful', 'optional'}
        v = v.lower().strip()
        if v not in valid_levels:
            raise ValueError(f"Requirement level must be one of {valid_levels}, got '{v}'")
        return v
    
    @field_validator('skill_components')
    @classmethod
    def normalize_skill_components(cls, v: List[str]) -> List[str]:
        """Normalize and validate skill components."""
        if not v:
            return []
        # Normalize: lowercase, strip whitespace, remove empty
        normalized = [skill.lower().strip() for skill in v if skill.strip()]
        # Remove duplicates while preserving order
        return list(dict.fromkeys(normalized))
    
    @model_validator(mode='after')
    def set_relationship_properties(self) -> 'RequiresKnowledgeRelationship':
        """Calculate relationship strength based on requirement level and importance."""
        # Validate source and target types
        if self.source_type != 'Question':
            raise ValueError("RequiresKnowledgeRelationship source must be a Question")
        if self.target_type != 'Topic':
            raise ValueError("RequiresKnowledgeRelationship target must be a Topic")
        
        # Calculate strength based on requirement level and importance
        level_weights = {
            'prerequisite': 1.0,
            'required': 0.8,
            'helpful': 0.5,
            'optional': 0.3
        }
        
        base_strength = level_weights.get(self.requirement_level, 0.5)
        # Adjust by importance weight
        self.strength = base_strength * self.importance_weight
        
        # Update metadata
        self.metadata.update({
            'requirement_level': self.requirement_level,
            'importance': self.importance_weight,
            'coverage': self.topic_coverage,
            'is_primary': self.is_primary_topic,
            'skill_count': len(self.skill_components)
        })
        
        # Add descriptive tags
        if self.is_primary_topic:
            self.tags.append('primary_topic')
        
        if self.requirement_level == 'prerequisite':
            self.tags.append('prerequisite')
        
        if self.importance_weight > 0.8:
            self.tags.append('high_importance')
        elif self.importance_weight < 0.3:
            self.tags.append('low_importance')
        
        if self.topic_coverage > 0.7:
            self.tags.append('comprehensive')
        elif self.topic_coverage < 0.3:
            self.tags.append('touches_on')
        
        return self
    
    def to_cypher_query(self) -> str:
        """
        Generate Cypher query for creating this relationship in Neo4j.
        
        Creates: (Question)-[:REQUIRES_KNOWLEDGE]->(Topic)
        """
        skill_components_str = str(self.skill_components).replace("'", '"')
        
        return f"""
        MATCH (q:Question {{id: '{self.source_id}'}})
        MATCH (t:Topic {{id: '{self.target_id}'}})
        CREATE (q)-[r:REQUIRES_KNOWLEDGE {{
            id: '{self.id}',
            created_at: datetime('{self.created_at.isoformat()}'),
            requirement_level: '{self.requirement_level}',
            importance_weight: {self.importance_weight},
            topic_coverage: {self.topic_coverage},
            complexity_contribution: {self.complexity_contribution},
            is_primary_topic: {str(self.is_primary_topic).lower()},
            skill_components: {skill_components_str},
            strength: {self.strength}
        }}]->(t)
        """
    
    @classmethod
    def create_from_question_analysis(
        cls,
        question_id: str,
        topic_id: str,
        question_content: str,
        topic_name: str,
        is_primary: bool = False
    ) -> 'RequiresKnowledgeRelationship':
        """
        Create a RequiresKnowledgeRelationship from question analysis.
        
        This is a simplified version - in production, this would use NLP
        to determine importance, coverage, etc.
        """
        # Simple heuristics for demonstration
        content_lower = question_content.lower()
        topic_lower = topic_name.lower()
        
        # Determine importance based on how prominently the topic appears
        if topic_lower in content_lower:
            if content_lower.startswith(f"what is {topic_lower}") or \
               content_lower.startswith(f"define {topic_lower}"):
                importance = 0.9
                requirement_level = 'required'
            else:
                importance = 0.7
                requirement_level = 'helpful'
        else:
            importance = 0.4
            requirement_level = 'optional'
        
        # If marked as primary, override
        if is_primary:
            importance = max(importance, 0.8)
            requirement_level = 'required'
        
        return cls(
            source_id=question_id,
            target_id=topic_id,
            requirement_level=requirement_level,
            importance_weight=importance,
            topic_coverage=0.5,  # Default, would be calculated by NLP
            is_primary_topic=is_primary,
            skill_components=[]  # Would be extracted by NLP
        )
    
    def update_from_answer_patterns(
        self,
        correct_answers_using_topic: int,
        total_answers: int
    ) -> None:
        """
        Update the relationship based on answer patterns.
        
        If users who know this topic do better on the question,
        increase the importance weight.
        """
        if total_answers > 0:
            success_rate = correct_answers_using_topic / total_answers
            
            # Adjust importance based on correlation
            if success_rate > 0.8:
                # Strong correlation - topic is very important
                self.importance_weight = min(1.0, self.importance_weight + 0.1)
                if self.requirement_level == 'helpful':
                    self.requirement_level = 'required'
            elif success_rate < 0.3:
                # Weak correlation - topic might not be as important
                self.importance_weight = max(0.1, self.importance_weight - 0.1)
                if self.requirement_level == 'required' and not self.is_primary_topic:
                    self.requirement_level = 'helpful'
            
            # Recalculate strength
            self.set_relationship_properties()


# Sub-task 2.4: MasteryRelationship
class MasteryLevel(str, Enum):
    """Mastery level categories."""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    PROFICIENT = "proficient"
    EXPERT = "expert"
    MASTER = "master"
    
    @classmethod
    def from_score(cls, score: float) -> 'MasteryLevel':
        """Convert mastery score (0-1) to level."""
        if score < 0.2:
            return cls.NOVICE
        elif score < 0.4:
            return cls.BEGINNER
        elif score < 0.6:
            return cls.INTERMEDIATE
        elif score < 0.75:
            return cls.PROFICIENT
        elif score < 0.9:
            return cls.EXPERT
        else:
            return cls.MASTER


class MasteryRelationship(BaseRelationship):
    """
    Represents a user's mastery level of a specific topic.
    
    This relationship tracks proficiency over time and adapts based on performance.
    Includes temporal decay to model knowledge retention/forgetting.
    """
    
    def __init__(self, **data):
        # Set relationship type and ensure correct source/target types
        data['relationship_type'] = RelationshipType.MASTERS
        data['source_type'] = 'User'
        data['target_type'] = 'Topic'
        super().__init__(**data)
    
    # Mastery metrics
    mastery_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current mastery score (0-1)"
    )
    
    mastery_level: Optional[MasteryLevel] = Field(
        default=None,
        description="Categorical mastery level (auto-computed)"
    )
    
    # Performance tracking
    questions_attempted: int = Field(
        default=0,
        ge=0,
        description="Total questions attempted for this topic"
    )
    
    questions_correct: int = Field(
        default=0,
        ge=0,
        description="Questions answered correctly"
    )
    
    questions_partial: int = Field(
        default=0,
        ge=0,
        description="Questions with partial credit"
    )
    
    avg_response_time: Optional[float] = Field(
        default=None,
        gt=0,
        description="Average response time for topic questions"
    )
    
    # Learning velocity
    learning_rate: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="How quickly the user learns this topic"
    )
    
    forgetting_rate: float = Field(
        default=0.01,
        ge=0,
        le=1.0,
        description="How quickly mastery decays over time"
    )
    
    # Temporal tracking
    last_practice: datetime = Field(
        default_factory=datetime.now,
        description="Last time this topic was practiced"
    )
    
    peak_mastery: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Highest mastery score achieved"
    )
    
    peak_mastery_date: Optional[datetime] = Field(
        default=None,
        description="When peak mastery was achieved"
    )
    
    # Consistency metrics
    practice_streak: int = Field(
        default=0,
        ge=0,
        description="Current practice streak in days"
    )
    
    consistency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How consistently the user practices"
    )
    
    # Difficulty adaptation
    recommended_difficulty: Optional[str] = Field(
        default=None,
        description="Recommended question difficulty for this user-topic pair"
    )
    
    @model_validator(mode='after')
    def compute_mastery_properties(self) -> 'MasteryRelationship':
        """Calculate derived properties and validate consistency."""
        # Validate source and target types
        if self.source_type != 'User':
            raise ValueError("MasteryRelationship source must be a User")
        if self.target_type != 'Topic':
            raise ValueError("MasteryRelationship target must be a Topic")
        
        # Ensure questions_correct doesn't exceed attempted
        if self.questions_correct > self.questions_attempted:
            self.questions_correct = self.questions_attempted
        
        # Ensure partial doesn't exceed remaining questions
        max_partial = self.questions_attempted - self.questions_correct
        if self.questions_partial > max_partial:
            self.questions_partial = max_partial
        
        # Auto-compute mastery level
        if self.mastery_level is None:
            self.mastery_level = MasteryLevel.from_score(self.mastery_score)
        
        # Update peak mastery
        if self.mastery_score > self.peak_mastery:
            self.peak_mastery = self.mastery_score
            self.peak_mastery_date = datetime.now()
        
        # Set strength to mastery score
        self.strength = self.mastery_score
        
        # Recommend difficulty based on mastery
        if self.mastery_score < 0.3:
            self.recommended_difficulty = "easy"
        elif self.mastery_score < 0.6:
            self.recommended_difficulty = "medium"
        elif self.mastery_score < 0.85:
            self.recommended_difficulty = "hard"
        else:
            self.recommended_difficulty = "expert"
        
        # Update metadata
        self.metadata.update({
            'mastery_level': self.mastery_level,
            'accuracy_rate': self.get_accuracy_rate(),
            'questions_attempted': self.questions_attempted,
            'peak_mastery': self.peak_mastery,
            'days_since_practice': self.days_since_practice(),
            'recommended_difficulty': self.recommended_difficulty
        })
        
        # Add descriptive tags
        if self.mastery_level in [MasteryLevel.EXPERT, MasteryLevel.MASTER]:
            self.tags.append('high_mastery')
        elif self.mastery_level == MasteryLevel.NOVICE:
            self.tags.append('needs_practice')
        
        if self.practice_streak > 7:
            self.tags.append('consistent_learner')
        
        if self.days_since_practice() > 30:
            self.tags.append('needs_review')
        
        return self
    
    def get_accuracy_rate(self) -> float:
        """Calculate accuracy rate including partial credit."""
        if self.questions_attempted == 0:
            return 0.0
        
        total_score = self.questions_correct + (0.5 * self.questions_partial)
        return total_score / self.questions_attempted
    
    def days_since_practice(self) -> int:
        """Calculate days since last practice."""
        return (datetime.now() - self.last_practice).days
    
    def apply_temporal_decay(self) -> float:
        """
        Apply forgetting curve to mastery score.
        Returns the decayed mastery score.
        """
        days_elapsed = self.days_since_practice()
        if days_elapsed == 0:
            return self.mastery_score
        
        # Exponential decay model
        # mastery = mastery * e^(-forgetting_rate * days)
        import math
        decay_factor = math.exp(-self.forgetting_rate * days_elapsed)
        decayed_score = self.mastery_score * decay_factor
        
        # Minimum retention (can't forget everything)
        min_retention = 0.1 * self.peak_mastery
        return max(min_retention, decayed_score)
    
    def update_mastery(
        self,
        was_correct: bool,
        was_partial: bool = False,
        response_time: Optional[float] = None,
        question_difficulty: float = 0.5
    ) -> None:
        """
        Update mastery based on a new answer attempt.
        
        Args:
            was_correct: Whether the answer was fully correct
            was_partial: Whether partial credit was given
            response_time: Time taken to answer
            question_difficulty: Difficulty of the question (0-1)
        """
        # Apply decay before updating
        self.mastery_score = self.apply_temporal_decay()
        
        # Update counts
        self.questions_attempted += 1
        if was_correct:
            self.questions_correct += 1
        elif was_partial:
            self.questions_partial += 1
        
        # Calculate performance score for this attempt
        if was_correct:
            performance = 1.0
        elif was_partial:
            performance = 0.5
        else:
            performance = 0.0
        
        # Adjust for question difficulty
        # Harder questions give more credit for success
        if was_correct:
            performance *= (1 + question_difficulty * 0.5)
        else:
            # Less penalty for failing hard questions
            performance += (1 - performance) * question_difficulty * 0.2
        
        # Update mastery score using exponential moving average
        # Faster learning for early attempts, slower for later
        adaptive_rate = self.learning_rate / (1 + self.questions_attempted * 0.01)
        self.mastery_score = self.mastery_score + adaptive_rate * (performance - self.mastery_score)
        
        # Ensure bounds
        self.mastery_score = max(0.0, min(1.0, self.mastery_score))
        
        # Update response time average
        if response_time and self.avg_response_time:
            self.avg_response_time = (
                self.avg_response_time * (self.questions_attempted - 1) + response_time
            ) / self.questions_attempted
        elif response_time:
            self.avg_response_time = response_time
        
        # Update practice tracking
        if self.days_since_practice() <= 1:
            self.practice_streak += 1
        else:
            self.practice_streak = 1
        
        self.last_practice = datetime.now()
        
        # Update consistency score
        # Based on practice frequency and performance
        self.consistency_score = min(1.0, 
            (self.practice_streak / 30) * 0.5 + 
            self.get_accuracy_rate() * 0.5
        )
        
        # Trigger recomputation
        self.compute_mastery_properties()
    
    def to_cypher_query(self) -> str:
        """Generate Cypher query for creating/updating this relationship."""
        return f"""
        MATCH (u:User {{id: '{self.source_id}'}})
        MATCH (t:Topic {{id: '{self.target_id}'}})
        MERGE (u)-[r:MASTERS {{id: '{self.id}'}}]->(t)
        SET r.mastery_score = {self.mastery_score},
            r.mastery_level = '{self.mastery_level}',
            r.questions_attempted = {self.questions_attempted},
            r.questions_correct = {self.questions_correct},
            r.questions_partial = {self.questions_partial},
            r.learning_rate = {self.learning_rate},
            r.forgetting_rate = {self.forgetting_rate},
            r.last_practice = datetime('{self.last_practice.isoformat()}'),
            r.peak_mastery = {self.peak_mastery},
            r.practice_streak = {self.practice_streak},
            r.consistency_score = {self.consistency_score},
            r.recommended_difficulty = '{self.recommended_difficulty}',
            r.updated_at = datetime()
        """
    
    @classmethod
    def create_initial(
        cls,
        user_id: str,
        topic_id: str,
        initial_performance: float = 0.0
    ) -> 'MasteryRelationship':
        """Create an initial mastery relationship for a new user-topic pair."""
        return cls(
            source_id=user_id,
            target_id=topic_id,
            mastery_score=initial_performance,
            learning_rate=0.15,  # Higher initial learning rate
            forgetting_rate=0.01  # Standard forgetting rate
        )


# Sub-task 2.5: Relationship validation rules
class RelationshipValidationRules:
    """
    Centralized validation rules and utilities for all relationship types.
    Ensures data integrity and consistency across the graph.
    """
    
    # ID format patterns
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    
    # Entity ID prefixes for different types
    ID_PREFIXES = {
        'User': 'usr_',
        'Question': 'q_',
        'Answer': 'a_',
        'Topic': 'topic_',
        'Session': 'session_'
    }
    
    # Valid entity type combinations for relationships
    VALID_RELATIONSHIP_PAIRS = {
        RelationshipType.ANSWERED: ('User', 'Question'),
        RelationshipType.REQUIRES_KNOWLEDGE: ('Question', 'Topic'),
        RelationshipType.MASTERS: ('User', 'Topic'),
        RelationshipType.FOLLOWS_UP: ('Question', 'Question'),
        RelationshipType.RELATED_TO: ('Topic', 'Topic'),
        RelationshipType.DERIVED_FROM: ('Answer', 'Answer'),
        RelationshipType.TEACHES: ('User', 'User')
    }
    
    @classmethod
    def validate_id_format(cls, entity_id: str, entity_type: Optional[str] = None) -> str:
        """
        Validate entity ID format.
        
        Args:
            entity_id: The ID to validate
            entity_type: Optional entity type for prefix validation
            
        Returns:
            The validated ID
            
        Raises:
            ValueError: If ID format is invalid
        """
        if not entity_id or not entity_id.strip():
            raise ValueError("Entity ID cannot be empty")
        
        entity_id = entity_id.strip()
        
        # Check if it's a UUID
        if cls.UUID_PATTERN.match(entity_id):
            return entity_id
        
        # Check for entity-specific prefix if type provided
        if entity_type and entity_type in cls.ID_PREFIXES:
            expected_prefix = cls.ID_PREFIXES[entity_type]
            if not entity_id.startswith(expected_prefix):
                raise ValueError(
                    f"{entity_type} ID must start with '{expected_prefix}', got '{entity_id}'"
                )
        
        # Basic format validation
        if len(entity_id) < 3:
            raise ValueError(f"Entity ID too short: '{entity_id}'")
        
        if len(entity_id) > 100:
            raise ValueError(f"Entity ID too long: '{entity_id}'")
        
        # Check for invalid characters
        if not re.match(r'^[a-zA-Z0-9_\-]+$', entity_id):
            raise ValueError(
                f"Entity ID contains invalid characters: '{entity_id}'. "
                "Only alphanumeric, underscore, and hyphen allowed."
            )
        
        return entity_id
    
    @classmethod
    def validate_relationship_types(
        cls,
        relationship_type: RelationshipType,
        source_type: str,
        target_type: str
    ) -> None:
        """
        Validate that source and target types are valid for the relationship type.
        
        Raises:
            ValueError: If the combination is invalid
        """
        expected = cls.VALID_RELATIONSHIP_PAIRS.get(relationship_type)
        if not expected:
            return  # No validation rule for this relationship type
        
        expected_source, expected_target = expected
        if source_type != expected_source or target_type != expected_target:
            raise ValueError(
                f"{relationship_type} relationships must connect "
                f"{expected_source} to {expected_target}, "
                f"got {source_type} to {target_type}"
            )
    
    @classmethod
    def validate_score_consistency(
        cls,
        scores: Dict[str, float],
        tolerance: float = 0.001
    ) -> None:
        """
        Validate that scores are consistent (e.g., probabilities sum to 1).
        
        Args:
            scores: Dictionary of score names to values
            tolerance: Acceptable deviation from expected sum
        """
        for name, score in scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {score}")
    
    @classmethod
    def validate_temporal_sequence(
        cls,
        timestamps: Dict[str, Optional[datetime]]
    ) -> None:
        """
        Validate that timestamps follow logical sequence.
        
        Args:
            timestamps: Dictionary of timestamp names to values
        """
        # Check created_at <= updated_at
        if 'created_at' in timestamps and 'updated_at' in timestamps:
            if timestamps['created_at'] and timestamps['updated_at']:
                if timestamps['created_at'] > timestamps['updated_at']:
                    raise ValueError("created_at cannot be after updated_at")
        
        # Check valid_from <= valid_until
        if 'valid_from' in timestamps and 'valid_until' in timestamps:
            if timestamps['valid_from'] and timestamps['valid_until']:
                if timestamps['valid_from'] > timestamps['valid_until']:
                    raise ValueError("valid_from cannot be after valid_until")
    
    @classmethod
    def validate_performance_metrics(
        cls,
        attempted: int,
        correct: int,
        partial: int = 0
    ) -> None:
        """
        Validate performance metric consistency.
        
        Args:
            attempted: Total attempts
            correct: Correct answers
            partial: Partial credit answers
        """
        if attempted < 0 or correct < 0 or partial < 0:
            raise ValueError("Performance metrics cannot be negative")
        
        if correct > attempted:
            raise ValueError(f"Correct ({correct}) cannot exceed attempted ({attempted})")
        
        if correct + partial > attempted:
            raise ValueError(
                f"Correct ({correct}) + partial ({partial}) cannot exceed "
                f"attempted ({attempted})"
            )


# Enhanced validators for BaseRelationship
@field_validator('source_id', 'target_id', mode='after')
def validate_entity_id_format(cls, v: str, info) -> str:
    """Enhanced ID validation using RelationshipValidationRules."""
    field_name = info.field_name
    entity_type = info.data.get(f"{field_name.replace('_id', '')}_type")
    return RelationshipValidationRules.validate_id_format(v, entity_type)


# Add this to BaseRelationship as a model validator
def add_enhanced_validation_to_base():
    """Add enhanced validation to BaseRelationship class."""
    
    @model_validator(mode='after')
    def validate_relationship_consistency(self) -> 'BaseRelationship':
        """Enhanced validation for all relationships."""
        # Validate entity type combination
        RelationshipValidationRules.validate_relationship_types(
            self.relationship_type,
            self.source_type,
            self.target_type
        )
        
        # Validate temporal consistency
        RelationshipValidationRules.validate_temporal_sequence({
            'created_at': self.created_at,
            'valid_from': self.valid_from,
            'valid_until': self.valid_until,
            'last_interaction': self.last_interaction
        })
        
        # Validate scores
        RelationshipValidationRules.validate_score_consistency({
            'strength': self.strength,
            'confidence': self.confidence
        })
        
        return self
    
    # Inject the validator into BaseRelationship
    BaseRelationship.__pydantic_decorators__.model_validators['validate_consistency'] = (
        validate_relationship_consistency,
        {'mode': 'after'}
    )


# Custom exceptions for relationship validation
class RelationshipValidationError(ValidationError):
    """Custom exception for relationship validation failures."""
    
    @classmethod
    def from_field_errors(cls, errors: Dict[str, str]) -> 'RelationshipValidationError':
        """Create validation error from field errors."""
        error_list = []
        for field, message in errors.items():
            error_list.append({
                'type': 'value_error',
                'loc': (field,),
                'msg': message,
                'input': None
            })
        return cls.from_exception_data("RelationshipValidation", error_list)


# Validation decorators for common patterns
def validate_score_field(field_name: str, min_val: float = 0.0, max_val: float = 1.0):
    """Decorator to validate score fields."""
    def decorator(cls):
        @field_validator(field_name)
        @classmethod
        def validate_score(cls, v: float) -> float:
            if not min_val <= v <= max_val:
                raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
            return v
        
        # Add the validator to the class
        cls.__pydantic_decorators__.field_validators[f'validate_{field_name}'] = (
            validate_score,
            {'mode': 'after'}
        )
        return cls
    return decorator


def validate_count_fields(*field_names: str):
    """Decorator to validate count fields are non-negative."""
    def decorator(cls):
        @field_validator(*field_names)
        @classmethod
        def validate_counts(cls, v: int) -> int:
            if v < 0:
                raise ValueError("Count fields cannot be negative")
            return v
        
        cls.__pydantic_decorators__.field_validators['validate_counts'] = (
            validate_counts,
            {'mode': 'after'}
        )
        return cls
    return decorator


# Apply enhanced validation to existing relationship classes
def apply_relationship_validations():
    """Apply validation enhancements to all relationship classes."""
    
    # Add to AnsweredRelationship
    @model_validator(mode='after')
    def validate_answered_consistency(self) -> 'AnsweredRelationship':
        """Validate AnsweredRelationship specific rules."""
        # Performance metrics validation
        if self.is_correct and self.is_partial:
            raise ValueError("Answer cannot be both correct and partial")
        
        # Response time validation
        if self.response_time_seconds > 3600:  # 1 hour
            raise ValueError("Response time seems unreasonably long (>1 hour)")
        
        return self
    
    # Add to RequiresKnowledgeRelationship  
    @model_validator(mode='after')
    def validate_knowledge_consistency(self) -> 'RequiresKnowledgeRelationship':
        """Validate RequiresKnowledgeRelationship specific rules."""
        # Topic coverage vs importance
        if self.topic_coverage > 0.8 and self.importance_weight < 0.5:
            raise ValueError(
                "High topic coverage (>0.8) inconsistent with low importance (<0.5)"
            )
        
        return self
    
    # Add to MasteryRelationship
    @model_validator(mode='after')
    def validate_mastery_consistency(self) -> 'MasteryRelationship':
        """Validate MasteryRelationship specific rules."""
        # Performance validation
        RelationshipValidationRules.validate_performance_metrics(
            self.questions_attempted,
            self.questions_correct,
            self.questions_partial
        )
        
        # Learning rate validation
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.forgetting_rate < 0:
            raise ValueError("Forgetting rate cannot be negative")
        
        # Mastery score vs level consistency
        expected_level = MasteryLevel.from_score(self.mastery_score)
        if self.mastery_level and self.mastery_level != expected_level:
            # Allow one level difference for hysteresis
            level_values = list(MasteryLevel)
            current_idx = level_values.index(self.mastery_level)
            expected_idx = level_values.index(expected_level)
            if abs(current_idx - expected_idx) > 1:
                raise ValueError(
                    f"Mastery level {self.mastery_level} inconsistent with "
                    f"score {self.mastery_score} (expected {expected_level})"
                )
        
        return self


# Initialize enhanced validations
add_enhanced_validation_to_base()
apply_relationship_validations()


# Sub-task 2.6: Relationship builder utilities
class RelationshipBuilder:
    """
    Builder utilities for creating relationships with common patterns.
    Simplifies relationship creation and ensures consistency.
    """
    
    def __init__(self):
        """Initialize the builder with caches for efficiency."""
        self._user_topic_mastery: Dict[tuple, MasteryRelationship] = {}
        self._question_topics: Dict[str, List[RequiresKnowledgeRelationship]] = {}
        self._answer_cache: Dict[str, AnsweredRelationship] = {}
    
    # Answer relationship builders
    def build_answer_relationship(
        self,
        user_id: str,
        question_id: str,
        answer_entity: Any,  # Would be AnswerEntity in practice
        start_time: datetime,
        end_time: datetime,
        attempt_number: int = 1,
        hint_used: bool = False
    ) -> AnsweredRelationship:
        """
        Build an AnsweredRelationship with calculated metrics.
        
        Args:
            user_id: User who answered
            question_id: Question that was answered
            answer_entity: The answer entity
            start_time: When user started answering
            end_time: When user submitted answer
            attempt_number: Which attempt this is
            hint_used: Whether a hint was used
            
        Returns:
            Fully constructed AnsweredRelationship
        """
        # Calculate response time
        response_time = (end_time - start_time).total_seconds()
        
        # Create the relationship
        relationship = AnsweredRelationship(
            source_id=user_id,
            source_type='User',
            target_id=question_id,
            target_type='Question',
            answer_id=answer_entity.id,
            session_id=answer_entity.session_id,
            is_correct=answer_entity.status == 'correct',
            is_partial=answer_entity.status == 'partial',
            response_time_seconds=response_time,
            confidence_score=answer_entity.confidence_score,
            evaluation_score=answer_entity.confidence_score,
            attempt_number=attempt_number,
            hint_used=hint_used,
            created_at=end_time  # Set creation time to submission time
        )
        
        # Cache for potential reuse
        cache_key = f"{user_id}_{question_id}_{attempt_number}"
        self._answer_cache[cache_key] = relationship
        
        return relationship
    
    # Knowledge requirement builders
    def build_question_topic_relationships(
        self,
        question_id: str,
        question_content: str,
        primary_topic: str,
        secondary_topics: List[str] = None,
        skill_requirements: Dict[str, List[str]] = None
    ) -> List[RequiresKnowledgeRelationship]:
        """
        Build all topic relationships for a question.
        
        Args:
            question_id: The question ID
            question_content: Question text for analysis
            primary_topic: Main topic being tested
            secondary_topics: Additional related topics
            skill_requirements: Topic-specific skills required
            
        Returns:
            List of RequiresKnowledgeRelationship objects
        """
        relationships = []
        
        # Primary topic relationship
        primary_rel = RequiresKnowledgeRelationship.create_from_question_analysis(
            question_id=question_id,
            topic_id=primary_topic,
            question_content=question_content,
            topic_name=primary_topic,
            is_primary=True
        )
        
        # Add skill components if provided
        if skill_requirements and primary_topic in skill_requirements:
            primary_rel.skill_components = skill_requirements[primary_topic]
        
        relationships.append(primary_rel)
        
        # Secondary topic relationships
        if secondary_topics:
            for i, topic in enumerate(secondary_topics):
                secondary_rel = RequiresKnowledgeRelationship.create_from_question_analysis(
                    question_id=question_id,
                    topic_id=topic,
                    question_content=question_content,
                    topic_name=topic,
                    is_primary=False
                )
                
                # Decrease importance for each secondary topic
                secondary_rel.importance_weight *= (0.8 - i * 0.1)
                
                # Add skills if provided
                if skill_requirements and topic in skill_requirements:
                    secondary_rel.skill_components = skill_requirements[topic]
                
                relationships.append(secondary_rel)
        
        # Cache for reuse
        self._question_topics[question_id] = relationships
        
        return relationships
    
    # Mastery relationship builders
    def build_or_update_mastery(
        self,
        user_id: str,
        topic_id: str,
        answer_was_correct: bool,
        answer_was_partial: bool = False,
        response_time: float = None,
        question_difficulty: float = 0.5,
        existing_mastery: Optional[MasteryRelationship] = None
    ) -> MasteryRelationship:
        """
        Build a new or update existing mastery relationship.
        
        Args:
            user_id: User ID
            topic_id: Topic ID
            answer_was_correct: Whether the answer was correct
            answer_was_partial: Whether partial credit was given
            response_time: Time taken to answer
            question_difficulty: Difficulty of the question (0-1)
            existing_mastery: Existing relationship to update
            
        Returns:
            New or updated MasteryRelationship
        """
        cache_key = (user_id, topic_id)
        
        # Check cache first
        if not existing_mastery and cache_key in self._user_topic_mastery:
            existing_mastery = self._user_topic_mastery[cache_key]
        
        if existing_mastery:
            # Update existing mastery
            existing_mastery.update_mastery(
                was_correct=answer_was_correct,
                was_partial=answer_was_partial,
                response_time=response_time,
                question_difficulty=question_difficulty
            )
            mastery = existing_mastery
        else:
            # Create new mastery relationship
            initial_score = 0.0
            if answer_was_correct:
                initial_score = 0.3  # Start with some mastery for correct answer
            elif answer_was_partial:
                initial_score = 0.15
            
            mastery = MasteryRelationship.create_initial(
                user_id=user_id,
                topic_id=topic_id,
                initial_performance=initial_score
            )
            
            # Apply the first update
            mastery.update_mastery(
                was_correct=answer_was_correct,
                was_partial=answer_was_partial,
                response_time=response_time,
                question_difficulty=question_difficulty
            )
        
        # Update cache
        self._user_topic_mastery[cache_key] = mastery
        
        return mastery
    
    # Batch builders
    def build_relationships_from_qa_interaction(
        self,
        user_id: str,
        question_entity: Any,  # Would be QuestionEntity
        answer_entity: Any,    # Would be AnswerEntity
        topic_entities: List[Any],  # Would be List[TopicEntity]
        start_time: datetime,
        end_time: datetime,
        existing_masteries: Dict[str, MasteryRelationship] = None
    ) -> Dict[str, List[BaseRelationship]]:
        """
        Build all relationships from a complete Q&A interaction.
        
        Args:
            user_id: User who answered
            question_entity: The question that was asked
            answer_entity: The answer provided
            topic_entities: Topics related to the question
            start_time: When interaction started
            end_time: When interaction ended
            existing_masteries: Existing mastery relationships by topic_id
            
        Returns:
            Dictionary with relationship types as keys and lists of relationships
        """
        relationships = {
            'answered': [],
            'requires_knowledge': [],
            'masters': []
        }
        
        # Build answer relationship
        answer_rel = self.build_answer_relationship(
            user_id=user_id,
            question_id=question_entity.id,
            answer_entity=answer_entity,
            start_time=start_time,
            end_time=end_time,
            attempt_number=1  # Would be calculated in practice
        )
        relationships['answered'].append(answer_rel)
        
        # Build question-topic relationships
        primary_topic = topic_entities[0] if topic_entities else None
        secondary_topics = [t.id for t in topic_entities[1:]] if len(topic_entities) > 1 else []
        
        if primary_topic:
            topic_rels = self.build_question_topic_relationships(
                question_id=question_entity.id,
                question_content=question_entity.content,
                primary_topic=primary_topic.id,
                secondary_topics=secondary_topics
            )
            relationships['requires_knowledge'].extend(topic_rels)
        
        # Build/update mastery relationships
        existing_masteries = existing_masteries or {}
        response_time = answer_rel.response_time_seconds
        
        for i, topic in enumerate(topic_entities):
            # Adjust question difficulty based on topic position
            # Primary topic gets full difficulty, secondary topics get reduced
            adjusted_difficulty = question_entity.difficulty * (1.0 - i * 0.1)
            
            mastery = self.build_or_update_mastery(
                user_id=user_id,
                topic_id=topic.id,
                answer_was_correct=answer_rel.is_correct,
                answer_was_partial=answer_rel.is_partial,
                response_time=response_time,
                question_difficulty=adjusted_difficulty,
                existing_mastery=existing_masteries.get(topic.id)
            )
            relationships['masters'].append(mastery)
        
        return relationships
    
    # Utility methods
    def calculate_topic_importance_from_history(
        self,
        question_id: str,
        topic_id: str,
        answer_history: List[AnsweredRelationship]
    ) -> float:
        """
        Calculate topic importance based on answer patterns.
        
        Args:
            question_id: Question to analyze
            topic_id: Topic to evaluate
            answer_history: Historical answers for this question
            
        Returns:
            Calculated importance weight (0-1)
        """
        if not answer_history:
            return 0.5  # Default importance
        
        # Group answers by user mastery of the topic
        high_mastery_correct = 0
        high_mastery_total = 0
        low_mastery_correct = 0
        low_mastery_total = 0
        
        for answer in answer_history:
            # This would need actual mastery lookup in practice
            # For now, simulate based on overall performance
            user_mastery = 0.7 if answer.strength > 0.6 else 0.3
            
            if user_mastery > 0.5:
                high_mastery_total += 1
                if answer.is_correct:
                    high_mastery_correct += 1
            else:
                low_mastery_total += 1
                if answer.is_correct:
                    low_mastery_correct += 1
        
        # Calculate success rates
        high_success = high_mastery_correct / high_mastery_total if high_mastery_total > 0 else 0
        low_success = low_mastery_correct / low_mastery_total if low_mastery_total > 0 else 0
        
        # Higher difference means topic is more important
        importance = min(1.0, max(0.1, high_success - low_success))
        
        return importance
    
    def get_recommended_difficulty_for_user(
        self,
        user_id: str,
        topic_ids: List[str]
    ) -> str:
        """
        Get recommended question difficulty based on user's mastery.
        
        Args:
            user_id: User to evaluate
            topic_ids: Topics to consider
            
        Returns:
            Recommended difficulty level
        """
        if not topic_ids:
            return "medium"
        
        # Get cached masteries
        masteries = []
        for topic_id in topic_ids:
            cache_key = (user_id, topic_id)
            if cache_key in self._user_topic_mastery:
                masteries.append(self._user_topic_mastery[cache_key])
        
        if not masteries:
            return "easy"  # Default for new user
        
        # Average mastery score
        avg_mastery = sum(m.mastery_score for m in masteries) / len(masteries)
        
        # Map to difficulty
        if avg_mastery < 0.3:
            return "easy"
        elif avg_mastery < 0.6:
            return "medium"
        elif avg_mastery < 0.85:
            return "hard"
        else:
            return "expert"
    
    def clear_cache(self):
        """Clear all internal caches."""
        self._user_topic_mastery.clear()
        self._question_topics.clear()
        self._answer_cache.clear()


# Global builder instance for convenience
relationship_builder = RelationshipBuilder()


# Helper functions for common operations
def create_qa_relationships(
    user_id: str,
    question_id: str,
    answer_content: str,
    is_correct: bool,
    topics: List[str],
    response_time: float,
    session_id: str
) -> Dict[str, List[BaseRelationship]]:
    """
    Convenience function to create all relationships for a Q&A interaction.
    
    This is a simplified version that doesn't require full entity objects.
    """
    from datetime import datetime, timedelta
    
    # Create mock entities for the builder
    class MockAnswer:
        def __init__(self):
            self.id = f"a_{hash(answer_content)}_{int(datetime.now().timestamp())}"
            self.status = 'correct' if is_correct else 'incorrect'
            self.confidence_score = 0.95 if is_correct else 0.6
            self.session_id = session_id
    
    class MockQuestion:
        def __init__(self):
            self.id = question_id
            self.content = "Mock question"
            self.difficulty = 0.5
    
    class MockTopic:
        def __init__(self, topic_id):
            self.id = topic_id
    
    # Create mock entities
    answer = MockAnswer()
    question = MockQuestion()
    topic_entities = [MockTopic(t) for t in topics]
    
    # Calculate times
    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=response_time)
    
    # Use the builder
    return relationship_builder.build_relationships_from_qa_interaction(
        user_id=user_id,
        question_entity=question,
        answer_entity=answer,
        topic_entities=topic_entities,
        start_time=start_time,
        end_time=end_time
    )