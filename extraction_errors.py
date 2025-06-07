"""
Error handling and fallback strategies for entity extraction.

This module provides robust error handling, retry logic, and fallback strategies
to ensure entity extraction continues even when individual operations fail.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

import logfire
from pydantic import BaseModel, Field

from entity_extraction import (
    AnswerEvaluation,
    AnswerPattern,
    AnswerStatus,
    ComplexityMetrics,
    DifficultyLevel,
    EntityExtractor,
    ExtractedTopic,
)
from graphiti_entities import (
    AnswerEntity,
    QuestionEntity,
    TopicEntity,
    UserEntity,
)


# Sub-task 3.6: Error types and handling
class ExtractionErrorType(str, Enum):
    """Types of errors that can occur during extraction."""
    
    INVALID_INPUT = "invalid_input"
    PARSING_ERROR = "parsing_error"
    MODEL_ERROR = "model_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


class ExtractionError(BaseModel):
    """Structured error information for extraction failures."""
    
    error_type: ExtractionErrorType
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = Field(default=0, ge=0)
    is_recoverable: bool = Field(default=True)
    fallback_used: Optional[str] = None
    
    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'error_type': self.error_type.value,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'retry_count': self.retry_count,
            'is_recoverable': self.is_recoverable,
            'fallback_used': self.fallback_used
        }


class ExtractionFallbackStrategy(BaseModel):
    """Defines a fallback strategy for extraction failures."""
    
    strategy_name: str
    priority: int = Field(ge=0, le=10)
    max_retries: int = Field(default=3, ge=0)
    retry_delay_seconds: float = Field(default=1.0, gt=0)
    exponential_backoff: bool = Field(default=True)
    
    def calculate_delay(self, retry_count: int) -> float:
        """Calculate retry delay based on strategy settings."""
        if self.exponential_backoff:
            return self.retry_delay_seconds * (2 ** retry_count)
        return self.retry_delay_seconds


class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling behavior."""
    
    # Global settings
    capture_errors: bool = Field(default=True)
    log_errors: bool = Field(default=True)
    raise_on_final_failure: bool = Field(default=False)
    
    # Retry settings
    default_max_retries: int = Field(default=3, ge=0)
    default_retry_delay: float = Field(default=1.0, gt=0)
    
    # Timeout settings
    extraction_timeout_seconds: float = Field(default=30.0, gt=0)
    batch_timeout_seconds: float = Field(default=300.0, gt=0)
    
    # Fallback settings
    enable_fallbacks: bool = Field(default=True)
    fallback_strategies: List[ExtractionFallbackStrategy] = Field(default_factory=list)
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.fallback_strategies:
            self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default fallback strategies."""
        self.fallback_strategies = [
            ExtractionFallbackStrategy(
                strategy_name="simple_retry",
                priority=1,
                max_retries=3,
                retry_delay_seconds=1.0
            ),
            ExtractionFallbackStrategy(
                strategy_name="reduced_complexity",
                priority=2,
                max_retries=2,
                retry_delay_seconds=0.5
            ),
            ExtractionFallbackStrategy(
                strategy_name="basic_extraction",
                priority=3,
                max_retries=1,
                retry_delay_seconds=0.1
            )
        ]


class RobustEntityExtractor(EntityExtractor):
    """
    Entity extractor with comprehensive error handling and fallback strategies.
    
    This class wraps EntityExtractor methods with error handling, retry logic,
    and fallback strategies to ensure robust extraction even when failures occur.
    """
    
    def __init__(
        self,
        extractor: Optional[EntityExtractor] = None,
        config: Optional[ErrorHandlingConfig] = None
    ):
        """
        Initialize robust extractor.
        
        Args:
            extractor: Base entity extractor to wrap
            config: Error handling configuration
        """
        # Initialize parent with same configuration as wrapped extractor
        if extractor:
            super().__init__(extractor.topics.values())
        else:
            super().__init__()
        
        self.base_extractor = extractor or EntityExtractor()
        self.config = config or ErrorHandlingConfig()
        self.error_history: List[ExtractionError] = []
        self._circuit_breaker_open = False
        self._consecutive_failures = 0
    
    def _handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        retry_count: int = 0
    ) -> ExtractionError:
        """
        Handle an extraction error.
        
        Args:
            error: The exception that occurred
            context: Context information about the operation
            retry_count: Number of retries attempted
            
        Returns:
            Structured error information
        """
        # Determine error type
        error_type = ExtractionErrorType.UNKNOWN_ERROR
        is_recoverable = True
        
        if isinstance(error, ValueError):
            error_type = ExtractionErrorType.VALIDATION_ERROR
        elif isinstance(error, asyncio.TimeoutError):
            error_type = ExtractionErrorType.TIMEOUT_ERROR
        elif "parse" in str(error).lower():
            error_type = ExtractionErrorType.PARSING_ERROR
        elif "model" in str(error).lower():
            error_type = ExtractionErrorType.MODEL_ERROR
        
        # Create error record
        extraction_error = ExtractionError(
            error_type=error_type,
            message=str(error),
            context=context,
            retry_count=retry_count,
            is_recoverable=is_recoverable
        )
        
        # Log error if configured
        if self.config.log_errors:
            logfire.error(
                "Extraction error occurred",
                **extraction_error.to_log_dict()
            )
        
        # Track error history
        if self.config.capture_errors:
            self.error_history.append(extraction_error)
        
        # Update circuit breaker
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            self._circuit_breaker_open = True
            logfire.warning("Circuit breaker opened due to consecutive failures")
        
        return extraction_error
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker on successful operation."""
        if self._consecutive_failures > 0:
            self._consecutive_failures = 0
            self._circuit_breaker_open = False
            logfire.info("Circuit breaker reset after successful operation")
    
    async def _retry_with_fallback(
        self,
        operation: Callable,
        args: tuple,
        kwargs: dict,
        context: Dict[str, Any],
        fallback_result: Any = None
    ) -> Any:
        """
        Execute operation with retry logic and fallback strategies.
        
        Args:
            operation: The operation to execute
            args: Positional arguments for operation
            kwargs: Keyword arguments for operation
            context: Context information
            fallback_result: Default result if all strategies fail
            
        Returns:
            Operation result or fallback result
        """
        if self._circuit_breaker_open:
            logfire.warning("Circuit breaker is open, using fallback")
            return fallback_result
        
        last_error = None
        
        for strategy in sorted(self.config.fallback_strategies, key=lambda s: s.priority):
            for retry in range(strategy.max_retries):
                try:
                    # Apply strategy-specific modifications
                    modified_kwargs = self._apply_strategy(strategy, kwargs)
                    
                    # Execute operation with timeout
                    result = await asyncio.wait_for(
                        self._execute_operation(operation, args, modified_kwargs),
                        timeout=self.config.extraction_timeout_seconds
                    )
                    
                    # Success - reset circuit breaker
                    self._reset_circuit_breaker()
                    return result
                    
                except Exception as e:
                    last_error = self._handle_error(
                        e,
                        {**context, 'strategy': strategy.strategy_name},
                        retry
                    )
                    
                    # Calculate retry delay
                    if retry < strategy.max_retries - 1:
                        delay = strategy.calculate_delay(retry)
                        await asyncio.sleep(delay)
        
        # All strategies failed
        if self.config.raise_on_final_failure and last_error:
            raise Exception(f"All extraction strategies failed: {last_error.message}")
        
        return fallback_result
    
    async def _execute_operation(
        self,
        operation: Callable,
        args: tuple,
        kwargs: dict
    ) -> Any:
        """Execute operation, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            return await asyncio.to_thread(operation, *args, **kwargs)
    
    def _apply_strategy(
        self,
        strategy: ExtractionFallbackStrategy,
        kwargs: dict
    ) -> dict:
        """
        Apply strategy-specific modifications to operation parameters.
        
        Args:
            strategy: The fallback strategy to apply
            kwargs: Original keyword arguments
            
        Returns:
            Modified keyword arguments
        """
        modified = kwargs.copy()
        
        if strategy.strategy_name == "reduced_complexity":
            # Reduce complexity of operations
            if 'max_topics' in modified:
                modified['max_topics'] = min(3, modified['max_topics'])
            if 'min_confidence' in modified:
                modified['min_confidence'] = max(0.5, modified['min_confidence'])
                
        elif strategy.strategy_name == "basic_extraction":
            # Use most basic extraction settings
            if 'max_topics' in modified:
                modified['max_topics'] = 1
            if 'use_nlp' in modified:
                modified['use_nlp'] = False
        
        return modified
    
    # Wrapped extraction methods with error handling
    async def extract_topics_safe(
        self,
        text: str,
        min_confidence: float = 0.3,
        max_topics: int = 5
    ) -> List[ExtractedTopic]:
        """
        Extract topics with error handling and fallbacks.
        
        Args:
            text: Text to extract topics from
            min_confidence: Minimum confidence threshold
            max_topics: Maximum topics to return
            
        Returns:
            List of extracted topics or empty list on failure
        """
        context = {
            'operation': 'extract_topics',
            'text_length': len(text),
            'min_confidence': min_confidence,
            'max_topics': max_topics
        }
        
        fallback_result = []  # Empty list as fallback
        
        return await self._retry_with_fallback(
            self.base_extractor.extract_topics,
            (text,),
            {'min_confidence': min_confidence, 'max_topics': max_topics},
            context,
            fallback_result
        )
    
    async def estimate_difficulty_safe(
        self,
        text: str,
        topics: Optional[List[ExtractedTopic]] = None
    ) -> Tuple[DifficultyLevel, float, Optional[ComplexityMetrics]]:
        """
        Estimate difficulty with error handling and fallbacks.
        
        Args:
            text: Question text
            topics: Pre-extracted topics
            
        Returns:
            Tuple of (level, score, metrics) with fallback values
        """
        context = {
            'operation': 'estimate_difficulty',
            'text_length': len(text),
            'has_topics': topics is not None
        }
        
        # Fallback: medium difficulty with basic metrics
        fallback_metrics = ComplexityMetrics(
            word_count=len(text.split()),
            avg_word_length=5.0,
            sentence_count=1,
            avg_sentence_length=float(len(text.split())),
            complex_word_count=0,
            technical_term_count=0,
            subordinate_clause_count=0,
            has_negation=False,
            has_comparison=False,
            has_multiple_parts=False,
            requires_calculation=False,
            requires_reasoning=False,
            concept_count=0,
            relationship_count=0,
            abstraction_level=0.5,
            linguistic_complexity=0.5,
            cognitive_complexity=0.5,
            overall_complexity=0.5
        )
        fallback_result = (DifficultyLevel.MEDIUM, 0.5, fallback_metrics)
        
        try:
            result = await self._retry_with_fallback(
                self.base_extractor.estimate_difficulty,
                (text,),
                {'topics': topics},
                context,
                None
            )
            
            if result is None:
                return fallback_result
            
            return result
            
        except Exception as e:
            logfire.error("Difficulty estimation failed completely", error=str(e))
            return fallback_result
    
    async def classify_answer_safe(
        self,
        answer: str,
        expected_patterns: List[AnswerPattern],
        question_type: Optional[Dict[str, bool]] = None
    ) -> AnswerEvaluation:
        """
        Classify answer with error handling and fallbacks.
        
        Args:
            answer: User's answer
            expected_patterns: Expected answer patterns
            question_type: Question type info
            
        Returns:
            AnswerEvaluation with fallback to unevaluated status
        """
        context = {
            'operation': 'classify_answer',
            'answer_length': len(answer),
            'pattern_count': len(expected_patterns),
            'has_question_type': question_type is not None
        }
        
        # Fallback: unevaluated status
        fallback_result = AnswerEvaluation(
            status=AnswerStatus.UNEVALUATED,
            confidence=0.0,
            matched_patterns=[],
            partial_matches=[],
            feedback="Unable to evaluate answer due to processing error",
            suggestions=["Please try rephrasing your answer"],
            score=0.0
        )
        
        result = await self._retry_with_fallback(
            self.base_extractor.answer_classifier.classify_answer,
            (answer,),
            {'expected_patterns': expected_patterns, 'question_type': question_type},
            context,
            fallback_result
        )
        
        return result
    
    async def extract_entities_safe(
        self,
        question_text: str,
        answer_text: Optional[str] = None,
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Extract all entities with comprehensive error handling.
        
        Args:
            question_text: Question text
            answer_text: Optional answer text
            user_id: User identifier
            
        Returns:
            Dictionary with extracted entities and metadata
        """
        result = {
            'success': False,
            'question': None,
            'answer': None,
            'topics': [],
            'difficulty': None,
            'errors': []
        }
        
        # Extract question entity
        try:
            topics = await self.extract_topics_safe(question_text)
            difficulty = await self.estimate_difficulty_safe(question_text, topics)
            
            question_entity = QuestionEntity(
                content=question_text,
                difficulty=difficulty[0],
                topics=[t.topic_name for t in topics]
            )
            
            result['question'] = question_entity
            result['topics'] = topics
            result['difficulty'] = difficulty
            
        except Exception as e:
            error = self._handle_error(
                e,
                {'phase': 'question_extraction'},
                0
            )
            result['errors'].append(error)
        
        # Extract answer entity if provided
        if answer_text and result['question']:
            try:
                # Create simple patterns for demo
                patterns = [
                    AnswerPattern(
                        pattern_type='fuzzy',
                        expected_values=[],
                        similarity_threshold=0.7
                    )
                ]
                
                evaluation = await self.classify_answer_safe(
                    answer_text,
                    patterns
                )
                
                answer_entity = AnswerEntity(
                    question_id=result['question'].id,
                    user_id=user_id,
                    session_id="robust_extraction",
                    content=answer_text,
                    status=evaluation.status,
                    confidence_score=evaluation.confidence,
                    response_time_seconds=1.0,
                    feedback=evaluation.feedback
                )
                
                result['answer'] = answer_entity
                
            except Exception as e:
                error = self._handle_error(
                    e,
                    {'phase': 'answer_extraction'},
                    0
                )
                result['errors'].append(error)
        
        result['success'] = len(result['errors']) == 0
        return result
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        if not self.error_history:
            return {
                'total_errors': 0,
                'error_types': {},
                'recoverable_errors': 0,
                'circuit_breaker_open': self._circuit_breaker_open
            }
        
        error_types = {}
        recoverable = 0
        
        for error in self.error_history:
            error_types[error.error_type.value] = error_types.get(error.error_type.value, 0) + 1
            if error.is_recoverable:
                recoverable += 1
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'recoverable_errors': recoverable,
            'circuit_breaker_open': self._circuit_breaker_open,
            'consecutive_failures': self._consecutive_failures,
            'last_error': self.error_history[-1].to_log_dict() if self.error_history else None
        }
    
    def clear_error_history(self):
        """Clear error history and reset circuit breaker."""
        self.error_history.clear()
        self._consecutive_failures = 0
        self._circuit_breaker_open = False
        logfire.info("Error history cleared and circuit breaker reset")


# Decorator for automatic error handling
def with_error_handling(
    fallback_result: Any = None,
    max_retries: int = 3,
    timeout_seconds: float = 30.0
):
    """
    Decorator to add error handling to extraction functions.
    
    Args:
        fallback_result: Result to return if function fails
        max_retries: Maximum retry attempts
        timeout_seconds: Timeout for function execution
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for retry in range(max_retries):
                try:
                    # Add timeout
                    if asyncio.iscoroutinefunction(func):
                        result = await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=timeout_seconds
                        )
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(func, *args, **kwargs),
                            timeout=timeout_seconds
                        )
                    
                    return result
                    
                except Exception as e:
                    last_error = e
                    logfire.warning(
                        f"Function {func.__name__} failed",
                        retry=retry,
                        error=str(e)
                    )
                    
                    if retry < max_retries - 1:
                        await asyncio.sleep(2 ** retry)  # Exponential backoff
            
            # All retries failed
            logfire.error(
                f"Function {func.__name__} failed after {max_retries} retries",
                error=str(last_error)
            )
            
            return fallback_result
        
        return wrapper
    return decorator


# Example usage functions
async def extract_with_monitoring(
    text: str,
    extractor: RobustEntityExtractor
) -> Dict[str, Any]:
    """
    Extract entities with performance monitoring and error tracking.
    
    Args:
        text: Text to extract from
        extractor: Robust extractor instance
        
    Returns:
        Extraction results with timing and error information
    """
    start_time = datetime.now()
    
    with logfire.span("monitored_extraction") as span:
        span.set_attribute("text_length", len(text))
        
        # Extract topics
        topics = await extractor.extract_topics_safe(text)
        span.set_attribute("topics_found", len(topics))
        
        # Estimate difficulty
        difficulty = await extractor.estimate_difficulty_safe(text, topics)
        span.set_attribute("difficulty_level", difficulty[0].value)
        span.set_attribute("difficulty_score", difficulty[1])
        
        # Get error summary
        errors = extractor.get_error_summary()
        span.set_attribute("total_errors", errors['total_errors'])
        
        elapsed = (datetime.now() - start_time).total_seconds()
        span.set_attribute("extraction_time_seconds", elapsed)
        
        return {
            'topics': topics,
            'difficulty': difficulty,
            'extraction_time': elapsed,
            'errors': errors
        }


# Global robust extractor instance
default_robust_extractor = RobustEntityExtractor()