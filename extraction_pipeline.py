"""
Batch processing pipelines for entity extraction.

This module provides efficient batch processing capabilities for extracting
entities from large volumes of Q&A data with progress tracking and parallelization.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import logfire
from pydantic import BaseModel, Field

from entity_extraction import (
    AnswerClassifier,
    AnswerEvaluation,
    AnswerPattern,
    ComplexityMetrics,
    EntityExtractor,
    ExtractedTopic,
)
from extraction_errors import (
    ErrorHandlingConfig,
    ExtractionError,
    ExtractionErrorType,
    RobustEntityExtractor,
    with_error_handling,
)
from graphiti_entities import (
    AnswerEntity,
    AnswerStatus,
    QuestionEntity,
    UserEntity,
)
from graphiti_relationships import RelationshipBuilder, create_qa_relationships


# Sub-task 3.4: Batch processing models
@dataclass
class QAItem:
    """Single Q&A item for batch processing."""
    
    question_text: str
    answer_text: Optional[str] = None
    correct_answers: Optional[List[str]] = None
    user_id: str = "anonymous"
    session_id: str = "batch"
    question_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BatchExtractionResult(BaseModel):
    """Result of batch extraction process."""
    
    total_items: int = Field(ge=0)
    successful: int = Field(ge=0)
    failed: int = Field(ge=0)
    
    questions: List[QuestionEntity] = Field(default_factory=list)
    answers: List[Tuple[AnswerEntity, AnswerEvaluation]] = Field(default_factory=list)
    
    processing_time: float = Field(ge=0.0)
    items_per_second: float = Field(ge=0.0)
    
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Aggregate statistics
    topic_distribution: Dict[str, int] = Field(default_factory=dict)
    difficulty_distribution: Dict[str, int] = Field(default_factory=dict)
    answer_status_distribution: Dict[str, int] = Field(default_factory=dict)


class ExtractionProgress:
    """Track extraction progress for long-running operations."""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = datetime.now()
        self._lock = asyncio.Lock()
        self._callbacks: List[Callable[[ExtractionProgress], None]] = []
    
    async def increment(self, success: bool = True):
        """Increment progress counter."""
        async with self._lock:
            self.processed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
            
            # Notify callbacks
            for callback in self._callbacks:
                callback(self)
    
    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.processed / self.total_items) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def items_per_second(self) -> float:
        """Get processing rate."""
        if self.elapsed_time == 0:
            return 0.0
        return self.processed / self.elapsed_time
    
    @property
    def estimated_time_remaining(self) -> float:
        """Estimate remaining time in seconds."""
        if self.items_per_second == 0:
            return 0.0
        remaining = self.total_items - self.processed
        return remaining / self.items_per_second
    
    def add_callback(self, callback: Callable[[ExtractionProgress], None]):
        """Add progress callback."""
        self._callbacks.append(callback)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        return {
            'total': self.total_items,
            'processed': self.processed,
            'successful': self.successful,
            'failed': self.failed,
            'percentage': round(self.percentage, 2),
            'elapsed_seconds': round(self.elapsed_time, 2),
            'items_per_second': round(self.items_per_second, 2),
            'eta_seconds': round(self.estimated_time_remaining, 2)
        }


class ExtractionPipeline:
    """Batch processing pipeline for entity extraction."""
    
    def __init__(
        self,
        extractor: Optional[EntityExtractor] = None,
        max_workers: int = 4,
        use_async: bool = True
    ):
        """
        Initialize extraction pipeline.
        
        Args:
            extractor: Entity extractor instance
            max_workers: Maximum parallel workers
            use_async: Use async processing
        """
        self.extractor = extractor or EntityExtractor()
        self.max_workers = max_workers
        self.use_async = use_async
        self.relationship_builder = RelationshipBuilder()
    
    async def extract_questions_batch(
        self,
        questions: List[str],
        progress_callback: Optional[Callable[[ExtractionProgress], None]] = None
    ) -> BatchExtractionResult:
        """
        Extract entities from a batch of questions.
        
        Args:
            questions: List of question texts
            progress_callback: Optional progress callback
            
        Returns:
            BatchExtractionResult with extracted entities
        """
        start_time = datetime.now()
        progress = ExtractionProgress(len(questions))
        
        if progress_callback:
            progress.add_callback(progress_callback)
        
        result = BatchExtractionResult(total_items=len(questions))
        
        with logfire.span("pipeline.extract_questions_batch") as span:
            span.set_attribute("batch_size", len(questions))
            
            # Process in parallel
            if self.use_async:
                tasks = [
                    self._extract_question_async(q, i, progress, result)
                    for i, q in enumerate(questions)
                ]
                await asyncio.gather(*tasks)
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(self._extract_question_sync, q, i, result)
                        for i, q in enumerate(questions)
                    ]
                    for future in futures:
                        future.result()
                        await progress.increment(True)
            
            # Calculate statistics
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.items_per_second = len(questions) / result.processing_time if result.processing_time > 0 else 0
            
            # Aggregate statistics
            for question in result.questions:
                # Topic distribution
                for topic in question.topics:
                    result.topic_distribution[topic] = result.topic_distribution.get(topic, 0) + 1
                
                # Difficulty distribution
                result.difficulty_distribution[question.difficulty.value] = \
                    result.difficulty_distribution.get(question.difficulty.value, 0) + 1
            
            span.set_attribute("successful", result.successful)
            span.set_attribute("failed", result.failed)
            
        return result
    
    async def _extract_question_async(
        self,
        question_text: str,
        index: int,
        progress: ExtractionProgress,
        result: BatchExtractionResult
    ):
        """Extract single question asynchronously."""
        try:
            question_entity = await asyncio.to_thread(
                self.extractor.extract_entities_from_question,
                question_text,
                f"q_batch_{index}"
            )
            result.questions.append(question_entity)
            result.successful += 1
            await progress.increment(True)
        except Exception as e:
            result.failed += 1
            result.errors.append({
                'index': index,
                'question': question_text[:100],
                'error': str(e)
            })
            await progress.increment(False)
            logfire.error(f"Failed to extract question {index}", error=str(e))
    
    def _extract_question_sync(
        self,
        question_text: str,
        index: int,
        result: BatchExtractionResult
    ):
        """Extract single question synchronously."""
        try:
            question_entity = self.extractor.extract_entities_from_question(
                question_text,
                f"q_batch_{index}"
            )
            result.questions.append(question_entity)
            result.successful += 1
        except Exception as e:
            result.failed += 1
            result.errors.append({
                'index': index,
                'question': question_text[:100],
                'error': str(e)
            })
    
    async def process_qa_batch(
        self,
        qa_items: List[QAItem],
        progress_callback: Optional[Callable[[ExtractionProgress], None]] = None
    ) -> BatchExtractionResult:
        """
        Process a batch of Q&A items.
        
        Args:
            qa_items: List of Q&A items
            progress_callback: Optional progress callback
            
        Returns:
            BatchExtractionResult with questions and answers
        """
        start_time = datetime.now()
        progress = ExtractionProgress(len(qa_items))
        
        if progress_callback:
            progress.add_callback(progress_callback)
        
        result = BatchExtractionResult(total_items=len(qa_items))
        
        with logfire.span("pipeline.process_qa_batch") as span:
            span.set_attribute("batch_size", len(qa_items))
            
            # Process in batches for memory efficiency
            batch_size = min(100, self.max_workers * 10)
            
            for i in range(0, len(qa_items), batch_size):
                batch = qa_items[i:i + batch_size]
                
                if self.use_async:
                    tasks = [
                        self._process_qa_item_async(item, idx + i, progress, result)
                        for idx, item in enumerate(batch)
                    ]
                    await asyncio.gather(*tasks)
                else:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = [
                            executor.submit(self._process_qa_item_sync, item, idx + i, result)
                            for idx, item in enumerate(batch)
                        ]
                        for future in futures:
                            future.result()
                            await progress.increment(True)
            
            # Calculate final statistics
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.items_per_second = len(qa_items) / result.processing_time if result.processing_time > 0 else 0
            
            # Answer status distribution
            for answer, evaluation in result.answers:
                status = evaluation.status.value
                result.answer_status_distribution[status] = \
                    result.answer_status_distribution.get(status, 0) + 1
            
            span.set_attribute("successful", result.successful)
            span.set_attribute("failed", result.failed)
        
        return result
    
    async def _process_qa_item_async(
        self,
        item: QAItem,
        index: int,
        progress: ExtractionProgress,
        result: BatchExtractionResult
    ):
        """Process single Q&A item asynchronously."""
        try:
            # Extract question
            question_entity = await asyncio.to_thread(
                self.extractor.extract_entities_from_question,
                item.question_text,
                item.question_id or f"q_batch_{index}"
            )
            result.questions.append(question_entity)
            
            # Extract answer if provided
            if item.answer_text:
                answer_entity, evaluation = await asyncio.to_thread(
                    self.extractor.extract_answer_entity,
                    item.answer_text,
                    item.question_text,
                    item.correct_answers,
                    item.user_id,
                    item.session_id,
                    item.metadata.get('response_time', 0.0)
                )
                result.answers.append((answer_entity, evaluation))
            
            result.successful += 1
            await progress.increment(True)
            
        except Exception as e:
            result.failed += 1
            result.errors.append({
                'index': index,
                'question': item.question_text[:100],
                'answer': item.answer_text[:100] if item.answer_text else None,
                'error': str(e)
            })
            await progress.increment(False)
            logfire.error(f"Failed to process Q&A item {index}", error=str(e))
    
    def _process_qa_item_sync(
        self,
        item: QAItem,
        index: int,
        result: BatchExtractionResult
    ):
        """Process single Q&A item synchronously."""
        try:
            # Extract question
            question_entity = self.extractor.extract_entities_from_question(
                item.question_text,
                item.question_id or f"q_batch_{index}"
            )
            result.questions.append(question_entity)
            
            # Extract answer if provided
            if item.answer_text:
                answer_entity, evaluation = self.extractor.extract_answer_entity(
                    item.answer_text,
                    item.question_text,
                    item.correct_answers,
                    item.user_id,
                    item.session_id,
                    item.metadata.get('response_time', 0.0)
                )
                result.answers.append((answer_entity, evaluation))
            
            result.successful += 1
            
        except Exception as e:
            result.failed += 1
            result.errors.append({
                'index': index,
                'question': item.question_text[:100],
                'answer': item.answer_text[:100] if item.answer_text else None,
                'error': str(e)
            })
    
    def create_pipeline_report(
        self,
        result: BatchExtractionResult
    ) -> str:
        """
        Create a human-readable report from batch results.
        
        Args:
            result: Batch extraction result
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=== Extraction Pipeline Report ===\n")
        
        # Summary
        report.append(f"Total items: {result.total_items}")
        report.append(f"Successful: {result.successful}")
        report.append(f"Failed: {result.failed}")
        report.append(f"Processing time: {result.processing_time:.2f}s")
        report.append(f"Rate: {result.items_per_second:.2f} items/second\n")
        
        # Topic distribution
        if result.topic_distribution:
            report.append("Topic Distribution:")
            for topic, count in sorted(result.topic_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / result.successful) * 100 if result.successful > 0 else 0
                report.append(f"  - {topic}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Difficulty distribution
        if result.difficulty_distribution:
            report.append("Difficulty Distribution:")
            for difficulty, count in sorted(result.difficulty_distribution.items()):
                percentage = (count / result.successful) * 100 if result.successful > 0 else 0
                report.append(f"  - {difficulty}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Answer status distribution
        if result.answer_status_distribution:
            report.append("Answer Status Distribution:")
            for status, count in sorted(result.answer_status_distribution.items()):
                percentage = (count / len(result.answers)) * 100 if result.answers else 0
                report.append(f"  - {status}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Errors
        if result.errors:
            report.append(f"Errors ({len(result.errors)}):")
            for error in result.errors[:5]:  # Show first 5 errors
                report.append(f"  - Item {error['index']}: {error['error']}")
            if len(result.errors) > 5:
                report.append(f"  ... and {len(result.errors) - 5} more errors")
        
        return "\n".join(report)


# Convenience functions
async def extract_questions_from_file(
    file_path: str,
    extractor: Optional[EntityExtractor] = None,
    progress_callback: Optional[Callable[[ExtractionProgress], None]] = None
) -> BatchExtractionResult:
    """
    Extract questions from a text file (one per line).
    
    Args:
        file_path: Path to text file
        extractor: Optional extractor instance
        progress_callback: Optional progress callback
        
    Returns:
        BatchExtractionResult
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    pipeline = ExtractionPipeline(extractor)
    return await pipeline.extract_questions_batch(questions, progress_callback)


async def process_qa_csv(
    csv_path: str,
    question_col: str = "question",
    answer_col: str = "answer",
    correct_answer_col: Optional[str] = "correct_answer",
    extractor: Optional[EntityExtractor] = None,
    progress_callback: Optional[Callable[[ExtractionProgress], None]] = None
) -> BatchExtractionResult:
    """
    Process Q&A data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        question_col: Question column name
        answer_col: Answer column name
        correct_answer_col: Optional correct answer column
        extractor: Optional extractor instance
        progress_callback: Optional progress callback
        
    Returns:
        BatchExtractionResult
    """
    import csv
    
    qa_items = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if question_col not in row:
                continue
            
            item = QAItem(
                question_text=row[question_col],
                answer_text=row.get(answer_col),
                correct_answers=[row[correct_answer_col]] if correct_answer_col and correct_answer_col in row else None,
                question_id=f"csv_{i}",
                metadata={'row_index': i}
            )
            qa_items.append(item)
    
    pipeline = ExtractionPipeline(extractor)
    return await pipeline.process_qa_batch(qa_items, progress_callback)


def print_progress(progress: ExtractionProgress):
    """Simple progress printer for console output."""
    print(f"\rProgress: {progress.percentage:.1f}% "
          f"({progress.processed}/{progress.total_items}) "
          f"Rate: {progress.items_per_second:.1f} items/s "
          f"ETA: {progress.estimated_time_remaining:.0f}s", end='', flush=True)


# Sub-task 3.6: Robust pipeline with error handling
class RobustExtractionPipeline(ExtractionPipeline):
    """
    Extraction pipeline with comprehensive error handling and recovery.
    
    This pipeline ensures batch processing continues even when individual
    items fail, providing detailed error reporting and fallback strategies.
    """
    
    def __init__(
        self,
        extractor: Optional[EntityExtractor] = None,
        max_workers: int = 4,
        use_async: bool = True,
        error_config: Optional[ErrorHandlingConfig] = None
    ):
        """
        Initialize robust extraction pipeline.
        
        Args:
            extractor: Entity extractor instance
            max_workers: Maximum parallel workers
            use_async: Use async processing
            error_config: Error handling configuration
        """
        # Create robust extractor wrapper
        self.error_config = error_config or ErrorHandlingConfig()
        self.robust_extractor = RobustEntityExtractor(extractor, self.error_config)
        
        # Initialize parent with robust extractor
        super().__init__(self.robust_extractor, max_workers, use_async)
        
        # Error tracking
        self.failed_items: List[Dict[str, Any]] = []
        self.partial_failures: List[Dict[str, Any]] = []
    
    async def process_qa_batch_robust(
        self,
        qa_items: List[QAItem],
        progress_callback: Optional[Callable[[ExtractionProgress], None]] = None,
        continue_on_error: bool = True,
        error_callback: Optional[Callable[[int, Exception], None]] = None
    ) -> BatchExtractionResult:
        """
        Process Q&A batch with robust error handling.
        
        Args:
            qa_items: List of Q&A items
            progress_callback: Progress callback
            continue_on_error: Whether to continue processing on errors
            error_callback: Callback for error notification
            
        Returns:
            BatchExtractionResult with error details
        """
        start_time = datetime.now()
        progress = ExtractionProgress(len(qa_items))
        
        if progress_callback:
            progress.add_callback(progress_callback)
        
        result = BatchExtractionResult(total_items=len(qa_items))
        
        # Reset error tracking
        self.failed_items.clear()
        self.partial_failures.clear()
        
        with logfire.span("robust_pipeline.process_batch") as span:
            span.set_attribute("batch_size", len(qa_items))
            span.set_attribute("continue_on_error", continue_on_error)
            
            # Process items with error isolation
            for i, item in enumerate(qa_items):
                try:
                    # Process single item with timeout
                    item_result = await asyncio.wait_for(
                        self._process_single_item_robust(item, i),
                        timeout=self.error_config.extraction_timeout_seconds
                    )
                    
                    # Update result based on item processing
                    if item_result['success']:
                        if item_result.get('question'):
                            result.questions.append(item_result['question'])
                        if item_result.get('answer_tuple'):
                            result.answers.append(item_result['answer_tuple'])
                        result.successful += 1
                    else:
                        result.failed += 1
                        self.failed_items.append({
                            'index': i,
                            'item': item,
                            'errors': item_result.get('errors', [])
                        })
                    
                    await progress.increment(item_result['success'])
                    
                except asyncio.TimeoutError:
                    # Handle timeout
                    result.failed += 1
                    error_info = {
                        'index': i,
                        'item': item,
                        'error': 'Processing timeout exceeded',
                        'error_type': ExtractionErrorType.TIMEOUT_ERROR
                    }
                    
                    self.failed_items.append(error_info)
                    result.errors.append(error_info)
                    
                    if error_callback:
                        error_callback(i, TimeoutError("Processing timeout"))
                    
                    await progress.increment(False)
                    
                    if not continue_on_error:
                        break
                        
                except Exception as e:
                    # Handle other errors
                    result.failed += 1
                    error_info = {
                        'index': i,
                        'item': item,
                        'error': str(e),
                        'error_type': ExtractionErrorType.UNKNOWN_ERROR
                    }
                    
                    self.failed_items.append(error_info)
                    result.errors.append(error_info)
                    
                    if error_callback:
                        error_callback(i, e)
                    
                    await progress.increment(False)
                    
                    if not continue_on_error:
                        break
            
            # Calculate final statistics
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.items_per_second = len(qa_items) / result.processing_time if result.processing_time > 0 else 0
            
            # Update statistics from successful items
            self._calculate_statistics(result)
            
            # Add error summary
            error_summary = self.robust_extractor.get_error_summary()
            span.set_attribute("total_errors", error_summary['total_errors'])
            span.set_attribute("circuit_breaker_open", error_summary['circuit_breaker_open'])
            
            # Include error summary in result metadata
            if result.errors:
                result.errors.append({
                    'summary': error_summary,
                    'failed_count': len(self.failed_items),
                    'partial_failures': len(self.partial_failures)
                })
        
        return result
    
    async def _process_single_item_robust(
        self,
        item: QAItem,
        index: int
    ) -> Dict[str, Any]:
        """
        Process single Q&A item with error isolation.
        
        Args:
            item: Q&A item to process
            index: Item index in batch
            
        Returns:
            Processing result with success status
        """
        result = {
            'success': False,
            'question': None,
            'answer_tuple': None,
            'errors': []
        }
        
        try:
            # Extract question with error handling
            question_result = await self.robust_extractor.extract_entities_safe(
                item.question_text,
                user_id=item.user_id
            )
            
            if question_result['success'] and question_result['question']:
                result['question'] = question_result['question']
                
                # Process answer if provided
                if item.answer_text:
                    answer_result = await self._process_answer_robust(
                        item,
                        question_result['question'],
                        question_result.get('topics', [])
                    )
                    
                    if answer_result['success']:
                        result['answer_tuple'] = (
                            answer_result['answer'],
                            answer_result['evaluation']
                        )
                    else:
                        # Partial failure - question extracted but answer failed
                        self.partial_failures.append({
                            'index': index,
                            'item': item,
                            'phase': 'answer_processing',
                            'errors': answer_result.get('errors', [])
                        })
                
                result['success'] = True
            else:
                result['errors'] = question_result.get('errors', [])
                
        except Exception as e:
            result['errors'].append({
                'phase': 'item_processing',
                'error': str(e),
                'error_type': ExtractionErrorType.UNKNOWN_ERROR
            })
        
        return result
    
    async def _process_answer_robust(
        self,
        item: QAItem,
        question_entity: QuestionEntity,
        topics: List[ExtractedTopic]
    ) -> Dict[str, Any]:
        """
        Process answer with robust error handling.
        
        Args:
            item: Q&A item containing answer
            question_entity: Extracted question entity
            topics: Extracted topics
            
        Returns:
            Answer processing result
        """
        result = {
            'success': False,
            'answer': None,
            'evaluation': None,
            'errors': []
        }
        
        try:
            # Create answer patterns if expected answers provided
            patterns = []
            if item.correct_answers:
                patterns = await asyncio.to_thread(
                    self.robust_extractor.answer_classifier.create_answer_patterns,
                    item.correct_answers,
                    self.robust_extractor.classify_question_type(item.question_text)
                )
            
            # Evaluate answer
            evaluation = await self.robust_extractor.classify_answer_safe(
                item.answer_text,
                patterns
            )
            
            # Create answer entity
            answer_entity = AnswerEntity(
                question_id=question_entity.id,
                user_id=item.user_id,
                session_id=item.session_id,
                content=item.answer_text,
                status=evaluation.status,
                confidence_score=evaluation.confidence,
                response_time_seconds=item.metadata.get('response_time', 1.0),
                feedback=evaluation.feedback
            )
            
            result['answer'] = answer_entity
            result['evaluation'] = evaluation
            result['success'] = True
            
        except Exception as e:
            result['errors'].append({
                'phase': 'answer_evaluation',
                'error': str(e)
            })
        
        return result
    
    def _calculate_statistics(self, result: BatchExtractionResult):
        """Calculate statistics for successful items."""
        # Topic distribution
        for question in result.questions:
            for topic in question.topics:
                result.topic_distribution[topic] = result.topic_distribution.get(topic, 0) + 1
        
        # Difficulty distribution
        for question in result.questions:
            result.difficulty_distribution[question.difficulty.value] = \
                result.difficulty_distribution.get(question.difficulty.value, 0) + 1
        
        # Answer status distribution
        for answer, evaluation in result.answers:
            status = evaluation.status.value
            result.answer_status_distribution[status] = \
                result.answer_status_distribution.get(status, 0) + 1
    
    def create_error_report(self) -> str:
        """
        Create detailed error report for the pipeline.
        
        Returns:
            Formatted error report
        """
        report = []
        report.append("=== Robust Pipeline Error Report ===\n")
        
        # Overall summary
        total_failed = len(self.failed_items)
        total_partial = len(self.partial_failures)
        
        report.append(f"Total failed items: {total_failed}")
        report.append(f"Partial failures: {total_partial}")
        
        # Extractor error summary
        error_summary = self.robust_extractor.get_error_summary()
        report.append(f"\nExtractor errors: {error_summary['total_errors']}")
        report.append(f"Circuit breaker status: {'OPEN' if error_summary['circuit_breaker_open'] else 'CLOSED'}")
        
        # Error type breakdown
        if error_summary['error_types']:
            report.append("\nError types:")
            for error_type, count in error_summary['error_types'].items():
                report.append(f"  - {error_type}: {count}")
        
        # Failed items detail (first 5)
        if self.failed_items:
            report.append(f"\nFailed items (showing first 5 of {total_failed}):")
            for item in self.failed_items[:5]:
                report.append(f"  Item {item['index']}: {item.get('error', 'Unknown error')}")
        
        # Partial failures detail
        if self.partial_failures:
            report.append(f"\nPartial failures (showing first 3 of {total_partial}):")
            for item in self.partial_failures[:3]:
                report.append(f"  Item {item['index']}: Failed at {item['phase']}")
        
        return "\n".join(report)
    
    async def retry_failed_items(
        self,
        max_retries: int = 2
    ) -> BatchExtractionResult:
        """
        Retry processing for failed items.
        
        Args:
            max_retries: Maximum retry attempts per item
            
        Returns:
            Results from retry attempts
        """
        if not self.failed_items:
            return BatchExtractionResult(total_items=0)
        
        retry_items = []
        for failed in self.failed_items:
            if 'item' in failed and isinstance(failed['item'], QAItem):
                retry_items.append(failed['item'])
        
        if not retry_items:
            return BatchExtractionResult(total_items=0)
        
        logfire.info(f"Retrying {len(retry_items)} failed items")
        
        # Clear error history for retry
        self.robust_extractor.clear_error_history()
        
        # Process with increased timeout
        original_timeout = self.error_config.extraction_timeout_seconds
        self.error_config.extraction_timeout_seconds *= 2
        
        try:
            result = await self.process_qa_batch_robust(
                retry_items,
                continue_on_error=True
            )
            return result
        finally:
            # Restore original timeout
            self.error_config.extraction_timeout_seconds = original_timeout


# Enhanced convenience functions with error handling
@with_error_handling(fallback_result=BatchExtractionResult(total_items=0))
async def extract_questions_from_file_robust(
    file_path: str,
    extractor: Optional[EntityExtractor] = None,
    progress_callback: Optional[Callable[[ExtractionProgress], None]] = None,
    error_config: Optional[ErrorHandlingConfig] = None
) -> BatchExtractionResult:
    """
    Extract questions from file with robust error handling.
    
    Args:
        file_path: Path to text file
        extractor: Optional extractor instance
        progress_callback: Progress callback
        error_config: Error handling configuration
        
    Returns:
        BatchExtractionResult with error details
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logfire.error(f"Failed to read file {file_path}", error=str(e))
        result = BatchExtractionResult(total_items=0)
        result.errors.append({
            'error': f"File read error: {str(e)}",
            'file_path': file_path
        })
        return result
    
    pipeline = RobustExtractionPipeline(extractor, error_config=error_config)
    
    # Convert questions to QAItems
    qa_items = [
        QAItem(
            question_text=q,
            question_id=f"file_q_{i}"
        )
        for i, q in enumerate(questions)
    ]
    
    return await pipeline.process_qa_batch_robust(
        qa_items,
        progress_callback,
        continue_on_error=True
    )