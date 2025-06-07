# Memory Updates Module

The memory updates module provides functionality for tracking and updating user performance data after answer evaluation. It integrates with Graphiti to maintain a comprehensive record of user progress, topic mastery, and learning trends.

## Overview

The module consists of three main components:

1. **MemoryUpdateService**: Core service for processing evaluation results
2. **EvaluationEventHandler**: Batch processing handler for evaluation events
3. **Data Models**: EvaluationResult and UserPerformanceUpdate

## Core Components

### EvaluationResult

A dataclass that captures all relevant information from an answer evaluation:

```python
@dataclass
class EvaluationResult:
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
```

### MemoryUpdateService

The main service class that handles all memory update operations:

```python
service = MemoryUpdateService(client=graphiti_client, config=config)

# Record an evaluation immediately
await service.record_evaluation(evaluation_result, immediate=True)

# Or batch for later processing
await service.record_evaluation(evaluation_result, immediate=False)
await service.flush_pending_updates()
```

#### Key Methods

- **record_evaluation**: Records an evaluation result (immediate or batched)
- **flush_pending_updates**: Processes all pending batch updates
- **get_user_progress_summary**: Retrieves comprehensive user statistics

### EvaluationEventHandler

Provides automatic batch processing of evaluation events:

```python
handler = EvaluationEventHandler(update_service)
handler._batch_size = 10  # Process every 10 events

await handler.handle_evaluation_event(
    question_id="q_123",
    answer_id="a_456",
    user_id="user_789",
    session_id="session_abc",
    correct=True,
    comment="Well done!",
    confidence=0.95,
    response_time=4.2,
    topics=["math", "algebra"],
    difficulty=DifficultyLevel.MEDIUM
)
```

## Update Operations

### 1. Answer Entity Update

Updates the answer entity with evaluation status and feedback:

```python
await storage.update_answer_evaluation(
    answer_id=result.answer_id,
    correct=result.correct,
    comment=result.evaluation_comment
)
```

### 2. Evaluation Relationship

Creates a relationship between answer and question with evaluation metadata:

```cypher
MATCH (q:Question {id: $question_id})
MATCH (a:Answer {id: $answer_id})
MERGE (a)-[r:EVALUATED {
    timestamp: datetime(),
    correct: $correct,
    confidence: $confidence,
    comment: $comment
}]->(q)
```

### 3. User Statistics

Updates overall user performance metrics:

```cypher
MATCH (u:User {id: $user_id})
SET u.total_questions = COALESCE(u.total_questions, 0) + 1,
    u.correct_answers = COALESCE(u.correct_answers, 0) + CASE WHEN $correct THEN 1 ELSE 0 END,
    u.last_active = datetime()
```

### 4. Session Statistics

Tracks performance within specific sessions:

```cypher
MERGE (s:Session {id: $session_id})
SET s.total_questions = COALESCE(s.total_questions, 0) + 1,
    s.correct_answers = COALESCE(s.correct_answers, 0) + CASE WHEN $correct THEN 1 ELSE 0 END,
    s.last_activity = datetime()
```

### 5. Topic Mastery

Updates user's mastery level for each topic:

```python
# Mastery calculation considers:
# - Historical accuracy
# - Question difficulty
# - Recent performance
# - Number of attempts

new_level = (current_level * (1 - weight)) + (recent_score * weight)
confidence = min(1.0, total_attempts / 10)
```

### 6. Performance Tracking

Creates performance events for trend analysis:

```cypher
CREATE (e:PerformanceEvent {
    id: $event_id,
    user_id: $user_id,
    timestamp: datetime(),
    correct: $correct,
    difficulty: $difficulty,
    response_time: $response_time,
    topics: $topics
})
```

### 7. Streak Management

Tracks consecutive correct answers:

```python
# For correct answers
SET u.current_streak = COALESCE(u.current_streak, 0) + 1,
    u.best_streak = CASE 
        WHEN COALESCE(u.current_streak, 0) + 1 > COALESCE(u.best_streak, 0)
        THEN COALESCE(u.current_streak, 0) + 1
        ELSE COALESCE(u.best_streak, 0)
    END

# For incorrect answers
SET u.current_streak = 0
```

## Progress Summary

The `get_user_progress_summary` method provides comprehensive user statistics:

```python
summary = await service.get_user_progress_summary("user_123")

# Returns:
{
    "overall": {
        "total_questions": 150,
        "correct_answers": 112,
        "accuracy": 0.747,
        "avg_response_time": 4.2,
        "current_streak": 7,
        "best_streak": 15,
        "topics_practiced": 12,
        "avg_mastery": 0.68
    },
    "recent_performance": {
        "questions_last_7_days": 35,
        "correct_last_7_days": 30,
        "recent_accuracy": 0.857,
        "recent_avg_time": 3.5
    },
    "topics": [
        {
            "name": "mathematics",
            "mastery": 0.82,
            "accuracy": 0.85,
            "attempts": 45,
            "last_practiced": "2024-01-15T10:30:00"
        }
    ]
}
```

## Mastery Level Calculation

The mastery level algorithm considers multiple factors:

1. **Historical Performance**: Overall accuracy rate
2. **Recent Performance**: Latest answer correctness
3. **Difficulty Weighting**: Harder questions have more impact
4. **Attempt Count**: Confidence increases with more attempts

### Difficulty Weights

- Easy: 0.2
- Medium: 0.3
- Hard: 0.4
- Expert: 0.5

### Performance Adjustments

- Consistent high performance (>80% accuracy): +10% boost
- Struggling performance (<30% accuracy): -10% reduction
- Confidence reaches 100% after 10 attempts

## Usage Patterns

### Immediate Processing

Best for real-time feedback and small-scale applications:

```python
result = EvaluationResult(...)
await service.record_evaluation(result, immediate=True)
```

### Batch Processing

Optimal for high-volume scenarios:

```python
# Collect evaluations
for eval_data in evaluations:
    await service.record_evaluation(eval_data, immediate=False)

# Process batch
successful, failed = await service.flush_pending_updates()
```

### Event-Driven Processing

Using the event handler for automatic batching:

```python
handler = EvaluationEventHandler(service)
handler._batch_size = 20

# Events are automatically batched
for event in evaluation_stream:
    await handler.handle_evaluation_event(**event)

# Flush remaining at end
await handler.flush()
```

## Error Handling

The module includes comprehensive error handling:

- Failed updates don't crash the system
- Batch processing continues even if individual updates fail
- All errors are logged with context
- Methods return success/failure indicators

## Performance Considerations

1. **Batch Size**: Adjust based on your workload (default: 10)
2. **Immediate vs Batch**: Use batch mode for high-throughput scenarios
3. **Query Optimization**: Updates are designed to minimize database queries
4. **Async Processing**: All operations are async for better performance

## Integration with Question Graph

The memory updates module integrates seamlessly with the question graph:

1. After the `Evaluate` node classifies an answer
2. Create an `EvaluationResult` from the evaluation
3. Pass it to `MemoryUpdateService`
4. Updates are reflected in future question generation

Example integration:

```python
# In the Evaluate node
evaluation = await evaluate_answer(answer_text, expected_answers)

# Create result
result = EvaluationResult(
    question_id=ctx.state.current_question.id,
    answer_id=answer_entity.id,
    user_id=ctx.state.user_id,
    session_id=ctx.state.session_id,
    correct=(evaluation.status == AnswerStatus.CORRECT),
    evaluation_comment=evaluation.feedback,
    confidence_score=evaluation.confidence,
    topics=ctx.state.current_question.topics,
    difficulty=ctx.state.current_question.difficulty
)

# Update memory
await memory_update_service.record_evaluation(result)
```

## Best Practices

1. **Always Include Topics**: Helps track subject-specific progress
2. **Record Response Time**: Valuable for performance analysis
3. **Use Meaningful Comments**: Helps users understand their mistakes
4. **Batch When Possible**: Improves performance for high-volume apps
5. **Monitor Progress**: Regularly check user summaries for insights
6. **Handle Failures Gracefully**: Don't let update failures break the flow

## Future Enhancements

Potential improvements to consider:

1. **Adaptive Difficulty**: Automatically adjust question difficulty based on mastery
2. **Learning Curves**: Track improvement rate over time
3. **Peer Comparison**: Compare user progress to cohort averages
4. **Recommendation Engine**: Suggest topics needing more practice
5. **Gamification**: Achievement system based on streaks and mastery