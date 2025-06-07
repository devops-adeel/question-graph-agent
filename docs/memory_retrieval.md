# Memory Retrieval for Question Generation

This document describes the memory retrieval functionality that enables intelligent, personalized question generation based on user history and performance.

## Overview

The memory retrieval system leverages the Graphiti knowledge graph to:

1. **Track User Performance**: Monitor accuracy, response times, and topic mastery
2. **Identify Weak Areas**: Detect topics where users need more practice
3. **Avoid Repetition**: Prevent asking recently asked questions
4. **Personalize Difficulty**: Adjust question difficulty based on performance
5. **Generate Context-Aware Questions**: Create questions tailored to individual learning needs

## Core Components

### MemoryRetrieval

The main class for retrieving information from the knowledge graph.

```python
from memory_retrieval import MemoryRetrieval
from graphiti_client import GraphitiClient

client = GraphitiClient(...)
retrieval = MemoryRetrieval(client=client)
```

#### Key Methods

- **`get_asked_questions(user_id, session_id, limit)`**: Retrieve previously asked questions
- **`get_user_performance(user_id)`**: Get comprehensive performance metrics
- **`get_weak_topics(user_id, threshold, min_attempts)`**: Identify topics needing practice
- **`get_topic_questions(topic_name, exclude_ids, difficulty, limit)`**: Get questions for specific topics
- **`get_recommended_questions(user_id, session_id, count)`**: Get personalized recommendations

### EnhancedQuestionAgent

An intelligent agent that generates questions using memory context.

```python
from enhanced_question_agent import EnhancedQuestionAgent, QuestionGenerationContext

agent = EnhancedQuestionAgent(graphiti_client=client)

context = QuestionGenerationContext(
    user_id="user123",
    session_id="session456",
    prefer_weak_topics=True
)

question = await agent.generate_question(context)
```

### MemoryEnhancedAsk

A graph node that integrates memory retrieval into the question-asking flow.

```python
from memory_integration import MemoryEnhancedAsk

node = MemoryEnhancedAsk()
# Used within a pydantic_graph flow
```

## Usage Examples

### 1. Basic Memory Retrieval

```python
# Get user's performance data
performance = await retrieval.get_user_performance("user123")
print(f"Accuracy: {performance['accuracy']:.1%}")
print(f"Recommended difficulty: {performance['recommended_difficulty']}")

# Get weak topics
weak_topics = await retrieval.get_weak_topics("user123")
for topic, accuracy in weak_topics:
    print(f"{topic}: {accuracy:.1%} accuracy")
```

### 2. Personalized Question Generation

```python
# Create context with user preferences
context = QuestionGenerationContext(
    user_id="user123",
    session_id="session789",
    prefer_weak_topics=True,
    avoid_recent=10,
    difficulty_override=DifficultyLevel.MEDIUM
)

# Generate personalized question
question = await agent.generate_question(context)
```

### 3. Integration with Question Graph

```python
from memory_integration import MemoryStateEnhancer, create_memory_enhanced_graph

# Enhance state with memory capabilities
state = QuestionState()
state = MemoryStateEnhancer.enhance_state(
    state,
    graphiti_client=client,
    user_id="user123",
    session_id="session789"
)

# Create memory-enhanced graph
graph = await create_memory_enhanced_graph()
```

## Memory Retrieval Flow

1. **User Identification**: Identify the user and session
2. **Performance Analysis**: Retrieve user's historical performance
3. **Topic Analysis**: Identify strong and weak topics
4. **Question Selection**: Select or generate appropriate questions
5. **Context Building**: Build context for question generation
6. **Personalization**: Apply user-specific adjustments

## Recommendation Algorithm

The recommendation system considers multiple factors:

1. **Difficulty Progression**:
   - Easy: < 70% accuracy on easy questions
   - Medium: > 80% on easy OR < 60% on medium
   - Hard: > 70% on medium AND attempted hard questions

2. **Topic Selection**:
   - Prioritize topics with < 50% accuracy
   - Balance between weak topics and general knowledge
   - Avoid over-focusing on single topics

3. **Question Freshness**:
   - Exclude questions asked in recent sessions
   - Prioritize questions with low ask counts
   - Consider time since last asked

## Performance Tracking

The system tracks various metrics:

- **Overall Accuracy**: Percentage of correct answers
- **Topic Accuracy**: Performance by subject area  
- **Difficulty Performance**: Success rate by difficulty level
- **Response Times**: Average time to answer
- **Learning Trends**: Improvement over time

## Best Practices

1. **Session Management**: Always provide session IDs for better tracking
2. **User Identification**: Use consistent user IDs across sessions
3. **Topic Tagging**: Ensure questions have appropriate topic tags
4. **Performance Monitoring**: Regularly check user performance metrics
5. **Difficulty Adjustment**: Let the system manage difficulty progression

## Configuration

Memory retrieval can be configured through `RuntimeConfig`:

```python
config = RuntimeConfig(
    memory_retrieval_enabled=True,
    max_recent_questions=50,
    weak_topic_threshold=0.5,
    min_topic_attempts=3,
    recommendation_count=5
)
```

## Troubleshooting

### No Recommendations Available

- Ensure the user has sufficient question history
- Check that questions have proper topic tags
- Verify the GraphitiClient connection

### Poor Difficulty Calibration

- Review the user's performance history
- Check difficulty distribution of asked questions
- Consider manual difficulty override if needed

### Repetitive Questions

- Increase the `avoid_recent` parameter
- Ensure question IDs are properly stored
- Check for duplicate questions in the database

## Future Enhancements

1. **Machine Learning Integration**: Use ML models for better question selection
2. **Collaborative Filtering**: Recommend based on similar users
3. **Adaptive Difficulty**: Real-time difficulty adjustment
4. **Learning Path Generation**: Create structured learning sequences
5. **Performance Prediction**: Predict success likelihood for questions