# Memory Retrieval Implementation

This sub-task implements memory retrieval functionality for intelligent question generation in the question-graph agent.

## Overview

The memory retrieval system enables the agent to:
- Access previously asked questions to avoid repetition
- Analyze user performance to adjust difficulty
- Identify weak topics that need more practice
- Generate personalized question recommendations
- Track learning progress over time

## Implementation Components

### 1. Memory Retrieval (`memory_retrieval.py`)

**MemoryRetrieval Class**
- `get_asked_questions()`: Retrieves questions previously asked to a user
- `get_user_performance()`: Gets comprehensive performance metrics
- `get_topic_questions()`: Finds questions for specific topics
- `get_weak_topics()`: Identifies topics where the user struggles
- `get_recommended_questions()`: Generates personalized recommendations
- `get_question_context()`: Retrieves context about a specific question

**QuestionSelector Class**
- `select_next_question()`: Intelligently selects the next question
- `_apply_context_filter()`: Applies contextual filtering to questions

### 2. Enhanced Question Agent (`enhanced_question_agent.py`)

**EnhancedQuestionAgent Class**
- Integrates memory retrieval with question generation
- Uses context to generate appropriate questions
- Falls back to standard generation if memory unavailable

**QuestionGenerationContext**
- Configurable context for question generation
- Supports difficulty override and topic preferences

### 3. Memory Integration (`memory_integration.py`)

**MemoryEnhancedAsk Node**
- Enhanced version of Ask node with memory capabilities
- Automatically adjusts difficulty based on performance
- Stores generated questions in memory

**MemoryContextBuilder**
- Builds rich context for question generation
- Aggregates user performance and topic data

**MemoryStateEnhancer**
- Enhances QuestionState with memory-related fields
- Adds user tracking and session management

## Key Features

### Personalized Difficulty Adjustment
```python
# Automatic difficulty recommendation based on performance
if accuracies["easy"] < 0.7:
    return DifficultyLevel.EASY
elif accuracies["medium"] < 0.6 or accuracies["easy"] > 0.8:
    return DifficultyLevel.MEDIUM
elif accuracies["medium"] > 0.7 and accuracies["hard"] < 0.5:
    return DifficultyLevel.HARD
```

### Weak Topic Identification
```python
# Find topics where user accuracy is below threshold
weak_topics = await retrieval.get_weak_topics(
    user_id="user1",
    threshold=0.5,
    min_attempts=3
)
```

### Smart Question Selection
```python
# Combines multiple factors for recommendation:
# 1. User performance history
# 2. Weak topics needing practice
# 3. Appropriate difficulty level
# 4. Avoiding recent questions
recommendations = await retrieval.get_recommended_questions(
    user_id="user1",
    session_id="session1",
    count=5
)
```

## Usage Example

```python
from graphiti_client import GraphitiClient
from memory_retrieval import MemoryRetrieval
from enhanced_question_agent import EnhancedQuestionAgent, QuestionGenerationContext

# Initialize
client = GraphitiClient(...)
agent = EnhancedQuestionAgent(graphiti_client=client)

# Create context
context = QuestionGenerationContext(
    user_id="user123",
    session_id="session456",
    prefer_weak_topics=True
)

# Generate personalized question
question = await agent.generate_question(context)
```

## Testing

Comprehensive test coverage includes:
- Memory retrieval operations
- Question selection logic
- Enhanced agent functionality
- Integration with graph nodes
- Mock-based testing for Graphiti interactions

## Documentation

- **User Guide**: `docs/memory_retrieval.md`
- **API Reference**: Inline documentation in source files
- **Examples**: `examples/memory_retrieval_example.py`

## Integration Points

1. **GraphitiClient**: Provides access to the knowledge graph
2. **QuestionState**: Enhanced with memory-related fields
3. **Graph Nodes**: Memory-enhanced versions of standard nodes
4. **Entity Extraction**: Automatic topic identification

## Future Enhancements

1. Machine learning for better question selection
2. Collaborative filtering based on similar users
3. Real-time difficulty adjustment during sessions
4. Structured learning path generation
5. Performance prediction models