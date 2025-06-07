# GraphitiClient Integration Guide

This guide explains how to use the GraphitiClient for managing temporal knowledge graph interactions in the question-graph agent.

## Overview

The GraphitiClient provides a high-level interface for:
- Storing questions, answers, and user interactions
- Managing temporal relationships between entities
- Creating episodes for Q&A sessions
- Retrieving user history and related questions
- Tracking user mastery of topics

## Architecture

### Components

1. **GraphitiClient**: Main client class for Graphiti interactions
2. **GraphitiSession**: Tracks session statistics and metadata
3. **QuestionState**: Enhanced with GraphitiClient instance
4. **Integration**: Automatic initialization in run modes

### State Management

The `QuestionState` now includes:
- `graphiti_client`: GraphitiClient instance (excluded from serialization)
- `current_user`: Current user entity (excluded from serialization)
- `session_id`: Session identifier for tracking

## Quick Start

### Basic Usage

```python
from graphiti_client import GraphitiClient
from graphiti_entities import QuestionEntity, UserEntity

# Create client
client = GraphitiClient()

# Connect to Graphiti
await client.connect()

# Store a question
question = QuestionEntity(
    id="q1",
    content="What is the capital of France?",
    difficulty=DifficultyLevel.EASY,
    topics=["geography", "europe"]
)
await client.store_question(question)

# Get session statistics
stats = client.get_session_stats()
print(f"Entities created: {stats['entity_count']}")
```

### Integration with Question Graph

```python
from question_graph import initialize_graphiti_state

# Initialize state with Graphiti
state = await initialize_graphiti_state(
    user_id="user123",
    enable_graphiti=True
)

# Run graph with Graphiti-enabled state
await question_graph.run(Ask(), state=state)
```

## Client Methods

### Connection Management

```python
# Connect to Graphiti service
success = await client.connect()

# Disconnect when done
await client.disconnect()

# Use context manager
async with client.session_context() as session:
    # Operations here
    pass
```

### Entity Storage

```python
# Store question
await client.store_question(question)

# Store answer with relationships
await client.store_answer(
    answer=answer_entity,
    question=question_entity,
    user=user_entity
)

# Create Q&A episode
await client.create_qa_episode(
    question=question,
    answer=answer,
    user=user,
    evaluation_correct=True
)
```

### Data Retrieval

```python
# Get user's question history
history = await client.get_user_history(
    user_id="user123",
    limit=10
)

# Get related questions by topic
questions = await client.get_related_questions(
    topic="mathematics",
    difficulty="MEDIUM",
    limit=5
)
```

### User Progress Tracking

```python
# Update user mastery after answering
await client.update_user_mastery(
    user=user_entity,
    topic=topic_entity,
    correct=True,
    time_taken=15.5
)
```

## State Initialization

### Automatic Initialization

The question graph automatically initializes Graphiti when running:

```python
# Continuous mode
await run_as_continuous()  # Graphiti enabled by default

# CLI mode
await run_as_cli(answer)  # Graphiti enabled by default
```

### Manual Initialization

```python
# Create new state with Graphiti
state = await initialize_graphiti_state(
    user_id="custom_user",
    session_id="custom_session",
    enable_graphiti=True
)

# Enhance existing state
existing_state = QuestionState()
state = await initialize_graphiti_state(
    state=existing_state,
    enable_graphiti=True
)

# Disable Graphiti
state = await initialize_graphiti_state(
    enable_graphiti=False
)
```

## Session Management

### Session Tracking

Each GraphitiClient maintains session statistics:

```python
stats = client.get_session_stats()
{
    "session_id": "uuid...",
    "user_id": "user123",
    "duration_seconds": 300,
    "episode_count": 5,
    "entity_count": 10,
    "relationship_count": 8,
    "fallback_active": false,
    "circuit_state": "closed"
}
```

### Session Context

Use the session context for automatic cleanup:

```python
async with client.session_context() as session:
    # Store entities
    await session.store_question(question)
    
    # Create episodes
    await session.create_qa_episode(...)
    
    # Stats available throughout session
    print(session.get_session_stats())
# Automatic disconnect on exit
```

## Error Handling

### Fallback Support

The client automatically uses fallback when Graphiti is unavailable:

```python
# With fallback enabled (default)
client = GraphitiClient(enable_fallback=True)

# Without fallback
client = GraphitiClient(enable_fallback=False)
```

### Circuit Breaker

Circuit breaker prevents cascading failures:

```python
# With circuit breaker (default)
client = GraphitiClient(enable_circuit_breaker=True)

# Check circuit state
stats = client.get_session_stats()
print(f"Circuit state: {stats['circuit_state']}")
```

### Health Checks

The client checks system health before connecting:

```python
if await client.connect():
    print("Connected successfully")
else:
    print("Connection failed - check fallback status")
    if client.fallback_manager.state.is_active:
        print(f"Fallback mode: {client.fallback_manager.state.mode}")
```

## Configuration

### Environment Variables

```env
# Graphiti configuration
GRAPHITI_ENDPOINT=http://localhost:8000
GRAPHITI_API_KEY=your-api-key
GRAPHITI_LLM_MODEL=gpt-4
GRAPHITI_EMBEDDER_MODEL=text-embedding-ada-002

# Neo4j configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### Custom Configuration

```python
from graphiti_config import RuntimeConfig

config = RuntimeConfig()
config.graphiti.llm_model = "gpt-4-turbo"

client = GraphitiClient(config=config)
```

## Integration Examples

### Node Integration

Nodes can access GraphitiClient through the state:

```python
class CustomNode(BaseNode[QuestionState]):
    async def run(self, ctx: GraphRunContext[QuestionState]):
        if ctx.state.graphiti_client:
            # Store entity
            await ctx.state.graphiti_client.store_question(...)
            
            # Get user history
            history = await ctx.state.graphiti_client.get_user_history(
                ctx.state.current_user.id
            )
```

### Custom Episodes

Create custom episodes for special interactions:

```python
from graphiti_registry import EpisodeBuilder

builder = EpisodeBuilder()

# Build custom episode
episode = builder.build_custom_episode(
    name="hint_requested",
    entities=[question, user],
    relationships=[],
    metadata={"hint_type": "partial"}
)

await client._graphiti.add_episode(episode)
```

### Batch Operations

Process multiple entities efficiently:

```python
questions = [
    QuestionEntity(...),
    QuestionEntity(...),
    # ...
]

# Store in batch
for question in questions:
    await client.store_question(question)

# Session stats show total
stats = client.get_session_stats()
print(f"Total entities: {stats['entity_count']}")
```

## Best Practices

### 1. Session Management

Always use session context for proper cleanup:

```python
# Good
async with client.session_context() as session:
    await session.store_question(question)

# Avoid
client = GraphitiClient()
await client.connect()
await client.store_question(question)
# May forget to disconnect
```

### 2. Error Handling

Check connection status and handle failures:

```python
if not await client.connect():
    logger.warning("Using fallback mode")
    # Continue with degraded functionality
```

### 3. Entity IDs

Use consistent ID generation:

```python
import uuid

question = QuestionEntity(
    id=f"q_{uuid.uuid4().hex[:8]}",
    content="..."
)
```

### 4. Relationship Management

Always create relationships when storing related entities:

```python
# Store answer with relationship
await client.store_answer(
    answer=answer,
    question=question,
    user=user  # Creates ANSWERED relationship
)
```

## Troubleshooting

### Connection Issues

```python
# Check system health
from graphiti_health import check_system_health

health = await check_system_health()
if health['components']['graphiti']['status'] != 'healthy':
    print("Graphiti service issues detected")
```

### Missing Entities

```python
# Verify entity was stored
if client.fallback_manager.state.is_active:
    print("Entity may be in fallback storage")
    entity = client.fallback_manager.get_entity(entity_id)
```

### Performance

```python
# Monitor session statistics
stats = client.get_session_stats()
print(f"Duration: {stats['duration_seconds']}s")
print(f"Entities/sec: {stats['entity_count'] / stats['duration_seconds']}")
```

## Future Enhancements

### Planned Features

1. **Bulk Operations**: Batch entity storage
2. **Query Builder**: Fluent API for complex queries
3. **Caching Layer**: Local cache for frequently accessed entities
4. **Analytics**: Advanced session analytics
5. **Export/Import**: Session data portability

### Integration Points

- Memory retrieval for question generation
- Adaptive difficulty based on mastery
- Session summaries and insights
- Learning path recommendations

## Summary

The GraphitiClient integration enhances the question-graph agent with:
- Persistent storage of Q&A interactions
- Temporal relationship tracking
- User progress monitoring
- Session management
- Automatic fallback handling

Use it to build intelligent, memory-aware Q&A systems that adapt to user performance over time.