# Entity Registration Guide

This guide explains how to register custom entities with Graphiti for the question-graph agent.

## Overview

The entity registration module (`graphiti_registry.py`) provides a bridge between our Pydantic entity models and Graphiti's knowledge graph system. It handles:

- Entity type mapping
- Entity conversion and adaptation
- Episode creation from interactions
- Batch registration operations

## Components

### EntityTypeRegistry

Maps our custom entity types to Graphiti's entity types:

```python
QUESTION → EntityType.generic
ANSWER → EntityType.generic  
USER → EntityType.person
TOPIC → EntityType.topic
```

### EntityAdapter

Converts Pydantic entities to Graphiti entities:

```python
from graphiti_registry import EntityAdapter, EntityTypeRegistry
from graphiti_entities import QuestionEntity, DifficultyLevel

# Create a question entity
question = QuestionEntity(
    content="What is machine learning?",
    difficulty=DifficultyLevel.MEDIUM,
    topics=["AI", "Computer Science"]
)

# Convert to Graphiti entity
graphiti_entity = EntityAdapter.to_graphiti_entity(
    question,
    EntityTypeRegistry.QUESTION
)
```

### RelationshipAdapter

Converts relationships to facts for Graphiti episodes:

```python
from graphiti_registry import RelationshipAdapter
from graphiti_relationships import MasteryRelationship

# Create mastery relationship
mastery = MasteryRelationship(
    source_id="user123",
    target_id="AI",
    mastery_score=0.75,
    learning_rate=0.1
)

# Convert to facts
facts = RelationshipAdapter.to_graphiti_facts(mastery)
# Results in:
# ["user123 has mastery of AI",
#  "Mastery score: 0.75",
#  "Learning rate: 0.10", ...]
```

### EpisodeBuilder

Creates Graphiti episodes from Q&A interactions:

```python
from graphiti_registry import EpisodeBuilder

# Build Q&A episode
episode_data = EpisodeBuilder.build_qa_episode(
    question=question_entity,
    answer=answer_entity,
    user=user_entity,
    evaluation={"correct": True, "feedback": "Well done!"}
)

# Build session summary
summary = EpisodeBuilder.build_session_summary_episode(
    user=user_entity,
    session_stats={
        "total_questions": 10,
        "correct_answers": 8,
        "success_rate": 0.8
    },
    topics_covered=["AI", "ML", "Data Science"]
)
```

### EntityRegistrar

Main class for registering entities with Graphiti:

```python
from graphiti_registry import EntityRegistrar
from graphiti_core import Graphiti

# Initialize Graphiti client
client = Graphiti(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# Create registrar
registrar = EntityRegistrar(client)

# Register entity types
await registrar.register_entity_types()

# Upsert entities
await registrar.upsert_entity(question, "question")
await registrar.upsert_entity(user, "user")

# Create episode
await registrar.upsert_episode(episode_data)
```

## Usage Patterns

### Basic Entity Registration

```python
async def register_question(registrar, question):
    """Register a question entity."""
    # Convert and register
    entity_id = await registrar.upsert_entity(
        question,
        EntityTypeRegistry.QUESTION
    )
    
    if entity_id:
        logger.info(f"Registered question: {entity_id}")
    else:
        logger.error("Failed to register question")
```

### Batch Registration

```python
async def register_batch(registrar, entities):
    """Register multiple entities."""
    entity_list = [
        (question1, "question"),
        (question2, "question"),
        (user1, "user"),
        (topic1, "topic")
    ]
    
    results = await registrar.batch_upsert_entities(entity_list)
    
    success_count = sum(1 for success in results.values() if success)
    logger.info(f"Registered {success_count}/{len(entity_list)} entities")
```

### Complete Q&A Flow

```python
async def process_qa_interaction(registrar, question, answer, user):
    """Process complete Q&A interaction."""
    # 1. Register entities
    await registrar.upsert_entity(question, "question")
    await registrar.upsert_entity(answer, "answer")
    await registrar.upsert_entity(user, "user")
    
    # 2. Evaluate answer
    evaluation = evaluate_answer(question, answer)
    
    # 3. Create Q&A episode
    episode_data = EpisodeBuilder.build_qa_episode(
        question, answer, user, evaluation
    )
    
    # 4. Register episode
    episode_id = await registrar.upsert_episode(episode_data)
    
    # 5. Update user mastery (if correct)
    if evaluation["correct"]:
        for topic in question.topics:
            mastery = calculate_new_mastery(user, topic)
            # Create mastery relationship...
    
    return episode_id
```

## Entity Facts Generation

The adapter generates facts (observations) for each entity type:

### Question Facts
- Content: The question text
- Difficulty: Easy/Medium/Hard
- Topics: Comma-separated list
- Asked count: Number of times asked
- Success rate: Percentage of correct answers

### Answer Facts
- Response: The answer text
- Status: Correct/Incorrect/Partial
- Response time: Time in seconds
- Confidence: Score if available
- Feedback: Evaluation feedback

### User Facts
- Total questions answered
- Correct answers count
- Average response time
- Mastered topics list

### Topic Facts
- Topic name
- Complexity score
- Parent topic (if hierarchical)
- Prerequisites list

## Episode Types

### Q&A Interaction Episodes
- Type: `EpisodeType.interaction`
- Contains: Question, answer, user, and evaluation
- References: All involved entities
- Metadata: Performance metrics

### Session Summary Episodes
- Type: `EpisodeType.summary`
- Contains: Session statistics and progress
- References: User and covered topics
- Metadata: Aggregate performance data

## Caching

The registrar maintains an entity cache for performance:

```python
# Check if entity is cached
cached_entity = registrar.get_cached_entity(entity_id)

if cached_entity:
    # Use cached version
    logger.info("Using cached entity")
else:
    # Fetch from Graphiti
    entity = await client.get_entity(entity_id)
```

## Error Handling

The registrar handles errors gracefully:

```python
# Registration returns None on failure
entity_id = await registrar.upsert_entity(entity, entity_type)

if entity_id is None:
    # Handle registration failure
    logger.error(f"Failed to register {entity_type}")
    # Implement fallback logic
```

## Best Practices

1. **Batch Operations**: Use batch registration for multiple entities
2. **Episode Context**: Include rich context in episodes for better graph traversal
3. **Entity Updates**: Re-register entities when properties change significantly
4. **Error Recovery**: Implement retry logic for transient failures
5. **Cache Management**: Clear cache periodically for long-running processes

## Integration with Question Graph

```python
from pydantic_graph import GraphContext
from graphiti_registry import EntityRegistrar

class QuestionGraphWithMemory:
    def __init__(self, graphiti_client):
        self.registrar = EntityRegistrar(graphiti_client)
        
    async def on_question_asked(self, state: QuestionState):
        """Hook when question is asked."""
        question = create_question_entity(state.current_question)
        await self.registrar.upsert_entity(question, "question")
    
    async def on_answer_evaluated(self, state: QuestionState, evaluation):
        """Hook when answer is evaluated."""
        # Create episode for the interaction
        episode = EpisodeBuilder.build_qa_episode(
            question=state.question_entity,
            answer=state.answer_entity,
            user=state.user_entity,
            evaluation=evaluation
        )
        await self.registrar.upsert_episode(episode)
```

## Troubleshooting

### Entity Registration Fails
- Check Graphiti client connection
- Verify entity validation passes
- Check Neo4j is running

### Episodes Not Linking
- Ensure entity IDs match registered entities
- Verify entity_references format
- Check timestamp formatting

### Performance Issues
- Use batch operations
- Enable entity caching
- Consider async operations
- Monitor Neo4j performance