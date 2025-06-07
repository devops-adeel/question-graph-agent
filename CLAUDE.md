# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This repository contains a single-file question-graph agent built with pydantic_graph and pydantic_ai. The application creates an interactive Q&A flow where an AI agent asks questions, users provide answers, and another AI agent evaluates the correctness of the responses.

### Core Components

- **QuestionState**: Dataclass that maintains the conversation state including the current question and message histories for both agents
- **Graph Nodes**: Four node types that form the question-answer cycle:
  - `Ask`: Generates new questions using OpenAI GPT-4o
  - `Answer`: Prompts user for input and captures their response
  - `Evaluate`: Evaluates answer correctness and provides feedback
  - `Reprimand`: Handles incorrect answers and loops back to Ask
- **State Persistence**: Uses FileStatePersistence to save/restore graph state to `question_graph.json`

### Key Dependencies

- `pydantic_graph`: Graph execution framework with state management
- `pydantic_ai`: AI agent framework with OpenAI integration
- `logfire`: Observability and logging (optional - uses 'if-token-present' mode)
- `groq`: BaseModel for structured outputs

## Common Commands

### Running the Application

```bash
# Generate a Mermaid diagram of the graph structure
uv run -m question_graph mermaid

# Run in continuous mode (single session)
uv run -m question_graph continuous

# Run in CLI mode (stateful with persistence)
uv run -m question_graph cli [answer]
```

### Development

The application requires Python with `uv` package manager and OpenAI API access. State is persisted in `question_graph.json` for the CLI mode, allowing sessions to be resumed.

## Graphiti Integration Implementation Plan

### Overview
The project is being enhanced with Graphiti integration to add persistent, graph-based memory capabilities. This enables adaptive learning, personalized question generation, and cross-session intelligence.

### Implementation Status

#### Phase 1: Core Entity Models ✅ COMPLETED
- [x] 1.1: Create base entity model with common fields (id, created_at, updated_at)
- [x] 1.2: Define QuestionEntity with content, difficulty enum, and topic list
- [x] 1.3: Define AnswerEntity with content, status enum, and response metrics
- [x] 1.4: Define UserEntity with session tracking and performance metrics
- [x] 1.5: Define TopicEntity with hierarchy support and complexity scoring
- [x] 1.6: Add custom validators for each entity (content cleaning, score bounds)
- [x] 1.7: Create entity factory functions for easy instantiation

**Status**: All entity models created in `graphiti_entities.py` with comprehensive validation and factory functions.

#### Phase 2: Relationship Models ✅ COMPLETED
- [x] 2.1: Create base relationship model with timestamp and metadata fields
- [x] 2.2: Define AnsweredRelationship linking users to questions via answers
- [x] 2.3: Define RequiresKnowledgeRelationship for question-topic connections
- [x] 2.4: Define MasteryRelationship for user-topic proficiency tracking
- [x] 2.5: Add relationship validation rules (e.g., score ranges, ID formats)
- [x] 2.6: Create relationship builder utilities for common patterns

**Status**: All relationship models completed in `graphiti_relationships.py` with temporal support, validation rules, and builder utilities.

#### Phase 3: Entity Extraction ✅ COMPLETED
- [x] 3.1: Create basic EntityExtractor class with simple keyword-based topic extraction
- [x] 3.2: Implement difficulty estimation based on question complexity metrics
- [x] 3.3: Add answer classification logic (correct/incorrect/partial)
- [x] 3.4: Create extraction pipelines for batch processing
- [x] 3.5: Add async NLP integration points for future enhancement
- [x] 3.6: Implement extraction error handling and fallback strategies

**Status**: Entity extraction module completed with:
- `entity_extraction.py`: Core extraction logic with keyword-based algorithms
- `extraction_pipeline.py`: Batch processing with async support
- `extraction_errors.py`: Comprehensive error handling framework

#### Phase 4: Graphiti Infrastructure ⏳ PENDING
- [ ] 4.1: Add Graphiti dependencies to project (create pyproject.toml)
- [ ] 4.2: Create environment configuration for Neo4j connection
- [ ] 4.3: Implement connection manager with retry logic
- [ ] 4.4: Register custom entity types with Graphiti
- [ ] 4.5: Create database initialization scripts
- [ ] 4.6: Add connection health checks and monitoring
- [ ] 4.7: Implement graceful fallback for when Graphiti is unavailable

#### Phase 5: Episode Building ⏳ PENDING
- [ ] 5.1: Define episode schema for Q&A interactions
- [ ] 5.2: Create narrative generator for episode descriptions
- [ ] 5.3: Implement context extraction from QuestionState
- [ ] 5.4: Add importance scoring for episodes
- [ ] 5.5: Create batch episode builder for session summaries
- [ ] 5.6: Implement episode validation and sanitization

#### Phase 6: Temporal Tracking ⏳ PENDING
- [ ] 6.1: Define temporal decay functions for knowledge retention
- [ ] 6.2: Implement mastery score calculations with time weighting
- [ ] 6.3: Create session-based performance tracking
- [ ] 6.4: Add temporal relationship versioning
- [ ] 6.5: Implement knowledge gap detection over time
- [ ] 6.6: Create temporal query utilities

#### Phase 7: Validation & Error Handling ⏳ PENDING
- [ ] 7.1: Create custom validation exceptions for entity creation
- [ ] 7.2: Implement validation middleware for all Graphiti operations
- [ ] 7.3: Add input sanitization for user-provided content
- [ ] 7.4: Create validation reports and logging
- [ ] 7.5: Implement transaction rollback for failed operations
- [ ] 7.6: Add validation performance monitoring

#### Phase 8: Query Models ⏳ PENDING
- [ ] 8.1: Define query result models (UserProgress, TopicMastery, etc.)
- [ ] 8.2: Create query builder classes for common patterns
- [ ] 8.3: Implement paginated query results
- [ ] 8.4: Add query caching layer
- [ ] 8.5: Create aggregate query functions (averages, trends)
- [ ] 8.6: Implement query performance optimization

#### Phase 9: Migration Utilities ⏳ PENDING
- [ ] 9.1: Create schema versioning system
- [ ] 9.2: Implement entity migration functions
- [ ] 9.3: Add relationship migration support
- [ ] 9.4: Create rollback mechanisms
- [ ] 9.5: Build migration testing framework
- [ ] 9.6: Add migration documentation generator

#### Phase 10: Testing Infrastructure ⏳ PENDING
- [ ] 10.1: Set up pytest infrastructure with async support
- [ ] 10.2: Create entity model unit tests with validation edge cases
- [ ] 10.3: Add relationship model tests
- [ ] 10.4: Implement extraction pipeline tests with fixtures
- [ ] 10.5: Create Graphiti integration tests with mocked Neo4j
- [ ] 10.6: Add end-to-end flow tests
- [ ] 10.7: Implement performance benchmarks
- [ ] 10.8: Create test data generators

### Key Design Decisions

1. **Pydantic-First Approach**: All entities and relationships use Pydantic models for validation and type safety
2. **Temporal Awareness**: Built-in support for Graphiti's bi-temporal model
3. **Privacy by Design**: Opt-in memory features with clear data control
4. **Incremental Integration**: Can be enabled/disabled without breaking existing functionality
5. **Performance Conscious**: Async operations, caching, and non-blocking design

### Technical Architecture

#### Completed Modules

1. **Entity Models** (`graphiti_entities.py`)
   - BaseEntity with temporal tracking
   - QuestionEntity with difficulty levels and topics
   - AnswerEntity with evaluation status
   - UserEntity with performance metrics
   - TopicEntity with hierarchy support

2. **Relationship Models** (`graphiti_relationships.py`)
   - BaseRelationship with temporal validity
   - AnsweredRelationship (User → Question)
   - RequiresKnowledgeRelationship (Question → Topic)
   - MasteryRelationship (User → Topic) with forgetting curves
   - RelationshipValidationRules and RelationshipBuilder utilities

3. **Entity Extraction** (`entity_extraction.py`)
   - EntityExtractor with keyword-based topic extraction
   - ComplexityMetrics for difficulty estimation
   - AnswerClassifier with multiple similarity algorithms
   - AsyncEntityExtractor with NLP placeholders

4. **Batch Processing** (`extraction_pipeline.py`)
   - ExtractionPipeline with async/sync support
   - BatchExtractionResult with aggregated statistics
   - ExtractionProgress for real-time monitoring
   - RobustExtractionPipeline with error handling

5. **Error Handling** (`extraction_errors.py`)
   - ExtractionError models with structured error info
   - RobustEntityExtractor with retry logic and circuit breaker
   - ErrorHandlingConfig for customizable behavior
   - Fallback strategies and recovery mechanisms

### Next Steps
Begin Phase 4 (Graphiti Infrastructure) to integrate the completed entity and relationship models with the actual Graphiti framework for persistent graph storage.