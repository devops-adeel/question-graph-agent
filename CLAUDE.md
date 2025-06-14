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

## Project Guidelines and Standards

### Development Workflow

1. **Issue Creation**: Use appropriate GitHub issue templates
   - Feature enhancements for new functionality
   - Phase sub-tasks for implementation work
   - Bug reports for issues
   - Testing tasks for coverage improvements
   - Documentation updates

2. **Branch Strategy**
   - Branch from latest `main`
   - Use descriptive branch names:
     - `feat/description` for features
     - `fix/description` for bugs
     - `docs/description` for documentation
     - `refactor/description` for refactoring
     - `test/description` for tests

3. **Commit Standards**: Follow Conventional Commits
   ```bash
   feat: add Neo4j connection manager
   fix: handle connection timeout error
   docs: update configuration guide
   test: add entity extraction tests
   ```

4. **Pull Request Process**
   - Use PR template with all sections filled
   - Link related issues with "Closes #XX"
   - Include comprehensive test coverage
   - Update documentation as needed
   - Ensure all quality checks pass

### Code Quality Standards

1. **Formatting and Linting**
   ```bash
   black .                 # Format code (line-length: 100)
   isort .                 # Sort imports
   mypy .                  # Type checking
   ruff check .            # Fast linting
   flake8                  # Additional linting
   pytest -v --cov=.       # Run tests with coverage
   ```

2. **Type Safety**
   - Use type hints for all functions
   - Leverage Pydantic models for validation
   - Define custom validators when needed

3. **Documentation**
   - Comprehensive docstrings for all public APIs
   - Include usage examples in docstrings
   - Update CLAUDE.md for architectural changes
   - Maintain README.md accuracy

4. **Testing Requirements**
   - Target >90% test coverage
   - Include unit and integration tests
   - Test edge cases and error scenarios
   - Use pytest fixtures for test data

### Project Structure

```
question-graph-agent/
├── Core Application
│   └── question_graph.py          # Main application
├── Graphiti Integration
│   ├── graphiti_entities.py       # Entity models
│   ├── graphiti_relationships.py  # Relationship models
│   ├── entity_extraction.py       # Extraction logic
│   ├── extraction_pipeline.py     # Batch processing
│   ├── extraction_errors.py       # Error handling
│   ├── graphiti_config.py         # Configuration
│   └── graphiti_connection.py     # Connection management
├── Configuration
│   ├── .env.example              # Environment template
│   ├── pyproject.toml            # Project configuration
│   └── requirements*.txt         # Dependencies
├── Documentation
│   ├── README.md                 # Project overview
│   ├── CLAUDE.md                 # This file
│   ├── CONTRIBUTING.md           # Contribution guide
│   └── docs/                     # Additional docs
├── Tests
│   └── tests/                    # Test suite
└── GitHub
    └── .github/                  # Templates and workflows
```

### Integration Points

1. **Neo4j Database**
   - Connection via `graphiti_connection.py`
   - Configuration in environment variables
   - Retry logic for resilience

2. **Graphiti Framework**
   - Entity and relationship persistence
   - Temporal knowledge tracking
   - Query capabilities

3. **AI Services**
   - OpenAI for question generation
   - Groq for structured outputs
   - Pydantic AI for agent framework

### Best Practices

1. **Error Handling**
   - Use custom exceptions for clarity
   - Implement retry logic for external services
   - Provide meaningful error messages
   - Include fallback strategies

2. **Performance**
   - Use async operations where beneficial
   - Implement connection pooling
   - Add caching for expensive operations
   - Monitor performance metrics

3. **Security**
   - Never commit sensitive data
   - Use environment variables for secrets
   - Validate all user inputs
   - Follow secure coding practices

4. **Maintenance**
   - Keep dependencies updated
   - Document breaking changes
   - Maintain backwards compatibility
   - Use semantic versioning

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

#### Phase 4: Graphiti Infrastructure ⏳ IN PROGRESS
- [x] 4.1: Add Graphiti dependencies to project (create pyproject.toml)
- [x] 4.2: Create environment configuration for Neo4j connection
- [x] 4.3: Implement connection manager with retry logic
- [ ] 4.4: Register custom entity types with Graphiti
- [ ] 4.5: Create database initialization scripts
- [ ] 4.6: Add connection health checks and monitoring
- [ ] 4.7: Implement graceful fallback for when Graphiti is unavailable

**Status**: Infrastructure foundation completed with configuration management and connection handling. Next: entity registration and database setup.

#### Phase 5: Memory Integration ⏳ PENDING
- [ ] 5.1: Modify QuestionState to include GraphitiClient instance
- [ ] 5.2: Create memory storage functions for Q&A pairs
- [ ] 5.3: Implement memory retrieval for question generation
- [ ] 5.4: Add memory updates after answer evaluation
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

#### Phase 4 Modules (In Progress)

6. **Configuration Management** (`graphiti_config.py`)
   - Neo4jConfig with connection settings and validation
   - GraphitiConfig with service configuration
   - ApplicationConfig with feature flags and paths
   - RuntimeConfig combining all configurations

7. **Connection Management** (`graphiti_connection.py`)
   - Neo4jConnectionManager with retry logic
   - GraphitiConnectionManager for HTTP services
   - ConnectionPool for efficient resource usage
   - Metrics tracking and state monitoring

8. **Project Dependencies** (`pyproject.toml`)
   - Modern Python packaging configuration
   - Development tool settings (black, mypy, pytest)
   - Optional dependency groups (dev, nlp, monitoring)

### GitHub Integration

#### Issue Templates
The project uses specialized GitHub issue templates for:
- Feature enhancements with phase integration
- Phase sub-tasks following numbered convention
- Bug reports with reproduction steps
- Testing tasks for coverage improvements
- Documentation updates

#### Pull Request Template
Comprehensive PR template ensuring:
- Clear summary and issue linking
- Testing and documentation updates
- Breaking change identification
- Performance and security considerations

### Next Steps
Continue Phase 4 by implementing entity registration with Graphiti and creating database initialization scripts.