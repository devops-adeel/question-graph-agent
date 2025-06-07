# Contributing to Question Graph Agent

Thank you for your interest in contributing to the Question Graph Agent project! This guide will help you get started with contributing to our Graphiti-integrated question-answering system.

## 🏗️ Project Structure

This project follows a 10-phase implementation plan for integrating Graphiti temporal knowledge graphs. Before contributing, familiarize yourself with:

- **CLAUDE.md**: AI development guidelines and implementation roadmap
- **Phase Structure**: Each phase has specific objectives and sub-tasks
- **Issue Tracking**: GitHub issues track all sub-tasks with clear numbering (e.g., 4.1, 4.2)

## 🚀 Getting Started

### Prerequisites

1. Python 3.10 or higher
2. Neo4j database (for Graphiti backend)
3. Git for version control
4. GitHub account

### Development Setup

```bash
# Clone the repository
git clone https://github.com/devops-adeel/question-graph-agent.git
cd question-graph-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (when available)
pre-commit install

# Copy environment variables
cp .env.example .env
# Edit .env with your configuration
```

## 📝 Development Workflow

### 1. Find or Create an Issue

- Check the [project board](https://github.com/devops-adeel/question-graph-agent/projects) for open tasks
- Look for issues labeled `good first issue` or `help wanted`
- If creating a new issue, use the appropriate template

### 2. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feat/phase4-connection-manager
# Or for bugs: fix/connection-timeout
# Or for docs: docs/update-readme
```

### Branch Naming Convention

- `feat/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### 3. Make Your Changes

Follow these coding standards:

#### Python Style

- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100)
- Use [isort](https://pydantic.dev/isort) for import sorting
- Follow [PEP 8](https://pep8.org/) guidelines
- Add type hints to all functions

```python
def extract_entities(text: str, config: ExtractionConfig) -> List[Entity]:
    """Extract entities from text using configured extractors.
    
    Args:
        text: Input text to process
        config: Extraction configuration
        
    Returns:
        List of extracted entities
        
    Raises:
        ExtractionError: If extraction fails
    """
    # Implementation
```

#### Pydantic Models

- Use descriptive field names
- Add field descriptions and examples
- Implement validators for complex logic

```python
class QuestionEntity(BaseEntity):
    """Question entity with validation and metadata."""
    
    content: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question text",
        example="What is the capital of France?"
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Question difficulty level"
    )
    
    @validator("content")
    def validate_question_format(cls, v: str) -> str:
        """Ensure question ends with question mark."""
        if not v.strip().endswith("?"):
            v = v.strip() + "?"
        return v
```

### 4. Write Tests

- Aim for >90% test coverage
- Write unit tests for all new functionality
- Include edge cases and error scenarios
- Use pytest fixtures for common test data

```python
class TestConnectionManager:
    """Test Neo4j connection manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        # Fixture implementation
    
    def test_connection_retry(self, mock_config):
        """Test connection retry on failure."""
        # Test implementation
```

### 5. Update Documentation

- Update docstrings for modified functions
- Update README.md if adding new features
- Update CLAUDE.md if changing implementation details
- Add examples for new functionality

### 6. Run Quality Checks

```bash
# Format code
black .
isort .

# Run type checking
mypy .

# Run linters
flake8
ruff check .

# Run tests
pytest -v --cov=. --cov-report=term-missing

# Check documentation
# mkdocs serve (when configured)
```

### 7. Commit Your Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Examples
git commit -m "feat: add Neo4j connection manager with retry logic"
git commit -m "fix: handle connection timeout in GraphitiClient"
git commit -m "docs: update configuration guide for Neo4j settings"
git commit -m "test: add integration tests for entity extraction"
```

### 8. Create Pull Request

1. Push your branch: `git push origin feat/your-feature`
2. Open a PR using the template
3. Link related issues
4. Wait for review

## 🎯 Contribution Guidelines

### What We're Looking For

- **Phase Implementation**: Help complete sub-tasks in the implementation phases
- **Bug Fixes**: Identify and fix issues
- **Documentation**: Improve guides and examples
- **Tests**: Increase test coverage
- **Performance**: Optimize existing code
- **Examples**: Create usage examples

### Code Review Process

1. **Automated Checks**: CI runs tests, linting, and type checking
2. **Peer Review**: At least one maintainer reviews the code
3. **Feedback**: Address review comments
4. **Approval**: PR is approved and merged

### Review Criteria

- Code follows project style guidelines
- Tests are comprehensive and passing
- Documentation is updated
- No breaking changes (or properly documented)
- Performance impact is considered
- Security implications are addressed

## 🐛 Reporting Bugs

Use the bug report template and include:

1. Clear description of the issue
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details
5. Error messages/stack traces

## 💡 Suggesting Features

Use the feature enhancement template and include:

1. Problem statement
2. Proposed solution
3. Implementation approach
4. Potential impact
5. Alternative approaches considered

## 🤝 Community

- Be respectful and inclusive
- Help others in issues and discussions
- Share your use cases and experiences
- Provide constructive feedback

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Tests pass locally
- [ ] Code is formatted and linted
- [ ] Documentation is updated
- [ ] PR template is filled out completely
- [ ] Related issues are linked
- [ ] Branch is up to date with main

## 🏆 Recognition

Contributors are recognized in:

- GitHub contributors page
- Release notes
- Project documentation

## 📫 Getting Help

- Check existing [issues](https://github.com/devops-adeel/question-graph-agent/issues)
- Start a [discussion](https://github.com/devops-adeel/question-graph-agent/discussions)
- Review the documentation
- Read CLAUDE.md for implementation details

## 🔐 Security

- Never commit sensitive information
- Report security issues privately to maintainers
- Follow secure coding practices
- Validate all user inputs

Thank you for contributing to Question Graph Agent! Your efforts help build a better AI-powered question-answering system with persistent memory capabilities.