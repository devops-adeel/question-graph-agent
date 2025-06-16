# Question Graph Agent

An intelligent Q&A agent with graph-based memory using pydantic_graph and Graphiti for adaptive learning and personalized question generation.

## Features

- 🤖 AI-powered question generation and answer evaluation
- 🧠 Temporal knowledge graph for long-term memory
- 📊 Adaptive difficulty based on user performance
- 🔄 Session persistence and cross-session learning
- 📈 Performance tracking with forgetting curves
- 🛡️ Robust error handling and fallback strategies

## Architecture

The system integrates several key components:

1. **Question Graph Flow**: Built with `pydantic_graph` for stateful Q&A interactions
2. **Entity Extraction**: Keyword-based and NLP-ready extraction pipeline
3. **Knowledge Graph**: Graphiti integration for temporal relationship tracking
4. **Adaptive Learning**: Mastery tracking with spaced repetition principles

## Installation

### Prerequisites

- Python 3.10 or higher
- Neo4j database (for Graphiti backend)
- OpenAI API key

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/question-graph-agent.git
cd question-graph-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Using uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/yourusername/question-graph-agent.git
cd question-graph-agent

# Install dependencies
uv pip install -r requirements.txt

# For development
uv pip install -r requirements-dev.txt
```

### Using pyproject.toml

```bash
# Install in editable mode
pip install -e .

# With all optional dependencies
pip install -e ".[all]"
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:
```env
OPENAI_API_KEY=your-api-key
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your-password
```

3. Start Neo4j database:
```bash
# Using Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:latest
```

## Usage

### Generate Mermaid Diagram
```bash
uv run -m question_graph mermaid
```

### Run in Continuous Mode
```bash
uv run -m question_graph continuous
```

### Run in CLI Mode (with persistence)
```bash
uv run -m question_graph cli [answer]
```

## Project Structure

```
question-graph-agent/
├── question_graph.py          # Main application flow
├── graphiti_entities.py       # Entity models for knowledge graph
├── graphiti_relationships.py  # Relationship models with temporal tracking
├── entity_extraction.py       # Entity extraction and classification
├── extraction_pipeline.py     # Batch processing pipelines
├── extraction_errors.py       # Error handling framework
├── pyproject.toml            # Project configuration and dependencies
├── requirements.txt          # Core dependencies
├── requirements-dev.txt      # Development dependencies
└── .env.example             # Example environment configuration
```

## Development

### Running Tests

#### Local Testing
```bash
pytest
```

#### Containerized Testing with OrbStack
```bash
# Run tests in containers (requires OrbStack on macOS)
./run-tests-orbstack.sh

# Run specific test file
./run-tests-orbstack.sh tests/test_enhanced_nodes.py
```

For detailed OrbStack setup and performance benefits, see [OrbStack Testing Documentation](docs/orbstack-testing.md).

### Code Formatting
```bash
black .
isort .
```

### Type Checking
```bash
mypy .
```

### Linting
```bash
flake8
pylint question_graph
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [pydantic](https://pydantic-docs.helpmanual.io/) for data validation
- Uses [pydantic_graph](https://github.com/pydantic/pydantic-graph) for graph workflows
- Integrates [Graphiti](https://github.com/getzep/graphiti) for temporal knowledge graphs
- Powered by [OpenAI](https://openai.com/) for AI capabilities