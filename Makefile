.PHONY: help install install-dev test lint format type-check clean run-continuous run-cli docker-neo4j

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make test          Run tests with coverage"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black and isort"
	@echo "  make type-check    Run mypy type checking"
	@echo "  make clean         Clean up cache and build files"
	@echo "  make run-continuous Run in continuous mode"
	@echo "  make run-cli       Run in CLI mode"
	@echo "  make docker-neo4j  Start Neo4j in Docker"
	@echo "  make db-init       Initialize Neo4j database"
	@echo "  make db-status     Check database status"
	@echo "  make db-reset      Reset and reinitialize database"
	@echo "  make db-setup      Start Neo4j and initialize"

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest -v --cov=. --cov-report=term-missing --cov-report=html

test-watch:
	pytest-watch -- -v

# Code quality
lint:
	flake8 .
	pylint question_graph.py graphiti_*.py entity_extraction.py extraction_*.py
	ruff check .

format:
	black .
	isort .
	ruff check --fix .

type-check:
	mypy question_graph.py graphiti_entities.py graphiti_relationships.py entity_extraction.py

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/

# Running the application
run-continuous:
	python -m question_graph continuous

run-cli:
	python -m question_graph cli

run-mermaid:
	python -m question_graph mermaid

# Docker commands
docker-neo4j:
	docker run -d \
		--name neo4j-graphiti \
		-p 7474:7474 -p 7687:7687 \
		-e NEO4J_AUTH=neo4j/password123 \
		-e NEO4J_PLUGINS='["graph-data-science", "apoc"]' \
		-v neo4j-data:/data \
		neo4j:5-enterprise

docker-stop:
	docker stop neo4j-graphiti

docker-logs:
	docker logs -f neo4j-graphiti

# Database commands
db-init:
	python scripts/init_database.py init

db-reset:
	python scripts/init_database.py reset --confirm --reinit

db-status:
	python scripts/init_database.py status

db-setup: docker-neo4j
	@echo "Waiting for Neo4j to start..."
	@sleep 10
	python scripts/init_database.py init

# Development workflow
dev: install-dev format type-check test

# CI/CD simulation
ci: lint type-check test