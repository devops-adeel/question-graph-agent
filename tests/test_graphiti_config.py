"""
Tests for the Graphiti configuration module.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import SecretStr, ValidationError

from graphiti_config import (
    Environment,
    LogLevel,
    Neo4jConfig,
    GraphitiConfig,
    ApplicationConfig,
    RuntimeConfig,
    get_config,
    reload_config,
    is_graphiti_enabled,
    is_async_enabled,
    get_data_path,
    get_cache_path,
    get_log_path,
)


class TestNeo4jConfig:
    """Test Neo4j configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = Neo4jConfig(password="test123")
        assert config.uri == "bolt://localhost:7687"
        assert config.user == "neo4j"
        assert config.database == "neo4j"
        assert not config.encrypted
        assert config.max_connection_pool_size == 50
    
    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {
            "NEO4J_URI": "bolt://remote:7687",
            "NEO4J_USER": "admin",
            "NEO4J_PASSWORD": "secret123",
            "NEO4J_DATABASE": "graphiti",
        }):
            config = Neo4jConfig()
            assert config.uri == "bolt://remote:7687"
            assert config.user == "admin"
            assert config.password.get_secret_value() == "secret123"
            assert config.database == "graphiti"
    
    def test_uri_validation(self):
        """Test URI validation."""
        # Valid URIs
        valid_uris = [
            "bolt://localhost:7687",
            "bolt+s://secure.neo4j.com:7687",
            "neo4j://localhost:7687",
            "neo4j+ssc://secure.neo4j.com:7687",
        ]
        for uri in valid_uris:
            config = Neo4jConfig(uri=uri, password="test")
            assert config.uri == uri
        
        # Invalid URI
        with pytest.raises(ValidationError):
            Neo4jConfig(uri="http://localhost:7687", password="test")
    
    def test_password_required(self):
        """Test password is required."""
        with pytest.raises(ValidationError):
            Neo4jConfig()


class TestGraphitiConfig:
    """Test Graphiti configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GraphitiConfig()
        assert config.endpoint == "http://localhost:8000"
        assert config.enable_cache is True
        assert config.cache_ttl == 3600
        assert config.max_retries == 3
        assert config.api_key is None
    
    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {
            "GRAPHITI_ENDPOINT": "https://api.graphiti.com",
            "GRAPHITI_API_KEY": "secret-key-123",
            "GRAPHITI_ENABLE_CACHE": "false",
            "GRAPHITI_MAX_RETRIES": "5",
        }):
            config = GraphitiConfig()
            assert config.endpoint == "https://api.graphiti.com"
            assert config.api_key.get_secret_value() == "secret-key-123"
            assert config.enable_cache is False
            assert config.max_retries == 5
    
    def test_retry_validation(self):
        """Test retry parameter validation."""
        # Valid retries
        config = GraphitiConfig(max_retries=0)
        assert config.max_retries == 0
        
        config = GraphitiConfig(max_retries=10)
        assert config.max_retries == 10
        
        # Invalid retries
        with pytest.raises(ValidationError):
            GraphitiConfig(max_retries=-1)
        
        with pytest.raises(ValidationError):
            GraphitiConfig(max_retries=11)


class TestApplicationConfig:
    """Test application configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ApplicationConfig()
        assert config.env == Environment.DEVELOPMENT
        assert config.debug is False
        assert config.log_level == LogLevel.INFO
        assert config.enable_graphiti_memory is True
        assert config.max_workers == 4
    
    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {
            "APP_ENV": "production",
            "APP_DEBUG": "true",
            "APP_LOG_LEVEL": "DEBUG",
            "APP_ENABLE_NLP_MODELS": "true",
            "APP_MAX_WORKERS": "8",
        }):
            config = ApplicationConfig()
            assert config.env == Environment.PRODUCTION
            assert config.debug is True
            assert config.log_level == LogLevel.DEBUG
            assert config.enable_nlp_models is True
            assert config.max_workers == 8
    
    def test_directory_creation(self, tmp_path):
        """Test directory creation."""
        data_dir = tmp_path / "data"
        cache_dir = tmp_path / "cache"
        log_dir = tmp_path / "logs"
        
        config = ApplicationConfig(
            data_dir=str(data_dir),
            cache_dir=str(cache_dir),
            log_dir=str(log_dir)
        )
        
        assert data_dir.exists()
        assert cache_dir.exists()
        assert log_dir.exists()
        assert config.data_dir == data_dir
        assert config.cache_dir == cache_dir
        assert config.log_dir == log_dir


class TestRuntimeConfig:
    """Test runtime configuration container."""
    
    @patch.dict(os.environ, {"NEO4J_PASSWORD": "test123"})
    def test_initialization(self):
        """Test runtime config initialization."""
        config = RuntimeConfig()
        assert isinstance(config.neo4j, Neo4jConfig)
        assert isinstance(config.graphiti, GraphitiConfig)
        assert isinstance(config.app, ApplicationConfig)
    
    @patch.dict(os.environ, {"NEO4J_PASSWORD": "test123", "APP_ENV": "production"})
    def test_environment_checks(self):
        """Test environment check properties."""
        config = RuntimeConfig()
        assert config.is_production is True
        assert config.is_development is False
        assert config.is_testing is False
    
    @patch.dict(os.environ, {"NEO4J_PASSWORD": "test123"})
    def test_neo4j_config_dict(self):
        """Test Neo4j configuration dictionary."""
        config = RuntimeConfig()
        neo4j_dict = config.get_neo4j_config()
        
        assert neo4j_dict["uri"] == "bolt://localhost:7687"
        assert neo4j_dict["auth"] == ("neo4j", "test123")
        assert neo4j_dict["database"] == "neo4j"
        assert "max_connection_pool_size" in neo4j_dict
    
    def test_graphiti_headers(self):
        """Test Graphiti API headers."""
        config = RuntimeConfig()
        
        # Without API key
        headers = config.get_graphiti_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers
        
        # With API key
        config.graphiti.api_key = SecretStr("test-key-123")
        headers = config.get_graphiti_headers()
        assert headers["Authorization"] == "Bearer test-key-123"
    
    @patch.dict(os.environ, {"NEO4J_PASSWORD": "test123"})
    def test_validation_methods(self):
        """Test configuration validation methods."""
        config = RuntimeConfig()
        
        # Neo4j validation
        assert config.validate_neo4j_connection() is True
        
        config.neo4j.password = None
        assert config.validate_neo4j_connection() is False
        
        # Graphiti validation
        assert config.validate_graphiti_connection() is True
        
        config.graphiti.endpoint = ""
        assert config.validate_graphiti_connection() is False
    
    @patch.dict(os.environ, {"NEO4J_PASSWORD": "test123"})
    def test_to_dict(self):
        """Test configuration dictionary export."""
        config = RuntimeConfig()
        
        # Without secrets
        config_dict = config.to_dict(include_secrets=False)
        assert "password" not in config_dict["neo4j"]
        assert config_dict["environment"] == "development"
        assert config_dict["features"]["graphiti_memory"] is True
        
        # With secrets
        config_dict = config.to_dict(include_secrets=True)
        assert config_dict["neo4j"]["password"] == "test123"


class TestConfigurationHelpers:
    """Test configuration helper functions."""
    
    @patch.dict(os.environ, {"NEO4J_PASSWORD": "test123"})
    def test_get_config(self):
        """Test get_config function."""
        config = get_config()
        assert isinstance(config, RuntimeConfig)
    
    @patch.dict(os.environ, {"NEO4J_PASSWORD": "test123"})
    def test_reload_config(self):
        """Test reload_config function."""
        config = get_config()
        original_env = config.app.env
        
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            reloaded = reload_config()
            assert reloaded.app.env == Environment.PRODUCTION
            assert reloaded is config  # Same instance
    
    @patch.dict(os.environ, {"NEO4J_PASSWORD": "test123"})
    def test_is_graphiti_enabled(self):
        """Test is_graphiti_enabled function."""
        config = get_config()
        
        # Enabled by default
        assert is_graphiti_enabled() is True
        
        # Disabled via feature flag
        config.app.enable_graphiti_memory = False
        assert is_graphiti_enabled() is False
        
        # Disabled via missing endpoint
        config.app.enable_graphiti_memory = True
        config.graphiti.endpoint = ""
        assert is_graphiti_enabled() is False
    
    @patch.dict(os.environ, {"NEO4J_PASSWORD": "test123"})
    def test_is_async_enabled(self):
        """Test is_async_enabled function."""
        config = get_config()
        
        assert is_async_enabled() is True
        
        config.app.enable_async_processing = False
        assert is_async_enabled() is False
    
    def test_path_helpers(self, tmp_path):
        """Test path helper functions."""
        config = get_config()
        config.app.data_dir = tmp_path / "data"
        config.app.cache_dir = tmp_path / "cache"
        config.app.log_dir = tmp_path / "logs"
        
        # Test data path
        data_path = get_data_path("models/embeddings")
        assert data_path.exists()
        assert data_path.parent == tmp_path / "data" / "models"
        
        # Test cache path
        cache_path = get_cache_path("responses/api")
        assert cache_path.exists()
        assert cache_path.parent == tmp_path / "cache" / "responses"
        
        # Test log path
        log_path = get_log_path("app/errors.log")
        assert log_path.parent.exists()
        assert log_path.parent == tmp_path / "logs" / "app"