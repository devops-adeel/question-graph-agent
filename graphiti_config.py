"""
Configuration module for Graphiti integration with Neo4j.

This module provides environment configuration management, connection settings,
and feature flags for the Graphiti temporal knowledge graph integration.
"""

import os
from enum import Enum
from typing import Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic.types import conint


# Load environment variables from .env file
load_dotenv()


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level configuration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Neo4jConfig(BaseSettings):
    """Neo4j database configuration settings."""
    
    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI"
    )
    user: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    password: SecretStr = Field(
        ...,
        description="Neo4j password"
    )
    database: str = Field(
        default="neo4j",
        description="Neo4j database name"
    )
    encrypted: bool = Field(
        default=False,
        description="Use encrypted connection"
    )
    trust: str = Field(
        default="TRUST_ALL_CERTIFICATES",
        description="Certificate trust strategy"
    )
    max_connection_lifetime: conint(gt=0) = Field(
        default=3600,
        description="Maximum connection lifetime in seconds"
    )
    max_connection_pool_size: conint(gt=0) = Field(
        default=50,
        description="Maximum connection pool size"
    )
    connection_acquisition_timeout: conint(gt=0) = Field(
        default=60,
        description="Connection acquisition timeout in seconds"
    )
    
    class Config:
        env_prefix = "NEO4J_"
        case_sensitive = False
        
    @validator("uri")
    def validate_uri(cls, v: str) -> str:
        """Validate Neo4j URI format."""
        valid_schemes = ["bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"]
        scheme = v.split("://")[0].lower()
        if scheme not in valid_schemes:
            raise ValueError(f"Invalid Neo4j URI scheme: {scheme}. Must be one of {valid_schemes}")
        return v


class GraphitiConfig(BaseSettings):
    """Graphiti service configuration settings."""
    
    api_key: Optional[SecretStr] = Field(
        None,
        description="Graphiti API key for authentication"
    )
    endpoint: str = Field(
        default="http://localhost:8000",
        description="Graphiti service endpoint"
    )
    enable_cache: bool = Field(
        default=True,
        description="Enable response caching"
    )
    cache_ttl: conint(gt=0) = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    request_timeout: conint(gt=0) = Field(
        default=30,
        description="Request timeout in seconds"
    )
    max_retries: conint(ge=0, le=10) = Field(
        default=3,
        description="Maximum number of retries"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Initial retry delay in seconds"
    )
    retry_backoff: float = Field(
        default=2.0,
        description="Retry backoff multiplier"
    )
    
    class Config:
        env_prefix = "GRAPHITI_"
        case_sensitive = False


class ApplicationConfig(BaseSettings):
    """Main application configuration settings."""
    
    env: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    
    # Feature flags
    enable_graphiti_memory: bool = Field(
        default=True,
        description="Enable Graphiti memory integration"
    )
    enable_async_processing: bool = Field(
        default=True,
        description="Enable async processing"
    )
    enable_nlp_models: bool = Field(
        default=False,
        description="Enable NLP model integration"
    )
    
    # Performance settings
    max_workers: conint(gt=0) = Field(
        default=4,
        description="Maximum worker threads"
    )
    batch_size: conint(gt=0) = Field(
        default=100,
        description="Default batch processing size"
    )
    extraction_timeout: conint(gt=0) = Field(
        default=30,
        description="Entity extraction timeout in seconds"
    )
    
    # Data paths
    data_dir: Path = Field(
        default=Path("./data"),
        description="Data directory path"
    )
    cache_dir: Path = Field(
        default=Path("./cache"),
        description="Cache directory path"
    )
    log_dir: Path = Field(
        default=Path("./logs"),
        description="Logs directory path"
    )
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False
        
    @validator("data_dir", "cache_dir", "log_dir", pre=True)
    def create_directories(cls, v: Union[str, Path]) -> Path:
        """Ensure directories exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class RuntimeConfig:
    """Runtime configuration container combining all config sections."""
    
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    graphiti: GraphitiConfig = field(default_factory=GraphitiConfig)
    app: ApplicationConfig = field(default_factory=ApplicationConfig)
    
    def __post_init__(self):
        """Initialize configuration from environment."""
        self.reload()
    
    def reload(self):
        """Reload configuration from environment variables."""
        self.neo4j = Neo4jConfig()
        self.graphiti = GraphitiConfig()
        self.app = ApplicationConfig()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app.env == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app.env == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.app.env == Environment.TESTING
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """Get Neo4j driver configuration dict."""
        return {
            "uri": self.neo4j.uri,
            "auth": (self.neo4j.user, self.neo4j.password.get_secret_value()),
            "database": self.neo4j.database,
            "encrypted": self.neo4j.encrypted,
            "trust": self.neo4j.trust,
            "max_connection_lifetime": self.neo4j.max_connection_lifetime,
            "max_connection_pool_size": self.neo4j.max_connection_pool_size,
            "connection_acquisition_timeout": self.neo4j.connection_acquisition_timeout,
        }
    
    def get_graphiti_headers(self) -> Dict[str, str]:
        """Get Graphiti API headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.graphiti.api_key:
            headers["Authorization"] = f"Bearer {self.graphiti.api_key.get_secret_value()}"
        return headers
    
    def validate_neo4j_connection(self) -> bool:
        """Validate Neo4j configuration is complete."""
        return bool(
            self.neo4j.uri and 
            self.neo4j.user and 
            self.neo4j.password
        )
    
    def validate_graphiti_connection(self) -> bool:
        """Validate Graphiti configuration is complete."""
        return bool(self.graphiti.endpoint)
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config = {
            "environment": self.app.env.value,
            "debug": self.app.debug,
            "features": {
                "graphiti_memory": self.app.enable_graphiti_memory,
                "async_processing": self.app.enable_async_processing,
                "nlp_models": self.app.enable_nlp_models,
            },
            "neo4j": {
                "uri": self.neo4j.uri,
                "user": self.neo4j.user,
                "database": self.neo4j.database,
                "encrypted": self.neo4j.encrypted,
            },
            "graphiti": {
                "endpoint": self.graphiti.endpoint,
                "cache_enabled": self.graphiti.enable_cache,
                "cache_ttl": self.graphiti.cache_ttl,
            }
        }
        
        if include_secrets:
            config["neo4j"]["password"] = self.neo4j.password.get_secret_value()
            if self.graphiti.api_key:
                config["graphiti"]["api_key"] = self.graphiti.api_key.get_secret_value()
                
        return config


# Global configuration instance
config = RuntimeConfig()


# Configuration helper functions
def get_config() -> RuntimeConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> RuntimeConfig:
    """Reload configuration from environment."""
    config.reload()
    return config


def is_graphiti_enabled() -> bool:
    """Check if Graphiti memory is enabled."""
    return config.app.enable_graphiti_memory and config.validate_graphiti_connection()


def is_async_enabled() -> bool:
    """Check if async processing is enabled."""
    return config.app.enable_async_processing


def get_data_path(subpath: str = "") -> Path:
    """Get path within data directory."""
    path = config.app.data_dir / subpath
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_path(subpath: str = "") -> Path:
    """Get path within cache directory."""
    path = config.app.cache_dir / subpath
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_log_path(subpath: str = "") -> Path:
    """Get path within log directory."""
    path = config.app.log_dir / subpath
    path.parent.mkdir(parents=True, exist_ok=True)
    return path