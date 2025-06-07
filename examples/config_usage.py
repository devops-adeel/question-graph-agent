"""
Example usage of the Graphiti configuration module.

This script demonstrates how to use the configuration system
for Neo4j and Graphiti integration.
"""

import logging
from graphiti_config import (
    get_config,
    reload_config,
    is_graphiti_enabled,
    is_async_enabled,
    get_data_path,
    get_cache_path,
    get_log_path,
)


def setup_logging():
    """Set up logging based on configuration."""
    config = get_config()
    
    log_level = getattr(logging, config.app.log_level.value)
    log_file = get_log_path("app.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main example function."""
    # Set up logging
    logger = setup_logging()
    
    # Get configuration
    config = get_config()
    
    # Log configuration status
    logger.info(f"Environment: {config.app.env.value}")
    logger.info(f"Debug mode: {config.app.debug}")
    logger.info(f"Neo4j URI: {config.neo4j.uri}")
    logger.info(f"Graphiti endpoint: {config.graphiti.endpoint}")
    
    # Check feature flags
    if is_graphiti_enabled():
        logger.info("Graphiti memory is enabled")
    else:
        logger.warning("Graphiti memory is disabled")
    
    if is_async_enabled():
        logger.info("Async processing is enabled")
    else:
        logger.info("Async processing is disabled")
    
    # Validate connections
    if not config.validate_neo4j_connection():
        logger.error("Neo4j configuration is incomplete!")
        return
    
    if not config.validate_graphiti_connection():
        logger.warning("Graphiti configuration is incomplete!")
    
    # Get Neo4j configuration for driver
    neo4j_config = config.get_neo4j_config()
    logger.info(f"Neo4j driver config ready for: {neo4j_config['uri']}")
    
    # Get Graphiti headers for API calls
    headers = config.get_graphiti_headers()
    logger.info(f"Graphiti API headers prepared: {list(headers.keys())}")
    
    # Example of using path helpers
    model_path = get_data_path("models/question_embeddings.pkl")
    logger.info(f"Model path: {model_path}")
    
    cache_file = get_cache_path("graphiti/entities.json")
    logger.info(f"Cache file: {cache_file}")
    
    # Export configuration (without secrets)
    config_dict = config.to_dict(include_secrets=False)
    logger.info(f"Configuration summary: {config_dict}")
    
    # Example of reloading configuration
    logger.info("Reloading configuration...")
    reload_config()
    logger.info("Configuration reloaded successfully")
    
    # Production-specific behavior
    if config.is_production:
        logger.info("Running in production mode - extra validations enabled")
        # Add production-specific checks here
    elif config.is_development:
        logger.info("Running in development mode - debug features available")
        # Add development-specific features here
    
    # Performance settings
    logger.info(f"Max workers: {config.app.max_workers}")
    logger.info(f"Batch size: {config.app.batch_size}")
    logger.info(f"Extraction timeout: {config.app.extraction_timeout}s")
    
    # Neo4j connection settings
    logger.info(f"Connection pool size: {config.neo4j.max_connection_pool_size}")
    logger.info(f"Connection timeout: {config.neo4j.connection_acquisition_timeout}s")
    
    # Graphiti retry settings
    logger.info(f"Max retries: {config.graphiti.max_retries}")
    logger.info(f"Retry delay: {config.graphiti.retry_delay}s")
    logger.info(f"Retry backoff: {config.graphiti.retry_backoff}x")


if __name__ == "__main__":
    main()