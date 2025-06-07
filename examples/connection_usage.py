"""
Example usage of the Graphiti connection manager.

This script demonstrates how to use the connection managers
for both Neo4j and Graphiti with automatic retry logic.
"""

import asyncio
import logging
from typing import List, Dict, Any

from graphiti_connection import (
    Neo4jConnectionManager,
    GraphitiConnectionManager,
    ConnectionPool,
    get_neo4j_connection,
    get_graphiti_connection,
    ConnectionState,
)
from graphiti_config import get_config


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def neo4j_example():
    """Example of using Neo4j connection manager."""
    logger.info("=== Neo4j Connection Example ===")
    
    # Get global connection manager
    neo4j = get_neo4j_connection()
    
    # Add state change callback
    def on_state_change(state: ConnectionState):
        logger.info(f"Neo4j connection state changed to: {state.value}")
    
    neo4j.add_state_callback(on_state_change)
    
    try:
        # Simple query execution
        logger.info("Executing simple query...")
        results = neo4j.execute_query(
            "RETURN 'Hello from Neo4j!' as message, datetime() as timestamp"
        )
        for record in results:
            logger.info(f"Result: {record}")
        
        # Using session context manager
        logger.info("Using session for transaction...")
        with neo4j.session() as session:
            # Create some test data
            session.run(
                "CREATE (n:TestNode {name: $name, created: datetime()})",
                {"name": "Example Node"}
            )
            
            # Query the data
            result = session.run("MATCH (n:TestNode) RETURN n.name as name")
            names = [record["name"] for record in result]
            logger.info(f"Found nodes: {names}")
            
            # Clean up
            session.run("MATCH (n:TestNode) DELETE n")
        
        # Check metrics
        metrics = neo4j.metrics
        logger.info(f"Connection metrics:")
        logger.info(f"  Total connections: {metrics.total_connections}")
        logger.info(f"  Success rate: {metrics.success_rate:.2%}")
        logger.info(f"  Total retries: {metrics.total_retries}")
        
    except Exception as e:
        logger.error(f"Neo4j error: {e}")
    finally:
        neo4j.close()


def graphiti_example():
    """Example of using Graphiti connection manager."""
    logger.info("\n=== Graphiti Connection Example ===")
    
    # Get global connection manager
    graphiti = get_graphiti_connection()
    
    try:
        # Make API requests with automatic retry
        logger.info("Making Graphiti API request...")
        
        # Health check
        response = graphiti.request("GET", "/health")
        logger.info(f"Health check status: {response.status_code}")
        
        # Example entity creation (adjust based on actual API)
        entity_data = {
            "type": "Question",
            "content": "What is the capital of France?",
            "metadata": {
                "difficulty": "easy",
                "category": "geography"
            }
        }
        
        # This would create an entity if the endpoint exists
        # response = graphiti.request("POST", "/api/entities", json=entity_data)
        # logger.info(f"Entity creation status: {response.status_code}")
        
        # Check metrics
        metrics = graphiti.metrics
        logger.info(f"Connection metrics:")
        logger.info(f"  Total connections: {metrics.total_connections}")
        logger.info(f"  Success rate: {metrics.success_rate:.2%}")
        logger.info(f"  Total retries: {metrics.total_retries}")
        
    except Exception as e:
        logger.error(f"Graphiti error: {e}")
    finally:
        graphiti.close()


async def async_example():
    """Example of async connection usage."""
    logger.info("\n=== Async Connection Example ===")
    
    # Neo4j async example
    neo4j = Neo4jConnectionManager()
    
    try:
        # Async query execution
        results = await neo4j.execute_query_async(
            "UNWIND range(1, 5) as n RETURN n, n * n as square"
        )
        logger.info("Async query results:")
        for record in results:
            logger.info(f"  {record['n']} squared = {record['square']}")
        
        # Async session usage
        async with neo4j.async_session() as session:
            result = await session.run(
                "RETURN $message as msg",
                {"message": "Hello from async Neo4j!"}
            )
            async for record in result:
                logger.info(f"Async message: {record['msg']}")
    
    finally:
        await neo4j.close_async()
    
    # Graphiti async example
    graphiti = GraphitiConnectionManager()
    
    try:
        # Async HTTP request
        response = await graphiti.request_async("GET", "/health")
        logger.info(f"Async health check: {response.status_code}")
    
    finally:
        await graphiti.close_async()


def connection_pool_example():
    """Example of using connection pools."""
    logger.info("\n=== Connection Pool Example ===")
    
    # Create a connection pool for Neo4j
    pool = ConnectionPool(Neo4jConnectionManager, size=3)
    
    try:
        # Simulate multiple concurrent operations
        logger.info("Acquiring connections from pool...")
        
        with pool.acquire() as conn1:
            logger.info(f"Connection 1 state: {conn1.state.value}")
            results = conn1.execute_query("RETURN 1 as num")
            logger.info(f"Connection 1 result: {results}")
            
            with pool.acquire() as conn2:
                logger.info(f"Connection 2 state: {conn2.state.value}")
                results = conn2.execute_query("RETURN 2 as num")
                logger.info(f"Connection 2 result: {results}")
        
        # Connections are returned to pool after use
        logger.info("Connections returned to pool")
        
        # Reuse connection
        with pool.acquire() as conn3:
            logger.info("Reusing connection from pool")
            results = conn3.execute_query("RETURN 3 as num")
            logger.info(f"Reused connection result: {results}")
    
    finally:
        pool.close()
        logger.info("Connection pool closed")


def error_handling_example():
    """Example of error handling and retry behavior."""
    logger.info("\n=== Error Handling Example ===")
    
    config = get_config()
    
    # Simulate connection to non-existent server
    config.neo4j.uri = "bolt://nonexistent:7687"
    neo4j = Neo4jConnectionManager(config)
    
    try:
        # This will retry 3 times before failing
        neo4j.connect()
    except Exception as e:
        logger.error(f"Expected connection failure: {e}")
        logger.info(f"Failed after {neo4j.metrics.total_connections} attempts")
        logger.info(f"Total retries: {neo4j.metrics.total_retries}")


def main():
    """Run all examples."""
    # Synchronous examples
    neo4j_example()
    graphiti_example()
    connection_pool_example()
    error_handling_example()
    
    # Asynchronous examples
    logger.info("\n=== Running Async Examples ===")
    asyncio.run(async_example())
    
    logger.info("\n=== All Examples Completed ===")


if __name__ == "__main__":
    main()