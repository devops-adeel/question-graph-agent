"""
Database initialization for Graphiti integration.

This module provides initialization functionality for setting up
the Neo4j database schema and indexes for Graphiti.
"""

import logging
from typing import Optional, List, Dict, Any

from graphiti_connection import Neo4jConnectionManager
from graphiti_config import get_config, RuntimeConfig


logger = logging.getLogger(__name__)


class GraphitiInitializer:
    """Handles database initialization for Graphiti."""
    
    def __init__(self,
                 connection_manager: Optional[Neo4jConnectionManager] = None,
                 config: Optional[RuntimeConfig] = None):
        """Initialize the initializer.
        
        Args:
            connection_manager: Neo4j connection manager
            config: Runtime configuration
        """
        self.connection_manager = connection_manager
        self.config = config or get_config()
    
    async def initialize(self) -> bool:
        """Initialize the database with required schema.
        
        Returns:
            True if successful
        """
        try:
            logger.info("Initializing Graphiti database schema...")
            
            # Create constraints
            await self._create_constraints()
            
            # Create indexes
            await self._create_indexes()
            
            # Initialize entity types
            await self._initialize_entity_types()
            
            logger.info("Database initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    async def _create_constraints(self) -> None:
        """Create database constraints."""
        constraints = [
            # Unique constraints for entity IDs
            "CREATE CONSTRAINT IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Answer) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
        ]
        
        if self.connection_manager:
            for constraint in constraints:
                try:
                    await self.connection_manager.execute_query_async(constraint)
                    logger.debug(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")
    
    async def _create_indexes(self) -> None:
        """Create database indexes for performance."""
        indexes = [
            # Indexes for common queries
            "CREATE INDEX IF NOT EXISTS FOR (q:Question) ON (q.difficulty)",
            "CREATE INDEX IF NOT EXISTS FOR (q:Question) ON (q.asked_count)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Answer) ON (a.user_id)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Answer) ON (a.timestamp)",
            "CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.session_id)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Episode) ON (e.session_id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Episode) ON (e.timestamp)",
        ]
        
        if self.connection_manager:
            for index in indexes:
                try:
                    await self.connection_manager.execute_query_async(index)
                    logger.debug(f"Created index: {index}")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")
    
    async def _initialize_entity_types(self) -> None:
        """Initialize entity type nodes."""
        entity_types = [
            {
                "name": "Question",
                "description": "A question asked to users",
                "properties": ["content", "difficulty", "topics", "asked_count", "correct_rate"]
            },
            {
                "name": "Answer",
                "description": "A user's answer to a question",
                "properties": ["content", "status", "timestamp", "response_time", "confidence_score"]
            },
            {
                "name": "User",
                "description": "A user of the system",
                "properties": ["session_id", "total_questions", "correct_answers", "average_response_time"]
            },
            {
                "name": "Topic",
                "description": "A knowledge topic or subject area",
                "properties": ["name", "complexity_score", "question_count"]
            }
        ]
        
        if self.connection_manager:
            for entity_type in entity_types:
                query = """
                MERGE (et:EntityType {name: $name})
                SET et.description = $description,
                    et.properties = $properties,
                    et.created_at = COALESCE(et.created_at, datetime()),
                    et.updated_at = datetime()
                """
                
                await self.connection_manager.execute_query_async(
                    query,
                    entity_type
                )
                
            logger.info(f"Initialized {len(entity_types)} entity types")
    
    async def verify_setup(self) -> Dict[str, Any]:
        """Verify the database setup.
        
        Returns:
            Dictionary with verification results
        """
        results = {
            "constraints": [],
            "indexes": [],
            "entity_types": [],
            "node_counts": {}
        }
        
        if not self.connection_manager:
            return results
        
        try:
            # Check constraints
            constraint_query = "SHOW CONSTRAINTS"
            constraints = await self.connection_manager.execute_query_async(constraint_query)
            results["constraints"] = [c["name"] for c in constraints if c.get("name")]
            
            # Check indexes
            index_query = "SHOW INDEXES"
            indexes = await self.connection_manager.execute_query_async(index_query)
            results["indexes"] = [i["name"] for i in indexes if i.get("name")]
            
            # Check entity types
            entity_query = "MATCH (et:EntityType) RETURN et.name as name"
            entities = await self.connection_manager.execute_query_async(entity_query)
            results["entity_types"] = [e["name"] for e in entities]
            
            # Get node counts
            for label in ["Question", "Answer", "User", "Topic"]:
                count_query = f"MATCH (n:{label}) RETURN count(n) as count"
                result = await self.connection_manager.execute_query_async(count_query)
                if result:
                    results["node_counts"][label] = result[0]["count"]
                else:
                    results["node_counts"][label] = 0
            
        except Exception as e:
            logger.error(f"Failed to verify setup: {e}")
        
        return results
    
    async def clear_data(self, confirm: bool = False) -> bool:
        """Clear all data from the database.
        
        Args:
            confirm: Must be True to actually clear data
            
        Returns:
            True if successful
        """
        if not confirm:
            logger.warning("Clear data called without confirmation")
            return False
        
        if not self.connection_manager:
            return False
        
        try:
            logger.warning("Clearing all data from database...")
            
            # Delete all nodes and relationships
            await self.connection_manager.execute_query_async(
                "MATCH (n) DETACH DELETE n"
            )
            
            logger.info("All data cleared from database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return False
