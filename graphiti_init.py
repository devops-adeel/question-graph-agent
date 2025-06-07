"""
Database initialization module for Graphiti and Neo4j.

This module provides initialization scripts for setting up the Neo4j database
with required indexes, constraints, and initial data for the question-graph agent.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import (
    ConstraintError,
    DatabaseError,
    Neo4jError,
)

from graphiti_config import get_config, RuntimeConfig
from graphiti_connection import Neo4jConnectionManager
from graphiti_entities import (
    TopicEntity,
    DifficultyLevel,
    EntityFactory,
)


logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization for Graphiti integration."""
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """Initialize with configuration.
        
        Args:
            config: Runtime configuration, uses global if not provided
        """
        self.config = config or get_config()
        self.connection_manager = Neo4jConnectionManager(self.config)
        self._initialized = False
    
    async def initialize_database(self, force: bool = False) -> bool:
        """Initialize the database with schema and initial data.
        
        Args:
            force: Force re-initialization even if already done
            
        Returns:
            True if successful, False otherwise
        """
        if self._initialized and not force:
            logger.info("Database already initialized")
            return True
        
        try:
            logger.info("Starting database initialization...")
            
            # Create schema
            await self._create_indexes()
            await self._create_constraints()
            
            # Initialize node labels
            await self._create_node_labels()
            
            # Create initial data
            await self._create_initial_topics()
            await self._create_system_user()
            
            # Verify initialization
            if await self._verify_initialization():
                self._initialized = True
                logger.info("Database initialization completed successfully")
                return True
            else:
                logger.error("Database initialization verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    async def _create_indexes(self):
        """Create database indexes for performance."""
        logger.info("Creating indexes...")
        
        indexes = [
            # Entity indexes
            ("Entity", "id"),
            ("Entity", "entity_type"),
            ("Entity", "created_at"),
            ("Entity", "updated_at"),
            
            # Specific entity type indexes
            ("Question", "difficulty"),
            ("Question", "created_at"),
            ("Answer", "status"),
            ("Answer", "user_id"),
            ("User", "session_id"),
            ("Topic", "name"),
            ("Topic", "complexity_score"),
            
            # Episode indexes
            ("Episode", "episode_type"),
            ("Episode", "timestamp"),
            ("Episode", "name"),
            
            # Relationship indexes
            ("ANSWERED", "timestamp"),
            ("REQUIRES_KNOWLEDGE", "relevance_score"),
            ("HAS_MASTERY", "mastery_score"),
        ]
        
        async with self.connection_manager.async_session() as session:
            for label, property_name in indexes:
                query = f"""
                CREATE INDEX IF NOT EXISTS
                FOR (n:{label})
                ON (n.{property_name})
                """
                try:
                    await session.run(query)
                    logger.debug(f"Created index on {label}.{property_name}")
                except Exception as e:
                    logger.warning(f"Failed to create index on {label}.{property_name}: {e}")
    
    async def _create_constraints(self):
        """Create database constraints for data integrity."""
        logger.info("Creating constraints...")
        
        constraints = [
            # Unique constraints
            ("Entity", "id", "UNIQUE"),
            ("User", "id", "UNIQUE"),
            ("Topic", "name", "UNIQUE"),
            
            # Node key constraints (composite uniqueness)
            ("Answer", ["question_id", "user_id", "timestamp"], "NODE KEY"),
        ]
        
        async with self.connection_manager.async_session() as session:
            for constraint in constraints:
                if len(constraint) == 3 and constraint[2] == "UNIQUE":
                    label, property_name, _ = constraint
                    query = f"""
                    CREATE CONSTRAINT IF NOT EXISTS
                    FOR (n:{label})
                    REQUIRE n.{property_name} IS UNIQUE
                    """
                    try:
                        await session.run(query)
                        logger.debug(f"Created unique constraint on {label}.{property_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create constraint on {label}.{property_name}: {e}")
                        
                elif len(constraint) == 3 and constraint[2] == "NODE KEY":
                    label, properties, _ = constraint
                    props_str = ", ".join([f"n.{p}" for p in properties])
                    constraint_name = f"{label}_{'_'.join(properties)}_key"
                    query = f"""
                    CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                    FOR (n:{label})
                    REQUIRE ({props_str}) IS NODE KEY
                    """
                    try:
                        await session.run(query)
                        logger.debug(f"Created node key constraint on {label} for {properties}")
                    except Exception as e:
                        logger.warning(f"Failed to create node key constraint: {e}")
    
    async def _create_node_labels(self):
        """Create initial node labels."""
        logger.info("Creating node labels...")
        
        labels = [
            "Entity",
            "Question", 
            "Answer",
            "User",
            "Topic",
            "Episode",
            "System",
        ]
        
        async with self.connection_manager.async_session() as session:
            for label in labels:
                # Create a temporary node to ensure label exists
                query = f"""
                MERGE (n:_Temp_{label} {{id: 'temp'}})
                SET n:{label}
                WITH n
                DELETE n
                """
                try:
                    await session.run(query)
                    logger.debug(f"Ensured label {label} exists")
                except Exception as e:
                    logger.warning(f"Failed to create label {label}: {e}")
    
    async def _create_initial_topics(self):
        """Create initial topic hierarchy."""
        logger.info("Creating initial topics...")
        
        # Define topic hierarchy
        topics_data = [
            {
                "name": "General Knowledge",
                "complexity": 0.5,
                "description": "Basic general knowledge questions",
                "children": [
                    {"name": "History", "complexity": 0.6},
                    {"name": "Geography", "complexity": 0.5},
                    {"name": "Science", "complexity": 0.7},
                    {"name": "Literature", "complexity": 0.6},
                    {"name": "Current Events", "complexity": 0.4},
                ]
            },
            {
                "name": "Mathematics",
                "complexity": 0.8,
                "description": "Mathematical concepts and problem solving",
                "children": [
                    {"name": "Arithmetic", "complexity": 0.3},
                    {"name": "Algebra", "complexity": 0.6},
                    {"name": "Geometry", "complexity": 0.7},
                    {"name": "Calculus", "complexity": 0.9},
                    {"name": "Statistics", "complexity": 0.7},
                ]
            },
            {
                "name": "Technology",
                "complexity": 0.7,
                "description": "Computer science and technology topics",
                "children": [
                    {"name": "Programming", "complexity": 0.8},
                    {"name": "AI/ML", "complexity": 0.9},
                    {"name": "Databases", "complexity": 0.7},
                    {"name": "Networks", "complexity": 0.6},
                    {"name": "Security", "complexity": 0.8},
                ]
            },
        ]
        
        async with self.connection_manager.async_session() as session:
            for parent_data in topics_data:
                # Create parent topic
                parent_query = """
                MERGE (t:Topic:Entity {name: $name})
                SET t.id = coalesce(t.id, randomUUID()),
                    t.complexity_score = $complexity,
                    t.description = $description,
                    t.created_at = coalesce(t.created_at, datetime()),
                    t.updated_at = datetime(),
                    t.entity_type = 'topic'
                RETURN t.id as topic_id
                """
                
                result = await session.run(
                    parent_query,
                    name=parent_data["name"],
                    complexity=parent_data["complexity"],
                    description=parent_data.get("description", "")
                )
                record = await result.single()
                parent_id = record["topic_id"] if record else None
                
                logger.debug(f"Created parent topic: {parent_data['name']}")
                
                # Create child topics
                for child_data in parent_data.get("children", []):
                    child_query = """
                    MERGE (t:Topic:Entity {name: $name})
                    SET t.id = coalesce(t.id, randomUUID()),
                        t.complexity_score = $complexity,
                        t.parent_topic = $parent_name,
                        t.created_at = coalesce(t.created_at, datetime()),
                        t.updated_at = datetime(),
                        t.entity_type = 'topic'
                    WITH t
                    MATCH (p:Topic {name: $parent_name})
                    MERGE (t)-[:CHILD_OF]->(p)
                    RETURN t.id as topic_id
                    """
                    
                    await session.run(
                        child_query,
                        name=child_data["name"],
                        complexity=child_data["complexity"],
                        parent_name=parent_data["name"]
                    )
                    logger.debug(f"Created child topic: {child_data['name']}")
    
    async def _create_system_user(self):
        """Create system user for automated operations."""
        logger.info("Creating system user...")
        
        query = """
        MERGE (u:User:Entity {id: 'system'})
        SET u.session_id = 'system',
            u.total_questions = 0,
            u.correct_answers = 0,
            u.average_response_time = 0.0,
            u.created_at = coalesce(u.created_at, datetime()),
            u.updated_at = datetime(),
            u.entity_type = 'person',
            u.is_system = true
        RETURN u.id as user_id
        """
        
        async with self.connection_manager.async_session() as session:
            result = await session.run(query)
            record = await result.single()
            if record:
                logger.debug("Created system user")
    
    async def _verify_initialization(self) -> bool:
        """Verify database initialization was successful."""
        logger.info("Verifying initialization...")
        
        checks = [
            # Check indexes exist
            """
            SHOW INDEXES
            YIELD name, labelsOrTypes, properties
            WHERE 'Entity' IN labelsOrTypes AND 'id' IN properties
            RETURN count(*) as count
            """,
            
            # Check constraints exist
            """
            SHOW CONSTRAINTS
            YIELD name, labelsOrTypes, properties
            WHERE 'Entity' IN labelsOrTypes
            RETURN count(*) as count
            """,
            
            # Check topics exist
            """
            MATCH (t:Topic)
            RETURN count(t) as count
            """,
            
            # Check system user exists
            """
            MATCH (u:User {id: 'system'})
            RETURN count(u) as count
            """,
        ]
        
        async with self.connection_manager.async_session() as session:
            for i, check_query in enumerate(checks):
                try:
                    result = await session.run(check_query)
                    record = await result.single()
                    if not record or record["count"] == 0:
                        logger.error(f"Verification check {i+1} failed")
                        return False
                except Exception as e:
                    logger.error(f"Verification check {i+1} error: {e}")
                    return False
        
        logger.info("All verification checks passed")
        return True
    
    async def reset_database(self, confirm: bool = False) -> bool:
        """Reset the database (DANGEROUS - deletes all data).
        
        Args:
            confirm: Must be True to actually perform reset
            
        Returns:
            True if successful, False otherwise
        """
        if not confirm:
            logger.warning("Database reset requested but not confirmed")
            return False
        
        logger.warning("RESETTING DATABASE - ALL DATA WILL BE DELETED")
        
        try:
            async with self.connection_manager.async_session() as session:
                # Delete all nodes and relationships
                await session.run("MATCH (n) DETACH DELETE n")
                logger.info("Deleted all nodes and relationships")
                
                # Drop constraints
                constraints_result = await session.run("SHOW CONSTRAINTS")
                constraints = await constraints_result.data()
                for constraint in constraints:
                    constraint_name = constraint.get("name")
                    if constraint_name:
                        await session.run(f"DROP CONSTRAINT {constraint_name}")
                        logger.debug(f"Dropped constraint: {constraint_name}")
                
                # Drop indexes
                indexes_result = await session.run("SHOW INDEXES")
                indexes = await indexes_result.data()
                for index in indexes:
                    index_name = index.get("name")
                    if index_name and not index_name.startswith("constraint_"):
                        await session.run(f"DROP INDEX {index_name}")
                        logger.debug(f"Dropped index: {index_name}")
            
            self._initialized = False
            logger.info("Database reset completed")
            return True
            
        except Exception as e:
            logger.error(f"Database reset failed: {e}")
            return False
    
    async def get_initialization_status(self) -> Dict[str, Any]:
        """Get current initialization status.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "initialized": self._initialized,
            "indexes": 0,
            "constraints": 0,
            "topics": 0,
            "users": 0,
            "episodes": 0,
            "verified": False,
        }
        
        try:
            async with self.connection_manager.async_session() as session:
                # Count indexes
                result = await session.run("SHOW INDEXES YIELD name RETURN count(*) as count")
                record = await result.single()
                status["indexes"] = record["count"] if record else 0
                
                # Count constraints
                result = await session.run("SHOW CONSTRAINTS YIELD name RETURN count(*) as count")
                record = await result.single()
                status["constraints"] = record["count"] if record else 0
                
                # Count entities
                for entity_type, key in [("Topic", "topics"), ("User", "users"), ("Episode", "episodes")]:
                    result = await session.run(f"MATCH (n:{entity_type}) RETURN count(n) as count")
                    record = await result.single()
                    status[key] = record["count"] if record else 0
                
                # Verify status
                status["verified"] = await self._verify_initialization()
                
        except Exception as e:
            logger.error(f"Failed to get initialization status: {e}")
        
        return status


async def initialize_database(config: Optional[RuntimeConfig] = None, force: bool = False) -> bool:
    """Initialize the database.
    
    Args:
        config: Runtime configuration
        force: Force re-initialization
        
    Returns:
        True if successful
    """
    initializer = DatabaseInitializer(config)
    return await initializer.initialize_database(force)


async def reset_database(config: Optional[RuntimeConfig] = None, confirm: bool = False) -> bool:
    """Reset the database (requires confirmation).
    
    Args:
        config: Runtime configuration
        confirm: Must be True to perform reset
        
    Returns:
        True if successful
    """
    initializer = DatabaseInitializer(config)
    return await initializer.reset_database(confirm)


async def get_database_status(config: Optional[RuntimeConfig] = None) -> Dict[str, Any]:
    """Get database initialization status.
    
    Args:
        config: Runtime configuration
        
    Returns:
        Status dictionary
    """
    initializer = DatabaseInitializer(config)
    return await initializer.get_initialization_status()