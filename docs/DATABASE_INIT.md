# Database Initialization Guide

This guide explains how to initialize and manage the Neo4j database for the Graphiti integration.

## Overview

The database initialization module (`graphiti_init.py`) provides:
- Schema creation (indexes and constraints)
- Initial data loading (topics and system user)
- Database reset functionality
- Status checking and verification

## Prerequisites

1. Neo4j database running and accessible
2. Environment variables configured in `.env`:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-password
   ```

## Quick Start

### 1. Check Database Status

```bash
python scripts/init_database.py status
```

Output:
```
==================================================
DATABASE STATUS
==================================================
Initialized: ❌ No
Verified: ❌ No

Schema Objects:
  Indexes: 0
  Constraints: 0

Data:
  Topics: 0
  Users: 0
  Episodes: 0
==================================================
```

### 2. Initialize Database

```bash
python scripts/init_database.py init
```

This will:
- Create indexes for performance
- Add constraints for data integrity
- Create initial topic hierarchy
- Add system user
- Verify initialization

### 3. Verify Success

```bash
python scripts/init_database.py status
```

Expected output:
```
==================================================
DATABASE STATUS
==================================================
Initialized: ✅ Yes
Verified: ✅ Yes

Schema Objects:
  Indexes: 15+
  Constraints: 4+

Data:
  Topics: 18
  Users: 1
  Episodes: 0
==================================================
```

## Database Schema

### Indexes

The initialization creates indexes on:

**Entity Indexes:**
- `Entity(id)` - Primary lookup
- `Entity(entity_type)` - Type filtering
- `Entity(created_at)` - Temporal queries
- `Entity(updated_at)` - Change tracking

**Type-Specific Indexes:**
- `Question(difficulty)` - Difficulty filtering
- `Answer(status)` - Answer status queries
- `User(session_id)` - Session lookups
- `Topic(name)` - Topic searches
- `Episode(timestamp)` - Temporal ordering

### Constraints

**Unique Constraints:**
- `Entity.id` - Ensures unique entity IDs
- `User.id` - Unique user identifiers
- `Topic.name` - Unique topic names

**Node Key Constraints:**
- `Answer(question_id, user_id, timestamp)` - Composite uniqueness

### Initial Data

#### Topic Hierarchy

```
General Knowledge
├── History
├── Geography
├── Science
├── Literature
└── Current Events

Mathematics
├── Arithmetic
├── Algebra
├── Geometry
├── Calculus
└── Statistics

Technology
├── Programming
├── AI/ML
├── Databases
├── Networks
└── Security
```

#### System User

- ID: `system`
- Purpose: Automated operations and testing
- Marked with `is_system: true`

## Command Reference

### Initialize Command

```bash
python scripts/init_database.py init [OPTIONS]
```

Options:
- `--force` - Force re-initialization even if already initialized

Example:
```bash
# First initialization
python scripts/init_database.py init

# Force re-initialization
python scripts/init_database.py init --force
```

### Reset Command

```bash
python scripts/init_database.py reset [OPTIONS]
```

Options:
- `--confirm` - Required flag to confirm reset
- `--reinit` - Re-initialize after reset

⚠️ **WARNING**: Reset deletes ALL data!

Example:
```bash
# Reset only (requires confirmation)
python scripts/init_database.py reset --confirm

# Reset and re-initialize
python scripts/init_database.py reset --confirm --reinit
```

### Status Command

```bash
python scripts/init_database.py status [OPTIONS]
```

Options:
- `--verbose`, `-v` - Show detailed status information

Example:
```bash
# Basic status
python scripts/init_database.py status

# Detailed status
python scripts/init_database.py status --verbose
```

## Programmatic Usage

### Basic Initialization

```python
import asyncio
from graphiti_init import initialize_database

async def setup():
    success = await initialize_database()
    if success:
        print("Database initialized!")
    else:
        print("Initialization failed!")

asyncio.run(setup())
```

### Custom Configuration

```python
from graphiti_config import RuntimeConfig
from graphiti_init import DatabaseInitializer

async def custom_init():
    # Custom config
    config = RuntimeConfig()
    config.neo4j.uri = "bolt://custom-host:7687"
    
    # Initialize
    initializer = DatabaseInitializer(config)
    await initializer.initialize_database()
```

### Check Status

```python
from graphiti_init import get_database_status

async def check():
    status = await get_database_status()
    print(f"Topics: {status['topics']}")
    print(f"Verified: {status['verified']}")

asyncio.run(check())
```

## Troubleshooting

### Connection Issues

Error: "Failed to connect to Neo4j"
- Check Neo4j is running: `neo4j status`
- Verify connection settings in `.env`
- Test with Neo4j Browser

### Initialization Failures

Error: "Verification check X failed"
1. Check Neo4j logs for errors
2. Try reset and reinit:
   ```bash
   python scripts/init_database.py reset --confirm --reinit
   ```
3. Verify Neo4j version compatibility

### Permission Issues

Error: "Constraint already exists"
- Database may be partially initialized
- Use `--force` flag to reinitialize
- Or reset database first

### Performance

For large databases:
1. Initialization may take time
2. Monitor Neo4j memory usage
3. Consider increasing heap size

## Best Practices

1. **Regular Backups**
   - Backup before major changes
   - Use Neo4j's backup tools

2. **Environment Separation**
   - Use different databases for dev/test/prod
   - Never reset production without backup

3. **Monitoring**
   - Check status after deployment
   - Monitor index usage
   - Track constraint violations

4. **Maintenance**
   - Periodically check database health
   - Update statistics for query optimization
   - Clean up old episodes if needed

## Docker Support

For Docker-based Neo4j:

```bash
# Start Neo4j
docker run -d \
  --name neo4j-graphiti \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -e NEO4J_PLUGINS='["graph-data-science", "apoc"]' \
  neo4j:5-enterprise

# Initialize database
docker exec -it neo4j-graphiti \
  python scripts/init_database.py init
```

## Advanced Topics

### Custom Indexes

Add custom indexes in `_create_indexes()`:

```python
async def _create_indexes(self):
    # ... existing indexes ...
    
    # Custom composite index
    await session.run("""
        CREATE INDEX custom_idx IF NOT EXISTS
        FOR (n:Question)
        ON (n.difficulty, n.created_at)
    """)
```

### Migration Support

For schema changes:

1. Create migration script
2. Version your schema
3. Apply incrementally
4. Update verification

### Performance Tuning

Optimize for your use case:
- Adjust index granularity
- Consider full-text indexes
- Use composite indexes for common queries
- Monitor query performance