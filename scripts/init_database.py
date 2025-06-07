#!/usr/bin/env python3
"""
Database initialization script for Graphiti integration.

This script provides a command-line interface for initializing,
resetting, and checking the Neo4j database for the question-graph agent.
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphiti_init import (
    initialize_database,
    reset_database,
    get_database_status,
)
from graphiti_config import get_config
from graphiti_connection import Neo4jConnectionManager


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_connection():
    """Check Neo4j connection before operations."""
    config = get_config()
    manager = Neo4jConnectionManager(config)
    
    try:
        logger.info("Checking Neo4j connection...")
        driver = manager.connect()
        logger.info("✅ Successfully connected to Neo4j")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {e}")
        logger.error("Please check your connection settings in .env file")
        return False
    finally:
        manager.close()


async def init_command(args):
    """Handle init command."""
    logger.info("🚀 Starting database initialization...")
    
    # Check connection first
    if not await check_connection():
        return 1
    
    # Run initialization
    success = await initialize_database(force=args.force)
    
    if success:
        logger.info("✅ Database initialization completed successfully!")
        
        # Show status
        status = await get_database_status()
        print_status(status)
        return 0
    else:
        logger.error("❌ Database initialization failed!")
        return 1


async def reset_command(args):
    """Handle reset command."""
    if not args.confirm:
        logger.error("❌ Database reset requires --confirm flag")
        logger.error("⚠️  WARNING: This will DELETE ALL DATA in the database!")
        logger.error("Run with: python scripts/init_database.py reset --confirm")
        return 1
    
    # Double confirmation
    print("\n⚠️  WARNING: This will DELETE ALL DATA in the database!")
    response = input("Type 'DELETE ALL DATA' to confirm: ")
    
    if response != "DELETE ALL DATA":
        logger.info("Reset cancelled")
        return 0
    
    logger.info("🔥 Starting database reset...")
    
    # Check connection first
    if not await check_connection():
        return 1
    
    # Run reset
    success = await reset_database(confirm=True)
    
    if success:
        logger.info("✅ Database reset completed!")
        
        # Re-initialize if requested
        if args.reinit:
            logger.info("🚀 Re-initializing database...")
            init_success = await initialize_database()
            if init_success:
                logger.info("✅ Database re-initialized successfully!")
                status = await get_database_status()
                print_status(status)
            else:
                logger.error("❌ Re-initialization failed!")
                return 1
        return 0
    else:
        logger.error("❌ Database reset failed!")
        return 1


async def status_command(args):
    """Handle status command."""
    logger.info("📊 Checking database status...")
    
    # Check connection first
    if not await check_connection():
        return 1
    
    # Get status
    status = await get_database_status()
    print_status(status, verbose=args.verbose)
    
    if not status["verified"]:
        logger.warning("⚠️  Database initialization is incomplete!")
        logger.info("Run: python scripts/init_database.py init")
        return 1
    
    return 0


def print_status(status: dict, verbose: bool = False):
    """Print database status in a formatted way."""
    print("\n" + "="*50)
    print("DATABASE STATUS")
    print("="*50)
    
    # Basic status
    print(f"Initialized: {'✅ Yes' if status['initialized'] else '❌ No'}")
    print(f"Verified: {'✅ Yes' if status['verified'] else '❌ No'}")
    
    # Counts
    print(f"\nSchema Objects:")
    print(f"  Indexes: {status['indexes']}")
    print(f"  Constraints: {status['constraints']}")
    
    print(f"\nData:")
    print(f"  Topics: {status['topics']}")
    print(f"  Users: {status['users']}")
    print(f"  Episodes: {status['episodes']}")
    
    if verbose:
        print(f"\nDetailed Status:")
        for key, value in status.items():
            if key not in ['initialized', 'verified', 'indexes', 'constraints', 'topics', 'users', 'episodes']:
                print(f"  {key}: {value}")
    
    print("="*50 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Database initialization for Graphiti integration"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize the database')
    init_parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-initialization even if already initialized'
    )
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset the database (DELETE ALL DATA)')
    reset_parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm database reset'
    )
    reset_parser.add_argument(
        '--reinit',
        action='store_true',
        help='Re-initialize after reset'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check database status')
    status_parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show verbose status'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run the appropriate command
    if args.command == 'init':
        return asyncio.run(init_command(args))
    elif args.command == 'reset':
        return asyncio.run(reset_command(args))
    elif args.command == 'status':
        return asyncio.run(status_command(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())