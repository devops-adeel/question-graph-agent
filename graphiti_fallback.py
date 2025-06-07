"""
Graceful fallback module for when Graphiti is unavailable.

This module provides fallback mechanisms and alternative storage options
when the Graphiti service is unavailable, ensuring the application can
continue to function with reduced capabilities.
"""

import asyncio
import json
import logging
import pickle
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
from contextlib import asynccontextmanager
import sqlite3
from collections import deque
import threading

from graphiti_health import (
    HealthStatus,
    ComponentType,
    GraphitiHealthChecker,
    HealthCheckResult,
)
from graphiti_entities import BaseEntity, QuestionEntity, AnswerEntity, UserEntity, TopicEntity
from graphiti_relationships import BaseRelationship
from graphiti_connection import ConnectionState
from graphiti_config import get_config, RuntimeConfig


logger = logging.getLogger(__name__)


T = TypeVar('T')


class FallbackMode(str, Enum):
    """Fallback operation modes."""
    DISABLED = "disabled"  # No fallback, fail if Graphiti unavailable
    READ_ONLY = "read_only"  # Allow reads from cache only
    LOCAL_STORAGE = "local_storage"  # Use local SQLite storage
    MEMORY_ONLY = "memory_only"  # Store in memory only
    QUEUE_WRITES = "queue_writes"  # Queue writes for later sync


@dataclass
class FallbackState:
    """Current fallback state."""
    mode: FallbackMode = FallbackMode.DISABLED
    is_active: bool = False
    activated_at: Optional[datetime] = None
    reason: Optional[str] = None
    queued_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def activate(self, mode: FallbackMode, reason: str):
        """Activate fallback mode."""
        self.mode = mode
        self.is_active = True
        self.activated_at = datetime.now()
        self.reason = reason
    
    def deactivate(self):
        """Deactivate fallback mode."""
        self.is_active = False
        self.reason = None


@dataclass
class QueuedOperation:
    """Represents a queued operation for later sync."""
    operation_type: str  # create, update, delete
    entity_type: str
    entity_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    last_error: Optional[str] = None


class LocalCache:
    """Local cache for entities and relationships."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize local cache."""
        self.cache_dir = cache_dir or Path.home() / ".graphiti_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                return self._memory_cache[key]
            
            # Check disk cache
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                        self._memory_cache[key] = data
                        return data
                except Exception as e:
                    logger.error(f"Failed to load cache file {cache_file}: {e}")
                    return None
        
        return None
    
    def set(self, key: str, value: Any, persist: bool = True):
        """Set item in cache."""
        with self._lock:
            self._memory_cache[key] = value
            
            if persist:
                cache_file = self.cache_dir / f"{key}.pkl"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(value, f)
                except Exception as e:
                    logger.error(f"Failed to save cache file {cache_file}: {e}")
    
    def delete(self, key: str):
        """Delete item from cache."""
        with self._lock:
            self._memory_cache.pop(key, None)
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
    
    def clear(self):
        """Clear all cache."""
        with self._lock:
            self._memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()


class LocalStorage:
    """Local SQLite storage for fallback."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize local storage."""
        self.db_path = db_path or Path.home() / ".graphiti_fallback.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        
        # Create tables
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                rel_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS queued_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                data TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                attempts INTEGER DEFAULT 0,
                last_error TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(rel_type);
            CREATE INDEX IF NOT EXISTS idx_queued_timestamp ON queued_operations(timestamp);
        """)
        self._conn.commit()
    
    def store_entity(self, entity: BaseEntity):
        """Store entity in local database."""
        data = json.dumps(asdict(entity))
        
        self._conn.execute("""
            INSERT OR REPLACE INTO entities (id, entity_type, data, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (entity.id, entity.entity_type, data))
        self._conn.commit()
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity from local database."""
        cursor = self._conn.execute(
            "SELECT data FROM entities WHERE id = ?",
            (entity_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return json.loads(row['data'])
        return None
    
    def store_relationship(self, rel: BaseRelationship):
        """Store relationship in local database."""
        data = json.dumps(asdict(rel))
        
        self._conn.execute("""
            INSERT OR REPLACE INTO relationships 
            (id, rel_type, source_id, target_id, data, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (rel.id, rel.relationship_type, rel.source_id, rel.target_id, data))
        self._conn.commit()
    
    def queue_operation(self, operation: QueuedOperation):
        """Queue operation for later sync."""
        data = json.dumps(operation.data)
        
        self._conn.execute("""
            INSERT INTO queued_operations 
            (operation_type, entity_type, entity_id, data, attempts, last_error)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            operation.operation_type,
            operation.entity_type,
            operation.entity_id,
            data,
            operation.attempts,
            operation.last_error
        ))
        self._conn.commit()
    
    def get_queued_operations(self, limit: int = 100) -> List[QueuedOperation]:
        """Get queued operations for sync."""
        cursor = self._conn.execute("""
            SELECT * FROM queued_operations
            ORDER BY timestamp ASC
            LIMIT ?
        """, (limit,))
        
        operations = []
        for row in cursor:
            op = QueuedOperation(
                operation_type=row['operation_type'],
                entity_type=row['entity_type'],
                entity_id=row['entity_id'],
                data=json.loads(row['data']),
                timestamp=datetime.fromisoformat(row['timestamp']),
                attempts=row['attempts'],
                last_error=row['last_error']
            )
            operations.append(op)
        
        return operations
    
    def delete_queued_operation(self, operation_id: int):
        """Delete queued operation after successful sync."""
        self._conn.execute(
            "DELETE FROM queued_operations WHERE id = ?",
            (operation_id,)
        )
        self._conn.commit()
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()


class FallbackManager:
    """Manages fallback operations when Graphiti is unavailable."""
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """Initialize fallback manager."""
        self.config = config or get_config()
        self.state = FallbackState()
        self.cache = LocalCache()
        self.storage = LocalStorage()
        self.health_checker = GraphitiHealthChecker(config)
        self._callbacks: List[Callable[[FallbackState], None]] = []
        self._sync_task: Optional[asyncio.Task] = None
        self._operation_queue: deque[QueuedOperation] = deque(maxlen=1000)
    
    def add_callback(self, callback: Callable[[FallbackState], None]):
        """Add callback for state changes."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all callbacks of state change."""
        for callback in self._callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Error in fallback callback: {e}")
    
    async def check_and_activate(self) -> bool:
        """Check Graphiti health and activate fallback if needed."""
        result = await self.health_checker.perform_check()
        
        if result.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
            if not self.state.is_active:
                # Determine fallback mode based on configuration
                mode = self._determine_fallback_mode(result)
                self.state.activate(mode, result.error or "Service unavailable")
                logger.warning(f"Activating fallback mode: {mode.value}. Reason: {self.state.reason}")
                self._notify_callbacks()
                
                # Start sync task if queueing writes
                if mode == FallbackMode.QUEUE_WRITES and not self._sync_task:
                    self._sync_task = asyncio.create_task(self._sync_loop())
            
            return True
        
        elif result.status == HealthStatus.HEALTHY and self.state.is_active:
            # Deactivate fallback
            await self._deactivate_fallback()
        
        return False
    
    def _determine_fallback_mode(self, health_result: HealthCheckResult) -> FallbackMode:
        """Determine appropriate fallback mode based on health status."""
        # This could be configured via environment or config
        if health_result.status == HealthStatus.DEGRADED:
            return FallbackMode.QUEUE_WRITES
        else:
            return FallbackMode.LOCAL_STORAGE
    
    async def _deactivate_fallback(self):
        """Deactivate fallback mode and sync queued operations."""
        logger.info("Deactivating fallback mode, syncing queued operations...")
        
        # Sync any queued operations
        if self.state.mode == FallbackMode.QUEUE_WRITES:
            await self._sync_queued_operations()
        
        # Stop sync task
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
        
        self.state.deactivate()
        self._notify_callbacks()
        logger.info("Fallback mode deactivated")
    
    async def _sync_loop(self):
        """Background task to sync queued operations."""
        while self.state.is_active:
            try:
                # Check if Graphiti is back online
                result = await self.health_checker.perform_check()
                if result.status == HealthStatus.HEALTHY:
                    await self._sync_queued_operations()
                
                # Wait before next sync attempt
                await asyncio.sleep(30)  # 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _sync_queued_operations(self):
        """Sync queued operations to Graphiti."""
        operations = self.storage.get_queued_operations()
        
        for op in operations:
            try:
                # Here you would call the actual Graphiti API
                # For now, just log the operation
                logger.info(f"Syncing operation: {op.operation_type} {op.entity_type} {op.entity_id}")
                
                # Delete from queue after successful sync
                # self.storage.delete_queued_operation(op.id)
                
            except Exception as e:
                logger.error(f"Failed to sync operation: {e}")
                op.attempts += 1
                op.last_error = str(e)
                # Update operation in queue
    
    def store_entity(self, entity: BaseEntity) -> bool:
        """Store entity with fallback handling."""
        if not self.state.is_active:
            return False
        
        if self.state.mode == FallbackMode.READ_ONLY:
            logger.warning("Cannot store entity in read-only mode")
            return False
        
        elif self.state.mode == FallbackMode.MEMORY_ONLY:
            # Store in memory cache only
            self.cache.set(f"entity:{entity.id}", entity, persist=False)
            return True
        
        elif self.state.mode == FallbackMode.LOCAL_STORAGE:
            # Store in local database
            self.storage.store_entity(entity)
            self.cache.set(f"entity:{entity.id}", entity)
            return True
        
        elif self.state.mode == FallbackMode.QUEUE_WRITES:
            # Queue for later sync
            operation = QueuedOperation(
                operation_type="create",
                entity_type=entity.entity_type,
                entity_id=entity.id,
                data=asdict(entity)
            )
            self.storage.queue_operation(operation)
            self.state.queued_operations += 1
            
            # Also store locally for immediate access
            self.storage.store_entity(entity)
            self.cache.set(f"entity:{entity.id}", entity)
            return True
        
        return False
    
    def get_entity(self, entity_id: str) -> Optional[BaseEntity]:
        """Get entity with fallback handling."""
        if not self.state.is_active:
            return None
        
        # Check cache first
        cached = self.cache.get(f"entity:{entity_id}")
        if cached:
            self.state.cache_hits += 1
            return cached
        
        self.state.cache_misses += 1
        
        # Check local storage
        if self.state.mode in [FallbackMode.LOCAL_STORAGE, FallbackMode.QUEUE_WRITES]:
            data = self.storage.get_entity(entity_id)
            if data:
                # Reconstruct entity from data
                # This is simplified - you'd need proper deserialization
                return data
        
        return None
    
    @asynccontextmanager
    async def fallback_context(self):
        """Context manager for operations with fallback."""
        # Check and potentially activate fallback
        await self.check_and_activate()
        
        try:
            yield self
        finally:
            # Check if we can deactivate fallback
            if self.state.is_active:
                result = await self.health_checker.perform_check()
                if result.status == HealthStatus.HEALTHY:
                    await self._deactivate_fallback()
    
    def get_status(self) -> Dict[str, Any]:
        """Get fallback status."""
        return {
            "mode": self.state.mode.value,
            "is_active": self.state.is_active,
            "activated_at": self.state.activated_at.isoformat() if self.state.activated_at else None,
            "reason": self.state.reason,
            "queued_operations": self.state.queued_operations,
            "cache_hits": self.state.cache_hits,
            "cache_misses": self.state.cache_misses,
            "cache_hit_rate": (
                self.state.cache_hits / (self.state.cache_hits + self.state.cache_misses)
                if (self.state.cache_hits + self.state.cache_misses) > 0
                else 0
            )
        }


class FallbackDecorator:
    """Decorator for methods that need fallback support."""
    
    def __init__(self, fallback_manager: FallbackManager):
        """Initialize decorator."""
        self.fallback_manager = fallback_manager
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate function with fallback support."""
        async def wrapper(*args, **kwargs):
            # Try normal operation first
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check if we should use fallback
                if await self.fallback_manager.check_and_activate():
                    logger.warning(f"Using fallback for {func.__name__}: {e}")
                    
                    # Try to handle with fallback
                    # This is simplified - real implementation would be more sophisticated
                    if func.__name__.startswith("create_") or func.__name__.startswith("store_"):
                        # Handle write operations
                        if len(args) > 1 and isinstance(args[1], BaseEntity):
                            return self.fallback_manager.store_entity(args[1])
                    
                    elif func.__name__.startswith("get_") or func.__name__.startswith("find_"):
                        # Handle read operations
                        if len(args) > 1 and isinstance(args[1], str):
                            return self.fallback_manager.get_entity(args[1])
                
                # Re-raise if fallback couldn't handle it
                raise
        
        return wrapper


# Global fallback manager
_fallback_manager: Optional[FallbackManager] = None


def get_fallback_manager() -> FallbackManager:
    """Get or create global fallback manager."""
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = FallbackManager()
    return _fallback_manager


def with_fallback(func: Callable) -> Callable:
    """Decorator to add fallback support to a function."""
    manager = get_fallback_manager()
    decorator = FallbackDecorator(manager)
    return decorator(func)