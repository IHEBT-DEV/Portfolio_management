import json
import logging
import os
from typing import Any, Optional, Union
import redis
from redis.exceptions import RedisError, ConnectionError

logger = logging.getLogger(__name__)


class RedisService:
    """
    High-performance Redis cache management infrastructure layer.
    Enforces automatic byte decoding, structured exception handling, and memory-safe expiration controls.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None
    ):
        """
        Initializes the persistent Redis socket connection pool.
        Resolves configurations dynamically from system environments to enforce zero-hardcode compliance.
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.db = db if db is not None else int(os.getenv("REDIS_DB", 0))
        self.password = password or os.getenv("REDIS_PASSWORD", None)

        try:
            # decode_responses=True handles low-level UTF-8 string conversions automatically
            # socket_timeout ensures the system breaks quickly on network hang-ups
            self.connection = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_timeout=3.0
            )
            # Send an instant ping command to verify cluster heartbeat health
            self.connection.ping()
            logger.info(f"Redis memory store successfully linked to space pool: [{self.host}:{self.port}/db={self.db}]")
        except ConnectionError as connection_fault:
            logger.critical(f"Fatal cache pool initialization error: Failed to map connection to Redis: {connection_fault}")
            raise connection_fault

    def set_data(self, key: str, value: Union[str, int, float], expire_seconds: Optional[int] = 3600) -> bool:
        """
        Writes primitive data keys cleanly. Enforces a 1-hour defensive default memory expiration limit.
        """
        try:
            return bool(self.connection.set(key, value, ex=expire_seconds))
        except RedisError as cache_error:
            logger.error(f"Redis write mutation exception encountered for key [{key}]: {cache_error}")
            return False

    def set_json_data(self, key: str, value: Any, expire_seconds: Optional[int] = 3600) -> bool:
        """
        Serializes and pushes nested complex objects or lists into standard JSON tracking frames.
        """
        try:
            serialized_payload = json.dumps(value)
            return bool(self.connection.set(key, serialized_payload, ex=expire_seconds))
        except (ValueError, TypeError) as stringify_fault:
            logger.error(f"Cache serialization anomaly: Failed formatting string representation for key [{key}]: {stringify_fault}")
            return False
        except RedisError as cache_error:
            logger.error(f"Redis bulk json mutation transaction failure for key [{key}]: {cache_error}")
            return False

    def get_data(self, key: str) -> Optional[str]:
        """
        Retrieves a clean, decoded primitive string match for the specified query key.
        """
        try:
            return self.connection.get(key)
        except RedisError as cache_error:
            logger.error(f"Redis tracking lookup exception over key [{key}]: {cache_error}")
            return None

    def get_json_data(self, key: str) -> Optional[Any]:
        """
        Retrieves, decodes, and expands structured nested objects or lists automatically.
        """
        try:
            cached_string = self.connection.get(key)
            if not cached_string:
                return None
            return json.loads(cached_string)
        except json.JSONDecodeError as parsing_fault:
            logger.error(f"Cache layout extraction exception: Corrupted data shape matched at key [{key}]: {parsing_fault}")
            return None
        except RedisError as cache_error:
            logger.error(f"Redis object tracking retrieval exception over key [{key}]: {cache_error}")
            return None

    def delete_data(self, key: str) -> int:
        """
        Deletes a specific data key path from the memory matrix. Returns count of removed items.
        """
        try:
            return self.connection.delete(key)
        except RedisError as cache_error:
            logger.error(f"Redis mutation deletion exception over key [{key}]: {cache_error}")
            return 0
