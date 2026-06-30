import logging
import os
from typing import Dict, Any, List, Optional, Union
import pymongo
from pymongo.errors import PyMongoError, ConnectionFailure, InvalidOperation

logger = logging.getLogger(__name__)


class MongoDbService:
    """
    Enterprise-grade MongoDB data loading engine.
    Implements robust connection state handling, structured logging controls, and transaction validation boundaries.
    """

    def __init__(self, connection_uri: Optional[str] = None, database_name: Optional[str] = None):
        """
        Initializes the MongoDB client infrastructure pool.
        Defaults to environment variable resolution for secure, zero-hardcode configuration.
        """
        # Resolve configurations dynamically from environment variables
        self.uri = connection_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.db_name = database_name or os.getenv("MONGO_DATABASE", "portfolio_db")

        try:
            # Enforce strict 5-second connection timeout boundaries to prevent systemic thread lockups
            self.client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Instantly verify connection health
            self.client.server_info()
            self.db = self.client[self.db_name]
            logger.info(f"Database driver successfully established connection to database: [{self.db_name}]")
        except ConnectionFailure as connection_error:
            logger.critical(
                f"Fatal database initialization error: Failed to connect to MongoDB cluster at URI: {connection_error}")
            raise connection_error

    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> bool:
        """
        Inserts a single document layer cleanly. Returns True on success, False on execution failure.
        """
        if not document:
            logger.warning(f"Aborted write trace: Document passed to collection [{collection_name}] is empty.")
            return False

        try:
            collection = self.db[collection_name]
            collection.insert_one(document)
            return True
        except PyMongoError as database_error:
            logger.error(f"MongoDB write anomaly in collection [{collection_name}]: {database_error}")
            return False

    def insert_many_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> int:
        """
        Executes an optimized batch load over large dataset arrays.
        Returns the count of successfully inserted documents, or 0 on operation failure.
        """
        if not documents:
            return 0

        try:
            collection = self.db[collection_name]
            result = collection.insert_many(documents, ordered=False)
            return len(result.inserted_ids)
        except InvalidOperation as invalid_op_error:
            logger.error(f"Bulk data payload configuration failure in [{collection_name}]: {invalid_op_error}")
            return 0
        except PyMongoError as database_error:
            logger.error(
                f"Bulk loading operation anomaly encountered in collection [{collection_name}]: {database_error}")
            return 0

    def find_document(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single structural document matching the target query parameters.
        """
        try:
            collection = self.db[collection_name]
            return collection.find_one(query)
        except PyMongoError as database_error:
            logger.error(f"Database read query anomaly in collection [{collection_name}]: {database_error}")
            return None

    def update_document(self, collection_name: str, query: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        """
        Mutates structural document configurations via standard $set operators.
        Returns the number of documents successfully matched and updated.
        """
        try:
            collection = self.db[collection_name]
            result = collection.update_one(query, {'$set': update_data})
            return result.modified_count
        except PyMongoError as database_error:
            logger.error(f"Database update transaction anomaly in collection [{collection_name}]: {database_error}")
            return 0

    def delete_document(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Removes documents matching the query parameters. Returns the count of deleted elements.
        """
        try:
            collection = self.db[collection_name]
            result = collection.delete_one(query)
            return result.deleted_count
        except PyMongoError as database_error:
            logger.error(f"Database deletion transaction anomaly in collection [{collection_name}]: {database_error}")
            return 0

    def close(self) -> None:
        """
        Gracefully tears down connection sockets and returns resources to the operating system.
        """
        self.client.close()
        logger.info("MongoDB client session pool closed cleanly.")
