import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Optional


class DatabaseManager:
    _instance = None
    _async_client: Optional[AsyncIOMotorClient] = None
    _sync_client: Optional[MongoClient] = None
    _database = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    async def connect_async(self):
        """Create async database connection - for FastAPI startup"""
        if self._async_client is None:
            try:
                self._async_client = AsyncIOMotorClient(
                    os.getenv("MONGODB_URI"),
                    maxPoolSize=10,
                    minPoolSize=1,
                    maxIdleTimeMS=30000,
                    serverSelectionTimeoutMS=5000,
                    socketTimeoutMS=20000,
                )
                self._database = self._async_client[os.getenv("DATABASE_NAME")]
                await self._async_client.admin.command('ping')
                print('Successfully connected to Database (async).')
            except Exception as e:
                print(f"Failed to connect to Database (async): {e}")
                raise e

    def connect_sync(self):
        """Create sync database connection - for service layer"""
        if self._sync_client is None:
            try:
                self._sync_client = MongoClient(os.getenv("MONGODB_URI"))
                self._database_sync = self._sync_client[os.getenv("DATABASE_NAME")]
                self._sync_client.admin.command('ping')
                print('Successfully connected to Database (sync).')
            except Exception as e:
                print(f"Failed to connect to Database (sync): {e}")
                raise e

    async def close_async(self):
        """Close async database connection"""
        if self._async_client:
            self._async_client.close()
            self._async_client = None
            print("Disconnected from MongoDB (async)")

    def close_sync(self):
        """Close sync database connection"""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
            print("Disconnected from MongoDB (sync)")

    @property
    def database_async(self):
        """Get async database instance"""
        if self._database is None:
            raise Exception("Async database not connected. Call connect_async() first.")
        return self._database

    @property
    def database_sync(self):
        """Get sync database instance"""
        if not hasattr(self, '_database_sync') or self._database_sync is None:
            self.connect_sync()
        return self._database_sync


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions for getting database
def get_database():
    """Get async database instance"""
    return db_manager.database_async


def get_database_sync():
    """Get sync database instance"""
    return db_manager.database_sync


# Startup and shutdown functions
async def connect_to_mongo():
    """Initialize database connection"""
    await db_manager.connect_async()
    db_manager.connect_sync()


async def close_mongo_connection():
    """Close database connection"""
    await db_manager.close_async()
    db_manager.close_sync()