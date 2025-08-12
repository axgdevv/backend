import os
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional

class DatabaseManager:
    _instance = None
    _client: Optional[AsyncIOMotorClient] = None
    _database = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    async def connect(self):
        """Create database connection - should be called once at startup"""
        if self._client is None:
            try:
                self._client = AsyncIOMotorClient(os.getenv("MONGODB_URI"),
                    maxPoolSize=10,
                    minPoolSize=1,
                    maxIdleTimeMS=30000,
                    serverSelectionTimeoutMS=5000,
                    socketTimeoutMS=20000,
                )
                self._database = self._client[os.getenv("DATABASE_NAME")]

                # Test the connection
                await self._client.admin.command('ping')
                print('Successfully connected to Database.')

            except Exception as e:
                print(f"Failed to connect to Database: {e}")
                raise e

    async def close(self):
        """Close database connection - should be called at shutdown"""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            print("Disconnected from MongoDB")

    @property
    def database(self):
        """Get database instance"""
        if self._database is None:
            raise Exception("Database not connected. Call connect() first.")
        return self._database

    @property
    def client(self):
        """Get client instance"""
        if self._client is None:
            raise Exception("Database client not connected. Call connect() first.")
        return self._client


# Global database manager instance
db_manager = DatabaseManager()

# Convenience function for getting database
def get_database():
    """Get database instance - synchronous function"""
    return db_manager.database


# Startup and shutdown functions
async def connect_to_mongo():
    """Initialize database connection"""
    await db_manager.connect()

async def close_mongo_connection():
    """Close database connection"""
    await db_manager.close()