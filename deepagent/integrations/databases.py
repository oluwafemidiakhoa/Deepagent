"""
Database Integrations

Provides persistent storage for structured data:
- PostgreSQL for working memory and tool statistics
- Redis for caching and fast lookups

Author: Oluwafemi Idiakhoa
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import json
import os


class DatabaseConnection(ABC):
    """Base class for database connections"""

    @abstractmethod
    def connect(self) -> bool:
        """Establish database connection"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Close database connection"""
        pass

    @abstractmethod
    def save(self, key: str, value: Any) -> bool:
        """Save data"""
        pass

    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """Load data"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data"""
        pass


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL database connection for structured memory storage"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "deepagent",
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "")
        self.connection = None
        self.cursor = None

    def connect(self) -> bool:
        """Connect to PostgreSQL"""
        try:
            import psycopg2

            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.connection.cursor()

            # Create tables if they don't exist
            self._create_tables()

            return True

        except ImportError:
            raise ImportError(
                "psycopg2 not installed. Install with: pip install psycopg2-binary"
            )
        except Exception as e:
            print(f"PostgreSQL connection error: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from PostgreSQL"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            return True
        except Exception:
            return False

    def _create_tables(self):
        """Create necessary tables"""
        # Working memory table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS working_memory (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255),
                subgoal TEXT,
                content TEXT,
                priority INTEGER,
                status VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
        """)

        # Tool statistics table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_stats (
                tool_name VARCHAR(255) PRIMARY KEY,
                total_calls INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                failures INTEGER DEFAULT 0,
                avg_time FLOAT DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Generic key-value store
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key VARCHAR(255) PRIMARY KEY,
                value JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.connection.commit()

    def save(self, key: str, value: Any) -> bool:
        """Save data to key-value store"""
        try:
            self.cursor.execute(
                """
                INSERT INTO kv_store (key, value)
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = %s, timestamp = CURRENT_TIMESTAMP
                """,
                (key, json.dumps(value), json.dumps(value))
            )
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False

    def load(self, key: str) -> Optional[Any]:
        """Load data from key-value store"""
        try:
            self.cursor.execute(
                "SELECT value FROM kv_store WHERE key = %s",
                (key,)
            )
            result = self.cursor.fetchone()
            if result:
                return json.loads(result[0])
            return None
        except Exception as e:
            print(f"Load error: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete data from key-value store"""
        try:
            self.cursor.execute(
                "DELETE FROM kv_store WHERE key = %s",
                (key,)
            )
            self.connection.commit()
            return True
        except Exception:
            return False

    def save_working_memory(self, session_id: str, entries: List[Dict[str, Any]]) -> bool:
        """Save working memory entries"""
        try:
            for entry in entries:
                self.cursor.execute(
                    """
                    INSERT INTO working_memory (session_id, subgoal, content, priority, status, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        entry.get("subgoal", ""),
                        entry.get("content", ""),
                        entry.get("priority", 0),
                        entry.get("status", "active"),
                        json.dumps(entry.get("metadata", {}))
                    )
                )
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Save working memory error: {e}")
            return False

    def load_working_memory(self, session_id: str) -> List[Dict[str, Any]]:
        """Load working memory entries"""
        try:
            self.cursor.execute(
                "SELECT subgoal, content, priority, status, metadata FROM working_memory WHERE session_id = %s",
                (session_id,)
            )
            results = self.cursor.fetchall()
            return [
                {
                    "subgoal": row[0],
                    "content": row[1],
                    "priority": row[2],
                    "status": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {}
                }
                for row in results
            ]
        except Exception as e:
            print(f"Load working memory error: {e}")
            return []

    def update_tool_stats(self, tool_name: str, success: bool, execution_time: float) -> bool:
        """Update tool statistics"""
        try:
            # Get current stats
            self.cursor.execute(
                "SELECT total_calls, successes, failures, avg_time FROM tool_stats WHERE tool_name = %s",
                (tool_name,)
            )
            result = self.cursor.fetchone()

            if result:
                total_calls, successes, failures, avg_time = result
                total_calls += 1
                if success:
                    successes += 1
                else:
                    failures += 1
                avg_time = (avg_time * (total_calls - 1) + execution_time) / total_calls

                self.cursor.execute(
                    """
                    UPDATE tool_stats
                    SET total_calls = %s, successes = %s, failures = %s, avg_time = %s, last_updated = CURRENT_TIMESTAMP
                    WHERE tool_name = %s
                    """,
                    (total_calls, successes, failures, avg_time, tool_name)
                )
            else:
                self.cursor.execute(
                    """
                    INSERT INTO tool_stats (tool_name, total_calls, successes, failures, avg_time)
                    VALUES (%s, 1, %s, %s, %s)
                    """,
                    (tool_name, 1 if success else 0, 0 if success else 1, execution_time)
                )

            self.connection.commit()
            return True
        except Exception as e:
            print(f"Update tool stats error: {e}")
            return False


class RedisConnection(DatabaseConnection):
    """Redis connection for caching and fast lookups"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.client = None

    def connect(self) -> bool:
        """Connect to Redis"""
        try:
            import redis

            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )

            # Test connection
            self.client.ping()
            return True

        except ImportError:
            raise ImportError(
                "redis not installed. Install with: pip install redis"
            )
        except Exception as e:
            print(f"Redis connection error: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from Redis"""
        try:
            if self.client:
                self.client.close()
            return True
        except Exception:
            return False

    def save(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Save data to Redis with optional TTL (time-to-live)"""
        try:
            serialized = json.dumps(value)
            if ttl:
                self.client.setex(key, ttl, serialized)
            else:
                self.client.set(key, serialized)
            return True
        except Exception as e:
            print(f"Redis save error: {e}")
            return False

    def load(self, key: str) -> Optional[Any]:
        """Load data from Redis"""
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Redis load error: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete data from Redis"""
        try:
            self.client.delete(key)
            return True
        except Exception:
            return False

    def cache_embedding(self, text: str, embedding: List[float], ttl: int = 3600) -> bool:
        """Cache embedding with TTL"""
        key = f"embedding:{hash(text)}"
        return self.save(key, embedding, ttl=ttl)

    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = f"embedding:{hash(text)}"
        return self.load(key)
