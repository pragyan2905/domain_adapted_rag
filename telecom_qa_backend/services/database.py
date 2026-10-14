"""
services/database.py
====================
Dual-engine persistence layer (SQLite for local zero-config, PostgreSQL for production).
Supports connection pooling to protect free-tier cloud databases from connection storms.

Enables tracking of:
  - User accounts (Google OAuth authentication)
  - Historical query metrics, confidence scores, and token latency
  - Cache hit vs cold pipeline runs
  - Guardrail violation rates and abstention events
"""

import os
import sqlite3
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.environ.get("SQLITE_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "telecom_qa.db"))
DATABASE_URL = os.environ.get("DATABASE_URL")


class DatabaseManager:
    """
    Unified persistence manager for query audit logging, user sessions, and analytics.
    Transparently scales between embedded SQLite and cloud PostgreSQL with connection pooling.
    """

    def __init__(self, db_path: str = DB_PATH, database_url: Optional[str] = None):
        self.db_path = db_path
        self.database_url = database_url or DATABASE_URL
        self.is_postgres = bool(self.database_url and ("postgres://" in self.database_url or "postgresql://" in self.database_url))
        self.pg_pool = None

        if self.is_postgres:
            self._init_pool()

        self._init_db()

    def _init_pool(self):
        """Initializes bounded connection pool for PostgreSQL to prevent connection exhaustion."""
        try:
            from psycopg2.pool import ThreadedConnectionPool
            # Cap at 8 connections to stay safely below free-tier 20-connection limits
            self.pg_pool = ThreadedConnectionPool(minconn=1, maxconn=8, dsn=self.database_url)
            print("[DatabaseManager] Initialized ThreadedConnectionPool for PostgreSQL (max: 8).")
        except ImportError:
            print("[DatabaseManager] psycopg2 not installed; falling back to SQLite.")
            self.is_postgres = False
        except Exception as e:
            print(f"[DatabaseManager] Pool initialization notice: {e}. Falling back to SQLite.")
            self.is_postgres = False

    @contextmanager
    def _get_connection(self):
        """Yields a database connection with guaranteed cleanup and pool return."""
        if self.is_postgres:
            conn = None
            try:
                if self.pg_pool:
                    conn = self.pg_pool.getconn()
                else:
                    import psycopg2
                    conn = psycopg2.connect(self.database_url)
                yield conn
            finally:
                if conn:
                    if self.pg_pool:
                        self.pg_pool.putconn(conn)
                    else:
                        conn.close()
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initializes tables for users, query logs, and feedback."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            if self.is_postgres:
                # Users table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        google_id TEXT UNIQUE NOT NULL,
                        email TEXT NOT NULL,
                        name TEXT,
                        picture_url TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Query logs table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS query_logs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                        session_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        answer TEXT,
                        is_abstained BOOLEAN,
                        abstention_reason TEXT,
                        confidence_score REAL,
                        cache_hit BOOLEAN,
                        latency_ms REAL,
                        citations_count INTEGER,
                        citations_json TEXT,
                        guardrails_json TEXT,
                        client_ip TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Feedback table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id SERIAL PRIMARY KEY,
                        query_id INTEGER REFERENCES query_logs(id),
                        session_id TEXT,
                        rating INTEGER,
                        comment TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                # SQLite Users table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        google_id TEXT UNIQUE NOT NULL,
                        email TEXT NOT NULL,
                        name TEXT,
                        picture_url TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # SQLite Query logs table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS query_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                        session_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        answer TEXT,
                        is_abstained BOOLEAN,
                        abstention_reason TEXT,
                        confidence_score REAL,
                        cache_hit BOOLEAN,
                        latency_ms REAL,
                        citations_count INTEGER,
                        citations_json TEXT,
                        guardrails_json TEXT,
                        client_ip TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # SQLite Feedback table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id INTEGER REFERENCES query_logs(id),
                        session_id TEXT,
                        rating INTEGER,
                        comment TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Migration safeguard: ensure user_id exists if query_logs was created in earlier schema
            if not self.is_postgres:
                try:
                    cur.execute("ALTER TABLE query_logs ADD COLUMN user_id INTEGER REFERENCES users(id)")
                except Exception:
                    pass

            # Indexes for fast lookups
            cur.execute("CREATE INDEX IF NOT EXISTS idx_session ON query_logs(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON query_logs(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id)")
            conn.commit()

    def upsert_user(self, google_id: str, email: str, name: str, picture_url: str = "") -> Dict[str, Any]:
        """Registers a new user or updates existing user's login timestamp."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

                if self.is_postgres:
                    cur.execute("""
                        INSERT INTO users (google_id, email, name, picture_url, last_login_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (google_id) DO UPDATE SET
                            email = EXCLUDED.email,
                            name = EXCLUDED.name,
                            picture_url = EXCLUDED.picture_url,
                            last_login_at = EXCLUDED.last_login_at
                        RETURNING id, google_id, email, name, picture_url, created_at, last_login_at
                    """, (google_id, email, name, picture_url, now_str))
                    row = cur.fetchone()
                    conn.commit()
                    return {
                        "id": row[0],
                        "google_id": row[1],
                        "email": row[2],
                        "name": row[3],
                        "picture_url": row[4],
                        "created_at": str(row[5]),
                        "last_login_at": str(row[6]),
                    }
                else:
                    cur.execute("SELECT id, google_id, email, name, picture_url, created_at FROM users WHERE google_id = ?", (google_id,))
                    existing = cur.fetchone()
                    if existing:
                        cur.execute("""
                            UPDATE users SET email = ?, name = ?, picture_url = ?, last_login_at = ?
                            WHERE google_id = ?
                        """, (email, name, picture_url, now_str, google_id))
                        conn.commit()
                        return {
                            "id": existing[0],
                            "google_id": existing[1],
                            "email": email,
                            "name": name,
                            "picture_url": picture_url,
                            "created_at": str(existing[5]),
                            "last_login_at": now_str,
                        }
                    else:
                        cur.execute("""
                            INSERT INTO users (google_id, email, name, picture_url, last_login_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (google_id, email, name, picture_url, now_str))
                        new_id = cur.lastrowid
                        conn.commit()
                        return {
                            "id": new_id,
                            "google_id": google_id,
                            "email": email,
                            "name": name,
                            "picture_url": picture_url,
                            "created_at": now_str,
                            "last_login_at": now_str,
                        }
        except Exception as e:
            print(f"[DatabaseManager] upsert_user error: {e}")
            return {"id": -1, "google_id": google_id, "email": email, "name": name, "picture_url": picture_url}

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves user profile by internal ID."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                ph = "%s" if self.is_postgres else "?"
                cur.execute(f"SELECT id, google_id, email, name, picture_url, created_at, last_login_at FROM users WHERE id = {ph}", (user_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "google_id": row[1],
                    "email": row[2],
                    "name": row[3],
                    "picture_url": row[4],
                    "created_at": str(row[5]),
                    "last_login_at": str(row[6]),
                }
        except Exception as e:
            print(f"[DatabaseManager] get_user_by_id error: {e}")
            return None

    def log_query(
        self,
        session_id: str,
        query: str,
        answer: str,
        user_id: Optional[int] = None,
        is_abstained: bool = False,
        abstention_reason: Optional[str] = None,
        confidence_score: float = 0.0,
        cache_hit: bool = False,
        latency_ms: float = 0.0,
        citations: Optional[List[str]] = None,
        guardrails: Optional[Dict[str, Any]] = None,
        client_ip: str = "127.0.0.1",
    ) -> int:
        """Logs a completed query transaction. Returns inserted row ID."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                ph = "%s" if self.is_postgres else "?"
                returning = " RETURNING id" if self.is_postgres else ""
                sql = f"""
                    INSERT INTO query_logs (
                        user_id, session_id, query, answer, is_abstained, abstention_reason,
                        confidence_score, cache_hit, latency_ms, citations_count,
                        citations_json, guardrails_json, client_ip
                    ) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}){returning}
                """
                cur.execute(sql, (
                    user_id,
                    session_id,
                    query,
                    answer[:2000] if answer else "",
                    is_abstained,
                    abstention_reason,
                    confidence_score,
                    cache_hit,
                    latency_ms,
                    len(citations or []),
                    json.dumps(citations or []),
                    json.dumps(guardrails or {}),
                    client_ip,
                ))
                conn.commit()
                if self.is_postgres:
                    row = cur.fetchone()
                    return row[0] if row else -1
                return cur.lastrowid
        except Exception as e:
            print(f"[DatabaseManager] Failed to log query: {e}")
            return -1

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Calculates aggregate production analytics for the system."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                
                # Totals
                cur.execute("SELECT COUNT(*) as total FROM query_logs")
                row = cur.fetchone()
                total_queries = row[0] if row else 0

                # Registered Users count
                cur.execute("SELECT COUNT(*) as users_count FROM users")
                u_row = cur.fetchone()
                total_users = u_row[0] if u_row else 0

                if total_queries == 0:
                    return {
                        "total_queries": 0,
                        "total_registered_users": total_users,
                        "cache_hit_rate": 0.0,
                        "abstention_rate": 0.0,
                        "avg_latency_ms": 0.0,
                        "avg_confidence": 0.0,
                        "citation_grounding_rate": 1.0,
                        "database_engine": "PostgreSQL" if self.is_postgres else "SQLite",
                    }

                # Cache hits
                cur.execute("SELECT COUNT(*) as hits FROM query_logs WHERE cache_hit = True" if self.is_postgres else "SELECT COUNT(*) as hits FROM query_logs WHERE cache_hit = 1")
                row = cur.fetchone()
                cache_hits = row[0] if row else 0

                # Abstentions
                cur.execute("SELECT COUNT(*) as abstained FROM query_logs WHERE is_abstained = True" if self.is_postgres else "SELECT COUNT(*) as abstained FROM query_logs WHERE is_abstained = 1")
                row = cur.fetchone()
                abstentions = row[0] if row else 0

                # Averages
                cur.execute("SELECT AVG(latency_ms) as avg_lat, AVG(confidence_score) as avg_conf FROM query_logs WHERE is_abstained = False" if self.is_postgres else "SELECT AVG(latency_ms) as avg_lat, AVG(confidence_score) as avg_conf FROM query_logs WHERE is_abstained = 0")
                row = cur.fetchone()
                avg_latency = round(float(row[0] or 0.0), 2) if row else 0.0
                avg_conf = round(float(row[1] or 0.0), 3) if row else 0.0

                return {
                    "total_queries": total_queries,
                    "total_registered_users": total_users,
                    "cache_hits": cache_hits,
                    "cache_hit_rate": round(cache_hits / total_queries, 3),
                    "abstentions": abstentions,
                    "abstention_rate": round(abstentions / total_queries, 3),
                    "avg_latency_ms": avg_latency,
                    "avg_confidence": avg_conf,
                    "database_engine": "PostgreSQL" if self.is_postgres else "SQLite",
                    "database_path": "Cloud PostgreSQL Instance" if self.is_postgres else self.db_path,
                }
        except Exception as e:
            print(f"[DatabaseManager] Metrics retrieval failed: {e}")
            return {"error": str(e)}
