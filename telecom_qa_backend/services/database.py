"""
services/database.py
====================
Embedded SQLite persistence layer for production query logging,
system metrics, and audit trails.

Zero external dependencies (uses standard library sqlite3).
Enables historical tracking of:
  - Query latency, confidence scores, and tokens
  - Cache hit vs cold pipeline runs
  - Guardrail violation rates (ungrounded numbers, citation hallucinations)
  - Abstention events
"""

import os
import sqlite3
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

DB_PATH = os.environ.get("SQLITE_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "telecom_qa.db"))
DATABASE_URL = os.environ.get("DATABASE_URL")


class DatabaseManager:
    """
    Unified persistence manager for query audit logging and system analytics.
    Supports embedded SQLite by default (zero-config, high performance local)
    and transparently scales to cloud PostgreSQL (Neon/Supabase/RDS) when
    DATABASE_URL is provided in the environment.
    """

    def __init__(self, db_path: str = DB_PATH, database_url: Optional[str] = None):
        self.db_path = db_path
        self.database_url = database_url or DATABASE_URL
        self.is_postgres = bool(self.database_url and ("postgres://" in self.database_url or "postgresql://" in self.database_url))
        self.pg_pool = None
        self._init_db()

    def _get_connection(self):
        if self.is_postgres:
            try:
                import psycopg2
                import psycopg2.extras
                conn = psycopg2.connect(self.database_url)
                return conn
            except ImportError:
                print("[DatabaseManager] psycopg2 not installed; falling back to SQLite.")
                self.is_postgres = False
            except Exception as e:
                print(f"[DatabaseManager] PostgreSQL connection error: {e}. Falling back to SQLite.")
                self.is_postgres = False

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables for query logs and feedback."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            if self.is_postgres:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS query_logs (
                        id SERIAL PRIMARY KEY,
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
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS query_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            cur.execute("CREATE INDEX IF NOT EXISTS idx_session ON query_logs(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON query_logs(created_at)")
            conn.commit()


    def log_query(
        self,
        session_id: str,
        query: str,
        answer: str,
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
                        session_id, query, answer, is_abstained, abstention_reason,
                        confidence_score, cache_hit, latency_ms, citations_count,
                        citations_json, guardrails_json, client_ip
                    ) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}){returning}
                """
                cur.execute(sql, (
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

                if total_queries == 0:
                    return {
                        "total_queries": 0,
                        "cache_hit_rate": 0.0,
                        "abstention_rate": 0.0,
                        "avg_latency_ms": 0.0,
                        "avg_confidence": 0.0,
                        "citation_grounding_rate": 1.0,
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

