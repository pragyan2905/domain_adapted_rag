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


class DatabaseManager:
    """
    Thread-safe SQLite manager for query audit logging and analytics.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables for query logs and feedback."""
        with self._get_connection() as conn:
            conn.execute("""
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER,
                    session_id TEXT,
                    rating INTEGER, -- +1 for helpful, -1 for unhelpful
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_id) REFERENCES query_logs(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON query_logs(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON query_logs(created_at)")
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
                cur.execute("""
                    INSERT INTO query_logs (
                        session_id, query, answer, is_abstained, abstention_reason,
                        confidence_score, cache_hit, latency_ms, citations_count,
                        citations_json, guardrails_json, client_ip
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
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
                total_queries = cur.fetchone()["total"]

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
                cur.execute("SELECT COUNT(*) as hits FROM query_logs WHERE cache_hit = 1")
                cache_hits = cur.fetchone()["hits"]

                # Abstentions
                cur.execute("SELECT COUNT(*) as abstained FROM query_logs WHERE is_abstained = 1")
                abstentions = cur.fetchone()["abstained"]

                # Averages
                cur.execute("SELECT AVG(latency_ms) as avg_lat, AVG(confidence_score) as avg_conf FROM query_logs WHERE is_abstained = 0")
                row = cur.fetchone()
                avg_latency = round(row["avg_lat"] or 0.0, 2)
                avg_conf = round(row["avg_conf"] or 0.0, 3)

                return {
                    "total_queries": total_queries,
                    "cache_hits": cache_hits,
                    "cache_hit_rate": round(cache_hits / total_queries, 3),
                    "abstentions": abstentions,
                    "abstention_rate": round(abstentions / total_queries, 3),
                    "avg_latency_ms": avg_latency,
                    "avg_confidence": avg_conf,
                    "database_path": self.db_path,
                }
        except Exception as e:
            print(f"[DatabaseManager] Metrics retrieval failed: {e}")
            return {"error": str(e)}
