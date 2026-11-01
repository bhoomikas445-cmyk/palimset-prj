import sqlite3
import json
from typing import List, Optional, Dict, Any
from palimpsest.storage.models import MemoryItem, Version, DecisionLog

class Database:
    def __init__(self, db_path: str = "palimpsest.db"):
        self.db_path = db_path
        self._shared_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:" or "mode=memory" in db_path:
            self._shared_conn = sqlite3.connect(db_path, check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys = ON;")
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Versions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                topic_label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                branch TEXT NOT NULL DEFAULT 'main',
                FOREIGN KEY(parent_id) REFERENCES versions(id)
            );
            """)

            # Memory Items table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                original_content TEXT,
                tokens INTEGER NOT NULL DEFAULT 0,
                importance REAL NOT NULL DEFAULT 0.5,
                relevance REAL NOT NULL DEFAULT 0.5,
                recency REAL NOT NULL DEFAULT 1.0,
                cost_score REAL NOT NULL DEFAULT 0.1,
                timestamp TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT 'verbatim',
                reason TEXT NOT NULL DEFAULT '',
                is_pinned INTEGER NOT NULL DEFAULT 0,
                version_id TEXT,
                FOREIGN KEY(version_id) REFERENCES versions(id)
            );
            """)

            # Decisions Log table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                item_id TEXT,
                action TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                reason TEXT NOT NULL DEFAULT ''
            );
            """)

            # Indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mem_session ON memory_items(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mem_version ON memory_items(version_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dec_session ON decisions_log(session_id);")
            conn.commit()

    # --- Memory Items ---
    def insert_memory_item(self, item: MemoryItem) -> MemoryItem:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO memory_items (
                id, session_id, type, content, original_content, tokens,
                importance, relevance, recency, cost_score, timestamp,
                label, reason, is_pinned, version_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id, item.session_id, item.type, item.content, item.original_content,
                item.tokens, item.importance, item.relevance, item.recency, item.cost_score,
                item.timestamp, item.label, item.reason, 1 if item.is_pinned else 0, item.version_id
            ))
            conn.commit()
        return item

    def update_memory_item(self, item: MemoryItem) -> MemoryItem:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE memory_items SET
                session_id = ?, type = ?, content = ?, original_content = ?,
                tokens = ?, importance = ?, relevance = ?, recency = ?,
                cost_score = ?, timestamp = ?, label = ?, reason = ?,
                is_pinned = ?, version_id = ?
            WHERE id = ?
            """, (
                item.session_id, item.type, item.content, item.original_content,
                item.tokens, item.importance, item.relevance, item.recency,
                item.cost_score, item.timestamp, item.label, item.reason,
                1 if item.is_pinned else 0, item.version_id, item.id
            ))
            conn.commit()
        return item

    def get_memory_item(self, item_id: str) -> Optional[MemoryItem]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_memory_item(row)

    def get_memory_items(
        self,
        session_id: Optional[str] = None,
        version_id: Optional[str] = None,
        active_only: bool = False,
        include_discarded: bool = False
    ) -> List[MemoryItem]:
        query = "SELECT * FROM memory_items WHERE 1=1"
        params = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if version_id:
            query += " AND version_id = ?"
            params.append(version_id)
        elif active_only:
            query += " AND version_id IS NULL"
        if not include_discarded:
            query += " AND label != 'discard'"
        query += " ORDER BY timestamp ASC"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_memory_item(r) for r in rows]

    def pin_memory_item(self, item_id: str, is_pinned: bool) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE memory_items SET is_pinned = ? WHERE id = ?", (1 if is_pinned else 0, item_id))
            conn.commit()
            return cursor.rowcount > 0

    def delete_memory_items_for_session(self, session_id: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memory_items WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM decisions_log WHERE session_id = ?", (session_id,))
            conn.commit()

    # --- Versions ---
    def insert_version(self, version: Version) -> Version:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO versions (id, parent_id, topic_label, created_at, branch)
            VALUES (?, ?, ?, ?, ?)
            """, (version.id, version.parent_id, version.topic_label, version.created_at, version.branch))
            conn.commit()
        return version

    def get_version(self, version_id: str) -> Optional[Version]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM versions WHERE id = ?", (version_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Version(
                id=row["id"],
                parent_id=row["parent_id"],
                topic_label=row["topic_label"],
                created_at=row["created_at"],
                branch=row["branch"]
            )

    def get_all_versions(self) -> List[Version]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM versions ORDER BY created_at ASC")
            rows = cursor.fetchall()
            return [
                Version(
                    id=r["id"],
                    parent_id=r["parent_id"],
                    topic_label=r["topic_label"],
                    created_at=r["created_at"],
                    branch=r["branch"]
                ) for r in rows
            ]

    # --- Decisions Log ---
    def log_decision(self, log: DecisionLog) -> DecisionLog:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO decisions_log (id, timestamp, session_id, item_id, action, metadata_json, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (log.id, log.timestamp, log.session_id, log.item_id, log.action, log.metadata_json, log.reason))
            conn.commit()
        return log

    def get_decision_logs(self, session_id: Optional[str] = None, limit: int = 100) -> List[DecisionLog]:
        query = "SELECT * FROM decisions_log"
        params = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                DecisionLog(
                    id=r["id"],
                    timestamp=r["timestamp"],
                    session_id=r["session_id"],
                    item_id=r["item_id"],
                    action=r["action"],
                    metadata_json=r["metadata_json"],
                    reason=r["reason"]
                ) for r in rows
            ]

    def _row_to_memory_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            session_id=row["session_id"],
            type=row["type"],
            content=row["content"],
            original_content=row["original_content"],
            tokens=row["tokens"],
            importance=row["importance"],
            relevance=row["relevance"],
            recency=row["recency"],
            cost_score=row["cost_score"],
            timestamp=row["timestamp"],
            label=row["label"],
            reason=row["reason"],
            is_pinned=bool(row["is_pinned"]),
            version_id=row["version_id"]
        )
