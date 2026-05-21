"""Session history using Lakebase PostgreSQL.

Short-term memory: full conversation per thread_id.
Long-term memory: key-value insights extracted across sessions.
Falls back to in-memory if Lakebase is unavailable.
"""

import json
import logging
import uuid
from datetime import datetime

import psycopg2

from config import LAKEBASE_DB, LAKEBASE_HOST, _cfg

logger = logging.getLogger(__name__)

_PG_USER = _cfg.get("lakebase_user", "vi_app_user")
_PG_PASSWORD = _cfg.get("lakebase_password", "")

# In-memory fallback
_memory_store: dict[str, list[dict]] = {}
_lakebase_available = False


def _get_conn():
    """Get a Lakebase connection using native PG credentials."""
    return psycopg2.connect(
        host=LAKEBASE_HOST,
        port=5432,
        dbname=LAKEBASE_DB,
        user=_PG_USER,
        password=_PG_PASSWORD,
        sslmode="require",
        connect_timeout=10,
    )


def _init_lakebase():
    """Check if Lakebase is reachable."""
    global _lakebase_available
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM threads")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        _lakebase_available = True
        logger.info(f"Lakebase connected. {count} existing threads.")
    except Exception as e:
        logger.error(f"Lakebase init failed: {e}", exc_info=True)
        _lakebase_available = False


_init_lakebase()


# --- Short-term memory (conversation history) ---

def create_session() -> str:
    session_id = str(uuid.uuid4())
    if _lakebase_available:
        try:
            conn = _get_conn()
            cur = conn.cursor()
            cur.execute("INSERT INTO threads (thread_id) VALUES (%s)", (session_id,))
            conn.commit()
            cur.close()
            conn.close()
            return session_id
        except Exception as e:
            logger.warning(f"Lakebase create_session failed: {e}")
    _memory_store[session_id] = []
    return session_id


def add_message(session_id: str, role: str, content: str, tool_calls=None, trace_steps=None):
    if _lakebase_available:
        try:
            conn = _get_conn()
            cur = conn.cursor()
            cur.execute("INSERT INTO threads (thread_id) VALUES (%s) ON CONFLICT DO NOTHING", (session_id,))
            cur.execute(
                "INSERT INTO messages (thread_id, role, content, trace_steps) VALUES (%s, %s, %s, %s)",
                (session_id, role, content, json.dumps(trace_steps) if trace_steps else None),
            )
            conn.commit()
            cur.close()
            conn.close()
            return
        except Exception as e:
            logger.warning(f"Lakebase add_message failed: {e}")

    if session_id not in _memory_store:
        _memory_store[session_id] = []
    _memory_store[session_id].append({
        "role": role, "content": content, "trace_steps": trace_steps,
        "created_at": datetime.now().isoformat(),
    })


def get_history(session_id: str) -> list[dict]:
    if _lakebase_available:
        try:
            conn = _get_conn()
            cur = conn.cursor()
            cur.execute("SELECT role, content FROM messages WHERE thread_id = %s ORDER BY created_at", (session_id,))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [{"role": r[0], "content": r[1]} for r in rows]
        except Exception as e:
            logger.warning(f"Lakebase get_history failed: {e}")
    messages = _memory_store.get(session_id, [])
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def list_sessions() -> list[dict]:
    if _lakebase_available:
        try:
            conn = _get_conn()
            cur = conn.cursor()
            cur.execute("""
                SELECT t.thread_id, t.created_at,
                       (SELECT content FROM messages m WHERE m.thread_id = t.thread_id AND m.role = 'user'
                        ORDER BY m.created_at LIMIT 1) as preview
                FROM threads t
                WHERE EXISTS (SELECT 1 FROM messages m WHERE m.thread_id = t.thread_id)
                ORDER BY t.created_at DESC LIMIT 50
            """)
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [{"session_id": r[0], "created_at": r[1].isoformat() if r[1] else "", "preview": r[2] or ""} for r in rows]
        except Exception as e:
            logger.warning(f"Lakebase list_sessions failed: {e}")

    sessions = []
    for sid, messages in _memory_store.items():
        preview = next((m["content"][:80] for m in messages if m["role"] == "user"), "")
        created = messages[0]["created_at"] if messages else ""
        sessions.append({"session_id": sid, "created_at": created, "preview": preview})
    return sorted(sessions, key=lambda x: x["created_at"], reverse=True)


def delete_session(session_id: str):
    if _lakebase_available:
        try:
            conn = _get_conn()
            cur = conn.cursor()
            cur.execute("DELETE FROM threads WHERE thread_id = %s", (session_id,))
            conn.commit()
            cur.close()
            conn.close()
            return
        except Exception as e:
            logger.warning(f"Lakebase delete_session failed: {e}")
    _memory_store.pop(session_id, None)


# --- Long-term memory (cross-session insights) ---

def store_memory(topic: str, content: str, user_id: str = "default"):
    if not _lakebase_available:
        return
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO memories (user_id, topic, content) VALUES (%s, %s, %s)
            ON CONFLICT (user_id, topic) DO UPDATE SET content = EXCLUDED.content, updated_at = NOW()
        """, (user_id, topic, content))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Lakebase store_memory failed: {e}")


def get_memories(user_id: str = "default") -> list[dict]:
    if not _lakebase_available:
        return []
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT topic, content, updated_at FROM memories WHERE user_id = %s ORDER BY updated_at DESC", (user_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{"topic": r[0], "content": r[1], "updated_at": r[2].isoformat()} for r in rows]
    except Exception as e:
        logger.warning(f"Lakebase get_memories failed: {e}")
    return []
