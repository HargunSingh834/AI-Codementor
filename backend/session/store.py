"""
In-memory session store keyed by conversation_id.
Each session holds the current problem and conversation history.
"""

from typing import Optional
import os

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# In-memory store: conversation_id -> session dict
_sessions: dict[str, dict] = {}


def _new_session() -> dict:
    return {
        "problem": None,
        "history": [],
    }


def get_session(conversation_id: str) -> dict:
    """Get or create a session for a conversation."""
    if conversation_id not in _sessions:
        _sessions[conversation_id] = _new_session()
    return _sessions[conversation_id]


def set_problem(conversation_id: str, problem: dict) -> None:
    """Store the identified problem in the session."""
    session = get_session(conversation_id)
    session["problem"] = problem


def get_problem(conversation_id: str) -> Optional[dict]:
    """Get the current problem from the session, or None."""
    return get_session(conversation_id)["problem"]


def add_to_history(conversation_id: str, role: str, content: str) -> None:
    """Append a message to conversation history, trimming if needed."""
    session = get_session(conversation_id)
    session["history"].append({"role": role, "content": content})
    # Keep only the last MAX_HISTORY_TURNS turns (each turn = 1 message)
    if len(session["history"]) > MAX_HISTORY_TURNS * 2:
        session["history"] = session["history"][-(MAX_HISTORY_TURNS * 2):]


def get_history(conversation_id: str) -> list[dict]:
    """Return the conversation history."""
    return get_session(conversation_id)["history"]


def clear_session(conversation_id: str) -> None:
    """Clear and reset a session."""
    _sessions[conversation_id] = _new_session()


def has_problem(conversation_id: str) -> bool:
    """Check if the session already has a problem identified."""
    return get_session(conversation_id)["problem"] is not None
