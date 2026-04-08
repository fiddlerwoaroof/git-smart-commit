"""Immutable append-only message store for lossless compaction."""

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StoredMessage:
    """A single immutable message record.

    Records are append-only: once created, they are never modified.
    Summaries link back to their originals via *parent_id*.
    """

    msg_id: str
    role: str
    content: Any  # str or list (Anthropic structured content)
    message: dict  # full original message dict
    turn: int
    parent_id: str | None = None
    kind: str = "original"  # "original" | "summary"


class MessageStore:
    """Append-only, thread-safe message store.

    Every message appended to the active conversation context is first
    persisted here.  Compaction replaces active-context entries with
    summaries but the originals remain retrievable by ID.
    """

    def __init__(self) -> None:
        self._log: list[StoredMessage] = []
        self._index: dict[str, int] = {}  # msg_id -> position in _log
        self._counter: int = 0
        self._lock = threading.Lock()

    def append(
        self,
        message: dict,
        *,
        turn: int = 0,
        parent_id: str | None = None,
        kind: str = "original",
    ) -> str:
        """Persist a message and return its stable msg_id."""
        role = message.get("role", "unknown")
        content = message.get("content", "")
        with self._lock:
            msg_id = f"m{self._counter}"
            self._counter += 1
            record = StoredMessage(
                msg_id=msg_id,
                role=role,
                content=content,
                message=dict(message),
                turn=turn,
                parent_id=parent_id,
                kind=kind,
            )
            self._log.append(record)
            self._index[msg_id] = len(self._log) - 1
        return msg_id

    def get(self, msg_id: str) -> StoredMessage | None:
        """Retrieve a stored message by ID, or None."""
        with self._lock:
            pos = self._index.get(msg_id)
            if pos is None:
                return None
            return self._log[pos]

    def get_content(self, msg_id: str) -> Any | None:
        """Shorthand: return just the content field, or None."""
        rec = self.get(msg_id)
        return rec.content if rec is not None else None

    def get_original(self, msg_id: str) -> StoredMessage | None:
        """Follow parent_id chain to find the root original record."""
        rec = self.get(msg_id)
        while rec is not None and rec.parent_id is not None:
            rec = self.get(rec.parent_id)
        return rec

    def list_ids(self) -> list[str]:
        """Return all message IDs in append order."""
        with self._lock:
            return [r.msg_id for r in self._log]

    def list_ids_by_role(self, role: str) -> list[str]:
        """Return message IDs filtered by role."""
        with self._lock:
            return [r.msg_id for r in self._log if r.role == role]

    def __len__(self) -> int:
        with self._lock:
            return len(self._log)
