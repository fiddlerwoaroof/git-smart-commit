"""Tests for steropes.store — immutable append-only message store."""

import threading

import pytest

from steropes.store import MessageStore, StoredMessage


class TestStoredMessage:
    def test_frozen(self):
        rec = StoredMessage(
            msg_id="m0", role="user", content="hello",
            message={"role": "user", "content": "hello"}, turn=0,
        )
        with pytest.raises(AttributeError):
            rec.content = "changed"  # type: ignore[misc]

    def test_defaults(self):
        rec = StoredMessage(
            msg_id="m0", role="tool", content="data",
            message={"role": "tool", "content": "data"}, turn=1,
        )
        assert rec.parent_id is None
        assert rec.kind == "original"


class TestMessageStoreAppend:
    def test_sequential_ids(self):
        store = MessageStore()
        id0 = store.append({"role": "system", "content": "sys"})
        id1 = store.append({"role": "user", "content": "hi"})
        id2 = store.append({"role": "assistant", "content": "hello"})
        assert id0 == "m0"
        assert id1 == "m1"
        assert id2 == "m2"

    def test_len(self):
        store = MessageStore()
        assert len(store) == 0
        store.append({"role": "user", "content": "a"})
        store.append({"role": "user", "content": "b"})
        assert len(store) == 2

    def test_stores_full_message_dict(self):
        msg = {"role": "tool", "content": "result", "tool_call_id": "call_123"}
        store = MessageStore()
        mid = store.append(msg)
        rec = store.get(mid)
        assert rec is not None
        assert rec.message == msg
        assert rec.message is not msg  # defensive copy


class TestMessageStoreGet:
    def test_round_trip(self):
        store = MessageStore()
        mid = store.append({"role": "user", "content": "hello world"})
        rec = store.get(mid)
        assert rec is not None
        assert rec.msg_id == mid
        assert rec.content == "hello world"
        assert rec.role == "user"

    def test_missing_returns_none(self):
        store = MessageStore()
        assert store.get("m999") is None

    def test_get_content_shorthand(self):
        store = MessageStore()
        mid = store.append({"role": "user", "content": "data"})
        assert store.get_content(mid) == "data"
        assert store.get_content("m999") is None


class TestGetOriginal:
    def test_follows_parent_chain(self):
        store = MessageStore()
        orig = store.append({"role": "tool", "content": "full result"}, turn=1)
        summary1 = store.append(
            {"role": "meta", "content": "summary of result"},
            turn=2, parent_id=orig, kind="summary",
        )
        summary2 = store.append(
            {"role": "meta", "content": "summary of summary"},
            turn=3, parent_id=summary1, kind="summary",
        )
        # Following from deepest summary should reach original
        root = store.get_original(summary2)
        assert root is not None
        assert root.msg_id == orig
        assert root.content == "full result"

    def test_original_returns_self(self):
        store = MessageStore()
        mid = store.append({"role": "user", "content": "hi"})
        root = store.get_original(mid)
        assert root is not None
        assert root.msg_id == mid

    def test_missing_returns_none(self):
        store = MessageStore()
        assert store.get_original("m999") is None


class TestListIds:
    def test_list_ids_ordered(self):
        store = MessageStore()
        store.append({"role": "system", "content": "s"})
        store.append({"role": "user", "content": "u"})
        store.append({"role": "tool", "content": "t"})
        assert store.list_ids() == ["m0", "m1", "m2"]

    def test_list_ids_by_role(self):
        store = MessageStore()
        store.append({"role": "system", "content": "s"})
        store.append({"role": "user", "content": "u1"})
        store.append({"role": "tool", "content": "t"})
        store.append({"role": "user", "content": "u2"})
        assert store.list_ids_by_role("user") == ["m1", "m3"]
        assert store.list_ids_by_role("tool") == ["m2"]
        assert store.list_ids_by_role("assistant") == []


class TestThreadSafety:
    def test_concurrent_appends(self):
        store = MessageStore()
        errors: list[Exception] = []

        def writer(n: int):
            try:
                for i in range(100):
                    store.append({"role": "user", "content": f"t{n}-{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store) == 400
        # All IDs should be unique
        ids = store.list_ids()
        assert len(set(ids)) == 400
