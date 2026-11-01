import os
import pytest
from palimpsest.storage.models import MemoryItem, Version, DecisionLog
from palimpsest.storage.db import Database

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_palimpsest.db"
    return Database(str(db_file))

def test_database_initialization(temp_db):
    assert os.path.exists(temp_db.db_path)
    assert temp_db.get_all_versions() == []
    assert temp_db.get_memory_items() == []

def test_version_crud(temp_db):
    v1 = Version(topic_label="Initial Session", branch="main")
    created = temp_db.insert_version(v1)
    assert created.id == v1.id

    fetched = temp_db.get_version(v1.id)
    assert fetched is not None
    assert fetched.topic_label == "Initial Session"
    assert fetched.branch == "main"
    assert fetched.parent_id is None

def test_memory_items_crud(temp_db):
    v1 = temp_db.insert_version(Version(topic_label="Session 1"))
    
    item1 = MemoryItem(
        session_id="sess_1",
        type="message",
        content="I prefer Python and concise responses.",
        tokens=8,
        importance=0.9,
        relevance=0.8,
        recency=1.0,
        cost_score=0.01,
        label="verbatim",
        reason="User constraint/preference",
        is_pinned=True,
        version_id=v1.id
    )
    temp_db.insert_memory_item(item1)
    
    item2 = MemoryItem(
        session_id="sess_1",
        type="message",
        content="Random banter about the weather.",
        tokens=6,
        importance=0.1,
        relevance=0.2,
        recency=0.9,
        cost_score=0.01,
        label="discard",
        reason="Small talk with no task value",
        is_pinned=False,
        version_id=v1.id
    )
    temp_db.insert_memory_item(item2)

    # By default, discard is excluded
    active_items = temp_db.get_memory_items(session_id="sess_1", include_discarded=False)
    assert len(active_items) == 1
    assert active_items[0].content == item1.content
    assert active_items[0].is_pinned is True

    # When including discarded
    all_items = temp_db.get_memory_items(session_id="sess_1", include_discarded=True)
    assert len(all_items) == 2

def test_pin_and_decisions(temp_db):
    v1 = temp_db.insert_version(Version(topic_label="Session 1"))
    item = temp_db.insert_memory_item(
        MemoryItem(
            session_id="sess_1",
            type="message",
            content="Important guideline",
            is_pinned=False,
            version_id=v1.id
        )
    )
    temp_db.pin_memory_item(item.id, True)
    updated = temp_db.get_memory_item(item.id)
    assert updated.is_pinned is True

    dlog = DecisionLog(
        session_id="sess_1",
        item_id=item.id,
        action="pin",
        metadata_json='{"manual": true}',
        reason="User manually pinned"
    )
    temp_db.log_decision(dlog)
    logs = temp_db.get_decision_logs(session_id="sess_1")
    assert len(logs) == 1
    assert logs[0].action == "pin"
