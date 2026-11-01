import pytest
from palimpsest.storage.db import Database
from palimpsest.storage.models import MemoryItem, Version
from palimpsest.versioning.version_chain import VersionChain

@pytest.fixture
def temp_vchain(tmp_path):
    db_file = tmp_path / "test_version_chain.db"
    db = Database(str(db_file))
    return VersionChain(db)

def test_trace_back(temp_vchain):
    v1 = temp_vchain.db.insert_version(Version(topic_label="Root Version", branch="main"))
    v2 = temp_vchain.db.insert_version(Version(parent_id=v1.id, topic_label="Intermediate", branch="main"))
    v3 = temp_vchain.db.insert_version(Version(parent_id=v2.id, topic_label="Current Tip", branch="main"))

    chain = temp_vchain.trace_back(v3.id)
    assert len(chain) == 3
    assert chain[0].id == v1.id
    assert chain[1].id == v2.id
    assert chain[2].id == v3.id

def test_restart_restores_prior_memory(temp_vchain):
    # Setup v1 with 2 items
    v1 = temp_vchain.db.insert_version(Version(topic_label="Checkpoint 1", branch="main"))
    item1 = temp_vchain.db.insert_memory_item(MemoryItem(
        session_id="s1", type="message", content="User instruction A", version_id=v1.id, is_pinned=True
    ))
    item2 = temp_vchain.db.insert_memory_item(MemoryItem(
        session_id="s1", type="message", content="User instruction B", version_id=v1.id, is_pinned=True
    ))

    # Later session has corrupted/divergent state
    v2 = temp_vchain.db.insert_version(Version(parent_id=v1.id, topic_label="Corrupted State", branch="main"))
    temp_vchain.db.insert_memory_item(MemoryItem(
        session_id="s2", type="message", content="Bad forgetful state", version_id=v2.id
    ))

    # Restart from v1
    restarted_version, restored_items = temp_vchain.restart(v1.id, session_id="s3")
    assert restarted_version.parent_id == v1.id
    assert len(restored_items) == 2
    contents = [it.content for it in restored_items]
    assert "User instruction A" in contents
    assert "User instruction B" in contents

def test_branch_and_merge_prefer_verbatim():
    db = Database(":memory:")
    vc = VersionChain(db)

    # Base snapshot
    base_v = db.insert_version(Version(topic_label="Base", branch="main"))
    db.insert_memory_item(MemoryItem(
        session_id="s1", type="message", content="Core rule: Always output JSON.", version_id=base_v.id, label="verbatim", is_pinned=True
    ))

    # Branch A: keeps verbatim rule
    branch_a_v = vc.branch(base_v.id, new_branch_name="branch_a", topic_label="Feature A")
    # Add a verbatim item to branch A
    db.insert_memory_item(MemoryItem(
        session_id="s_a", type="message", content="Database is configured on port 5432.", version_id=branch_a_v.id, label="verbatim"
    ))

    # Branch B: summarizes that same database item
    branch_b_v = vc.branch(base_v.id, new_branch_name="branch_b", topic_label="Feature B")
    db.insert_memory_item(MemoryItem(
        session_id="s_b", type="message", content="Summary: Database configured on port 5432.", original_content="Database is configured on port 5432.", version_id=branch_b_v.id, label="summary"
    ))

    # Merge branch_b into branch_a
    result = vc.merge(branch_a="branch_a", branch_b="branch_b", target_branch="branch_a")
    assert result.merged_version.branch == "branch_a"
    
    # Conflict resolution should prefer verbatim
    contents = [it.content for it in result.merged_items]
    labels = [it.label for it in result.merged_items]
    assert "Database is configured on port 5432." in contents
    assert "verbatim" in labels

def test_merge_flags_contradictory_conflicts():
    db = Database(":memory:")
    vc = VersionChain(db)

    # Base version
    base_v = db.insert_version(Version(topic_label="Base", branch="main"))

    # Branch A specifies port 8000
    va = vc.branch(base_v.id, new_branch_name="api_a")
    db.insert_memory_item(MemoryItem(
        session_id="s_a", type="message", content="API port is set to 8000.", original_content="API port config", version_id=va.id, label="verbatim", importance=0.8
    ))

    # Branch B specifies port 9000
    vb = vc.branch(base_v.id, new_branch_name="api_b")
    db.insert_memory_item(MemoryItem(
        session_id="s_b", type="message", content="API port is set to 9000.", original_content="API port config", version_id=vb.id, label="verbatim", importance=0.9
    ))

    result = vc.merge(branch_a="api_a", branch_b="api_b")
    assert len(result.conflicts) > 0
    assert any(c.flagged for c in result.conflicts)
