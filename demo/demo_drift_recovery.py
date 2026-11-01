"""
PALIMPSEST DRIFT-RECOVERY DEMO
Demonstrates:
1. Objective & constraint established (Snapshot V1).
2. Drift Injection: A bad summary/discard causes the agent to forget its core directive.
3. Drift Detection: Automated audit detects missing critical constraint.
4. Transparent Rollback: Agent restores memory state to Snapshot V1 using version_chain.restart().
5. Resumption: Agent successfully recalls and executes original directive.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
from palimpsest.agent.palimpsest_agent import PalimpsestAgent
from palimpsest.storage.db import Database
from palimpsest.storage.models import MemoryItem

def run_drift_recovery_demo():
    print("=" * 75)
    print("PALIMPSEST: OBJECTIVE DRIFT DETECTION & VERSION ROLLBACK DEMO")
    print("=" * 75)

    # Isolated test database
    db = Database(":memory:")
    agent = PalimpsestAgent(db=db)
    session_id = "drift_demo_sess"

    # -------------------------------------------------------------
    # STEP 1: Establish Core Objective & Create Checkpoint V1
    # -------------------------------------------------------------
    print("\n[STEP 1] User establishes core objective and compliance constraint...")
    
    msg_1 = "Objective: Implement crypto payment fraud monitoring. Strict rule: Flag all transfers over €10,000 immediately."
    t1 = agent.process_turn(
        session_id=session_id,
        user_message=msg_1,
        current_objective="Crypto Fraud Pipeline",
        create_version_after=True,
        version_label="V1: Core Compliance Rule Established"
    )
    v1_id = t1.version_id
    print(f"  -> Agent Response: {t1.agent_response}")
    print(f"  -> Version Snapshot Created: ID={v1_id[:8]} ('V1: Core Compliance Rule Established')")
    print(f"  -> Active Memory Items: {t1.active_memory_count}")

    # -------------------------------------------------------------
    # STEP 2: Drift Injection (Corrupt Summary / Accidental Discard)
    # -------------------------------------------------------------
    print("\n[STEP 2] Simulating Drift: Replacing rule with a lossy/corrupted summary...")
    
    # Locate all active memory items containing the constraint and corrupt them
    active_items = db.get_memory_items(session_id=session_id, include_discarded=False)
    rule_items = [it for it in active_items if "10,000" in it.content]

    # Simulate an external bad summarization that drops the threshold
    bad_summary = "Summary: General discussion about payment monitoring; no specific threshold specified."
    for rule_item in rule_items:
        rule_item.content = bad_summary
        rule_item.label = "summary"
        rule_item.is_pinned = False
        rule_item.reason = "Bad lossy rolling compression corrupted constraint."
        db.update_memory_item(rule_item)
    
    # Create a corrupted snapshot V2
    snap_v2 = agent.version_chain.create_snapshot(
        session_id=session_id,
        topic_label="V2: Corrupted Memory State (Drifted)",
        parent_id=v1_id
    )
    print(f"  -> Corrupted Memory State Created: ID={snap_v2.id[:8]}")

    # Now query the agent with the corrupted memory
    probe_query = "A customer initiated a transfer of €25,000. What is our required action?"
    print(f"\n  User Query: '{probe_query}'")
    t2 = agent.process_turn(
        session_id=session_id,
        user_message=probe_query,
        current_objective="Crypto Fraud Pipeline"
    )
    print(f"  -> DRIFTED Agent Response: '{t2.agent_response}'")

    # -------------------------------------------------------------
    # STEP 3: Automated Drift Detection
    # -------------------------------------------------------------
    print("\n[STEP 3] Running Automated Objective & Constraint Verification...")
    
    # Check active memory items for the required €10,000 rule
    current_memories = db.get_memory_items(session_id=session_id, include_discarded=False)
    has_threshold = any("10,000" in it.content for it in current_memories)

    if not has_threshold:
        print("  [ALERT DETECTED]: Critical constraint '€10,000' missing from active context!")
        print("  Cause: Lossy summary in Version V2 wiped out mandatory threshold.")
    else:
        print("  Constraint found.")

    # -------------------------------------------------------------
    # STEP 4: Version Chain Traceback & Rollback to V1
    # -------------------------------------------------------------
    print("\n[STEP 4] Executing PALIMPSEST Version Rollback...")
    print(f"  Tracing lineage from V2 ({snap_v2.id[:8]}):")
    lineage = agent.version_chain.trace_back(snap_v2.id)
    for v in lineage:
        print(f"    - Version [{v.id[:8]}] : '{v.topic_label}'")

    print(f"\n  Restoring memory state back to V1 [{v1_id[:8]}]...")
    new_v, restored = agent.version_chain.restart(
        version_id=v1_id,
        session_id="drift_recovered_sess",
        topic_label="V3: Restored from V1 (Drift Corrected)"
    )
    print(f"  -> State Restored successfully! {len(restored)} items recovered to active context.")

    # -------------------------------------------------------------
    # STEP 5: Confirm Agent Resumes Original Task Correctly
    # -------------------------------------------------------------
    print("\n[STEP 5] Re-testing agent on the original query with restored memory...")
    t3 = agent.process_turn(
        session_id="drift_recovered_sess",
        user_message=probe_query,
        current_objective="Crypto Fraud Pipeline"
    )
    print(f"  User Query: '{probe_query}'")
    print(f"  -> RESTORED Agent Response: '{t3.agent_response}'")

    # Verification assertion
    recovered_items = db.get_memory_items(session_id="drift_recovered_sess", include_discarded=False)
    recovered_rule = any("10,000" in it.content for it in recovered_items)
    print("\n" + "=" * 75)
    if recovered_rule:
        print("DEMO RESULT: SUCCESS! Drift detected, version restored, original objective resumed.")
    else:
        print("DEMO RESULT: FAILED.")
    print("=" * 75)

if __name__ == "__main__":
    run_drift_recovery_demo()
