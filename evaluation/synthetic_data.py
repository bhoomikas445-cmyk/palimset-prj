"""
Synthetic multi-session conversational dataset with planted early instructions,
preferences, constraints, technical discussions, and verification recall probes.
"""

from typing import List, Dict, Any
from pydantic import BaseModel

class ConversationTurn(BaseModel):
    session_id: str
    role: str
    content: str
    is_probe: bool = False
    ground_truth: str = ""
    topic_label: str = ""

def generate_synthetic_conversations() -> List[List[ConversationTurn]]:
    """
    Generates 5 sequential sessions with planted constraints in Session 1
    and recall probes in Session 5.
    """
    sessions = []

    # --- SESSION 1: Inception & Core Directives ---
    s1 = [
        ConversationTurn(
            session_id="session_1",
            role="user",
            content="Hello! My name is Dr. Marcus Vance. I am the lead architect on Project Titan, a financial analytics service.",
            topic_label="Introduction"
        ),
        ConversationTurn(
            session_id="session_1",
            role="assistant",
            content="Pleasure to meet you, Dr. Vance. I am ready to assist with Project Titan."
        ),
        ConversationTurn(
            session_id="session_1",
            role="user",
            content="Remember: Our strict instruction is that all financial amounts must always be formatted in EUR with two decimal places (e.g., €1,250.00).",
            topic_label="Constraint: Currency Format"
        ),
        ConversationTurn(
            session_id="session_1",
            role="assistant",
            content="Understood. All financial amounts will be strictly formatted in EUR with two decimal places (€X.XX)."
        ),
        ConversationTurn(
            session_id="session_1",
            role="user",
            content="Also, my preference is that all system timestamps must use strict ISO-8601 formatting.",
            topic_label="Preference: Timestamp Format"
        ),
        ConversationTurn(
            session_id="session_1",
            role="assistant",
            content="Noted: strict ISO-8601 timestamps will be maintained across all specifications."
        ),
        ConversationTurn(
            session_id="session_1",
            role="user",
            content="Crucial security rule: The production encryption key identifier is KEY-TITAN-9921. Never leak or modify this key ID.",
            topic_label="Security Rule: Encryption Key"
        ),
        ConversationTurn(
            session_id="session_1",
            role="assistant",
            content="Understood. The production encryption key ID KEY-TITAN-9921 is stored securely as a mandatory constraint."
        ),
    ]
    sessions.append(s1)

    # --- SESSION 2: Database Schema & Entity Modeling ---
    s2 = [
        ConversationTurn(
            session_id="session_2",
            role="user",
            content="Let's design the ledger table schema in PostgreSQL with partition keys on account_id and transaction_date.",
            topic_label="Data Architecture"
        ),
        ConversationTurn(
            session_id="session_2",
            role="assistant",
            content="I recommend partitioning by RANGE on transaction_date and sub-partitioning by HASH on account_id."
        ),
        ConversationTurn(
            session_id="session_2",
            role="user",
            content="Haha that sounds awesome, thanks! Nice weather today for coding schemas."
        ),
        ConversationTurn(
            session_id="session_2",
            role="assistant",
            content="Glad to help! Let's ensure the foreign key constraints remain performant on partitioned tables."
        ),
        ConversationTurn(
            session_id="session_2",
            role="user",
            content="We should also add an audit_trail table tracking user modifications and IP addresses."
        ),
        ConversationTurn(
            session_id="session_2",
            role="assistant",
            content="Added the audit_trail entity with immutable append-only logs."
        ),
    ]
    sessions.append(s2)

    # --- SESSION 3: API Specification & gRPC Services ---
    s3 = [
        ConversationTurn(
            session_id="session_3",
            role="user",
            content="Let's specify the high-throughput gRPC streaming service for real-time market data ingestion.",
            topic_label="API & Protocols"
        ),
        ConversationTurn(
            session_id="session_3",
            role="assistant",
            content="We can define bidirectional protobuf streams with backpressure handling using token bucket rate limiters."
        ),
        ConversationTurn(
            session_id="session_3",
            role="user",
            content="Good morning! Hope you are having a wonderful day. Cool stuff."
        ),
        ConversationTurn(
            session_id="session_3",
            role="assistant",
            content="Thank you! Let's examine payload compression with snappy or zstd."
        ),
        ConversationTurn(
            session_id="session_3",
            role="user",
            content="Zstandard compression at level 3 is optimal for our network throughput."
        ),
        ConversationTurn(
            session_id="session_3",
            role="assistant",
            content="Configured zstd level 3 for streaming gRPC frames."
        ),
    ]
    sessions.append(s3)

    # --- SESSION 4: Infrastructure & Kubernetes Deployment ---
    s4 = [
        ConversationTurn(
            session_id="session_4",
            role="user",
            content="Let's review our Kubernetes manifest: 4 replicas across 2 availability zones with pod anti-affinity.",
            topic_label="Infrastructure"
        ),
        ConversationTurn(
            session_id="session_4",
            role="assistant",
            content="The pod anti-affinity rule ensures high availability during zone failures."
        ),
        ConversationTurn(
            session_id="session_4",
            role="user",
            content="We'll allocate 2 CPUs and 4Gi memory requests per pod, with horizontal pod autoscaling at 75% CPU."
        ),
        ConversationTurn(
            session_id="session_4",
            role="assistant",
            content="Resource quotas and HPA parameters are registered."
        ),
    ]
    sessions.append(s4)

    # --- SESSION 5: Verification & Early Instruction Recall Probes ---
    s5 = [
        ConversationTurn(
            session_id="session_5",
            role="user",
            content="Can you confirm how financial figures must be formatted, and show an example of displaying 1250?",
            is_probe=True,
            ground_truth="EUR with two decimal places (e.g., €1,250.00)",
            topic_label="Recall Probe: Currency"
        ),
        ConversationTurn(
            session_id="session_5",
            role="user",
            content="What is our production encryption key identifier?",
            is_probe=True,
            ground_truth="KEY-TITAN-9921",
            topic_label="Recall Probe: Key ID"
        ),
        ConversationTurn(
            session_id="session_5",
            role="user",
            content="What timestamp standard did Dr. Marcus Vance specify for the system?",
            is_probe=True,
            ground_truth="ISO-8601",
            topic_label="Recall Probe: Timestamp"
        ),
    ]
    sessions.append(s5)

    return sessions
