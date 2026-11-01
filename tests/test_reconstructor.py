import pytest
from palimpsest.storage.models import MemoryItem
from palimpsest.reconstructor.reconstructor import ContextReconstructor
from palimpsest.llm.mock_llm import MockLLMClient

def test_reconstructor_priority_and_budget():
    llm = MockLLMClient()
    reconstructor = ContextReconstructor(token_budget=80, llm=llm)

    pinned = MemoryItem(
        id="pin-1",
        session_id="s1",
        type="message",
        content="CRITICAL INSTRUCTION: Always use Python 3.14.",
        tokens=10,
        is_pinned=True,
        label="verbatim"
    )

    relevant_verb = MemoryItem(
        id="verb-1",
        session_id="s1",
        type="message",
        content="We configured the database engine with SQLite WAL.",
        tokens=12,
        is_pinned=False,
        label="verbatim"
    )

    summary = MemoryItem(
        id="sum-1",
        session_id="s1",
        type="message",
        content="Long general discussion about cloud deployment architectures and Kubernetes.",
        tokens=14,
        is_pinned=False,
        label="summary",
        importance=0.4,
        recency=0.5
    )

    items = [(pinned, 1.0), (relevant_verb, 0.8), (summary, 0.5)]
    result = reconstructor.assemble(
        query="What is our configuration?",
        retrieved_items=items,
        base_system_prompt="Assistant"
    )

    # Pinned MUST be included
    assert "pin-1" in result.included_item_ids
    assert "CRITICAL INSTRUCTION" in result.context_text
    assert result.total_tokens <= reconstructor.token_budget

def test_reconstructor_drops_low_priority_when_budget_exceeded():
    llm = MockLLMClient()
    # Very tight budget
    reconstructor = ContextReconstructor(token_budget=40, llm=llm)

    pinned = MemoryItem(
        id="pin-1",
        session_id="s1",
        type="message",
        content="USER PREFERENCE: Be brief.",
        tokens=6,
        is_pinned=True,
        label="verbatim"
    )

    bulk_summary = MemoryItem(
        id="sum-heavy",
        session_id="s1",
        type="message",
        content="A" * 300,  # Huge content
        tokens=75,
        is_pinned=False,
        label="summary"
    )

    items = [(pinned, 1.0), (bulk_summary, 0.2)]
    result = reconstructor.assemble(
        query="Hello",
        retrieved_items=items,
        base_system_prompt="AI"
    )

    assert "pin-1" in result.included_item_ids
    assert "sum-heavy" in result.omitted_item_ids
    assert result.total_tokens <= 40
