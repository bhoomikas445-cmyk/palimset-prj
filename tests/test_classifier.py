import pytest
from palimpsest.storage.models import MemoryItem
from palimpsest.classifier.classifier import Classifier
from palimpsest.llm.mock_llm import MockLLMClient

class FailingLLMClient(MockLLMClient):
    def generate(self, *args, **kwargs) -> str:
        raise RuntimeError("Simulated API failure or timeout")

def test_classifier_keeps_user_instructions_verbatim():
    classifier = Classifier(llm=MockLLMClient())
    item = MemoryItem(
        session_id="sess_1",
        type="message",
        content="Remember: Always use Python 3.14 and format code with black.",
        tokens=12
    )
    result = classifier.classify(item, current_objective="Write a backend script")
    assert result.label == "verbatim"
    assert result.importance >= 0.9
    assert result.is_pinned is True
    assert "verbatim" in result.reason.lower() or "constraint" in result.reason.lower()

def test_classifier_discards_smalltalk():
    classifier = Classifier(llm=MockLLMClient())
    item = MemoryItem(
        session_id="sess_1",
        type="message",
        content="Haha thanks! Good morning, nice weather today.",
        tokens=8
    )
    result = classifier.classify(item, current_objective="Implement database engine")
    assert result.label == "discard"
    assert result.importance <= 0.3
    assert result.is_pinned is False
    assert len(result.reason) > 5

def test_classifier_summarizes_informative_context():
    classifier = Classifier(llm=MockLLMClient())
    item = MemoryItem(
        session_id="sess_1",
        type="message",
        content="We researched SQLite WAL mode and confirmed it improves multi-threaded reads.",
        tokens=14
    )
    result = classifier.classify(item, current_objective="Optimize database performance")
    assert result.label == "summary"
    assert len(result.reason) > 5

def test_classifier_heuristic_fallback():
    # Force failure in LLM client
    failing_llm = FailingLLMClient()
    classifier = Classifier(llm=failing_llm)
    
    # Instruction item
    item_instruction = MemoryItem(
        session_id="sess_1",
        type="message",
        content="My preferred database is SQLite in memory.",
        tokens=9
    )
    result1 = classifier.classify(item_instruction, current_objective="Setup DB")
    assert result1.label == "verbatim"
    assert result1.is_pinned is True
    assert "fallback" in result1.reason.lower()

    # Smalltalk item
    item_smalltalk = MemoryItem(
        session_id="sess_1",
        type="message",
        content="hello there",
        tokens=2
    )
    result2 = classifier.classify(item_smalltalk, current_objective="Setup DB")
    assert result2.label == "discard"
    assert "fallback" in result2.reason.lower()
