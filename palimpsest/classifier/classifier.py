import json
import math
import re
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel
from palimpsest.storage.models import MemoryItem, MemoryLabel
from palimpsest.llm.interface import BaseLLMClient
from palimpsest.llm.providers import get_llm_client
from palimpsest.config import PalimpsestConfig, default_config

class ClassificationResult(BaseModel):
    importance: float
    relevance: float
    recency: float
    cost_score: float
    label: MemoryLabel
    reason: str
    is_pinned: bool = False

INSTRUCTION_KEYWORDS = [
    "prefer", "preference", "must always", "always use", "never use",
    "constraint", "instruction", "rule:", "remember:", "strict rule",
    "format as", "my name is", "my budget is", "call me", "do not",
    "guideline", "objective:", "target:"
]

DISCARD_KEYWORDS = [
    "good morning", "hello", "hi there", "how are you", "lol", "haha",
    "cool", "nice", "sounds good", "thanks", "thank you", "bye", "see ya"
]

class Classifier:
    def __init__(
        self,
        llm: Optional[BaseLLMClient] = None,
        config: PalimpsestConfig = default_config
    ):
        self.llm = llm or get_llm_client()
        self.config = config

    def classify(
        self,
        item: MemoryItem,
        current_objective: str = "",
        turn_index: int = 0,
        total_turns: int = 1
    ) -> ClassificationResult:
        """
        Classifies a memory item using LLM with structured JSON output,
        enforcing verbatim preservation for user instructions/preferences,
        and falling back gracefully to heuristic scoring on any failure.
        """
        tokens = item.tokens if item.tokens > 0 else self.llm.count_tokens(item.content)
        # Token cost score (0.0 to 1.0)
        cost_score = min(1.0, round(tokens / 500.0, 3))
        
        # Recency score calculation with decay
        turn_distance = max(0, total_turns - turn_index)
        half_life = self.config.recency_half_life_turns
        recency = round(math.exp(-0.693 * (turn_distance / max(1.0, half_life))), 3)

        # Check for mandatory verbatim rule for instructions/preferences
        is_user_instruction = self._is_user_instruction_or_preference(item)

        try:
            llm_result = self._classify_with_llm(item, current_objective, recency, cost_score)
            if is_user_instruction:
                # Enforce rule: Always keep user instructions and preferences verbatim
                llm_result.label = "verbatim"
                llm_result.importance = max(0.9, llm_result.importance)
                llm_result.is_pinned = True
                if not llm_result.reason.startswith("[Constraint Verified]"):
                    llm_result.reason = f"[Constraint Verified] {llm_result.reason}"
            return llm_result
        except Exception as e:
            # Fallback to heuristic
            return self._heuristic_fallback(
                item, current_objective, recency, cost_score, is_user_instruction, str(e)
            )

    def _is_user_instruction_or_preference(self, item: MemoryItem) -> bool:
        """Determines if the item contains a user constraint, instruction, or preference."""
        content_lower = item.content.lower()
        if item.type == "message" and any(k in content_lower for k in INSTRUCTION_KEYWORDS):
            return True
        return False

    def _classify_with_llm(
        self,
        item: MemoryItem,
        current_objective: str,
        recency: float,
        cost_score: float
    ) -> ClassificationResult:
        prompt = f"""You are PALIMPSEST's memory classifier.
Analyze this context item and score it for long-term agent memory.

CURRENT OBJECTIVE: "{current_objective or 'General assistant operation'}"
ITEM TYPE: {item.type}
ITEM CONTENT:
\"\"\"{item.content}\"\"\"

PRE-COMPUTED METRICS:
- Recency Score: {recency} (1.0 = newest)
- Token Cost Burden: {cost_score} (0.0 = minimal, 1.0 = very heavy)

RULES:
1. Always keep user instructions, constraints, and preferences verbatim with high importance (>=0.9).
2. For discardable greetings, small talk, or redundant acknowledgments, label as "discard".
3. For informative intermediate steps, documents, or context that can be compressed, label as "summary".
4. Output MUST be valid JSON with keys:
   - "importance": float between 0.0 and 1.0
   - "relevance": float between 0.0 and 1.0
   - "label": "verbatim" | "summary" | "discard"
   - "reason": exactly one concise sentence explaining why this label was chosen.
"""
        raw_response = self.llm.generate(prompt=prompt, json_mode=True, temperature=0.0)
        
        # Parse JSON
        parsed = json.loads(raw_response)
        importance = float(parsed.get("importance", 0.5))
        relevance = float(parsed.get("relevance", 0.5))
        label_raw = str(parsed.get("label", "summary")).lower().strip()
        reason = str(parsed.get("reason", "Classified by LLM.")).strip()

        if label_raw not in ["verbatim", "summary", "discard"]:
            label_raw = "summary"

        return ClassificationResult(
            importance=max(0.0, min(1.0, importance)),
            relevance=max(0.0, min(1.0, relevance)),
            recency=recency,
            cost_score=cost_score,
            label=label_raw,  # type: ignore
            reason=reason,
            is_pinned=(label_raw == "verbatim" and importance >= 0.9)
        )

    def _heuristic_fallback(
        self,
        item: MemoryItem,
        current_objective: str,
        recency: float,
        cost_score: float,
        is_user_instruction: bool,
        error_context: str
    ) -> ClassificationResult:
        content_lower = item.content.lower()

        if is_user_instruction:
            return ClassificationResult(
                importance=0.95,
                relevance=0.90,
                recency=recency,
                cost_score=cost_score,
                label="verbatim",
                reason="Always keep user instructions and preferences verbatim (heuristic fallback).",
                is_pinned=True
            )

        # Check for small talk / discard
        is_smalltalk = any(w in content_lower for w in DISCARD_KEYWORDS) and len(content_lower.split()) < 10
        if is_smalltalk:
            return ClassificationResult(
                importance=0.10,
                relevance=0.10,
                recency=recency,
                cost_score=cost_score,
                label="discard",
                reason="Conversational filler without long-term task relevance (heuristic fallback).",
                is_pinned=False
            )

        # Measure lexical relevance to objective
        relevance = 0.5
        if current_objective:
            obj_words = set(re.findall(r'\w+', current_objective.lower()))
            content_words = set(re.findall(r'\w+', content_lower))
            overlap = len(obj_words & content_words)
            relevance = min(1.0, overlap / max(1, len(obj_words)))

        importance = 0.6 if relevance > 0.4 else 0.4
        
        # If very high relevance or recent, verbatim, else summary
        label: MemoryLabel = "verbatim" if (relevance >= 0.7 and recency >= 0.8) else "summary"
        
        return ClassificationResult(
            importance=round(importance, 2),
            relevance=round(relevance, 2),
            recency=recency,
            cost_score=cost_score,
            label=label,
            reason=f"Heuristic fallback: importance={importance}, relevance={relevance}.",
            is_pinned=False
        )
