import json
import re
from typing import Optional
from palimpsest.llm.interface import BaseLLMClient

class MockLLMClient(BaseLLMClient):
    """
    Deterministic simulated LLM for unit tests, offline evaluations,
    and fallback operation when external API keys are not supplied.
    """
    def __init__(self, model_name: str = "mock-gpt-4o"):
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0
    ) -> str:
        prompt_lower = prompt.lower()
        system_lower = (system or "").lower()

        # 1. LLM Judge / Evaluation request (check first)
        if "judge" in prompt_lower or "evaluation judge" in prompt_lower or "probe question" in prompt_lower:
            return self._mock_judge(prompt)

        # 2. Classification request
        if "memory classifier" in prompt_lower or "pre-computed metrics" in prompt_lower or "classify" in prompt_lower:
            return self._mock_classification(prompt)

        # 3. Compression / Summarization request (only when specifically instructed to summarize)
        if "content to summarize:" in prompt_lower or "updated single-paragraph summary:" in prompt_lower or "condensed summary:" in prompt_lower:
            return self._mock_summarization(prompt)

        # 4. Agent answering based on context
        return self._mock_agent_response(prompt)

    def _mock_classification(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        # Extract item content from prompt if present
        match = re.search(r'item content:\s*"""(.*?)"""', prompt_lower, re.DOTALL)
        content_lower = match.group(1).strip() if match else prompt_lower

        # Check if content has explicit user instructions or preferences
        is_instruction_or_pref = any(k in content_lower for k in [
            "prefer", "preference", "must always", "always use", "never use",
            "instruction", "constraint", "my name is", "my api key", "my budget",
            "rule:", "remember:", "strict rule", "format as", "tone:"
        ])
        
        # Check if casual banter or chit-chat
        is_smalltalk = any(k in content_lower for k in [
            "weather is nice", "hello there", "how are you", "good morning",
            "lol", "haha", "thanks!", "bye", "see you later"
        ])

        if is_instruction_or_pref:
            data = {
                "importance": 0.95,
                "relevance": 0.90,
                "recency": 1.0,
                "cost_score": 0.05,
                "label": "verbatim",
                "reason": "Crucial user instruction/preference required verbatim across turns."
            }
        elif is_smalltalk:
            data = {
                "importance": 0.15,
                "relevance": 0.10,
                "recency": 0.5,
                "cost_score": 0.02,
                "label": "discard",
                "reason": "Casual conversational small talk with no operational memory value."
            }
        else:
            # Informational / intermediate results -> summary
            data = {
                "importance": 0.65,
                "relevance": 0.70,
                "recency": 0.8,
                "cost_score": 0.15,
                "label": "summary",
                "reason": "Informative conversation details suitable for compressed context."
            }
        return json.dumps(data)

    def _mock_summarization(self, prompt: str) -> str:
        # Extract text to summarize
        match = re.search(r'Content to summarize:\s*(.*?)(?:\n\n|\Z)', prompt, re.DOTALL | re.IGNORECASE)
        content = match.group(1).strip() if match else prompt
        # Generate a concise summary
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if not lines:
            return "Summary: Discussion of operational details."
        first_line = lines[0][:120]
        return f"Summary: Discussed {first_line}."

    def _mock_judge(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        # Look for whether the answer includes the target ground truth
        match_target = re.search(r'expected ground truth:\s*(.*?)(?:\n|$)', prompt_lower)
        expected = match_target.group(1).strip() if match_target else ""
        
        match_answer = re.search(r'actual answer:\s*(.*?)(?:\n|$)', prompt_lower)
        actual = match_answer.group(1).strip() if match_answer else prompt_lower

        if expected and (expected in actual or any(w in actual for w in expected.split() if len(w) > 3)):
            score = 10
            reason = f"Answer correctly recalled and adhered to target: '{expected}'"
        else:
            score = 2
            reason = f"Answer failed to recall or violated expected target: '{expected}'"

        return json.dumps({
            "score": score,
            "max_score": 10,
            "recalled": score >= 7,
            "reason": reason
        })

    def _mock_agent_response(self, prompt: str) -> str:
        p_lower = prompt.lower()
        q_lower = p_lower
        if "user question:" in p_lower:
            q_lower = p_lower.split("user question:")[-1]
        elif "\nuser:" in p_lower:
            q_lower = p_lower.split("\nuser:")[-1]

        # 1. Encryption Key probe
        if "encryption key" in q_lower or "key identifier" in q_lower:
            if "key-titan-9921" in p_lower:
                return "The production encryption key identifier is KEY-TITAN-9921."
            else:
                return "No production encryption key identifier is found in the current context."

        # 2. Timestamp standard probe
        if "timestamp" in q_lower or "marcus" in q_lower:
            if "iso-8601" in p_lower:
                return "Dr. Marcus Vance specified that all records must use strict ISO-8601 formatting."
            else:
                return "System timestamps default to Unix epochs or standard time."

        # 3. Financial formatting probe
        if "financial" in q_lower or "1250" in q_lower:
            if "eur" in p_lower or "€" in p_lower:
                return "All financial amounts must always be formatted in EUR with two decimal places (e.g., €1,250.00)."
            else:
                return "Amounts are typically displayed as standard numbers, for instance $1,250 or 1250.00."

        # Fraud threshold question
        if "25,000" in q_lower or ("transfer" in q_lower and "25,000" in p_lower):
            if "10,000" in p_lower:
                return "ALERT: The transfer of €25,000 exceeds our strict €10,000 threshold. Flagging immediately for compliance review."
            else:
                return "No specific threshold found in active memory. Proceeding with standard transfer without flagging."
        if "color" in p_lower or "favorite color" in p_lower:
            m = re.search(r'favorite color is (\w+)', p_lower)
            color = m.group(1) if m else "blue"
            return f"Your favorite color is {color}."

        if "language" in p_lower or "preferred language" in p_lower:
            m = re.search(r'preferred language is (\w+)', p_lower)
            lang = m.group(1) if m else ("Python" if uses_python else "Python")
            return f"Your preferred programming language is {lang}."

        if "budget" in p_lower:
            m = re.search(r'budget is \$?([0-9,]+)', p_lower)
            budget = m.group(1) if m else "5000"
            return f"Your allocated project budget is ${budget}."

        return "Understood. I have reviewed the active context and will proceed following your established instructions."
