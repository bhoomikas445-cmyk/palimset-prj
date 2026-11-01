import time
from typing import List, Dict, Any, Tuple
from palimpsest.llm.interface import BaseLLMClient
from palimpsest.llm.providers import get_llm_client
from palimpsest.config import PalimpsestConfig, default_config

class FullHistoryBaseline:
    """
    Baseline 1: Full-history prompting.
    Maintains all previous turns verbatim in the context.
    Context size grows unbounded with conversation length.
    """
    def __init__(self, llm: BaseLLMClient, config: PalimpsestConfig = default_config):
        self.llm = llm
        self.config = config
        self.history: List[Tuple[str, str]] = [] # (role, content)

    def process_turn(self, role: str, content: str, is_query: bool = False) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        # Build full context
        formatted_history = "\n".join([f"{r.capitalize()}: {c}" for r, c in self.history])
        prompt = f"""You are an assistant. Use the full prior conversation history to respond.

CONVERSATION HISTORY:
{formatted_history}

User: {content}
Assistant:"""

        input_tokens = self.llm.count_tokens(prompt)
        response = self.llm.generate(prompt=prompt, temperature=0.1) if is_query else "Acknowledged."
        output_tokens = self.llm.count_tokens(response)

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        cost_usd = round(
            (input_tokens / 1000.0 * self.config.cost_per_1k_input_tokens) +
            (output_tokens / 1000.0 * self.config.cost_per_1k_output_tokens),
            6
        )

        self.history.append((role, content))
        if is_query:
            self.history.append(("assistant", response))

        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd
        }


class RollingSummarizationBaseline:
    """
    Baseline 2: Plain rolling summarization.
    Compresses all prior context into a single rolling summary paragraph.
    No classification, no verbatim pinning, no versioning.
    Subject to detail loss and instruction drift.
    """
    def __init__(self, llm: BaseLLMClient, config: PalimpsestConfig = default_config):
        self.llm = llm
        self.config = config
        self.running_summary: str = ""

    def process_turn(self, role: str, content: str, is_query: bool = False) -> Dict[str, Any]:
        start_time = time.perf_counter()

        prompt = f"""You are an assistant with rolling memory.

ROLLING SUMMARY OF PRIOR CONVERSATION:
{self.running_summary or '[No previous summary]'}

User: {content}
Assistant:"""

        input_tokens = self.llm.count_tokens(prompt)
        response = self.llm.generate(prompt=prompt, temperature=0.1) if is_query else "Acknowledged."
        output_tokens = self.llm.count_tokens(response)

        # Update the rolling summary naively
        summary_prompt = f"""Update this rolling summary with the new turn:
Current Summary: {self.running_summary}
New {role}: {content}

Updated Single-Paragraph Summary:"""
        try:
            self.running_summary = self.llm.generate(prompt=summary_prompt, temperature=0.0).strip()
        except Exception:
            self.running_summary += f" Also discussed {content[:60]}."

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        cost_usd = round(
            (input_tokens / 1000.0 * self.config.cost_per_1k_input_tokens) +
            (output_tokens / 1000.0 * self.config.cost_per_1k_output_tokens),
            6
        )

        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd
        }
