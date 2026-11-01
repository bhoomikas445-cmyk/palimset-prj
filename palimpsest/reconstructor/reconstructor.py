from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel
from palimpsest.storage.models import MemoryItem
from palimpsest.llm.interface import BaseLLMClient
from palimpsest.llm.providers import get_llm_client

class ReconstructedContext(BaseModel):
    system_prompt: str
    context_text: str
    total_tokens: int
    token_budget: int
    included_item_ids: List[str]
    omitted_item_ids: List[str]
    budget_utilization: float

class ContextReconstructor:
    def __init__(self, token_budget: int = 1500, llm: Optional[BaseLLMClient] = None):
        self.token_budget = token_budget
        self.llm = llm or get_llm_client()

    def assemble(
        self,
        query: str,
        retrieved_items: List[Tuple[MemoryItem, float]],
        all_active_items: Optional[List[MemoryItem]] = None,
        base_system_prompt: str = "You are an intelligent assistant with PALIMPSEST memory."
    ) -> ReconstructedContext:
        """
        Assembles context within the configured token budget adhering to strict priority rules:
        1. Pinned instructions & user constraints (Highest priority)
        2. Retrieved verbatim items (ordered by relevance)
        3. Compressed summaries (ordered by importance * recency)
        """
        # Deduplicate items by ID
        seen_ids = set()
        items_with_scores: List[Tuple[MemoryItem, float]] = []
        for it, score in retrieved_items:
            if it.id not in seen_ids:
                seen_ids.add(it.id)
                items_with_scores.append((it, score))

        # Add any active summaries not in retrieved if budget permits
        if all_active_items:
            for it in all_active_items:
                if it.id not in seen_ids and it.label == "summary":
                    seen_ids.add(it.id)
                    items_with_scores.append((it, 0.2))

        # Separate into priority groups
        tier_1_pinned: List[Tuple[MemoryItem, float]] = []
        tier_2_verbatim: List[Tuple[MemoryItem, float]] = []
        tier_3_summaries: List[Tuple[MemoryItem, float]] = []

        for it, score in items_with_scores:
            if it.is_pinned:
                tier_1_pinned.append((it, score))
            elif it.label == "verbatim":
                tier_2_verbatim.append((it, score))
            elif it.label == "summary":
                tier_3_summaries.append((it, score))

        # Sort tier 2 by relevance descending
        tier_2_verbatim.sort(key=lambda x: x[1], reverse=True)
        
        # Sort tier 3 by (importance * recency) descending
        tier_3_summaries.sort(key=lambda x: (x[0].importance * x[0].recency), reverse=True)

        system_tokens = self.llm.count_tokens(base_system_prompt)
        query_tokens = self.llm.count_tokens(query)
        remaining_budget = max(0, self.token_budget - (system_tokens + query_tokens))

        included: List[MemoryItem] = []
        omitted: List[MemoryItem] = []
        current_tokens = 0

        # Helper to pack items within budget
        def pack_tier(tier_items: List[Tuple[MemoryItem, float]], allow_truncation: bool = False):
            nonlocal current_tokens, remaining_budget
            for it, score in tier_items:
                item_toks = it.tokens if it.tokens > 0 else self.llm.count_tokens(it.content)
                if current_tokens + item_toks <= remaining_budget:
                    included.append(it)
                    current_tokens += item_toks
                elif allow_truncation and remaining_budget - current_tokens > 20:
                    # Truncate content to fit remaining tokens
                    available_tokens = remaining_budget - current_tokens
                    chars_allowed = max(20, available_tokens * 4)
                    truncated_content = it.content[:chars_allowed] + " ...[truncated]"
                    truncated_item = it.model_copy(update={
                        "content": truncated_content,
                        "tokens": self.llm.count_tokens(truncated_content)
                    })
                    included.append(truncated_item)
                    current_tokens += truncated_item.tokens
                else:
                    omitted.append(it)

        # 1. Pack Pinned (Never omit unless physically exceeding total budget)
        pack_tier(tier_1_pinned, allow_truncation=False)
        
        # 2. Pack Retrieved Verbatim
        pack_tier(tier_2_verbatim, allow_truncation=True)

        # 3. Pack Summaries
        pack_tier(tier_3_summaries, allow_truncation=False)

        # Construct final formatted context blocks
        pinned_section = "\n".join([f"- [PINNED] {it.content}" for it in included if it.is_pinned])
        verbatim_section = "\n".join([f"- [VERBATIM] {it.content}" for it in included if not it.is_pinned and it.label == "verbatim"])
        summary_section = "\n".join([f"- [SUMMARY] {it.content}" for it in included if it.label == "summary"])

        blocks = []
        if pinned_section:
            blocks.append(f"<established_rules_and_preferences>\n{pinned_section}\n</established_rules_and_preferences>")
        if summary_section:
            blocks.append(f"<memory_summaries>\n{summary_section}\n</memory_summaries>")
        if verbatim_section:
            blocks.append(f"<relevant_prior_context>\n{verbatim_section}\n</relevant_prior_context>")

        context_text = "\n\n".join(blocks)
        total_tokens = system_tokens + current_tokens + query_tokens
        utilization = round(min(1.0, total_tokens / max(1, self.token_budget)), 3)

        return ReconstructedContext(
            system_prompt=base_system_prompt,
            context_text=context_text,
            total_tokens=total_tokens,
            token_budget=self.token_budget,
            included_item_ids=[it.id for it in included],
            omitted_item_ids=[it.id for it in omitted],
            budget_utilization=utilization
        )
