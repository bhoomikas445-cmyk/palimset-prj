from typing import Optional, List, Tuple
from palimpsest.storage.models import MemoryItem, DecisionLog
from palimpsest.llm.interface import BaseLLMClient
from palimpsest.llm.providers import get_llm_client

class Compressor:
    def __init__(self, llm: Optional[BaseLLMClient] = None):
        self.llm = llm or get_llm_client()

    def summarize_content(self, content: str, current_objective: str = "") -> str:
        """Summarizes text into a high-density, concise memory bullet."""
        prompt = f"""You are PALIMPSEST's memory compressor.
Condense the following conversation or document content into 1-2 concise, fact-dense sentences.
Preserve all key facts, numbers, configurations, and decisions.

CURRENT OBJECTIVE: {current_objective or 'Maintain relevant context'}
Content to summarize:
{content}

Condensed Summary:"""
        try:
            summary = self.llm.generate(prompt=prompt, temperature=0.0).strip()
            if summary.startswith("Condensed Summary:"):
                summary = summary.replace("Condensed Summary:", "").strip()
            return summary or content
        except Exception:
            # Fallback heuristic summary: extract first sentence or 100 characters
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            first_line = lines[0] if lines else content
            return f"Summary: {first_line[:120]}..."

    def process_item(
        self,
        item: MemoryItem,
        current_objective: str = ""
    ) -> Tuple[Optional[MemoryItem], DecisionLog]:
        """
        Processes a single item according to its label:
        - verbatim: keep intact
        - summary: compress content, preserve original_content
        - discard: drop item (return None), but create a decision log
        """
        old_tokens = item.tokens if item.tokens > 0 else self.llm.count_tokens(item.content)

        if item.label == "verbatim":
            log = DecisionLog(
                session_id=item.session_id,
                item_id=item.id,
                action="keep_verbatim",
                metadata_json=f'{{"tokens": {old_tokens}, "is_pinned": {str(item.is_pinned).lower()}}}',
                reason=item.reason or "Retained verbatim."
            )
            return item, log

        elif item.label == "summary":
            # Save original content if not already saved
            if not item.original_content:
                item.original_content = item.content
            
            summary_text = self.summarize_content(item.content, current_objective)
            item.content = summary_text
            item.tokens = self.llm.count_tokens(summary_text)
            compression_ratio = round(item.tokens / max(1, old_tokens), 3)

            log = DecisionLog(
                session_id=item.session_id,
                item_id=item.id,
                action="summarize",
                metadata_json=f'{{"original_tokens": {old_tokens}, "compressed_tokens": {item.tokens}, "ratio": {compression_ratio}}}',
                reason=f"Compressed from {old_tokens} to {item.tokens} tokens ({compression_ratio*100:.1f}%)."
            )
            return item, log

        else: # discard
            log = DecisionLog(
                session_id=item.session_id,
                item_id=item.id,
                action="discard",
                metadata_json=f'{{"dropped_tokens": {old_tokens}, "content_preview": {repr(item.content[:60])}}}',
                reason=item.reason or "Dropped item deemed non-essential."
            )
            return None, log

    def compress_all(
        self,
        items: List[MemoryItem],
        current_objective: str = ""
    ) -> Tuple[List[MemoryItem], List[DecisionLog]]:
        """
        Processes a batch of memory items:
        summarizes 'summary', retains 'verbatim', drops 'discard' while keeping all logs.
        """
        retained: List[MemoryItem] = []
        logs: List[DecisionLog] = []

        for item in items:
            processed_item, log = self.process_item(item, current_objective)
            logs.append(log)
            if processed_item is not None:
                retained.append(processed_item)

        return retained, logs
