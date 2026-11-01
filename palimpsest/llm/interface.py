from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import tiktoken

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0
    ) -> str:
        """Generate response text from LLM."""
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (cl100k_base) or fallback heuristic."""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Approx 4 chars per token fallback
            return max(1, len(text) // 4)
