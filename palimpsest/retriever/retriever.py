import os
import math
import re
from typing import List, Dict, Any, Tuple, Optional
from palimpsest.storage.models import MemoryItem

class EmbeddingModelWrapper:
    """
    Embedding provider supporting sentence-transformers and
    a lightning-fast pure-Python subword/N-gram vectorizer for instant execution.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.st_model = None
        # Only load heavy PyTorch/HuggingFace model if explicitly enabled via env var
        # or if sentence_transformers is requested
        if os.getenv("USE_SENTENCE_TRANSFORMERS", "0").lower() in ("1", "true", "yes"):
            self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.model_name)
        except Exception:
            self.st_model = None

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self.st_model is not None:
            try:
                embeddings = self.st_model.encode(texts, convert_to_numpy=True)
                return [emb.tolist() for emb in embeddings]
            except Exception:
                pass
        
        # Pure Python N-Gram Vectorizer
        return [self._vectorize_text(t) for t in texts]

    def _vectorize_text(self, text: str, dim: int = 128) -> List[float]:
        vec = [0.0] * dim
        tokens = re.findall(r'\w+', text.lower())
        if not tokens:
            return vec
        for t in tokens:
            h = hash(t) % dim
            vec[h] += 1.0
        # 3-gram char tokens for typo and subword robustness
        for i in range(len(text) - 2):
            trigram = text[i:i+3].lower()
            h = hash(trigram) % dim
            vec[h] += 0.5
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))

class Retriever:
    def __init__(self, embedding_model: Optional[EmbeddingModelWrapper] = None):
        self.embedder = embedding_model or EmbeddingModelWrapper()

    def retrieve(
        self,
        query: str,
        items: List[MemoryItem],
        top_k: int = 5
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieves top-k relevant memory items based on query embedding similarity,
        ALWAYS including pinned verbatim items regardless of top_k cutoff.
        Returns list of (item, relevance_score) sorted by relevance.
        """
        if not items:
            return []

        # Partition into pinned items and candidate pool
        pinned_items = [it for it in items if it.is_pinned]
        unpinned_candidates = [it for it in items if not it.is_pinned]

        # Embed query and candidate items
        query_vec = self.embedder.encode([query])[0]
        candidate_texts = [it.content for it in unpinned_candidates]
        candidate_vecs = self.embedder.encode(candidate_texts) if candidate_texts else []

        scored_candidates: List[Tuple[MemoryItem, float]] = []
        for it, vec in zip(unpinned_candidates, candidate_vecs):
            score = cosine_similarity(query_vec, vec)
            scored_candidates.append((it, round(score, 4)))

        # Sort candidates descending by similarity score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[:top_k]

        # Score pinned items for completeness (pinned items always score at least 1.0)
        pinned_scored: List[Tuple[MemoryItem, float]] = []
        pinned_texts = [it.content for it in pinned_items]
        pinned_vecs = self.embedder.encode(pinned_texts) if pinned_texts else []
        for it, vec in zip(pinned_items, pinned_vecs):
            sim = cosine_similarity(query_vec, vec)
            pinned_scored.append((it, round(max(1.0, sim + 1.0), 4)))

        # Combined result: pinned first, followed by top unpinned candidates
        result = pinned_scored + top_candidates
        return result
