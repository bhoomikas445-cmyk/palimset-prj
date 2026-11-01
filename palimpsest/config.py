from pydantic import BaseModel, Field
import os

class PalimpsestConfig(BaseModel):
    db_path: str = os.getenv("PALIMPSEST_DB", "palimpsest.db")
    token_budget: int = 1500  # Target context token budget
    top_k_retrieval: int = 5   # Number of relevant memories to retrieve
    recency_half_life_turns: float = 10.0 # Recency decay half-life in turns
    
    # Cost model per 1,000 tokens (approximate standard agent pricing)
    cost_per_1k_input_tokens: float = 0.0015
    cost_per_1k_output_tokens: float = 0.0020
    
    # Priority weights for heuristic classifier
    weight_importance: float = 0.35
    weight_relevance: float = 0.35
    weight_recency: float = 0.20
    weight_cost_penalty: float = 0.10
    
    # Embedding model name or fallback
    embedding_model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"

default_config = PalimpsestConfig()
