from typing import Optional, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

MemoryType = Literal["message", "document", "tool_output"]
MemoryLabel = Literal["verbatim", "summary", "discard"]

class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    type: MemoryType
    content: str
    original_content: Optional[str] = None
    tokens: int = 0
    importance: float = 0.5
    relevance: float = 0.5
    recency: float = 1.0
    cost_score: float = 0.1
    timestamp: str = Field(default_factory=utc_now_iso)
    label: MemoryLabel = "verbatim"
    reason: str = ""
    is_pinned: bool = False
    version_id: Optional[str] = None

class Version(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    topic_label: str
    created_at: str = Field(default_factory=utc_now_iso)
    branch: str = "main"

class DecisionLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=utc_now_iso)
    session_id: str
    item_id: Optional[str] = None
    action: str  # e.g., 'classify', 'compress', 'discard', 'restore', 'branch', 'merge', 'pin'
    metadata_json: str = "{}"
    reason: str = ""

