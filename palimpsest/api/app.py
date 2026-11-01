from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from palimpsest.agent.palimpsest_agent import PalimpsestAgent, AgentTurnResult
from palimpsest.storage.models import MemoryItem, Version, DecisionLog
from palimpsest.versioning.version_chain import MergeResult, ConflictInfo

app = FastAPI(
    title="PALIMPSEST API",
    description="Transparent, Version-Controlled Context Memory System for Long-Running LLM Agents",
    version="0.1.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = PalimpsestAgent()

# --- Request / Response Models ---
class ChatRequest(BaseModel):
    session_id: str = "default_session"
    message: str
    current_objective: Optional[str] = "General assistance"
    create_snapshot: bool = False
    snapshot_label: Optional[str] = None
    branch: str = "main"

class PinRequest(BaseModel):
    is_pinned: bool = True

class TrimRequest(BaseModel):
    manual_summary: Optional[str] = None

class BranchRequest(BaseModel):
    new_branch_name: str
    topic_label: Optional[str] = None

class MergeRequest(BaseModel):
    branch_a: str
    branch_b: str
    target_branch: Optional[str] = None
    topic_label: Optional[str] = None

class RestartRequest(BaseModel):
    session_id: str
    topic_label: Optional[str] = None

# --- Endpoints ---

@app.get("/api/health")
def health_check():
    return {"status": "ok", "system": "PALIMPSEST"}

@app.post("/api/chat", response_model=AgentTurnResult)
def chat(req: ChatRequest):
    try:
        result = agent.process_turn(
            session_id=req.session_id,
            user_message=req.message,
            current_objective=req.current_objective or "",
            create_version_after=req.create_snapshot,
            version_label=req.snapshot_label,
            current_branch=req.branch
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory", response_model=List[MemoryItem])
def get_memory(
    session_id: Optional[str] = None,
    version_id: Optional[str] = None,
    include_discarded: bool = False
):
    return agent.db.get_memory_items(
        session_id=session_id,
        version_id=version_id,
        include_discarded=include_discarded
    )

@app.post("/api/memory/{item_id}/pin")
def pin_memory_item(item_id: str, req: PinRequest):
    agent.pin_item(item_id, is_pinned=req.is_pinned)
    return {"status": "success", "item_id": item_id, "is_pinned": req.is_pinned}

@app.post("/api/memory/{item_id}/trim")
def trim_memory_item(item_id: str, req: TrimRequest):
    agent.trim_item(item_id, manual_summary=req.manual_summary)
    return {"status": "success", "item_id": item_id}

@app.post("/api/memory/{item_id}/expand")
def expand_memory_item(item_id: str):
    agent.expand_item(item_id)
    return {"status": "success", "item_id": item_id}

@app.get("/api/versions", response_model=List[Version])
def get_versions():
    return agent.db.get_all_versions()

@app.get("/api/versions/{version_id}/trace", response_model=List[Version])
def trace_version(version_id: str):
    chain = agent.version_chain.trace_back(version_id)
    if not chain:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found.")
    return chain

@app.post("/api/versions/{version_id}/restart")
def restart_version(version_id: str, req: RestartRequest):
    try:
        new_v, items = agent.version_chain.restart(
            version_id=version_id,
            session_id=req.session_id,
            topic_label=req.topic_label
        )
        return {"status": "success", "new_version": new_v, "restored_item_count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/versions/{version_id}/branch", response_model=Version)
def branch_version(version_id: str, req: BranchRequest):
    try:
        return agent.version_chain.branch(
            version_id=version_id,
            new_branch_name=req.new_branch_name,
            topic_label=req.topic_label
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/versions/merge", response_model=MergeResult)
def merge_branches(req: MergeRequest):
    try:
        return agent.version_chain.merge(
            branch_a=req.branch_a,
            branch_b=req.branch_b,
            target_branch=req.target_branch,
            topic_label=req.topic_label
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/decisions", response_model=List[DecisionLog])
def get_decisions(session_id: Optional[str] = None, limit: int = 50):
    return agent.db.get_decision_logs(session_id=session_id, limit=limit)
