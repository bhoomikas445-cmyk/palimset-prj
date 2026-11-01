import uuid
import time
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel
from palimpsest.storage.models import MemoryItem, Version, DecisionLog
from palimpsest.storage.db import Database
from palimpsest.classifier.classifier import Classifier, ClassificationResult
from palimpsest.compressor.compressor import Compressor
from palimpsest.retriever.retriever import Retriever
from palimpsest.reconstructor.reconstructor import ContextReconstructor, ReconstructedContext
from palimpsest.versioning.version_chain import VersionChain, MergeResult
from palimpsest.llm.interface import BaseLLMClient
from palimpsest.llm.providers import get_llm_client
from palimpsest.config import PalimpsestConfig, default_config

class AgentTurnResult(BaseModel):
    session_id: str
    user_query: str
    agent_response: str
    latency_ms: float
    reconstructed_context: ReconstructedContext
    active_memory_count: int
    version_id: Optional[str] = None
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float

class PalimpsestAgent:
    def __init__(
        self,
        db: Optional[Database] = None,
        llm: Optional[BaseLLMClient] = None,
        config: PalimpsestConfig = default_config
    ):
        self.config = config
        self.db = db or Database(self.config.db_path)
        self.llm = llm or get_llm_client()
        self.classifier = Classifier(llm=self.llm, config=self.config)
        self.compressor = Compressor(llm=self.llm)
        self.retriever = Retriever()
        self.reconstructor = ContextReconstructor(token_budget=self.config.token_budget, llm=self.llm)
        self.version_chain = VersionChain(self.db)
        self.turn_counter = 0

    def process_turn(
        self,
        session_id: str,
        user_message: str,
        current_objective: str = "",
        create_version_after: bool = False,
        version_label: Optional[str] = None,
        current_branch: str = "main"
    ) -> AgentTurnResult:
        start_time = time.perf_counter()
        self.turn_counter += 1

        # 1. Fetch active memory items for this session
        active_items = self.db.get_memory_items(session_id=session_id, active_only=True, include_discarded=False)

        # 2. Add User Message as memory item
        user_item = MemoryItem(
            session_id=session_id,
            type="message",
            content=user_message,
            tokens=self.llm.count_tokens(user_message)
        )
        
        # 3. Classify User Item
        clf_result = self.classifier.classify(
            item=user_item,
            current_objective=current_objective,
            turn_index=self.turn_counter,
            total_turns=self.turn_counter + 1
        )
        user_item.importance = clf_result.importance
        user_item.relevance = clf_result.relevance
        user_item.recency = clf_result.recency
        user_item.cost_score = clf_result.cost_score
        user_item.label = clf_result.label
        user_item.reason = clf_result.reason
        user_item.is_pinned = clf_result.is_pinned

        # 4. Process through Compressor
        proc_user_item, user_log = self.compressor.process_item(user_item, current_objective)
        self.db.log_decision(user_log)
        
        if proc_user_item is not None:
            self.db.insert_memory_item(proc_user_item)
            active_items.append(proc_user_item)

        # 5. Retrieve top-k relevant items (pinned are always retrieved)
        retrieved_scored = self.retriever.retrieve(
            query=user_message,
            items=active_items,
            top_k=self.config.top_k_retrieval
        )

        # 6. Reconstruct Context within budget
        reconstructed = self.reconstructor.assemble(
            query=user_message,
            retrieved_items=retrieved_scored,
            all_active_items=active_items,
            base_system_prompt="You are an intelligent agent operating with transparent, version-controlled context memory (PALIMPSEST)."
        )

        # 7. Generate Agent Response
        prompt_full = f"{reconstructed.context_text}\n\nUser Question:\n{user_message}"
        response_text = self.llm.generate(
            prompt=prompt_full,
            system=reconstructed.system_prompt,
            temperature=0.1
        )
        
        output_tokens = self.llm.count_tokens(response_text)
        input_tokens = reconstructed.total_tokens

        # 8. Record Agent Response in memory
        agent_item = MemoryItem(
            session_id=session_id,
            type="message",
            content=f"Assistant: {response_text}",
            tokens=output_tokens
        )
        agent_clf = self.classifier.classify(
            item=agent_item,
            current_objective=current_objective,
            turn_index=self.turn_counter,
            total_turns=self.turn_counter + 1
        )
        agent_item.importance = agent_clf.importance
        agent_item.relevance = agent_clf.relevance
        agent_item.recency = agent_clf.recency
        agent_item.cost_score = agent_clf.cost_score
        agent_item.label = agent_clf.label
        agent_item.reason = agent_clf.reason
        agent_item.is_pinned = agent_clf.is_pinned

        proc_agent_item, agent_log = self.compressor.process_item(agent_item, current_objective)
        self.db.log_decision(agent_log)
        if proc_agent_item is not None:
            self.db.insert_memory_item(proc_agent_item)

        # 9. Optional Version Snapshot
        version_id = None
        if create_version_after or version_label:
            snapshot = self.version_chain.create_snapshot(
                session_id=session_id,
                topic_label=version_label or f"Turn {self.turn_counter} Snapshot",
                branch=current_branch
            )
            version_id = snapshot.id

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        cost_usd = round(
            (input_tokens / 1000.0 * self.config.cost_per_1k_input_tokens) +
            (output_tokens / 1000.0 * self.config.cost_per_1k_output_tokens),
            6
        )

        all_active_now = self.db.get_memory_items(session_id=session_id, active_only=True, include_discarded=False)

        return AgentTurnResult(
            session_id=session_id,
            user_query=user_message,
            agent_response=response_text,
            latency_ms=latency_ms,
            reconstructed_context=reconstructed,
            active_memory_count=len(all_active_now),
            version_id=version_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost_usd
        )

    # Manual Memory Control Operations for UI & API
    def pin_item(self, item_id: str, is_pinned: bool = True):
        self.db.pin_memory_item(item_id, is_pinned)
        self.db.log_decision(DecisionLog(
            session_id="ui_control",
            item_id=item_id,
            action="pin" if is_pinned else "unpin",
            reason=f"User manually {'pinned' if is_pinned else 'unpinned'} item."
        ))

    def trim_item(self, item_id: str, manual_summary: Optional[str] = None):
        item = self.db.get_memory_item(item_id)
        if not item:
            return
        if not item.original_content:
            item.original_content = item.content
        summary = manual_summary or self.compressor.summarize_content(item.content)
        item.content = summary
        item.tokens = self.llm.count_tokens(summary)
        item.label = "summary"
        item.reason = "Manually trimmed/compressed by user."
        self.db.update_memory_item(item)
        self.db.log_decision(DecisionLog(
            session_id=item.session_id,
            item_id=item.id,
            action="manual_trim",
            reason="User manually trimmed item."
        ))

    def expand_item(self, item_id: str):
        item = self.db.get_memory_item(item_id)
        if not item or not item.original_content:
            return
        item.content = item.original_content
        item.tokens = self.llm.count_tokens(item.content)
        item.label = "verbatim"
        item.reason = "Restored to full verbatim original content."
        self.db.update_memory_item(item)
        self.db.log_decision(DecisionLog(
            session_id=item.session_id,
            item_id=item.id,
            action="manual_expand",
            reason="User expanded item back to verbatim original content."
        ))
