import json
import time
from typing import List, Dict, Any, Tuple
from palimpsest.agent.palimpsest_agent import PalimpsestAgent
from palimpsest.storage.db import Database
from palimpsest.llm.interface import BaseLLMClient
from palimpsest.llm.providers import get_llm_client
from palimpsest.config import PalimpsestConfig, default_config
from evaluation.synthetic_data import generate_synthetic_conversations, ConversationTurn
from evaluation.baselines import FullHistoryBaseline, RollingSummarizationBaseline

class LLMJudge:
    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    def score_recall(self, question: str, expected_ground_truth: str, actual_answer: str) -> Dict[str, Any]:
        prompt = f"""You are an objective evaluation judge assessing whether an agent successfully recalled and followed early user instructions/preferences.

USER PROBE QUESTION: {question}
EXPECTED GROUND TRUTH: {expected_ground_truth}
ACTUAL ANSWER: {actual_answer}

Evaluate whether the actual answer contains the expected ground truth and adheres to the constraint.
Output MUST be valid JSON with keys:
- "score": integer from 0 (completely forgotten or wrong) to 10 (perfect exact recall and execution)
- "recalled": boolean (true if score >= 7, false otherwise)
- "reason": concise explanation
"""
        raw = self.llm.generate(prompt=prompt, json_mode=True, temperature=0.0)
        try:
            return json.loads(raw)
        except Exception:
            # Deterministic string matching fallback
            recalled = expected_ground_truth.lower() in actual_answer.lower()
            return {
                "score": 10 if recalled else 2,
                "recalled": recalled,
                "reason": "String match ground truth check."
            }

class BenchmarkRunner:
    def __init__(self, llm: BaseLLMClient, config: PalimpsestConfig = default_config):
        self.llm = llm
        self.config = config
        self.judge = LLMJudge(llm)

    def run_benchmark(self) -> Dict[str, Any]:
        sessions = generate_synthetic_conversations()

        # Initialize systems
        full_history = FullHistoryBaseline(self.llm, self.config)
        rolling_summary = RollingSummarizationBaseline(self.llm, self.config)
        
        # Palimpsest agent with isolated in-memory DB
        pal_db = Database(":memory:")
        palimpsest = PalimpsestAgent(db=pal_db, llm=self.llm, config=self.config)

        results = {
            "full_history": {"total_tokens": 0, "total_cost": 0.0, "total_latency": 0.0, "probe_results": []},
            "rolling_summary": {"total_tokens": 0, "total_cost": 0.0, "total_latency": 0.0, "probe_results": []},
            "palimpsest": {"total_tokens": 0, "total_cost": 0.0, "total_latency": 0.0, "probe_results": []}
        }

        workflow_session = "project_titan"

        # Process Sessions 1 through 4 (Context Building)
        for s_idx, session in enumerate(sessions[:4]):
            session_num = s_idx + 1
            for turn in session:
                # Full history
                fh_res = full_history.process_turn(turn.role, turn.content, is_query=False)
                results["full_history"]["total_tokens"] += fh_res["total_tokens"]
                results["full_history"]["total_cost"] += fh_res["cost_usd"]
                results["full_history"]["total_latency"] += fh_res["latency_ms"]

                # Rolling summary
                rs_res = rolling_summary.process_turn(turn.role, turn.content, is_query=False)
                results["rolling_summary"]["total_tokens"] += rs_res["total_tokens"]
                results["rolling_summary"]["total_cost"] += rs_res["cost_usd"]
                results["rolling_summary"]["total_latency"] += rs_res["latency_ms"]

                # Palimpsest: process user inputs into memory
                if turn.role == "user":
                    create_snap = (turn.topic_label != "")
                    pal_res = palimpsest.process_turn(
                        session_id=workflow_session,
                        user_message=turn.content,
                        current_objective="Project Titan Architecture",
                        create_version_after=create_snap,
                        version_label=f"Session {session_num}: {turn.topic_label}" if create_snap else None
                    )
                    results["palimpsest"]["total_tokens"] += (pal_res.input_tokens + pal_res.output_tokens)
                    results["palimpsest"]["total_cost"] += pal_res.estimated_cost_usd
                    results["palimpsest"]["total_latency"] += pal_res.latency_ms

        # Process Session 5 (Recall Probes)
        probe_session = sessions[4]
        for probe in probe_session:
            # 1. Full History
            fh_turn = full_history.process_turn("user", probe.content, is_query=True)
            fh_eval = self.judge.score_recall(probe.content, probe.ground_truth, fh_turn["response"])
            results["full_history"]["total_tokens"] += fh_turn["total_tokens"]
            results["full_history"]["total_cost"] += fh_turn["cost_usd"]
            results["full_history"]["total_latency"] += fh_turn["latency_ms"]
            results["full_history"]["probe_results"].append({
                "question": probe.content,
                "ground_truth": probe.ground_truth,
                "answer": fh_turn["response"],
                "score": fh_eval["score"],
                "recalled": fh_eval["recalled"],
                "tokens": fh_turn["total_tokens"]
            })

            # 2. Rolling Summary
            rs_turn = rolling_summary.process_turn("user", probe.content, is_query=True)
            rs_eval = self.judge.score_recall(probe.content, probe.ground_truth, rs_turn["response"])
            results["rolling_summary"]["total_tokens"] += rs_turn["total_tokens"]
            results["rolling_summary"]["total_cost"] += rs_turn["cost_usd"]
            results["rolling_summary"]["total_latency"] += rs_turn["latency_ms"]
            results["rolling_summary"]["probe_results"].append({
                "question": probe.content,
                "ground_truth": probe.ground_truth,
                "answer": rs_turn["response"],
                "score": rs_eval["score"],
                "recalled": rs_eval["recalled"],
                "tokens": rs_turn["total_tokens"]
            })

            # 3. Palimpsest
            pal_turn = palimpsest.process_turn(
                session_id=workflow_session,
                user_message=probe.content,
                current_objective="Verify project standards & recall"
            )
            pal_eval = self.judge.score_recall(probe.content, probe.ground_truth, pal_turn.agent_response)
            results["palimpsest"]["total_tokens"] += (pal_turn.input_tokens + pal_turn.output_tokens)
            results["palimpsest"]["total_cost"] += pal_turn.estimated_cost_usd
            results["palimpsest"]["total_latency"] += pal_turn.latency_ms
            results["palimpsest"]["probe_results"].append({
                "question": probe.content,
                "ground_truth": probe.ground_truth,
                "answer": pal_turn.agent_response,
                "score": pal_eval["score"],
                "recalled": pal_eval["recalled"],
                "tokens": pal_turn.input_tokens + pal_turn.output_tokens
            })

        # Calculate averages
        for sys_name in results:
            probes = results[sys_name]["probe_results"]
            avg_score = sum(p["score"] for p in probes) / max(1, len(probes))
            recall_rate = sum(1 for p in probes if p["recalled"]) / max(1, len(probes)) * 100.0
            results[sys_name]["avg_score"] = round(avg_score, 2)
            results[sys_name]["recall_rate"] = round(recall_rate, 1)
            results[sys_name]["total_cost"] = round(results[sys_name]["total_cost"], 5)
            results[sys_name]["total_latency"] = round(results[sys_name]["total_latency"], 1)

        return results
