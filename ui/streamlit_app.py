import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from palimpsest.agent.palimpsest_agent import PalimpsestAgent
from palimpsest.storage.models import MemoryItem, Version, DecisionLog
from palimpsest.config import default_config

st.set_page_config(
    page_title="PALIMPSEST - Transparent Context Memory",
    page_icon="📜",
    layout="wide"
)

# Initialize Agent in Streamlit Session State
if "agent" not in st.session_state:
    st.session_state.agent = PalimpsestAgent()
if "session_id" not in st.session_state:
    st.session_state.session_id = "session_alpha"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_branch" not in st.session_state:
    st.session_state.current_branch = "main"

agent: PalimpsestAgent = st.session_state.agent
session_id = st.session_state.session_id

# --- Sidebar ---
st.sidebar.title("📜 PALIMPSEST")
st.sidebar.caption("Transparent, Version-Controlled Context Memory for Long-Running LLM Agents")

st.session_state.session_id = st.sidebar.text_input("Active Session ID", value=st.session_state.session_id)
current_objective = st.sidebar.text_input("Current Agent Objective", value="Build a high-performance backend system")
current_branch = st.sidebar.text_input("Active Branch", value=st.session_state.current_branch)
st.session_state.current_branch = current_branch

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Memory Budget & Weights")
token_budget = st.sidebar.slider("Context Token Budget", min_value=200, max_value=4000, value=agent.config.token_budget, step=100)
agent.config.token_budget = token_budget
agent.reconstructor.token_budget = token_budget

# --- Main App Layout ---
tab_chat, tab_memory, tab_versions, tab_decisions = st.tabs([
    "💬 Agent Chat",
    "🧠 Memory Inspector & Controls",
    "🌲 Version Graph & Operations",
    "📋 Decisions & Audit Log"
])

# ==========================================
# TAB 1: AGENT CHAT
# ==========================================
with tab_chat:
    st.subheader("Interactive Agent Chat")
    st.caption("Send prompts, instruct the agent, and observe real-time classification, compression, and retrieval.")

    col_chat, col_inspect = st.columns([1.2, 0.8])

    with col_chat:
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "meta" in msg:
                    st.caption(f"⏱️ {msg['meta'].get('latency_ms', 0)}ms | 🪙 {msg['meta'].get('tokens', 0)} tokens | 💵 ${msg['meta'].get('cost', 0):.5f}")

        # Chat inputs
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Your message or instruction:", placeholder="e.g. Remember: Always use Python 3.14 and write concise tests.")
            c1, c2 = st.columns([1, 1])
            with c1:
                take_snapshot = st.checkbox("Create Version Snapshot after turn", value=False)
            with c2:
                snapshot_label = st.text_input("Snapshot Milestone Label", placeholder="e.g. Session 1: Requirements Established")
            submitted = st.form_submit_button("Send to Agent", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Processing memory, classifying, compressing & reconstructing context..."):
                turn_result = agent.process_turn(
                    session_id=session_id,
                    user_message=user_input,
                    current_objective=current_objective,
                    create_version_after=take_snapshot,
                    version_label=snapshot_label if take_snapshot else None,
                    current_branch=st.session_state.current_branch
                )
            
            st.session_state.last_turn = turn_result
            st.session_state.messages.append({
                "role": "assistant",
                "content": turn_result.agent_response,
                "meta": {
                    "latency_ms": turn_result.latency_ms,
                    "tokens": turn_result.input_tokens + turn_result.output_tokens,
                    "cost": turn_result.estimated_cost_usd,
                    "budget_utilization": turn_result.reconstructed_context.budget_utilization
                }
            })
            st.rerun()

    with col_inspect:
        st.subheader("Live Context Telemetry")
        if "last_turn" in st.session_state:
            last = st.session_state.last_turn
            ctx = last.reconstructed_context
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Token Budget", f"{ctx.total_tokens} / {ctx.token_budget}")
            m2.metric("Utilization", f"{ctx.budget_utilization * 100:.1f}%")
            m3.metric("Latency", f"{last.latency_ms:.0f} ms")

            st.markdown("**Assembled Context Payload Sent to LLM:**")
            st.code(ctx.context_text or "[Empty Context]", language="xml")

            st.markdown(f"**Included Items ({len(ctx.included_item_ids)}) vs Omitted ({len(ctx.omitted_item_ids)})**")
        else:
            st.info("Send a message in chat to inspect live context reconstruction and token budget utilization.")

# ==========================================
# TAB 2: MEMORY INSPECTOR & CONTROLS
# ==========================================
with tab_memory:
    st.subheader("Active Memory Inspector")
    st.caption("Inspect how each memory item was classified, view reasons, and manually pin, trim, or expand items.")

    show_discarded = st.checkbox("Show Discarded Items", value=False)
    memory_items = agent.db.get_memory_items(session_id=session_id, include_discarded=show_discarded)

    if not memory_items:
        st.info("No active memory items in this session. Start a chat or load sample data.")
    else:
        st.write(f"Total Items: **{len(memory_items)}**")

        for item in memory_items:
            with st.container():
                # Color code badge
                badge = "📌 PINNED" if item.is_pinned else ("🟢 VERBATIM" if item.label == "verbatim" else ("🟡 SUMMARY" if item.label == "summary" else "🔴 DISCARD"))
                
                c_title, c_actions = st.columns([3, 2])
                with c_title:
                    st.markdown(f"**{badge}** | Type: `{item.type}` | Tokens: `{item.tokens}` | Importance: `{item.importance}` | Recency: `{item.recency}`")
                    st.markdown(f"> **Content:** {item.content}")
                    if item.reason:
                        st.caption(f"💡 *Reason:* {item.reason}")
                
                with c_actions:
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    with btn_col1:
                        if item.is_pinned:
                            if st.button("Unpin", key=f"unpin_{item.id}"):
                                agent.pin_item(item.id, False)
                                st.rerun()
                        else:
                            if st.button("Pin", key=f"pin_{item.id}"):
                                agent.pin_item(item.id, True)
                                st.rerun()
                    with btn_col2:
                        if item.label != "summary":
                            if st.button("Trim/Compress", key=f"trim_{item.id}"):
                                agent.trim_item(item.id)
                                st.rerun()
                    with btn_col3:
                        if item.original_content and item.label == "summary":
                            if st.button("Expand Verbatim", key=f"expand_{item.id}"):
                                agent.expand_item(item.id)
                                st.rerun()
                st.divider()

# ==========================================
# TAB 3: VERSION GRAPH & OPERATIONS
# ==========================================
with tab_versions:
    st.subheader("Version Graph & Milestones")
    st.caption("Topic-labelled snapshot lineage, branching, merging, and drift recovery.")

    versions = agent.db.get_all_versions()

    v_col1, v_col2 = st.columns([1.4, 1.0])

    with v_col1:
        if not versions:
            st.info("No version snapshots recorded yet. Create a snapshot in chat or create one below.")
        else:
            # Render Version Tree with Plotly
            edge_x, edge_y = [], []
            node_x, node_y, node_text, node_color = [], [], [], []
            
            # Map versions to 2D coordinates
            v_map = {v.id: i for i, v in enumerate(versions)}
            branches = list(set(v.branch for v in versions))
            b_map = {b: i for i, b in enumerate(branches)}

            for i, v in enumerate(versions):
                x = i
                y = b_map.get(v.branch, 0)
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"<b>{v.topic_label}</b><br>Branch: {v.branch}<br>ID: {v.id[:8]}")
                node_color.append("#10b981" if v.branch == "main" else "#6366f1")

                if v.parent_id and v.parent_id in v_map:
                    p_idx = v_map[v.parent_id]
                    edge_x.extend([p_idx, x, None])
                    edge_y.extend([b_map.get(versions[p_idx].branch, 0), y, None])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=2, color="#94a3b8"),
                hoverinfo="none"
            ))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                marker=dict(size=24, color=node_color, line=dict(width=2, color="#ffffff")),
                text=[v.topic_label[:18] for v in versions],
                textposition="top center",
                hovertext=node_text,
                hoverinfo="text"
            ))
            fig.update_layout(
                title="Version DAG (Lineage)",
                showlegend=False,
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Traceback View
            selected_v_id = st.selectbox(
                "Select a Version to Trace / Inspect:",
                options=[v.id for v in versions],
                format_func=lambda vid: f"{next(v.topic_label for v in versions if v.id == vid)} ({vid[:8]})"
            )
            if selected_v_id:
                lineage = agent.version_chain.trace_back(selected_v_id)
                st.write("**Lineage Traceback:**")
                st.write(" ➔ ".join([f"`{v.topic_label}`" for v in lineage]))

    with v_col2:
        st.subheader("Version Controls")
        
        # Snapshot creator
        with st.expander("📸 Create New Snapshot"):
            manual_label = st.text_input("Milestone Topic Label", value="Manual Checkpoint")
            if st.button("Create Snapshot"):
                snap = agent.version_chain.create_snapshot(
                    session_id=session_id,
                    topic_label=manual_label,
                    branch=st.session_state.current_branch
                )
                st.success(f"Snapshot created: {snap.topic_label}")
                st.rerun()

        # Restart / Restore
        with st.expander("⏪ Restart / Restore From Version"):
            if versions:
                restore_target = st.selectbox(
                    "Target Version to Restore:",
                    options=[v.id for v in versions],
                    format_func=lambda vid: f"{next(v.topic_label for v in versions if v.id == vid)}"
                )
                if st.button("Restore Selected Version"):
                    new_v, items = agent.version_chain.restart(restore_target, session_id=session_id)
                    st.success(f"Successfully restored state from version! {len(items)} items restored.")
                    st.rerun()

        # Branching
        with st.expander("🌿 Branch from Version"):
            if versions:
                branch_base = st.selectbox(
                    "Base Version:",
                    options=[v.id for v in versions],
                    format_func=lambda vid: f"{next(v.topic_label for v in versions if v.id == vid)}"
                )
                new_b_name = st.text_input("New Branch Name", placeholder="e.g. experimental-sql")
                if st.button("Create Branch"):
                    bv = agent.version_chain.branch(branch_base, new_branch_name=new_b_name)
                    st.session_state.current_branch = new_b_name
                    st.success(f"Branched to '{new_b_name}' at version {bv.topic_label}!")
                    st.rerun()

        # Merge
        with st.expander("🔀 Merge Branches"):
            all_b = list(set(v.branch for v in versions))
            if len(all_b) >= 2:
                b_src = st.selectbox("Source Branch (to merge from):", options=all_b, index=1)
                b_tgt = st.selectbox("Target Branch (to merge into):", options=all_b, index=0)
                if st.button("Merge Branches"):
                    try:
                        m_res = agent.version_chain.merge(branch_a=b_tgt, branch_b=b_src)
                        st.success(m_res.summary_message)
                        if m_res.conflicts:
                            st.write(f"**Conflicts Handled ({len(m_res.conflicts)}):**")
                            for c in m_res.conflicts:
                                st.warning(f"Resolution: {c.resolution}")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            else:
                st.info("Need at least two branches to merge. Create a branch first.")

# ==========================================
# TAB 4: AUDIT DECISIONS LOG
# ==========================================
with tab_decisions:
    st.subheader("Audit & Decisions Log")
    st.caption("Transparent audit trail of every classify, summarize, discard, pin, branch, and merge action.")

    logs = agent.db.get_decision_logs(session_id=session_id, limit=50)
    if not logs:
        st.info("No decisions logged yet.")
    else:
        df_logs = pd.DataFrame([{
            "Timestamp": l.timestamp,
            "Action": l.action.upper(),
            "Item ID": l.item_id[:8] if l.item_id else "N/A",
            "Reason": l.reason,
            "Metadata": l.metadata_json
        } for l in logs])
        st.dataframe(df_logs, use_container_width=True)

if __name__ == "__main__":
    import os
    import sys
    # If run directly via `python ui/streamlit_app.py`, redirect to `streamlit run`
    if not os.environ.get("STREAMLIT_RUN_DIRECTLY"):
        from streamlit.web import cli as stcli
        os.environ["STREAMLIT_RUN_DIRECTLY"] = "1"
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
