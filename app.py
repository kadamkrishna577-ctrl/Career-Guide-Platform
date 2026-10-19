# app.py
import tempfile
import uuid
import streamlit as st
from src.graph import build_career_graph
from src.config import get_llm

st.set_page_config(
    page_title="Agentic Career Intelligence Platform",
    page_icon="🧠",
    layout="wide",
)

@st.cache_resource
def get_compiled_graph():
    return build_career_graph("career_sessions.sqlite")

graph = get_compiled_graph()

# ==========================================
# SIDEBAR: Session Control & Inputs
# ==========================================
with st.sidebar:
    st.title("🧠 Career Intelligence")
    st.caption("7-Agent LangGraph Pipeline + HITL + SQLite Persistence")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "session-" + str(uuid.uuid4())[:8]

    thread_input = st.text_input(
        "🔑 Session ID (Copy to resume later)",
        value=st.session_state.thread_id,
        help="Your entire multi-agent state is persisted in SQLite under this ID."
    )
    st.session_state.thread_id = thread_input
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    if st.button("➕ Start Fresh Session", use_container_width=True):
        st.session_state.thread_id = "session-" + str(uuid.uuid4())[:8]
        st.session_state.mock_messages = []
        st.rerun()

    st.divider()
    uploaded_pdf = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
    target_role = st.text_input("🎯 Target Role", value="AI / Machine Learning Engineer")
    job_description = st.text_area(
        "📋 Paste Target Job Description",
        height=180,
        placeholder="Paste the full Job Description requirements here..."
    )

    run_btn = st.button("🚀 Run 7-Agent Pipeline", type="primary", use_container_width=True)

# ==========================================
# GRAPH EXECUTION TRIGGER
# ==========================================
if run_btn:
    if not uploaded_pdf or not job_description.strip():
        st.sidebar.error("⚠️ Please upload a PDF resume and paste a Job Description.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_pdf_path = tmp.name

        initial_state = {
            "pdf_path": tmp_pdf_path,
            "target_role": target_role,
            "job_description": job_description,
            "rewrite_iterations": 0,
            "hitl_status": "pending",
            "hitl_user_edits": "",
            "ats_history": [],
        }

        with st.status("⚙️ Running Multi-Agent Graph (Parallel Fan-Out Active)...", expanded=True) as status:
            for event in graph.stream(initial_state, config=config, stream_mode="updates"):
                for node_name in event.keys():
                    st.write(f"✅ Agent Node Completed: **`{node_name}`**")
            status.update(
                label="⏸️ Graph Paused at HITL Checkpoint! Review Tab 2 to approve or revise bullets.",
                state="complete",
                expanded=False,
            )

# ==========================================
# LOAD PERSISTED STATE FROM SQLITE
# ==========================================
snapshot = graph.get_state(config)
state_vals = snapshot.values if snapshot else {}
next_nodes = snapshot.next if snapshot else ()

if not state_vals:
    st.title("🧠 Agentic Career Intelligence Platform")
    st.info("👈 Upload your PDF Resume and paste a Target Job Description in the sidebar to launch the 7-agent pipeline.")
    st.stop()

parsed = state_vals.get("parsed_resume")
ats = state_vals.get("ats_score")
gaps = state_vals.get("gap_analysis")
proposal = state_vals.get("rewrite_proposal")
prep = state_vals.get("interview_prep")
market = state_vals.get("market_intel")
plan = state_vals.get("career_action_plan")

# ==========================================
# TOP KPI BANNER
# ==========================================
st.title(f"📊 Career Intelligence Report: {parsed.candidate_name if parsed else 'Candidate'}")
if "hitl_gate" in next_nodes:
    st.warning("⏸️ **ACTION REQUIRED (Human-in-the-Loop):** The graph is paused at `hitl_gate`. Go to **Tab 2 (HITL Resume Rewriter)** to approve or refine your bullet upgrades and unlock Agent 7's final roadmap!")

k1, k2, k3, k4, k5 = st.columns(5)
if ats:
    k1.metric("🏆 Composite ATS", f"{ats.composite_score}/100")
    k2.metric("🔤 Keyword (BM25)", f"{ats.keyword_score}/100")
    k3.metric("🧠 Semantic Vector", f"{ats.semantic_score}/100")
if gaps:
    k4.metric("⏱️ Gap Study Time", f"{gaps.total_estimated_hours} hrs")
if market:
    k5.metric("📈 Market Demand", market.demand_level)

st.divider()

# ==========================================
# 5 INTERACTIVE TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ ATS & Deep Gap Blueprints",
    "2️⃣ HITL Resume Rewriter",
    "3️⃣ Principal Interview Prep",
    "4️⃣ Market Intel & 4-Week Roadmap",
    "5️⃣ Live Mock Interviewer",
])

# ------------------------------------------
# TAB 1: ATS & GAP ANALYSIS
# ------------------------------------------
with tab1:
    if ats:
        st.subheader("🎯 Dual ATS Diagnostic Summary")
        st.write(ats.scoring_summary)
        c_match, c_miss = st.columns(2)
        with c_match:
            st.success(f"**✅ Matched JD Keywords ({len(ats.matched_keywords)}):**\n\n" + ", ".join(ats.matched_keywords))
        with c_miss:
            st.error(f"**❌ Missing JD Keywords ({len(ats.missing_keywords)}):**\n\n" + ", ".join(ats.missing_keywords))

    st.divider()
    if gaps:
        st.subheader("🛠️ Prioritized Skill Gaps & Project Upgrade Blueprints")
        st.caption(gaps.gap_summary)
        for g in gaps.prioritized_gaps:
            badge = "🔴" if g.priority == "Critical" else ("🟠" if g.priority == "High" else "🟡")
            with st.expander(f"{badge} [{g.priority}] {g.skill} — Estimated: {g.estimated_hours} Hours", expanded=True):
                st.markdown(f"**💡 Why It Matters in Production:**\n{g.why_it_matters_for_role}")
                st.info(f"**🚀 How to Bolt This onto Your Existing Resume Project:**\n\n{g.upgrade_existing_project_idea}")
                st.markdown("**📚 Step-by-Step Engineering Learning Path:**")
                for idx, step in enumerate(g.learning_path, 1):
                    st.markdown(f"{idx}. {step}")

# ------------------------------------------
# TAB 2: HUMAN-IN-THE-LOOP RESUME REWRITER
# ------------------------------------------
with tab2:
    if proposal:
        st.subheader(f"✍️ Agent 4: Impact Bullet Upgrades (Iteration #{state_vals.get('rewrite_iterations', 1)})")
        for idx, rw in enumerate(proposal.rewrites, 1):
            with st.container(border=True):
                st.markdown(f"**#{idx} | Section:** `{rw.source_section}` | **Confidence Score:** `{int(rw.confidence_score * 100)}%`")
                col_before, col_after = st.columns(2)
                with col_before:
                    st.error(f"**❌ Original Bullet:**\n\n{rw.original_bullet}")
                with col_after:
                    st.success(f"**✅ Upgraded Bullet:**\n\n{rw.rewritten_bullet}")
                if getattr(rw, "keywords_injected", None):
                    st.caption(f"🔑 **Keywords Injected:** {', '.join(rw.keywords_injected)}")
                st.markdown(f"**🧠 Senior Engineer Reasoning:** {rw.reasoning}")

        st.divider()
        if "hitl_gate" in next_nodes:
            st.subheader("🛑 Human-in-the-Loop (HITL) Approval Checkpoint")
            st.write("Provide real metrics to replace placeholders like `[X]%` or `[N]ms`, or approve directly to trigger **Agent 7 (Career Strategist)**.")
            user_feedback = st.text_area(
                "📝 Your Real Metrics / Revision Instructions (Optional):",
                placeholder="e.g., In the Face Stylization FastAPI server, batching reduced latency to 310ms and accuracy was 94.2%."
            )
            col_app, col_rev = st.columns(2)
            with col_app:
                if st.button("✅ Approve Bullets & Generate 4-Week Roadmap", type="primary", use_container_width=True):
                    # FIX: Added as_node="hitl_gate"
                    graph.update_state(
                        config,
                        {"hitl_status": "approved", "hitl_user_edits": user_feedback},
                        as_node="hitl_gate",
                    )
                    with st.spinner("🧠 Agent 7 (Career Strategist) is synthesizing your 4-week execution roadmap..."):
                        for _ in graph.stream(None, config=config, stream_mode="updates"):
                            pass
                    st.rerun()
            with col_rev:
                if st.button("🔄 Send Feedback Back to Agent 4 (Re-Write)", use_container_width=True):
                    # FIX: Added as_node="hitl_gate"
                    graph.update_state(
                        config,
                        {"hitl_status": "revision_requested", "hitl_user_edits": user_feedback},
                        as_node="hitl_gate",
                    )
                    with st.spinner("🔄 Agent 4 is rewriting bullets with your custom metrics..."):
                        for _ in graph.stream(None, config=config, stream_mode="updates"):
                            pass
                    st.rerun()
        else:
            st.success("✅ **HITL Checkpoint Approved!** Agent 7 has synthesized your final strategy in Tab 4.")

# ------------------------------------------
# TAB 3: PRINCIPAL INTERVIEW PREP
# ------------------------------------------
with tab3:
    if prep:
        st.subheader("🎙️ Principal-Level Technical & Architecture Grilling")
        for idx, q in enumerate(prep.questions, 1):
            with st.expander(f"Q{idx} [{q.category}]: {q.question}", expanded=(idx == 1)):
                st.caption(f"🎯 **Targets:** {q.target_gap_or_bullet}")
                if getattr(q, "interviewer_hidden_intent", None):
                    st.warning(f"**🕵️ Interviewer's Hidden Intent:**\n\n{q.interviewer_hidden_intent}")
                st.markdown("### 📘 Full STAR Answer Blueprint")
                st.write(q.star_answer_framework)
                if getattr(q, "follow_up_trap_question", None):
                    st.error(f"**⚠️ Follow-Up Trap Question & How to Handle It:**\n\n{q.follow_up_trap_question}")

# ------------------------------------------
# TAB 4: MARKET INTEL & 4-WEEK ROADMAP
# ------------------------------------------
with tab4:
    if market:
        st.subheader("🌐 Agent 6: Live Market Intelligence (via Tavily Web Search)")
        m1, m2, m3 = st.columns(3)
        m1.info(f"**🇮🇳 India Compensation:**\n\n{market.salary_range_india}")
        m2.info(f"**🇺🇸 US / Global Compensation:**\n\n{market.salary_range_us}")
        m3.info(f"**🔥 Top Hiring Domains:**\n\n{', '.join(getattr(market, 'top_hiring_domains', ['AI/ML']))}")
        st.markdown(f"**📈 Trending Skills in Live Job Postings:** `{', '.join(market.trending_skills)}`")
        st.write(market.market_insights)

    st.divider()
    if plan:
        st.subheader("🗺️ Agent 7: 4-Week Engineering Execution Plan")
        st.markdown("### Executive Strategy Diagnosis")
        st.write(plan.executive_summary)

        for w in plan.four_week_roadmap:
            with st.container(border=True):
                st.markdown(f"### 📅 Week {w.week}: {w.focus_title}")
                if getattr(w, "technical_architecture_to_build", None):
                    st.info(f"**🏗️ Technical Architecture to Build This Week:**\n\n{w.technical_architecture_to_build}")
                st.markdown("**📋 Day-by-Day Engineering Action Items:**")
                for item in w.action_items:
                    st.markdown(f"- {item}")
                st.success(f"**🏆 Sunday Night Verifiable Milestone:** {w.milestone}")
    elif "hitl_gate" in next_nodes:
        st.warning("⏸️ **Locked:** Please click **Approve Bullets** in **Tab 2** so Agent 7 can generate your final 4-week roadmap!")

# ------------------------------------------
# TAB 5: INTERACTIVE MOCK INTERVIEWER CHAT
# ------------------------------------------
with tab5:
    st.subheader("💬 Technical Mock Interview")

    if "mock_messages" not in st.session_state or not st.session_state.mock_messages:
        first_question = (
            prep.questions[0].question
            if (prep and prep.questions)
            else "Walk me through the architecture of your most complex machine learning project and its scaling bottlenecks."
        )
        st.session_state.mock_messages = [
            {
                "role": "assistant",
                "content": f"**Technical Question:**\n\n{first_question}"
            }
        ]

    for msg in st.session_state.mock_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_answer = st.chat_input("Type your technical answer here...")
    if user_answer:
        st.session_state.mock_messages.append({"role": "user", "content": user_answer})
        with st.chat_message("user"):
            st.markdown(user_answer)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing technical depth..."):
                llm = get_llm(temperature=0.2)
                history_text = "\n".join(
                    [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.mock_messages[-6:]]
                )

                # Build candidate context for targeted follow-ups
                candidate_context = ""
                if parsed:
                    project_titles = [p.title for p in parsed.projects] if parsed.projects else []
                    candidate_context += f"Candidate: {parsed.candidate_name}\nProjects: {', '.join(project_titles)}\n"
                if gaps:
                    gap_skills = [g.skill for g in gaps.prioritized_gaps]
                    candidate_context += f"Key Skill Gaps: {', '.join(gap_skills)}\n"

                critique_prompt = f"""You are a Principal Systems & Machine Learning Engineer conducting a technical interview for the role of {state_vals.get('target_role')}.

{candidate_context}
Recent Interview Transcript:
{history_text}

Instructions:
- Provide strictly high-signal technical content.
- DO NOT output scores, letter grades, ratings, or conversational pleasantries.
- Output ONLY the following two sections:

### Technical Feedback
- In 1-3 bullet points, highlight the exact technical, algorithmic, or architectural trade-offs the candidate missed or answered incorrectly.

### Follow-up Question
- Ask a direct, rigorous follow-up question targeting those specific gaps or bottlenecks.
"""
                response = llm.invoke(critique_prompt)
                reply_text = response.content if hasattr(response, "content") else str(response)
                st.markdown(reply_text)
                st.session_state.mock_messages.append({"role": "assistant", "content": reply_text})