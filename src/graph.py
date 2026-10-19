# src/graph.py
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import CareerGraphState
from src.agents.parser_agent import resume_parser_node
from src.agents.ats_agent import ats_scorer_node
from src.agents.gap_agent import gap_analyzer_node
from src.agents.rewriter_agent import resume_rewriter_node, hitl_approval_node
from src.agents.interview_agent import interview_prep_node
from src.agents.market_agent import market_intel_node
from src.agents.strategist_agent import career_strategist_node

def route_after_hitl(state: CareerGraphState) -> str:
    """
    Conditional edge after the Human-in-the-Loop checkpoint:
    - If the human requested revisions (and we haven't looped more than 3 times), route back to Agent 4.
    - Otherwise, proceed to Agent 7 (Career Strategist).
    """
    if state.get("hitl_status") == "revision_requested" and state.get("rewrite_iterations", 0) < 3:
        return "resume_rewriter"
    return "career_strategist"

def build_career_graph(db_path: str = "career_sessions.sqlite"):
    builder = StateGraph(CareerGraphState)

    # 1. Register all 7 Agent Nodes + HITL Gate
    builder.add_node("resume_parser", resume_parser_node)
    builder.add_node("ats_scorer", ats_scorer_node)
    builder.add_node("gap_analyzer", gap_analyzer_node)
    builder.add_node("resume_rewriter", resume_rewriter_node)
    builder.add_node("hitl_gate", hitl_approval_node)
    builder.add_node("interview_prep", interview_prep_node)
    builder.add_node("market_intel", market_intel_node)
    builder.add_node("career_strategist", career_strategist_node)

    # 2. Sequential Foundation: START -> Agent 1 -> Agent 2 -> Agent 3
    builder.add_edge(START, "resume_parser")
    builder.add_edge("resume_parser", "ats_scorer")
    builder.add_edge("ats_scorer", "gap_analyzer")

    # 3. Parallel Fan-Out: Agent 3 -> [Agent 4, Agent 5, Agent 6] concurrently!
    builder.add_edge("gap_analyzer", "resume_rewriter")
    builder.add_edge("gap_analyzer", "interview_prep")
    builder.add_edge("gap_analyzer", "market_intel")

    # 4. Agent 4 -> HITL Gate -> Conditional Loop or Advance
    builder.add_edge("resume_rewriter", "hitl_gate")
    builder.add_conditional_edges(
        "hitl_gate",
        route_after_hitl,
        {
            "resume_rewriter": "resume_rewriter",
            "career_strategist": "career_strategist",
        },
    )

    # 5. Fan-In to Agent 7 (Career Strategist waits for all parallel branches)
    builder.add_edge("interview_prep", "career_strategist")
    builder.add_edge("market_intel", "career_strategist")
    builder.add_edge("career_strategist", END)

    # 6. Persistent SQLite Checkpointer
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    # Pause execution automatically right before 'hitl_gate'
    return builder.compile(checkpointer=memory, interrupt_before=["hitl_gate"])