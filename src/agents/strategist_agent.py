# src/agents/strategist_agent.py
from src.config import get_llm
from src.schemas import CareerStrategyPlan

def career_strategist_node(state: dict) -> dict:
    """
    LangGraph Node for Agent 7 (Career Strategist).
    Synthesizes outputs from Agents 1-6 into a concrete 4-week phased action plan.
    """
    parsed = state["parsed_resume"]
    ats = state["ats_score"]
    gaps = state["gap_analysis"]
    rewrites = state["rewrite_proposal"]
    market = state["market_intel"]
    target_role = state["target_role"]

    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(CareerStrategyPlan)

    prompt = f"""You are Agent 7 (Master Career Strategist). Synthesize the findings from all 6 previous 
    agents into a personalized, concrete 4-Week Phased Action Plan for {parsed.candidate_name} targeting {target_role}.

    1. ATS COMPOSITE SCORE: {ats.composite_score}/100 (Missing: {ats.missing_keywords})
    2. PRIORITIZED SKILL GAPS ({gaps.total_estimated_hours} total hours):
       {[f"{g.skill} ({g.priority}, {g.estimated_hours}h)" for g in gaps.prioritized_gaps]}
    3. RESUME REWRITES COMPLETED: {len(rewrites.rewrites)} high-impact bullets upgraded.
    4. LIVE MARKET INTEL: Demand is {market.demand_level} | Trending skills: {market.trending_skills}

    Instructions:
    1. Write an encouraging, strategic `executive_summary` assessing their current readiness and immediate leverage points.
    2. Build a `four_week_roadmap` (Weeks 1, 2, 3, and 4). For each week, specify:
       - `week` (1 to 4)
       - `focus_title` (e.g., 'Week 1: Resume Finalization & Critical Gap Foundations')
       - `action_items` (3 to 4 concrete tasks combining their skill gaps, project upgrades, and STAR interview prep)
       - `milestone` (A tangible deliverable by Sunday night of that week, e.g., 'Deploy Dockerized FastAPI endpoint to GitHub')
    """

    strategy_plan: CareerStrategyPlan = structured_llm.invoke(prompt)
    return {"career_action_plan": strategy_plan}