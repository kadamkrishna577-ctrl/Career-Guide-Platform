# src/agents/gap_agent.py
from src.config import get_llm
from src.schemas import GapAnalysisReport

def gap_analyzer_node(state: dict) -> dict:
    """
    LangGraph Node for Agent 3 (Gap Analyzer).
    Combines ParsedResume (Agent 1) + ATSScoreResult (Agent 2) + Job Description
    to output prioritized skill gaps, concrete learning paths, and time estimates.
    """
    parsed_resume = state["parsed_resume"]
    ats_score = state["ats_score"]
    target_role = state["target_role"]
    jd_text = state["job_description"]

    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(GapAnalysisReport)

    prompt = f"""You are Agent 3 (Gap Analyzer). Analyze the skill gaps between this candidate's 
    current profile and the Target Role ({target_role}).

    CANDIDATE CURRENT SKILLS:
    {parsed_resume.skills}

    CANDIDATE EXISTING PROJECTS:
    {[f"{p.title}: {', '.join(p.tech_stack)}" for p in parsed_resume.projects]}

    MISSING KEYWORDS FLAGGED BY ATS SCORER (AGENT 2):
    {ats_score.missing_keywords}

    TARGET JOB DESCRIPTION:
    {jd_text}

    Instructions:
    1. Identify the 4 to 6 most important skill/technology gaps the candidate needs to close for this role.
    2. Assign each a priority: 'Critical' (must-have blocker), 'High' (strong differentiator), or 'Medium' (nice-to-have).
    3. Estimate realistic `estimated_hours` required for a CS student/developer to become interview-ready in that skill.
    4. Provide a 3-step `learning_path` for each skill (Step 1: Core concept, Step 2: Hands-on implementation, Step 3: How to integrate it into one of their existing projects).
    5. Sum the hours into `total_estimated_hours` and write an actionable `gap_summary`.
    """

    gap_report: GapAnalysisReport = structured_llm.invoke(prompt)

    return {"gap_analysis": gap_report}