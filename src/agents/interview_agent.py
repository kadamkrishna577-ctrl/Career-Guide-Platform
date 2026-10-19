# src/agents/interview_agent.py
from src.config import get_llm
from src.schemas import InterviewPrepPack

def interview_prep_node(state: dict) -> dict:
    """
    LangGraph Node for Agent 5 (Interview Prep).
    Generates tailored technical & behavioral questions + STAR answer frameworks
    grounded in the candidate's resume projects and identified skill gaps.
    """
    parsed_resume = state["parsed_resume"]
    gap_analysis = state["gap_analysis"]
    target_role = state["target_role"]
    jd_text = state["job_description"]

    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(InterviewPrepPack)

    projects_summary = "\n".join(
        [f"- {p.title} (Stack: {', '.join(p.tech_stack)}): {' '.join(p.bullets)}" for p in parsed_resume.projects]
    )
    gaps_summary = "\n".join(
        [f"- [{g.priority}] {g.skill}" for g in gap_analysis.prioritized_gaps]
    )

    prompt = f"""You are Agent 5 (Interview Prep Coach). Create a tailored interview preparation pack 
    for this candidate applying for {target_role}.

    CANDIDATE PROJECTS:
    {projects_summary}

    CANDIDATE SKILL GAPS (FROM AGENT 3):
    {gaps_summary}

    JOB DESCRIPTION:
    {jd_text}

    Instructions:
    1. Generate 6 high-signal interview questions covering all 4 categories:
       - 'Project Deep-Dive' (grilling their actual resume projects, architecture choices, and scaling bottlenecks)
       - 'Technical' (testing core JD requirements and their prioritized skill gaps)
       - 'System Design' (designing a production system relevant to {target_role})
       - 'Behavioral STAR' (handling technical trade-offs, tight deadlines, or debugging failures)
    2. For each question, specify `target_gap_or_bullet` (which exact project or gap triggered this question).
    3. Write a detailed `star_answer_framework` (Situation, Task, Action, Result) showing the candidate how to answer using their real background.
    """

    prep_pack: InterviewPrepPack = structured_llm.invoke(prompt)
    return {"interview_prep": prep_pack}