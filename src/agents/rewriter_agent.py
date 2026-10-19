# src/agents/rewriter_agent.py
from src.config import get_llm
from src.schemas import ResumeRewriteProposal

def resume_rewriter_node(state: dict) -> dict:
    """
    LangGraph Node for Agent 4 (Resume Rewriter).
    Generates before/after bullet point upgrades with confidence scores.
    Also incorporates human feedback if looping back from the HITL checkpoint.
    """
    parsed_resume = state["parsed_resume"]
    ats_score = state["ats_score"]
    target_role = state["target_role"]
    jd_text = state["job_description"]
    user_feedback = state.get("hitl_user_edits", "")
    iterations = state.get("rewrite_iterations", 0)

    # Safely collect existing bullets from experience and projects
    all_bullets = []

    if parsed_resume.experience:
        for exp_item in parsed_resume.experience:
            for bullet in exp_item.bullets:
                all_bullets.append(f"[Experience: {exp_item.company} - {exp_item.role}] {bullet}")

    if parsed_resume.projects:
        for proj_item in parsed_resume.projects:
            for bullet in proj_item.bullets:
                all_bullets.append(f"[Project: {proj_item.title}] {bullet}")

    # Fallback if the parser put everything in raw text instead of structured bullets
    if not all_bullets:
        all_bullets.append(state["raw_resume_text"][:1500])

    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(ResumeRewriteProposal)

    feedback_instruction = ""
    if user_feedback:
        feedback_instruction = f"""
        IMPORTANT HUMAN-IN-THE-LOOP REVISION FEEDBACK:
        The candidate reviewed your previous draft and provided these real metrics/instructions:
        "{user_feedback}"
        You MUST incorporate this exact feedback into the rewritten bullets!
        """

    prompt = f"""You are Agent 4 (Resume Rewriter). Upgrade the candidate's weakest resume bullet points 
    to maximize impact for the Target Role ({target_role}) and naturally weave in relevant JD keywords.

    CANDIDATE EXISTING BULLETS:
    {chr(10).join(all_bullets)}

    MISSING JD KEYWORDS TO INCORPORATE (ONLY WHERE TRUTHFUL/APPLICABLE):
    {ats_score.missing_keywords}

    TARGET JOB DESCRIPTION:
    {jd_text}
    {feedback_instruction}

    Instructions:
    1. Select 4 to 6 bullet points from the candidate's projects or experience that can be strengthened.
    2. Rewrite each using strong action verbs and the XYZ/STAR impact formula (Accomplished [X] as measured by [Y], by doing [Z]).
    3. Do NOT invent fake metrics—if a metric is needed, use a clear placeholder like `[X]%` or `[N]ms` so the human can fill it in at the HITL checkpoint.
    4. Assign a `confidence_score` (0.0 to 1.0) reflecting how grounded the rewrite is in their actual tech stack.
    5. Explain `reasoning` for each before/after transformation.
    """

    proposal: ResumeRewriteProposal = structured_llm.invoke(prompt)

    return {
        "rewrite_proposal": proposal,
        "rewrite_iterations": iterations + 1,
    }

def hitl_approval_node(state: dict) -> dict:
    """
    Dedicated Human-in-the-Loop (HITL) Gate Node.
    LangGraph will pause execution BEFORE this node (`interrupt_before=["hitl_gate"]`)
    until the user approves or requests revisions in the UI.
    """
    status = state.get("hitl_status", "approved")
    return {"hitl_status": status}