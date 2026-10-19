# src/state.py
import operator
from typing import TypedDict, List, Optional, Annotated, Dict
from src.schemas import (
    ParsedResume,
    ATSScoreResult,
    GapAnalysisReport,
    ResumeRewriteProposal,
    InterviewPrepPack,
    MarketIntelReport,
    CareerStrategyPlan,
)

class CareerGraphState(TypedDict):
    # Raw User Inputs
    pdf_path: str
    target_role: str
    job_description: str
    
    # Agent 1 (Resume Parser)
    ocr_used: bool
    raw_resume_text: str
    parsed_resume: Optional[ParsedResume]
    
    # Agent 2 (Dual ATS Scorer)
    ats_score: Optional[ATSScoreResult]
    ats_history: Annotated[List[float], operator.add]
    
    # Agent 3 (Gap Analyzer)
    gap_analysis: Optional[GapAnalysisReport]
    
    # Agent 4 (Resume Rewriter) + Human-in-the-Loop (HITL) State
    rewrite_proposal: Optional[ResumeRewriteProposal]
    rewrite_iterations: int
    hitl_status: str              # "pending", "approved", or "revision_requested"
    hitl_user_edits: str          # Real metrics/instructions provided by the human
    
    # Agent 5 (Interview Prep)
    interview_prep: Optional[InterviewPrepPack]
    
    # Agent 6 (Market Intel)
    market_intel: Optional[MarketIntelReport]
    
    # Agent 7 (Career Strategist)
    career_action_plan: Optional[CareerStrategyPlan]