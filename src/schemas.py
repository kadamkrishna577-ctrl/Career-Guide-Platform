# src/schemas.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# ==========================================
# 1. Agent 1: Resume Parser Output Schema
# ==========================================
class ExperienceItem(BaseModel):
    company: str = Field(description="Company or organization name")
    role: str = Field(description="Role or job title")
    duration: str = Field(description="Date range, e.g., May 2025 - Jul 2025")
    bullets: List[str] = Field(description="Action bullets under this experience")

class ProjectItem(BaseModel):
    title: str = Field(description="Name of the project")
    tech_stack: List[str] = Field(description="Frameworks, languages, and tools used")
    bullets: List[str] = Field(description="Bullet points describing what was built")

class ParsedResume(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    email: Optional[str] = Field(default=None, description="Email address if present")
    skills: List[str] = Field(description="Comprehensive list of technical skills extracted")
    education: List[str] = Field(description="Degrees, institutions, graduation years, GPA")
    experience: List[ExperienceItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)

# ==========================================
# 2. Agent 2: ATS Scorer Output Schema
# ==========================================
class ATSScoreResult(BaseModel):
    keyword_score: float = Field(description="Exact/BM25 keyword match score from 0 to 100")
    semantic_score: float = Field(description="Dense embedding cosine similarity score from 0 to 100")
    composite_score: float = Field(description="Weighted overall ATS score from 0 to 100")
    matched_keywords: List[str] = Field(description="Important JD keywords found in the resume")
    missing_keywords: List[str] = Field(description="Important JD keywords missing from the resume")
    scoring_summary: str = Field(
        description="Detailed 2-paragraph technical breakdown of why the resume scored this way, analyzing section-by-section strengths and exact keyword density gaps."
    )

# ==========================================
# 3. Agent 3: Gap Analyzer Output Schema
# ==========================================
class SkillGapItem(BaseModel):
    skill: str = Field(description="Name of the missing or weak skill")
    priority: Literal["Critical", "High", "Medium"]
    estimated_hours: int = Field(description="Estimated hours needed to master this skill")
    why_it_matters_for_role: str = Field(
        description="Deep technical explanation of how this skill is used in production for the target role."
    )
    upgrade_existing_project_idea: str = Field(
        description="Specific architectural blueprint showing how to bolt this missing technology directly onto one of the candidate's ACTUAL existing resume projects (mention the project by name and exact modules/APIs to add)."
    )
    learning_path: List[str] = Field(
        description="4 detailed, step-by-step engineering tasks with specific libraries, CLI commands, and sub-topics to master it."
    )

class GapAnalysisReport(BaseModel):
    prioritized_gaps: List[SkillGapItem]
    total_estimated_hours: int
    gap_summary: str = Field(description="Detailed technical diagnosis of the candidate's engineering gaps.")

# ==========================================
# 4. Agent 4: Resume Rewriter Output Schema
# ==========================================
class BulletRewriteItem(BaseModel):
    source_section: str = Field(description="Which project or company this bullet came from")
    original_bullet: str
    rewritten_bullet: str = Field(
        description="Upgraded impact-driven bullet using XYZ formula, weaving in JD keywords and specific technical mechanisms"
    )
    keywords_injected: List[str] = Field(description="Exact JD keywords woven into this rewrite")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(
        description="Detailed explanation of what was weak in the original bullet and why the new phrasing passes both ATS filters and Senior Engineer screening."
    )

class ResumeRewriteProposal(BaseModel):
    rewrites: List[BulletRewriteItem]

# ==========================================
# 5. Agent 5: Interview Prep Output Schema
# ==========================================
class InterviewQuestionItem(BaseModel):
    category: Literal["Technical", "Project Deep-Dive", "Behavioral STAR", "System Design"]
    question: str = Field(description="Challenging, senior-level interview question tailored to their resume")
    target_gap_or_bullet: str = Field(description="Which exact resume project or gap this tests")
    interviewer_hidden_intent: str = Field(
        description="What the interviewer is secretly testing (e.g., GIL limitations, CUDA OOM handling, vector index HNSW vs IVFFlat trade-offs)."
    )
    star_answer_framework: str = Field(
        description="Comprehensive, multi-paragraph technical answer script broken down into: [Situation & Architecture], [Task & Bottleneck], [Action & Implementation Details (mentioning specific classes/methods/configs)], and [Result & Quantifiable Metrics]."
    )
    follow_up_trap_question: str = Field(
        description="The brutal follow-up curveball question the interviewer will ask right after, and the exact technical way to answer it."
    )

class InterviewPrepPack(BaseModel):
    questions: List[InterviewQuestionItem]

# ==========================================
# 6. Agent 6: Market Intel Output Schema
# ==========================================
class MarketIntelReport(BaseModel):
    salary_range_india: str = Field(description="Live estimated salary range in India (LPA) broken down by Tier-1 / Product / Startup")
    salary_range_us: str = Field(description="Live estimated salary range in US/Global (USD)")
    demand_level: Literal["High Demand", "Moderate Demand", "Competitive"]
    trending_skills: List[str] = Field(description="Top trending skills appearing in live web search")
    top_hiring_domains: List[str] = Field(description="Specific sub-sectors actively hiring for this role right now")
    market_insights: str = Field(
        description="Detailed 2-paragraph market intelligence briefing analyzing hiring trends, what separates top 5% candidates from average applicants, and compensation leverage points."
    )

# ==========================================
# 7. Agent 7: Career Strategist Output Schema
# ==========================================
class PhasePlan(BaseModel):
    week: int
    focus_title: str
    technical_architecture_to_build: str = Field(
        description="Detailed technical spec of what the candidate will code/build this week (exact stack, endpoints, Docker/DB schema, and how it extends their existing resume projects)."
    )
    action_items: List[str] = Field(
        description="5 highly specific, day-by-day engineering action items (e.g. 'Days 1-2: Write a multi-stage Dockerfile for your FastAPI e4e server and benchmark latency with Locust'). No generic fluff."
    )
    milestone: str = Field(description="Verifiable engineering artifact and benchmark to achieve by Sunday night")

class CareerStrategyPlan(BaseModel):
    executive_summary: str = Field(
        description="Exhaustive 3-paragraph strategic diagnosis covering: (1) Current engineering baseline, (2) Architectural bridge to close the JD gaps, and (3) High-leverage positioning strategy."
    )
    four_week_roadmap: List[PhasePlan]