import re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from src.config import get_llm, get_embeddings
from src.schemas import ATSScoreResult


def tokenize(text: str) -> list[str]:
    """Splits text into clean lowercase alphanumeric tokens (preserving C++, C#, .NET, etc.)."""
    return re.findall(r"\b[a-zA-Z0-9+#.-]+\b", text.lower())

def compute_bm25_keyword_score(resume_text: str, jd_text: str) -> float:
    """
    Computes Part 1 of Dual Scoring: Lexical Keyword Overlap + BM25 Term Frequency.
    """
    resume_tokens = tokenize(resume_text)
    jd_tokens = tokenize(jd_text)
    if not jd_tokens or not resume_tokens:
        return 0.0

    # Keep meaningful words (length > 2) from JD
    jd_keywords = [w for w in set(jd_tokens) if len(w) > 2]
    resume_vocab = set(resume_tokens)

    # Exact token overlap ratio
    overlap_ratio = sum(1 for w in jd_keywords if w in resume_vocab) / max(len(jd_keywords), 1)

    # BM25 saturation across resume lines
    corpus = [tokenize(line) for line in resume_text.splitlines() if line.strip()]
    if corpus:
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(jd_keywords)
        bm25_normalized = min(float(np.mean(bm25_scores)) * 12.0, 100.0)
    else:
        bm25_normalized = 0.0

    score = (0.70 * (overlap_ratio * 100.0)) + (0.30 * bm25_normalized)
    return round(min(max(score, 0.0), 100.0), 1)
    
def compute_semantic_similarity_score(resume_text: str, jd_text: str) -> float:
    """
    Computes Part 2 of Dual Scoring: Dense Vector Cosine Similarity via Gemini Embeddings.
    """
    embeddings_model = get_embeddings()
    vectors = embeddings_model.embed_documents([resume_text, jd_text])
    raw_cosine = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    # Normalize typical text-embedding-004 cosine similarity [0.35, 0.88] -> [0, 100] scale
    scaled = np.clip((raw_cosine - 0.35) / (0.88 - 0.35) * 100.0, 0.0, 100.0)
    return round(float(scaled), 1)


def ats_scorer_node(state: dict) -> dict:
    """
    LangGraph Node for Agent 2 (ATS Scorer).
    Reads 'raw_resume_text' and 'job_description', runs dual scoring,
    and returns a validated ATSScoreResult Pydantic object.
    """
    resume_text = state["raw_resume_text"]
    jd_text = state["job_description"]

    # 1. Calculate mathematical dual scores
    bm25_score = compute_bm25_keyword_score(resume_text, jd_text)
    semantic_score = compute_semantic_similarity_score(resume_text, jd_text)

    # 2. Use Gemini to extract exact matched/missing technical keywords & diagnostic summary
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ATSScoreResult)

    prompt = f"""You are Agent 2 (ATS Scorer). Compare the Candidate Resume against the Target Job Description.

    Mathematical Scores Computed:
    - Lexical / BM25 Keyword Score: {bm25_score}/100
    - Semantic Embedding Similarity Score: {semantic_score}/100

    Instructions:
    1. Identify all core technical skills, tools, frameworks, and concepts required by the Job Description.
    2. List the ones present in the resume under `matched_keywords`.
    3. List the important ones missing from the resume under `missing_keywords`.
    4. Set `keyword_score` (0 to 100) by blending {bm25_score} with the percentage of core technical JD keywords matched.
    5. Set `semantic_score` to {semantic_score}.
    6. Set `composite_score` = round(0.55 * keyword_score + 0.45 * {semantic_score}, 1).
    7. Provide a clear `scoring_summary` explaining why the candidate received this score and what is missing.

    JOB DESCRIPTION:
    {jd_text}

    RESUME TEXT:
    {resume_text}
    """

    ats_result: ATSScoreResult = structured_llm.invoke(prompt)
    # Enforce exact mathematical consistency on the Pydantic output
    ats_result.semantic_score = semantic_score
    ats_result.composite_score = round(
        (0.55 * ats_result.keyword_score) + (0.45 * semantic_score), 1
    )

    return {"ats_score": ats_result}