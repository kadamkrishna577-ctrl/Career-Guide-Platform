# src/agents/market_agent.py
import os
from tavily import TavilyClient
from src.config import get_llm
from src.schemas import MarketIntelReport

def fetch_live_market_data(target_role: str) -> str:
    """Queries Tavily API for live salary, hiring demand, and trending skills."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily API key not configured; use general market estimates."

    try:
        client = TavilyClient(api_key=api_key)
        query = f"{target_role} salary range India LPA US USD hiring demand trending skills 2026"
        response = client.search(query=query, search_depth="basic", max_results=5)
        snippets = [
            f"- {item.get('title')}: {item.get('content')}"
            for item in response.get("results", [])
        ]
        return "\n".join(snippets) if snippets else "No live search snippets returned."
    except Exception as e:
        return f"Web search fallback triggered ({str(e)})."

def market_intel_node(state: dict) -> dict:
    """
    LangGraph Node for Agent 6 (Market Intel).
    Performs live web search via Tavily and extracts structured salary, demand, and trending skill insights.
    """
    target_role = state["target_role"]
    jd_text = state["job_description"]

    live_search_context = fetch_live_market_data(target_role)

    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(MarketIntelReport)

    prompt = f"""You are Agent 6 (Market Intelligence Analyst). Synthesize the live web search results 
    below to provide accurate market intelligence for the role: {target_role}.

    LIVE WEB SEARCH RESULTS (TAVILY):
    {live_search_context}

    JOB DESCRIPTION CONTEXT:
    {jd_text}

    Instructions:
    1. Provide realistic `salary_range_india` (in LPA, e.g., '12 - 28 LPA') and `salary_range_us` (in USD).
    2. Classify `demand_level` ('High Demand', 'Moderate Demand', or 'Competitive').
    3. Extract 5 to 7 `trending_skills` that top companies are pairing with {target_role} right now.
    4. Write a concise, data-backed `market_insights` summary.
    """

    market_report: MarketIntelReport = structured_llm.invoke(prompt)
    return {"market_intel": market_report}