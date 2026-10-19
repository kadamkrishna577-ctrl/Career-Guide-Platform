# 🧠 Career Intelligence Platform

An intelligent, multi-agent career platform built with **LangGraph**, **Google Gemini**, and **Streamlit**. It automates resume analysis, evaluates ATS compatibility, provides principal-level interview coaching, and builds personalized 4-week engineering roadmaps.

---

## 🌟 Key Features

* **7-Agent LangGraph Pipeline:** Combines sequential analysis with parallel fan-out execution to evaluate resumes efficiently.
* **Dual ATS Scorer:** Analyzes keyword matching and semantic vector similarity against target job descriptions.
* **Human-in-the-Loop (HITL) Checkpoint:** Pauses execution to let you review and customize AI-generated resume bullet points before building your final strategy.
* **Principal Interview Prep:** Generates deep technical interview questions with STAR answer frameworks and interviewer hidden intents.
* **Resilient API Key Rotation:** Thread-safe round-robin API rotator with automatic failover for rate limits (`429`), server traffic spikes (`503`), and invalid keys (`400`).
* **Session Persistence:** Saves all agent outputs locally using SQLite (`SqliteSaver`), allowing you to resume sessions seamlessly.

---

## 🛠️ Architecture & Agent Flow

1. **Agent 1 (Resume Parser):** Extracts clean text, skills, education, and project details from uploaded PDF resumes.
2. **Agent 2 (Dual ATS Scorer):** Computes keyword density and dense embedding similarity against the target job description.
3. **Agent 3 (Gap Analyzer):** Identifies skill gaps and provides concrete project upgrade blueprints.
4. **Agent 4 (Resume Rewriter):** Upgrades weak resume bullets using impact formulas and keyword injection.
5. **Agent 5 (Interview Prep):** Creates rigorous, scenario-based technical interview questions and STAR guides.
6. **Agent 6 (Market Intel):** Gathers real-time salary ranges and trending skills using Tavily Web Search.
7. **Agent 7 (Career Strategist):** Synthesizes all data into an exhaustive 4-week technical execution roadmap.

---

## 📂 Project Structure

```text
career-intelligence-platform/
├── src/
│   ├── agents/          # Individual agent nodes (Parser, ATS, Gap, Rewriter, etc.)
│   ├── config.py        # Rotating multi-key Gemini wrapper & error failover
│   ├── graph.py         # LangGraph state machine & parallel execution flow
│   ├── schemas.py       # Pydantic data models for structured LLM outputs
│   └── state.py         # Central state dictionary definition
├── app.py               # 5-Tab Streamlit interactive UI dashboard
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
