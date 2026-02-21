# Multi-Agent Resume Screening System with Hybrid RAG

A production-ready AI-powered ATS that uses a 3-agent pipeline to automatically screen resumes, rank candidates, and generate personalized feedback reports.

**DeepEval Score: 92.35% | Status: PRODUCTION READY**

## Architecture
```
Resume + JD
     |
     v
Agent 1 — Parser (Groq LLaMA 3.3 70B)
     |
     v
Agent 2 — Hybrid Search + ATS Scorer
     |    BGE-large + BM25 + Qdrant RRF + Cross-Encoder
     |
     v
Agent 3 — Feedback Generator (Groq LLaMA 3.3 70B)
```

## ATS Scoring Formula

| Component | Weight |
|---|---|
| Required Skills Match | 40% |
| Semantic Match | 30% |
| Experience Score | 15% |
| Preferred Skills Match | 10% |
| Certifications | 5% |

## Evaluation Results (DeepEval)

| Metric | Score | Status |
|---|---|---|
| Answer Relevancy | 0.96 | PASSED |
| Faithfulness | 0.88 | PASSED |
| Contextual Precision | 0.98 | PASSED |
| Contextual Recall | 1.00 | PASSED |
| Contextual Relevancy | 0.80 | PASSED |
| Overall | 0.9235 | PRODUCTION READY |

## Tech Stack

- LLM: Groq LLaMA 3.3 70B
- Embeddings: BAAI/bge-large-en-v1.5
- Sparse Retrieval: BM25
- Vector Database: Qdrant Cloud
- Fusion Ranking: RRF
- Reranking: Cross-Encoder ms-marco-MiniLM-L-6-v2
- Agent Framework: CrewAI Flows
- Evaluation: DeepEval
- Frontend: Streamlit

## Setup
```bash
git clone https://github.com/sapavatcharan/resume-matcher.git
cd resume-matcher
python3 -m venv venv
source venv/bin/activate
pip install streamlit groq qdrant-client sentence-transformers rank-bm25 numpy requests PyPDF2 crewai deepeval pydantic
streamlit run app/streamlit_app.py
```

## Notebooks

| Notebook | Description |
|---|---|
| NB1_Resume_JD_Parser | Resume and JD parsing with Pydantic schemas |
| NB2_Hybrid_Search_Engine | BGE + BM25 + RRF + Cross-Encoder pipeline |
| NB3_CrewAI_Flows_Agents | 3-agent CrewAI automation pipeline |
| NB4_DeepEval_Evaluation | Evaluation across 5 metrics, 92.35% score |

## Features

- PDF upload or text paste for resume and JD
- Real-time 3-agent progress tracking
- Visual ATS score display with color coding
- Skill tags — green matched, red missing, blue preferred
- Progress bars for each scoring component
- Downloadable personalized feedback report
- Dual dashboard — Candidate portal and Recruiter dashboard

## Author

Sapavat Charan Kumar
Final Year Student, IIIT Kota
GitHub: https://github.com/sapavatcharan
