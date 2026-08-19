# 🔎 TruthLens AI

### Explainable Agentic Fact-Checking Assistant

TruthLens AI is an AI-powered fact-checking system that verifies factual claims using **hybrid evidence retrieval, semantic filtering, web search, LLM reasoning, and an agentic retry loop**.

Unlike a simple LLM-based chatbot, TruthLens retrieves evidence, evaluates its relevance and source quality, reasons over the evidence, and performs additional searches when confidence is insufficient.

---

## 🚀 Key Features

- 🔍 **Claim Analysis**
  - Analyzes factual, temporal, subjective, and question-based claims.
  - Generates targeted search queries.

- 📚 **Hybrid Evidence Retrieval**
  - Retrieves evidence from a local knowledge base using FAISS.
  - Searches the public web using DDGS.

- 🔬 **Semantic Evidence Filtering**
  - Filters irrelevant evidence using sentence-transformer embeddings.

- 🏆 **Source Quality Analysis**
  - Evaluates the reliability of evidence sources.

- 🤖 **LLM-Based Reasoning**
  - Uses Groq-powered LLM reasoning to determine whether evidence supports or contradicts a claim.

- 🔁 **Agentic Retry Loop**
  - Searches again when confidence is low or evidence is insufficient.
  - Refines the search query based on missing evidence.

- 📊 **Trust-Aware Confidence**
  - Combines:
    - LLM confidence
    - Evidence relevance
    - Source quality
    - Evidence agreement

- 💡 **Explainable Results**
  - Provides:
    - Verdict
    - Confidence
    - Explanation
    - Evidence
    - Missing evidence
    - Agent iterations
    - Search statistics

- ⚡ **FastAPI Backend**
- 🖥️ **Streamlit Frontend**
- 🧪 **Automated Testing with Pytest**

---

## 🧠 How It Works

```text
User Claim
    │
    ▼
Claim Analysis
    │
    ▼
Query Planning
    │
    ▼
Hybrid Evidence Retrieval
    ├───────────────┐
    ▼               ▼
Local FAISS      Web Search
Evidence         DDGS
    │               │
    └───────┬───────┘
            ▼
Semantic Evidence Filtering
            │
            ▼
      LLM Reasoning
            │
            ▼
   Confidence Evaluation
            │
       ┌────┴────┐
       │         │
      Low      High
       │         │
       ▼         ▼
 Refine Query  Final Verdict
       │
       └──► Search Again
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| FastAPI | REST API backend |
| Streamlit | Web interface |
| Groq | LLM reasoning |
| Sentence Transformers | Semantic embeddings |
| FAISS | Vector similarity search |
| DDGS | Web search |
| BeautifulSoup | Web content extraction |
| Pydantic | Data validation |
| Pytest | Automated testing |
| python-dotenv | Environment configuration |

---

## 📂 Project Structure

```text
TruthLensAi/
│
├── agents/
│   ├── evidence_agent.py
│   ├── fact_check_agent.py
│   └── search_agent.py
│
├── backend/
│   ├── main.py
│   └── config_test.py
│
├── data/
│   └── sample_dataset.json
│
├── evaluation/
│   └── metrics.py
│
├── frontend/
│   └── app.py
│
├── rag/
│   └── reasoner.py
│
├── retriever/
│   ├── embedder.py
│   ├── evidence_filter.py
│   └── faiss_index.py
│
├── tests/
│   └── test_*.py
│
├── utils/
│   ├── claim_analyzer.py
│   ├── claim_extractor.py
│   ├── confidence.py
│   ├── config.py
│   ├── dataset_loader.py
│   ├── pipeline.py
│   ├── query_planner.py
│   ├── schemas.py
│   └── source_quality.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/aalokTiwar/TruthLensAi.git
cd TruthLensAi
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

The repository contains `.env.example` as a safe configuration template.

**Never commit your `.env` file or API keys to GitHub.**

---

## ▶️ Running the Application

TruthLens AI uses a FastAPI backend and Streamlit frontend.

### Start the FastAPI backend

Open **Terminal 1**:

```bash
uvicorn backend.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Start the Streamlit frontend

Open **Terminal 2**:

```bash
streamlit run frontend/app.py
```

Frontend:

```text
http://localhost:8501
```

---

## 🔌 API

### Health Check

```http
GET /health
```

Example:

```json
{
  "status": "healthy"
}
```

### Verify a Claim

```http
POST /verify
```

Request:

```json
{
  "text": "Humans landed on Mars in 2025."
}
```

The API returns:

- Verdict label
- Explanation
- Confidence score
- Evidence
- Missing evidence
- Confidence level
- Evidence statistics
- Agent iterations
- Search statistics

Possible verdicts:

```text
TRUE
FALSE
NOT_ENOUGH_EVIDENCE
```

---

## 📊 Confidence Model

TruthLens uses multiple signals to calculate trust-aware confidence:

```text
LLM Confidence
Evidence Relevance
Source Quality
Evidence Agreement
```

The final confidence is used to classify results into:

```text
VERY_HIGH
HIGH
MEDIUM
LOW
VERY_LOW
```

This allows the system to communicate not only **what it believes**, but also **how confident it is**.

---

## 🔁 Agentic Verification

TruthLens does not necessarily stop after the first search.

When evidence is insufficient or confidence is low, the agent:

1. Analyzes the missing evidence.
2. Refines the search query.
3. Performs another retrieval.
4. Filters the new evidence.
5. Reasons over the updated evidence.
6. Returns the strongest available verdict.

This makes TruthLens an **agentic research workflow rather than a single LLM prompt**.

---

## 🧪 Testing

The project includes automated tests for:

- API endpoints
- Claim analysis
- Claim extraction
- Confidence calculation
- Dataset loading
- Embeddings
- FAISS retrieval
- Evidence filtering
- Agentic retry behavior
- Query planning
- RAG reasoning
- Web search
- Source quality
- Data schemas

Run the complete test suite:

```bash
pytest tests/ -q
```

Current test result:

```text
78 passed
```

---

## 🖥️ Example Claims

TruthLens can verify claims such as:

```text
Humans landed on Mars in 2025.
```

```text
The Earth revolves around the Sun.
```

```text
Humans are mammals.
```

```text
Cycling improves stamina.
```

```text
WWE is owned by John Cena.
```

The system returns an explainable verdict supported by retrieved evidence rather than simply generating a TRUE/FALSE response.

---

## 🔒 Security

- API keys are stored in environment variables.
- `.env` is excluded from Git using `.gitignore`.
- `.env.example` is provided as a safe template.
- API credentials are never required to be stored in source code.

**Never expose your Groq API key in source code or public repositories.**

---

## ⚠️ Limitations

TruthLens AI is an experimental AI/ML project and should not be treated as an authoritative source of truth.

Potential limitations include:

- Web search availability
- Search result quality
- Webpage accessibility
- LLM reasoning errors
- Semantic retrieval errors
- Conflicting online sources
- Time-sensitive information
- Limited local knowledge-base coverage

The system therefore exposes evidence and confidence information to make the verification process more transparent.

---

## 🚀 Future Improvements

- More authoritative source prioritization
- Improved claim decomposition
- Parallel web retrieval
- Evidence caching
- Citation extraction and source highlighting
- Cross-source contradiction detection
- Larger benchmark datasets
- Advanced evaluation metrics
- Multi-language fact checking
- Docker deployment
- Cloud deployment

---

## 🎯 Skills Demonstrated

This project demonstrates practical experience in:

- Artificial Intelligence
- Agentic AI
- Retrieval-Augmented Generation (RAG)
- Natural Language Processing
- Semantic Search
- Vector Databases
- LLM Integration
- Web Information Retrieval
- Explainable AI
- Confidence Modeling
- FastAPI
- Streamlit
- Automated Testing
- Python Software Development

---

## 👨‍💻 Author

**Aalok Tiwari**

MSc Computer Science  
Data Science | Machine Learning | AI & NLP

GitHub:  
https://github.com/aalokTiwar

---

## 📄 License

This project is intended for educational and research purposes.