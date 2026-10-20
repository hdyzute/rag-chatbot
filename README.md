# VietGuard AI: Automated Legal Contract Risk Auditor & Redlining Copilot

VietGuard AI is a domain-specific LegalTech system designed to protect Vietnamese employees, tenants, and freelancers from unlawful, predatory, or high-risk clauses in contracts (e.g., labor contracts, residential lease agreements, NDA/training bonds).

Instead of generic document Q&A, VietGuard AI functions as an end-to-end contract auditor. It segments contracts clause-by-clause, cross-references each clause against Vietnamese statutory laws via Hybrid Search, detects illegal traps, cites governing legal articles, and generates counter-clauses (redline suggestions) for negotiation.

## Core Capabilities

* **Structure-Aware Legal Clause Segmentation**: Unlike standard character-based chunking that breaks sentences arbitrarily, the custom `LegalClauseParser` uses legal syntax boundaries (`Điều [0-9]+`, `Khoản`) to preserve each clause intact with its complete legal context.
* **Hybrid Legal Knowledge Base**: Integrates core Vietnamese statutory documents (Labor Code 2019, Civil Code 2015, Housing Law 2023) using an `EnsembleRetriever` combining dense vector search (ChromaDB + `nomic-embed-text`) and sparse keyword matching (BM25). This ensures exact matching of specific legal terms and statutory articles.
* **Risk Categorization & Redline Generation**: Classifies clauses into three risk tiers:
  * RED: Unlawful or legally void provisions (e.g., withholding original citizen ID/diplomas, illegal wage deductions, excessive probation periods, unauthorized premises entry).
  * YELLOW: Highly one-sided or predatory terms (e.g., unconditional 100% deposit forfeiture, unverified internal training penalties).
  * GREEN: Compliant, standard legal terms.
  Generates negotiation-ready counter-proposals that users can copy directly to negotiate with employers or landlords.
* **Interactive Legal Assistant**: Embedded session-based RAG chat allowing users to ask contextual follow-up questions regarding potential litigation risks and statutory rights.
* **Automated Evaluation Benchmark**: Includes `evaluate_legal_benchmark.py` running against a curated test suite of 10 real-world contract traps, achieving 100% recall on severe illegal clauses (RED).

## System Architecture

The pipeline processes contracts through five stages:
1. Ingestion: Accepts PDF uploads, plain text files, or raw text input.
2. Clause Parsing: Isolates each legal provision while maintaining structural metadata.
3. Statutory Retrieval: Executes parallel dense and sparse retrieval across indexed Vietnamese laws.
4. Risk Assessment: Evaluates compliance, cites governing articles, and drafts alternative text.
5. Interactive Q&A: In-memory ChromaDB vectorstore per session for multi-turn advisory chat.

## Tech Stack

* Backend: Flask, Flask-CORS
* Orchestration: LangChain (LCEL)
* Local LLM & Embeddings: Ollama (`qwen2.5:3b`, `nomic-embed-text`)
* Vector Store: ChromaDB (persistent legal knowledge base and ephemeral session stores)
* Sparse Keyword Search: rank_bm25
* Document Parsing: pypdf, regex-based statutory tokenizer
* Frontend: HTML5, Tailwind CSS, Lucide Icons

## Setup & Installation

1. Prerequisites:
Ensure Python 3.11 is installed. Download and install [Ollama](https://ollama.com/).

2. Download required local models:
```bash
ollama serve
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

3. Setup virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open `index.html` or navigate to `http://localhost:5000` in your web browser. You can click on the pre-loaded sample contracts (e.g., Labor Contract with Illegal Deductions, Rental Agreement with Deposit Traps) to test the analysis immediately.

## Running the Benchmark Evaluation

To reproduce the benchmark evaluation and verify model recall on real-world contract traps:
```bash
python evaluate_legal_benchmark.py
```
This runs the test suite in `legal_trap_benchmark.json` and exports full precision and recall metrics to `legal_benchmark_results.json`.
