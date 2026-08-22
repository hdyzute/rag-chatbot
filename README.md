# Local RAG Chatbot with Hybrid Search & Agentic Routing

A session-based RAG chatbot built with Flask, LangChain, and Ollama. It runs entirely locally for data privacy and features an intelligent routing system to seamlessly switch between document retrieval, web search, and direct conversation.

## Features

* **Hybrid Search Retrieval**: Combines semantic search (ChromaDB) and sparse keyword search (BM25) via an Ensemble Retriever. This solves the common issue of vector databases failing at exact keyword matches, ensuring high accuracy when retrieving specific product codes or financial figures.
* **Agentic Query Routing**: Uses the local LLM to classify user intents and route queries to either `RETRIEVE` (document search), `WEB_SEARCH` (DuckDuckGo integration for real-time information), or `DIRECT` (general chat).
* **Automated Evaluation Pipeline**: Includes a standalone `evaluate.py` script implementing an LLM-as-a-judge framework. It automatically benchmarks Context Relevance, Faithfulness, and Answer Relevance against a ground-truth dataset (e.g., corporate annual reports).
* **Contextual Memory**: Maintains chat history per session and leverages a rephrase chain to convert follow-up questions into standalone queries, optimizing retrieval performance.

## Tech Stack

* **Backend**: Flask
* **Orchestration**: LangChain (LCEL architecture)
* **Local LLM & Embeddings**: Ollama (`qwen2.5:3b` for generation/routing, `nomic-embed-text` for vector embeddings)
* **Database**: ChromaDB (persistent storage per session)
* **Keyword Search**: rank_bm25
* **Web Search**: DuckDuckGo API

## Setup Instructions

1. Install Python 3.11 or 3.12 (Highly recommended for optimal compatibility with underlying C-extensions on Windows).
2. Install [Ollama](https://ollama.com/) and download the required models by running:
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```
3. Clone the repository and navigate into it.
4. Create a virtual environment and install the dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

Start the backend server:
```bash
python app.py
```
Then, simply open the `index.html` file in your web browser to access the chat interface.

## Running the Evaluation

To benchmark the RAG pipeline using the included sample corporate dataset, run:
```bash
python evaluate.py
```
The script will output the evaluation process to the console and save detailed metrics to `evaluation_results.json`.
