# Codebase RAG

Multi-stage RAG system for intelligent code search and analysis.

## Overview

Project exploring production RAG patterns: ensemble retrieval (vector + BM25), cross-encoder re-ranking, and multi-agent orchestration with LangGraph.

Built to understand how AI-powered code assistants work internally, as a cost-effective alternative to full-context LLM approaches.

## Features

- 🔍 **Ensemble Retrieval**: Combines semantic (vector) and keyword (BM25) search
- 🎯 **Cross-Encoder Re-Ranking**: Improves result relevance with domain-specific models
- 🤖 **Multi-Agent System**: LangGraph orchestration with specialized tools
- 💰 **Cost-Effective**: ~60x cheaper than full-context approaches
- 🔒 **Privacy-First**: Local embeddings, optional local LLM support

## Tech Stack

- **LLM**: Gemini API (swappable)
- **Embeddings**: Ollama (mxbai-embed-large)
- **Vector Store**: ChromaDB
- **Re-Ranking**: mixedbread-ai reranker
- **Orchestration**: LangGraph
- **Framework**: LangChain

## Architecture
```
Query → Multi-Query Expansion → Ensemble Retrieval (Vector + BM25) 
→ Cross-Encoder Re-Ranking → Agent Tool Selection → Response
```

## Use Case

Developer onboarding assistant - helps new developers understand codebases through natural language queries.

## Status

🚧 Active development - learning project, not production-ready.

Built to explore RAG internals and agentic workflows.

## Setup
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/codebase-rag.git

# Install dependencies
pip install -r requirements.txt

# Run
python agent.py
```

## Learning Goals

- Production RAG patterns (retrieval, re-ranking, chunking)
- Multi-agent orchestration with LangGraph
- Trade-offs: cost vs accuracy, local vs API, speed vs quality

---