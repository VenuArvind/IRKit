# 🚀 IRKit: Distributed Hybrid Information Retrieval Toolkit

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

**IRKit** is a modular, pluggable Python SDK designed for building high-performance search systems. It combines the power of **Keyword Search (BM25)** and **Semantic Search (Vector Embeddings)** into a single, cohesive engine.

---

## ✨ Key Features

- **🔍 Hybrid Search**: Combines BM25 and Semantic search using **Reciprocal Rank Fusion (RRF)** for superior relevance.
- **🌐 Pluggable Architecture**: Easily swap between different Data Sources, Embedders, Rankers, and Storage backends.
- **⚡ Performance First**: Integrated **Latency Metrics** (P50, P95, P99) to track search speeds in real-time.
- **💎 Distributed Sharding**: Scalable persistent storage using **Consistent Hashing** (ready for Redis/Postgres).
- **🚀 Web-Ready**: Built-in **FastAPI** serving layer with auto-generated Swagger documentation.
- **💻 Modern CLI**: Professional interactive terminal interface powered by **Typer** and **Rich**.
- **🎨 Search Demo**: A beautiful React-based frontend with real-time "Advanced Reranking" toggles.
- **🧠 Two-Stage Retrieval**: State-of-the-art pipeline using **Cross-Encoders** (MiniLM) for high-precision document re-scoring.

---

## 🎨 Frontend Demo

Experience the engine visually through our React-powered search interface.

![IRKit Demo](https://raw.githubusercontent.com/VenuArvind/IRKit/main/demo/src/assets/hero.png)

1. **Start the API**: `irkit serve --max-docs 500 --source arxiv`
2. **Start the UI**: `cd demo && npm run dev`
3. **Open**: `http://localhost:5173`

---

## 📊 Benchmarks

*Verified actual numbers (M3 Pro MacBook).*

### 📚 Scientific (ArXiv - 500 docs)
| Ranker | p50 (ms) | p95 (ms) | p99 (ms) |
|--------|----------|----------|----------|
| **Hybrid** | 3.48 | 3.86 | 4.41 |
| **Reranked** | 73.86 | 125.66 | 235.05 |

### 📖 General (Wikipedia - 100 docs)
| Ranker | p50 (ms) | p95 (ms) |
|--------|----------|----------|
| **Hybrid** | 3.34 | 17.53 |
| **Reranked** | 43.39 | 63.03 |

### 📰 Real-time (News RSS - 100 docs)
| Ranker | p50 (ms) | p95 (ms) |
|--------|----------|----------|
| **Hybrid** | 3.34 | 4.92 |
| **Reranked** | 56.06 | 96.37 |

---

## 🧪 Testing

We maintain high standards for search accuracy and system reliability.

```bash
# Run full test suite with coverage
pytest tests/ -v --cov=irkit
```

---

## 🏗️ Architecture

IRKit is built on four core pillars:

1.  **Sources**: Ingest data from ArXiv, Wikipedia, or your own local JSON/CSV files.
2.  **Embedders**: Generate vector embeddings using local (HuggingFace) or API (OpenAI) models.
3.  **Rankers**: Search using traditional keywords (BM25) or conceptual meaning (Semantic).
4.  **Storage**: Persist documents in memory or distributed Redis clusters.

---

Made with ❤️ by Venu Arvind
