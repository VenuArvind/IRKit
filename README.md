# 🚀 IRKit: Distributed Hybrid Information Retrieval Toolkit

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

**IRKit** is a modular, high-performance Information Retrieval (IR) library built for the modern era of Semantic Search. It combines **BM25 Keyword Search** with **Vector Embeddings** and **Cross-Encoder Reranking** to provide "Google-tier" search precision.

---

## ✨ Key Features

- **🔍 Hybrid Search**: Combines BM25 and Semantic search using **Reciprocal Rank Fusion (RRF)**.
- **⚡ Semantic Caching**: Vector-based query caching that skips redundant model inference (8.2x faster search).
- **🧠 Two-Stage Retrieval**: State-of-the-art pipeline using **Cross-Encoders** (MiniLM) for high-precision document re-scoring.
- **📊 Quality Metrics**: Built-in support for **MRR (Mean Reciprocal Rank)** and **nDCG (Normalized Discounted Cumulative Gain)**.
- **⚡ Performance First**: Integrated **Latency Metrics** (P50, P95, P99) and automated benchmark suites.
- **🌐 Pluggable Architecture**: Easily swap between different Data Sources (ArXiv, Wikipedia, RSS), Embedders, and Storage modes.
- **🚀 Web-Ready**: Built-in **FastAPI** serving layer with a beautiful **React-powered Demo UI**.

---

## 🎨 Frontend Demo

Experience the engine visually through our React-powered search interface.

![IRKit Demo](https://raw.githubusercontent.com/VenuArvind/IRKit/main/demo/src/assets/hero.png)

1. **Start the API**: `irkit serve --source arxiv --max-docs 500`
2. **Start the UI**: `cd demo && npm run dev`
3. **Open**: `http://localhost:5173`

---

## 📊 Benchmarks & Quality

*Verified actual numbers (M5 MacBook Pro).*

### ⚡ Search Speed (Semantic Caching)
| Search Type | Latency (ms) | Speedup |
|-------------|--------------|---------|
| **Cold Search** | 27.61 | 1.0x |
| **Semantic Cache (Similar Query)** | 19.14 | **1.4x** |
| **Hot Cache (Exact Match)** | 3.36 | **8.2x** |

### 🔬 Scientific Precision (ArXiv - 200 docs)
| Ranking Mode | Mean MRR | Mean nDCG@10 |
|--------------|----------|--------------|
| BM25 Only | 0.8750 | 1.0000 |
| Semantic Only | 1.0000 | 1.0000 |
| **Hybrid + Reranking** | **1.0000** | **1.0000** |

---

## 🏗️ Technical Architecture

IRKit is built on four core pillars:

1.  **Sources**: Native ingestors for ArXiv, Wikipedia, News RSS, and Local Files.
2.  **Embedders**: Support for local HuggingFace models or OpenAI API.
3.  **Rankers**: Advanced relevance algorithms including BM25, Semantic FAISS, Hybrid RRF, and Cross-Encoder Reranking.
4.  **Core Services**: Latency tracking, Semantic Caching, and Scientific Evaluation modules.

---

## 🧪 Testing

```bash
# Run full test suite with coverage
pytest tests/ -v --cov=irkit

# Run performance & quality benchmarks
python3 scripts/evaluate_quality.py
python3 scripts/benchmark_cache.py
```

---

Made with ❤️ by Venu Arvind
