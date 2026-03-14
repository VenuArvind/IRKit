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
- **🎨 Search Demo**: A beautiful React-based frontend to showcase the search capabilities.

---

## 🎨 Frontend Demo

Experience the engine visually through our React-powered search interface.

![IRKit Demo](https://raw.githubusercontent.com/VenuArvind/IRKit/main/demo/src/assets/hero.png)

1. **Start the API**: `irkit serve --max-docs 500`
2. **Start the UI**: `cd demo && npm run dev`
3. **Open**: `http://localhost:5173`

---

## 📊 Benchmarks

*Tested on 1,000 ArXiv papers, M3 Pro MacBook, Single Node.*

| Ranker | p50 (ms) | p95 (ms) | p99 (ms) |
|--------|----------|----------|----------|
| **BM25** | 1.2 | 4.5 | 8.2 |
| **Semantic** | 12.4 | 18.2 | 25.1 |
| **Hybrid (RRF)** | 14.1 | 21.5 | 32.4 |

---

## 🧪 Testing

We maintain high standards for search accuracy and system reliability.

```bash
# Run full test suite with coverage
pytest tests/ -v --cov=irkit
```

- **Unit Tests**: Verified Ranking logic, Consistent Hashing, and Metric calculations.
- **Integration Tests**: End-to-end `Source -> Indexer -> Storage -> Ranker` validation.

---

## 🏗️ Architecture

IRKit is built on four core pillars:

1.  **Sources**: Ingest data from ArXiv, Wikipedia, or your own local JSON/CSV files.
2.  **Embedders**: Generate vector embeddings using local (HuggingFace) or API (OpenAI) models.
3.  **Rankers**: Search using traditional keywords (BM25) or conceptual meaning (Semantic).
4.  **Storage**: Persist documents in memory or distributed Redis clusters.

---

## 📖 Python Usage

```python
from irkit import IndexEngine, ArxivSource, HybridRanker, BM25Ranker, SemanticRanker, HuggingFaceEmbedder, MemoryStorage

# 1. Initialize Components
embedder = HuggingFaceEmbedder()
ranker = HybridRanker(rankers=[BM25Ranker(), SemanticRanker(embedder)])
engine = IndexEngine(ranker=ranker, storage=MemoryStorage())

# 2. Index Data
engine.index(ArxivSource(), max_docs=50)

# 3. Search
results = engine.search("quantum computing")
for res in results:
    print(f"[{res.score:.4f}] {res.title}")

# 4. Check Stats
print(engine.stats())
```

---

## 🤝 Contributing

Contributions are welcome! Whether you want to add a new Data Source or a specialized Ranker, please feel free to open a Pull Request.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
