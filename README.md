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

---

## 🛠️ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/VenuArvind/IRKit.git
cd IRKit

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Search Server

Start the engine and ingest real research papers from ArXiv:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 cli/main.py serve --max-docs 100
```

Now visit [http://localhost:8000/docs](http://localhost:8000/docs) to explore the API!

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
