import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import datasets
import numpy as np

from irkit import (
    IndexEngine,
    HybridRanker,
    BM25Ranker,
    SemanticRanker,
    SentenceTransformerEmbedder,
    InMemoryStorage,
    CrossEncoderRanker,
)
from irkit.core.evaluation import calculate_mrr, calculate_ndcg
from irkit.sources.base import BaseSource, Document

class SciFactSource(BaseSource):
    """Loads BEIR SciFact corpus from Hugging Face MTEB dataset."""

    def __init__(self, corpus_ds):
        self.corpus_ds = corpus_ds

    def load(self, max_docs: int = None) -> List[Document]:
        docs = []
        limit = max_docs if max_docs is not None else len(self.corpus_ds)
        for i in range(min(limit, len(self.corpus_ds))):
            item = self.corpus_ds[i]
            docs.append(
                Document(
                    id=str(item["_id"]),
                    title=item["title"] or "",
                    text=(item["title"] or "") + " " + (item["text"] or ""),
                    metadata={"doc_id": str(item["_id"])},
                )
            )
        return docs

def run_scifact_evaluation(sample_queries: int = 150):
    print("=" * 75)
    print("BEIR SciFact Benchmark Evaluation: Multi-Stage Hybrid IR Precision")
    print("=" * 75)

    print("\n[1/4] Downloading BEIR SciFact dataset (MTEB)...")
    corpus_ds = datasets.load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = datasets.load_dataset("mteb/scifact", "queries", split="queries")
    qrels_ds = datasets.load_dataset("mteb/scifact", "default", split="test")

    print(f"Loaded {len(corpus_ds)} documents and {len(queries_ds)} candidate queries.")

    # Build ground truth dictionary: query_id -> set of relevant corpus_ids
    ground_truth: Dict[str, Set[str]] = defaultdict(set)
    for row in qrels_ds:
        if row["score"] > 0:
            ground_truth[str(row["query-id"])].add(str(row["corpus-id"]))

    # Map query_id -> query text
    query_text_map: Dict[str, str] = {str(row["_id"]): row["text"] for row in queries_ds}

    # Filter to test queries that exist in ground truth
    eval_queries = [(qid, query_text_map[qid], ground_truth[qid]) for qid in ground_truth if qid in query_text_map]
    if sample_queries and len(eval_queries) > sample_queries:
        eval_queries = eval_queries[:sample_queries]

    print(f"Evaluating across {len(eval_queries)} labeled SciFact queries...")

    # [2/4] Index Corpus
    print("\n[2/4] Indexing SciFact corpus into IRKit engine...")
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    bm25 = BM25Ranker()
    semantic = SemanticRanker(embedder)
    hybrid = HybridRanker(rankers=[bm25, semantic])
    reranker = CrossEncoderRanker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    engine = IndexEngine(ranker=hybrid, storage=InMemoryStorage(), reranker=reranker)

    t0_idx = time.perf_counter()
    source = SciFactSource(corpus_ds)
    engine.index(source)
    idx_time = time.perf_counter() - t0_idx
    print(f"Successfully indexed {len(corpus_ds)} documents in {idx_time:.2f} s.")

    # [3/4] Evaluate across configurations
    modes = [
        ("BM25 Only", bm25, False),
        ("Dense (MiniLM-L6)", semantic, False),
        ("Hybrid (RRF)", hybrid, False),
        ("Hybrid + Reranking", hybrid, True),
    ]

    metrics_report = {}
    print("\n[3/4] Executing retrieval pipelines...")

    for mode_name, ranker_obj, use_rerank in modes:
        engine.ranker = ranker_obj
        ndcg_list = []
        mrr_list = []
        latencies = []

        for qid, qtext, gold_ids in eval_queries:
            t0 = time.perf_counter()
            results = engine.search(qtext, top_k=10, rerank=use_rerank)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            ndcg = calculate_ndcg(gold_ids, results, k=10)
            mrr = calculate_mrr(gold_ids, results)
            ndcg_list.append(ndcg)
            mrr_list.append(mrr)

        mean_ndcg = float(np.mean(ndcg_list))
        mean_mrr = float(np.mean(mrr_list))
        p50_lat = float(np.percentile(latencies, 50))
        p95_lat = float(np.percentile(latencies, 95))

        metrics_report[mode_name] = {
            "mean_ndcg_at_10": round(mean_ndcg, 4),
            "mean_mrr": round(mean_mrr, 4),
            "p50_latency_ms": round(p50_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
        }

    # [4/4] Output & Save Report
    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)
    report_file = output_dir / "beir_scifact_evaluation.json"
    with open(report_file, "w") as f:
        json.dump(metrics_report, f, indent=2)

    print("\n" + "=" * 75)
    print("BEIR SCIFACT RETRIEVAL EVALUATION REPORT")
    print("=" * 75)
    print(f"{'Retrieval Pipeline':<24} | {'nDCG@10':<10} | {'MRR':<10} | {'P50 Latency':<12} | {'P95 Latency':<12}")
    print("-" * 75)
    for mode, data in metrics_report.items():
        print(
            f"{mode:<24} | {data['mean_ndcg_at_10']:<10.4f} | {data['mean_mrr']:<10.4f} | "
            f"{data['p50_latency_ms']:>8.2f} ms | {data['p95_latency_ms']:>8.2f} ms"
        )
    print("=" * 75)
    print(f"Results exported to: {report_file}")

if __name__ == "__main__":
    run_scifact_evaluation(sample_queries=150)

