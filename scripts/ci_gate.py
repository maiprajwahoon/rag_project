import sys
from pathlib import Path

from rag_project.evaluation.gate import check_thresholds
from rag_project.evaluation.golden import load_golden
from rag_project.evaluation.metrics import first_hit_rank
from rag_project.indexing.embedder import Embedder
from rag_project.indexing.store import QdrantStore

K = 5
MINIMUMS = {"hit_rate": 0.90, "mrr": 0.70}

items = [i for i in load_golden(Path("eval/golden.jsonl")) if i.answerable]
store = QdrantStore()
embedder = Embedder()
hits = 0
reciprocal_ranks: list[float] = []
for item in items:
    hits_for_item = store.search(embedder.embed_query(item.question), K)
    pages = [chunk.source_path for chunk, _ in hits_for_item]
    rank = first_hit_rank(pages, item.pages)
    hits += rank is not None
    reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)

metrics = {
    "hit_rate": hits / len(items),
    "mrr": sum(reciprocal_ranks) / len(items),
}
print(f"vector retrieval on {len(items)} answerable questions (k={K})")
for name, value in metrics.items():
    print(f"  {name}: {value:.3f} (minimum {MINIMUMS[name]:.2f})")
failures = check_thresholds(metrics, MINIMUMS)
for failure in failures:
    print("FAIL:", failure)
sys.exit(1 if failures else 0)
