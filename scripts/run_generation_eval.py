import hashlib
import sys
from pathlib import Path

from rag_project.evaluation.generation_eval import (
    CachedLLM,
    ItemOutcome,
    evaluate_generation,
    format_generation_report,
    save_generation_report,
)
from rag_project.evaluation.golden import load_golden
from rag_project.generation.generator import Generator
from rag_project.generation.llm import DEFAULT_MODEL, GeminiClient
from rag_project.indexing.embedder import Embedder
from rag_project.indexing.store import QdrantStore
from rag_project.rag import RagService

K = 5
use_judge = "--judge" in sys.argv
interval = 6.0
if "--interval" in sys.argv:
    interval = float(sys.argv[sys.argv.index("--interval") + 1])

golden_path = Path("eval/golden.jsonl")
sha = hashlib.sha256(golden_path.read_bytes()).hexdigest()[:10]
items = load_golden(golden_path)

llm = CachedLLM(
    GeminiClient(),
    Path("eval/cache/llm_cache.json"),
    min_interval=interval,
    namespace=DEFAULT_MODEL,
)
service = RagService(Embedder(), QdrantStore(), Generator(llm), k=K)


def progress(index: int, total: int, outcome: ItemOutcome) -> None:
    state = "refused" if outcome.refused else "answered"
    print(
        f"[{index}/{total}] {outcome.id} {state} (real calls so far: {llm.real_calls})",
        flush=True,
    )


report = evaluate_generation(
    service, items, llm if use_judge else None, sha, DEFAULT_MODEL, K, progress
)
print()
print(format_generation_report(report))
print("saved:", save_generation_report(report, Path("eval/results")))
