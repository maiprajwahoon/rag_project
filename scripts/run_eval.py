import hashlib
from pathlib import Path

from rag_project.evaluation.golden import load_golden
from rag_project.evaluation.runner import (
    evaluate_retrieval,
    format_report,
    save_results,
)
from rag_project.retrieval.factory import build_default_retriever

K = 5
golden_path = Path("eval/golden.jsonl")
sha = hashlib.sha256(golden_path.read_bytes()).hexdigest()[:10]
items = load_golden(golden_path)
retriever = build_default_retriever(k=K)
report = evaluate_retrieval(retriever, items, k=K, golden_sha=sha)
print(format_report(report))
print("saved:", save_results(report, Path("eval/results")))
