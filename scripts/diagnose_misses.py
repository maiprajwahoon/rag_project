from pathlib import Path

from rag_project.evaluation.golden import load_golden
from rag_project.retrieval.factory import build_default_retriever

wanted = {"q04", "q20", "h01", "h02", "h05", "h06"}
items = [i for i in load_golden(Path("eval/golden.jsonl")) if i.id in wanted]
retriever = build_default_retriever(k=5)
for item in items:
    print(f"[{item.id}] {item.question}")
    print("   expected:", item.pages)
    for mode in ("vector", "hybrid", "hybrid_rerank"):
        pages = [c.source_path for c, _ in retriever.retrieve(item.question, mode)]
        print(f"   {mode:<14}", list(dict.fromkeys(pages)))
