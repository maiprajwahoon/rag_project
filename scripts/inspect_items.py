import sys
from pathlib import Path

from rag_project.evaluation.generation_eval import CachedLLM
from rag_project.evaluation.golden import load_golden
from rag_project.generation.generator import Generator
from rag_project.generation.llm import DEFAULT_MODEL, GeminiClient
from rag_project.indexing.embedder import Embedder
from rag_project.indexing.store import QdrantStore
from rag_project.rag import RagService

wanted = set(sys.argv[1:])
items = [i for i in load_golden(Path("eval/golden.jsonl")) if i.id in wanted]
llm = CachedLLM(
    GeminiClient(), Path("eval/cache/llm_cache.json"), namespace=DEFAULT_MODEL
)
service = RagService(Embedder(), QdrantStore(), Generator(llm), k=5)
for item in items:
    result = service.ask(item.question)
    print("=" * 70)
    print(f"[{item.id}] {item.question}")
    print("expected pages:", item.pages)
    shown = {c.chunk_id for c in result.citations}
    for rank, cid in enumerate(result.retrieved_ids, start=1):
        print(f"  {rank}. {cid}{'  <-- cited' if cid in shown else ''}")
    print("refused:", result.refused, "| real calls:", llm.real_calls)
    print("answer:", result.answer)