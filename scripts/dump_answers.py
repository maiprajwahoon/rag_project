from pathlib import Path

from rag_project.evaluation.generation_eval import CachedLLM
from rag_project.evaluation.golden import load_golden
from rag_project.generation.generator import Generator
from rag_project.generation.llm import DEFAULT_MODEL, GeminiClient
from rag_project.indexing.embedder import Embedder
from rag_project.indexing.store import QdrantStore
from rag_project.rag import RagService

llm = CachedLLM(
    GeminiClient(), Path("eval/cache/llm_cache.json"), namespace=DEFAULT_MODEL
)
service = RagService(Embedder(), QdrantStore(), Generator(llm), k=5)
lines: list[str] = []
for item in load_golden(Path("eval/golden.jsonl")):
    if not item.answerable:
        continue
    result = service.ask(item.question)
    if result.refused:
        continue
    lines.append("=" * 70)
    lines.append(f"[{item.id}] {item.question}")
    lines.append("ANSWER: " + result.answer)
    for chunk in result.citations:
        lines.append(f"--- cited {chunk.chunk_id} | {chunk.heading_path}")
        lines.append(chunk.text)
    lines.append("")
Path("eval/cache/answers_review.txt").write_text("\n".join(lines), encoding="utf-8")
print("wrote eval/cache/answers_review.txt | real calls:", llm.real_calls)