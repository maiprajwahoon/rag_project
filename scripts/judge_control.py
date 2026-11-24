from pathlib import Path

from rag_project.evaluation.generation_eval import CachedLLM, judge_answer
from rag_project.evaluation.golden import load_golden
from rag_project.generation.generator import Generator
from rag_project.generation.llm import DEFAULT_MODEL, GeminiClient
from rag_project.indexing.embedder import Embedder
from rag_project.indexing.store import QdrantStore
from rag_project.rag import RagService

PLANTED = " Also, FastAPI requires you to restart the server after every request."
llm = CachedLLM(
    GeminiClient(),
    Path("eval/cache/llm_cache.json"),
    min_interval=6.0,
    namespace=DEFAULT_MODEL,
)
service = RagService(Embedder(), QdrantStore(), Generator(llm), k=5)
checked = 0
for item in load_golden(Path("eval/golden.jsonl")):
    if not item.answerable:
        continue
    result = service.ask(item.question)
    if result.refused:
        continue
    honest = judge_answer(llm, result)
    planted = judge_answer(llm, result.model_copy(update={"answer": result.answer + PLANTED}))
    print(
        item.id,
        "| honest supported:", honest.supported if honest else None,
        "| planted-error supported:", planted.supported if planted else None,
    )
    checked += 1
    if checked == 4:
        break
print("real calls:", llm.real_calls)