import sys

from rag_project.generation.generator import Generator
from rag_project.generation.llm import GeminiClient
from rag_project.indexing.embedder import Embedder
from rag_project.indexing.store import QdrantStore
from rag_project.rag import RagService

question = (
    " ".join(sys.argv[1:]) or "How do I declare a query parameter with validation?"
)
service = RagService(Embedder(), QdrantStore(), Generator(GeminiClient()))
result = service.ask(question)
print(
    "refused:",
    result.refused,
    "| reason:",
    result.refusal_reason,
    "| attempts:",
    result.attempts,
)
print("scores:", [round(s, 3) for s in result.retrieval_scores])
print("answer:", result.answer)
for chunk in result.citations:
    print("  cite:", chunk.chunk_id, "|", chunk.heading_path)
