import sys

from rag_project.retrieval.factory import build_default_retriever

query = " ".join(sys.argv[1:]) or "How do I declare a query parameter with validation?"
retriever = build_default_retriever()
print("QUERY:", query)
for mode in ("vector", "hybrid", "hybrid_rerank"):
    print(f"\n=== {mode} ===")
    for chunk, score in retriever.retrieve(query, mode):
        print(f"{score:8.3f} | {chunk.chunk_id} | {chunk.heading_path}")
