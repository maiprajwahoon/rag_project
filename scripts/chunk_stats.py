from collections import Counter
from pathlib import Path

from rag_project.ingestion.chunker import chunk_documents
from rag_project.ingestion.loader import load_documents

docs = load_documents(Path("data/raw/fastapi/docs/en/docs"))
chunks = chunk_documents(docs)
per_file = Counter(c.source_path for c in chunks)
print("top files by chunk count:")
for path, n in per_file.most_common(8):
    print(f"  {n:5d}  {path}")
print("chunks under 20 tokens:", sum(c.token_count < 20 for c in chunks))
print("chunks under 50 tokens:", sum(c.token_count < 50 for c in chunks))
print("chunks with include lines ({* ... *}):", sum("{*" in c.text for c in chunks))
