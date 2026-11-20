from pathlib import Path
from statistics import mean

from rag_project.ingestion.chunker import chunk_documents
from rag_project.ingestion.loader import load_documents

root = Path("data/raw/fastapi/docs/en/docs")
docs = load_documents(root, docs_src_root=Path("data/raw/fastapi/docs_src"))
chunks = chunk_documents(docs)
counts = [c.token_count for c in chunks]
print(len(docs), "docs,", len(chunks), "chunks")
print("tokens min/avg/max:", min(counts), round(mean(counts)), max(counts))
print("chunks with include lines:", sum("{*" in c.text for c in chunks))
print("chunks under 30 tokens:", sum(c.token_count < 30 for c in chunks))
