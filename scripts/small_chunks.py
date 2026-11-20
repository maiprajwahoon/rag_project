from pathlib import Path

from rag_project.ingestion.chunker import chunk_documents
from rag_project.ingestion.loader import load_documents

chunks = chunk_documents(load_documents(Path("data/raw/fastapi/docs/en/docs")))
small = sorted((c for c in chunks if c.token_count < 30), key=lambda c: c.token_count)
print("chunks under 30 tokens:", len(small))
for c in small[:10]:
    print(c.token_count, "|", c.source_path, "|", c.heading_path)
    print("   ", repr(c.text[:100]))
