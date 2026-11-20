from pathlib import Path

from rag_project.ingestion.chunker import chunk_documents
from rag_project.ingestion.loader import load_documents

docs = load_documents(
    Path("data/raw/fastapi/docs/en/docs"),
    docs_src_root=Path("data/raw/fastapi/docs_src"),
)
chunks = chunk_documents(docs)

print("total chunks:", len(chunks))
print("under 30 tokens:", sum(c.token_count < 30 for c in chunks))

print("--- context around remaining '{*' ---")
for c in chunks:
    i = c.text.find("{*")
    if i != -1:
        print(c.source_path, "|", repr(c.text[max(0, i - 40): i + 80]))

unbalanced = [c for c in chunks if c.text.count("```") % 2 == 1]
print("--- chunks with an unbalanced code fence:", len(unbalanced))
for c in unbalanced[:5]:
    print(c.chunk_id, c.token_count)

print("--- one resolved code chunk, around '# File:' ---")
for c in chunks:
    i = c.text.find("# File:")
    if i != -1:
        print(c.chunk_id)
        print(c.text[max(0, i - 60): i + 300])
        break
