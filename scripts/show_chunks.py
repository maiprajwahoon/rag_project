import sys

from rag_project.indexing.store import QdrantStore

chunks = {c.chunk_id: c for c in QdrantStore().all_chunks()}
for arg in sys.argv[1:]:
    if arg.endswith(".md"):
        ids = sorted(cid for cid in chunks if cid.startswith(arg + "::"))
        print(f"{arg}: {len(ids)} chunks -> {', '.join(i.split('::')[1] for i in ids)}")
        continue
    chunk = chunks.get(arg)
    print("=" * 70)
    if chunk is None:
        print("NOT FOUND:", arg)
        continue
    print(chunk.chunk_id, "|", chunk.heading_path, "|", chunk.token_count, "tokens")
    print(chunk.text[:1100])