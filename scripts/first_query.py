from rag_project.indexing.embedder import Embedder
from rag_project.indexing.store import QdrantStore

store = QdrantStore()
vec = Embedder().embed_query("How do I declare a query parameter with validation?")
for chunk, score in store.search(vec, k=5):
    print(f"{score:.3f} | {chunk.chunk_id} | {chunk.heading_path}")
    print("   ", chunk.text[:150].replace("\n", " "))
