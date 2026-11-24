from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

from rag_chunking import chunkenizar, preparar_documentos


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

# Se obtienen los chunks desde rag_chunking.py y se convierten en textos vectorizables.
preparar_documentos(DOCS_DIR)
chunks = chunkenizar(DOCS_DIR, chunk_size=600, chunk_overlap=100)
textos = [chunk.page_content for chunk in chunks]

if not chunks:
    raise ValueError(f"No hay chunks disponibles en {DOCS_DIR}")

print(f"Chunks disponibles desde rag_chunking.py: {len(chunks)}")
for chunk in chunks[:5]:
    print(chunk.metadata["chunk_id"], "|", chunk.page_content[:80].replace("\n", " "))

# ==========================
# 1. Vectorización de los chunks
# =========================
modelo_emb = SentenceTransformer("BAAI/bge-m3")   # la primera vez tarda: descarga ~2,3 GB

# Con muchos chunks conviene ir por lotes y ver el progreso
vectores = modelo_emb.encode(textos, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
print("Forma de la matriz:", vectores.shape)   # (número de chunks, 1024)

# ==========================
# 2. Guardar los vectores y metadatos en ChromaDB
# ==========================

cliente = chromadb.PersistentClient(path="./chroma_rag")

if "apuntes_rag" in [c.name for c in cliente.list_collections()]:
    cliente.delete_collection("apuntes_rag")

coleccion = cliente.create_collection(name="apuntes_rag", metadata={"hnsw:space": "cosine"})

# Chroma solo admite metadatos str/int/float/bool: nos quedamos con los que nos interesan
def metadatos_chroma(c):
    m = {"source": c.metadata["source"], "doc_key": c.metadata["doc_key"],
         "start_index": c.metadata.get("start_index", 0)}
    if "page" in c.metadata:
        m["page"] = c.metadata["page"]
    return m

LOTE = 1000

for i in range(0, len(chunks), LOTE):
    coleccion.add(
        ids=[c.metadata["chunk_id"] for c in chunks[i:i + LOTE]],
        embeddings=vectores[i:i + LOTE].tolist(),
        documents=textos[i:i + LOTE],
        metadatas=[metadatos_chroma(c) for c in chunks[i:i + LOTE]],
    )
print("Chunks guardados:", coleccion.count())

