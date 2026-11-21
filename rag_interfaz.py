"""Chat RAG por consola: vectoriza la pregunta, consulta ChromaDB y genera la respuesta con un LLM.

Antes de ejecutarlo hay que haber ejecutado rag_vectorizar.py (crea ./chroma_rag).
Uso:  python rag_interfaz.py
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag_chunking import llm, extraer_texto

BASE_DIR = Path(__file__).resolve().parent
K = 3                   # chunks que se pasan al LLM
UMBRAL = 0.40           # similitud mínima (coseno) para considerar un chunk relevante

INSTRUCCIONES = """Eres un asistente que responde preguntas usando ÚNICAMENTE los documentos proporcionados.
- Si la respuesta no está en los documentos, di exactamente: "No tengo esa información en los documentos."
- Indica entre corchetes el id del documento que usas, tal como aparece, por ejemplo [apuntes:p3:c12].
- Responde en el mismo idioma en que está escrita la pregunta, de forma breve."""

print("Cargando modelo de embeddings (BGE-M3)...")
modelo_emb = SentenceTransformer("BAAI/bge-m3")
coleccion = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_rag")).get_collection("apuntes_rag")
print(f"Colección cargada: {coleccion.count()} chunks.\n")


def recuperar(pregunta: str) -> list[dict]:
    """Vectoriza la pregunta con el MISMO modelo que los chunks y busca los K más parecidos."""
    vector = modelo_emb.encode([pregunta], normalize_embeddings=True).tolist()
    res = coleccion.query(query_embeddings=vector, n_results=K,
                          include=["documents", "metadatas", "distances"])
    encontrados = []
    for doc_id, texto, meta, dist in zip(res["ids"][0], res["documents"][0],
                                         res["metadatas"][0], res["distances"][0]):
        fuente = meta["doc_key"] + (f", pág. {meta['page']}" if "page" in meta else "")
        encontrados.append({"id": doc_id, "texto": texto, "fuente": fuente,
                            "similitud": round(1 - dist, 2)})
    return encontrados


def responder(pregunta: str) -> tuple[str, list[dict]]:
    docs = recuperar(pregunta)
    relevantes = [d for d in docs if d["similitud"] >= UMBRAL]

    # Sin chunks relevantes ni siquiera llamamos al LLM: así no puede inventar
    if not relevantes:
        return "No tengo esa información en los documentos.", docs

    contexto = "\n\n".join(f"[{d['id']}] (fuente: {d['fuente']})\n{d['texto']}" for d in relevantes)
    mensajes = [
        ("system", INSTRUCCIONES),
        ("human", f"DOCUMENTOS:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ]
    return extraer_texto(llm.invoke(mensajes).content), docs


def main() -> None:
    print("Chat RAG. Escribe tu pregunta (o 'salir' para terminar).")
    while True:
        try:
            pregunta = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if not pregunta:
            continue

        respuesta, docs = responder(pregunta)
        print("\nChunks recuperados:")
        for d in docs:
            marca = "✓" if d["similitud"] >= UMBRAL else "✗"
            print(f"  {marca} {d['similitud']:.2f} | [{d['id']}] ({d['fuente']}) "
                  f"{d['texto'][:90].replace(chr(10), ' ')}")
        print(f"\nAsistente: {respuesta}")
    print("¡Hasta luego!")


if __name__ == "__main__":
    main()