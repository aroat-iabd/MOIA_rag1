"""Compara varios tamaños de chunk y solapamiento evaluando el ciclo RAG completo
(recuperación en ChromaDB + umbral de relevancia + generación con LLM).

Uso: python rag_experimentos.py
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag_chunking import chunkenizar, preparar_documentos, llm, extraer_texto

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

# Configuraciones a comparar: (chunk_size, chunk_overlap)
CONFIGS = [
    (300, 50),
    (600, 100),
    (900, 150),
    (2000, 300),
]

PREGUNTAS = [
    # 1. Prueba sobre .md (tabla de rutas)
    "¿Cuántos kilómetros tiene el Camino Primitivo y desde dónde sale?",
    # 2. Prueba sobre .pdf de 2 páginas (cita con número de página)
    "¿En qué año se aprobó el Estatuto de Autonomía de Galicia y cuál es su capital?",
    # 3. Prueba sobre .txt en gallego
    "Como se prepara e se serve o polbo á feira?",
    # 4. Prueba sobre .docx con tabla
    "¿Qué subzona de la D.O. Rías Baixas es la cuna del albariño y dónde se celebra su fiesta?",
    # 5. Pregunta fuera de la documentación (debe responder que no tiene información)
    "¿Cuál es la capital de Australia?",
]

K = 3
UMBRAL = 0.40

INSTRUCCIONES = """Eres un asistente que responde preguntas usando ÚNICAMENTE los documentos proporcionados.
- Si la respuesta no está en los documentos, di exactamente: "No tengo esa información en los documentos."
- Indica entre corchetes el id del documento que usas, tal como aparece, por ejemplo [apuntes:p3:c12].
- Responde en el mismo idioma en que está escrita la pregunta, de forma breve."""


def generar_respuesta(pregunta: str, docs_recuperados: list[dict]) -> str:
    relevantes = [d for d in docs_recuperados if d["similitud"] >= UMBRAL]
    if not relevantes:
        return "No tengo esa información en los documentos."

    contexto = "\n\n".join(
        f"[{d['id']}] (fuente: {d['fuente']})\n{d['texto']}" for d in relevantes
    )
    mensajes = [
        ("system", INSTRUCCIONES),
        ("human", f"DOCUMENTOS:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ]
    return extraer_texto(llm.invoke(mensajes).content)


def main() -> None:
    preparar_documentos(DOCS_DIR)
    print("Cargando modelo de embeddings (BGE-M3)...")
    modelo = SentenceTransformer("BAAI/bge-m3")
    cliente = chromadb.EphemeralClient()  # En memoria: no modifica ./chroma_rag

    resumen = []

    for size, overlap in CONFIGS:
        print("\n" + "=" * 80)
        print(f"EXPERIMENTO: chunk_size={size} | chunk_overlap={overlap}")
        print("=" * 80)

        chunks = chunkenizar(DOCS_DIR, chunk_size=size, chunk_overlap=overlap)
        if not chunks:
            continue

        nombre_col = f"exp_{size}_{overlap}"
        col = cliente.create_collection(nombre_col, metadata={"hnsw:space": "cosine"})

        textos = [c.page_content for c in chunks]
        vectores = modelo.encode(textos, normalize_embeddings=True, batch_size=16)

        LOTE = 1000
        for i in range(0, len(chunks), LOTE):
            col.add(
                ids=[c.metadata["chunk_id"] for c in chunks[i:i + LOTE]],
                embeddings=vectores[i:i + LOTE].tolist(),
                documents=textos[i:i + LOTE],
                metadatas=[
                    {
                        "doc_key": c.metadata["doc_key"],
                        **({"page": c.metadata["page"]} if "page" in c.metadata else {}),
                    }
                    for c in chunks[i:i + LOTE]
                ],
            )

        for p in PREGUNTAS:
            vec_p = modelo.encode([p], normalize_embeddings=True).tolist()
            r = col.query(
                query_embeddings=vec_p,
                n_results=K,
                include=["documents", "metadatas", "distances"],
            )

            encontrados = []
            for doc_id, texto, meta, dist in zip(
                r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]
            ):
                fuente = meta["doc_key"] + (f", pág. {meta['page']}" if "page" in meta else "")
                encontrados.append({
                    "id": doc_id,
                    "texto": texto,
                    "fuente": fuente,
                    "similitud": round(1 - dist, 2),
                })

            print(f"\nPregunta: {p}")
            for d in encontrados:
                marca = "✓" if d["similitud"] >= UMBRAL else "✗"
                print(f"  {marca} {d['similitud']:.2f} | [{d['id']}] ({d['fuente']}) {d['texto'][:75].replace(chr(10), ' ')}")

            respuesta = generar_respuesta(p, encontrados)
            print(f"Respuesta LLM: {respuesta}")

            resumen.append({
                "config": f"{size}/{overlap}",
                "total_chunks": len(chunks),
                "pregunta": p[:42] + ("..." if len(p) > 42 else ""),
                "max_sim": encontrados[0]["similitud"] if encontrados else 0.0,
            })

    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO DE CONFIGURACIONES")
    print("=" * 80)
    print(f"{'Chunk/Overlap':<15} | {'Chunks':<7} | {'Sim. Máx':<9} | {'Pregunta'}")
    print("-" * 80)
    for fila in resumen:
        print(f"{fila['config']:<15} | {fila['total_chunks']:<7} | {fila['max_sim']:<9.2f} | {fila['pregunta']}")


if __name__ == "__main__":
    main()