import os
from typing import Literal

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

import logging
from pathlib import Path
from typing import List

import docx  # pip install python-docx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)

load_dotenv()  # Cargar las variables del archivo .env inmediatamente


# ==========================================
# 1. FUNCIONES: Selección de LLM y extracción de texto
# ==========================================
def get_llm(provider: Literal["gemini", "ollama"] = "gemini", model_name: str = "gemini-2.5-flash"):
    """
    Fábrica de LLM para intercambiar modelos rápidamente entre Gemini y Ollama local.
    """
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model_name, api_key=os.getenv("GEMINI_API_KEY"))
    elif provider == "ollama":
        return ChatOllama(model=model_name, temperature=0.0)
    else:
        raise ValueError(f"Proveedor no soportado: {provider}")

# Cambia 'LLM_PROVIDER' y 'LLM_MODEL' en el archivo .env para alternar entre Gemini y Ollama, y entre modelos.
provider = os.getenv("LLM_PROVIDER", "gemini")
default_model = "gemini-3.5-flash" if provider == "gemini" else "llama3.1"
llm = get_llm(provider=provider, model_name=os.getenv("LLM_MODEL", default_model))


def extraer_texto(content) -> str:
    """Devuelve solo el texto de la respuesta, sea str o lista de bloques."""
    if isinstance(content, str):
        return content
    return "".join(
        bloque.get("text", "")
        for bloque in content
        if isinstance(bloque, dict) and bloque.get("type") == "text"
    )


# ==========================================
# 2. CARGA Y CHUNKING DE DOCUMENTOS
# ==========================================

DOCS_DIR = Path("./docs")
SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}

# Garantizar la existencia de los directorios iniciales
for directory in [DOCS_DIR]:
    os.makedirs(directory, exist_ok=True)

def _meta(path: Path, **extra) -> dict:
    return {"source": str(path), "doc_key": path.name, **extra}


def load_txt_md(path: Path) -> List[Document]:
    try:
        txt = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        txt = path.read_text(encoding="cp1252", errors="replace")
    if not txt.strip():
        return []
    return [Document(page_content=txt, metadata=_meta(path))]

def load_pdf(path: Path) -> List[Document]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")  # PDFs con contraseña vacía
    docs: List[Document] = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            docs.append(Document(page_content=txt, metadata=_meta(path, page=i + 1)))
    return docs

def load_docx(path: Path) -> List[Document]:
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    txt = "\n".join(parts).strip()
    if not txt:
        return []
    return [Document(page_content=txt, metadata=_meta(path))]


LOADERS = {".txt": load_txt_md, ".md": load_txt_md, ".pdf": load_pdf, ".docx": load_docx}

def load_file(path: Path) -> List[Document]:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        return []
    try:
        return loader(path)
    except Exception as e:
        logger.warning("No se pudo leer %s: %s", path, e)
        return []


# Una vez cargados los documentos, se dividen en chunks para RAG. Se puede ajustar el tamaño y solapamiento de los chunks.
def chunkenizar(docs_dir: Path = DOCS_DIR, chunk_size: int = 900, chunk_overlap: int = 150) -> List[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")

    docs_d = Path(docs_dir).resolve()
    files = sorted(
        p for p in docs_d.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
        and not p.name.startswith(("~$", "."))   # ignora temporales de Word y ocultos
    )
    if not files:
        logger.warning("No hay documentos en %s (pdf/docx/txt/md).", docs_d)
        return []

    raw_docs: List[Document] = []
    for p in files:
        raw_docs.extend(load_file(p))
    if not raw_docs:
        logger.warning("No se pudo extraer texto útil de los documentos.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(raw_docs)

    for i, c in enumerate(chunks):
        pagina = f":p{c.metadata['page']}" if "page" in c.metadata else ""
        c.metadata["chunk_id"] = f"{Path(c.metadata['doc_key']).stem}{pagina}:c{i}"

    print(f"{len(files)} archivos -> {len(raw_docs)} documentos/páginas -> {len(chunks)} chunks")
    return chunks


# Documentos de ejemplo: solo se crean si docs/ está vacía
EJEMPLOS = {
    "cocina_paella.txt":     "La paella valenciana se cocina con arroz, pollo, conejo, judía verde y azafrán.",
    "cocina_tortilla.txt":   "Para una buena tortilla de patatas, fríe las patatas a fuego lento antes de mezclarlas con el huevo.",
    "cocina_polbo.txt":      "O polbo á feira sérvese sobre madeira con aceite de oliva, sal grosa e pemento.",
    "deporte_depor.txt":     "El Deportivo de La Coruña ganó la Liga en la temporada 1999-2000.",
    "deporte_correr.txt":    "Correr tres veces por semana mejora la resistencia cardiovascular.",
    "deporte_triple.txt":    "En baloncesto, un triple vale tres puntos si se lanza desde fuera de la línea.",
    "tecnologia_python.txt": "Python es un lenguaje de programación muy usado en ciencia de datos e inteligencia artificial.",
    "tecnologia_gpu.txt":    "Una GPU acelera el entrenamiento de redes neuronales porque hace muchas operaciones en paralelo.",
    "tecnologia_git.txt":    "Git permite guardar versiones del código y trabajar en equipo sin pisarse los cambios.",
    "viajes_santiago.txt":   "Santiago de Compostela es el final del Camino de Santiago y su catedral es Patrimonio de la Humanidad.",
    "viajes_equipaje.txt":   "Para volar con equipaje de mano, revisa las medidas máximas que permite la aerolínea.",
    "viajes_cies.txt":       "Para visitar as Illas Cíes no verán hai que solicitar autorización á Xunta e reservar o barco.",
}


def preparar_documentos(docs_dir: Path = DOCS_DIR) -> None:
    docs_dir.mkdir(exist_ok=True)
    if not any(p.is_file() for p in docs_dir.rglob("*")):
        for nombre, texto in EJEMPLOS.items():
            (docs_dir / nombre).write_text(texto, encoding="utf-8")
        print(f"Creados {len(EJEMPLOS)} documentos de ejemplo en {docs_dir}")
    else:
        print(f"Usando los documentos que ya hay en {docs_dir}")


def main() -> None:
    preparar_documentos(DOCS_DIR)
    chunks = chunkenizar(DOCS_DIR, chunk_size=900, chunk_overlap=150)
    textos = [c.page_content for c in chunks]

    for c in chunks[:5]:
        print(c.metadata["chunk_id"], "|", c.page_content[:80].replace("\n", " "))

    return textos


if __name__ == "__main__":
    main()

