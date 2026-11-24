# MOIA – Asistente RAG sobre Galicia

Sistema RAG (*Retrieval-Augmented Generation*) que responde preguntas sobre una colección de documentos de texto en varios formatos (**txt, md, docx y pdf**) sobre Galicia: gastronomía, Camino de Santiago, Costa da Morte, historia, lengua, clima, vino y deporte. Recibe la pregunta por consola, busca los fragmentos más parecidos en **ChromaDB** y un LLM redacta la respuesta usando solo esos fragmentos. Si en los documentos no hay información relevante, lo indica.

Los textos se vectorizan con **BGE-M3** (multilingüe) y el modelo de lenguaje es intercambiable entre **Google Gemini** (en la nube) y **Ollama** (local) cambiando solo el archivo `.env`.

## Cómo funciona

```
INDEXACIÓN   docs/ ──► rag_chunking.py ──► BGE-M3 ──► ChromaDB (chroma_rag/)
                      (chunking recursivo)  (rag_vectorizar.py)

CONSULTA     Usuario ──► rag_interfaz.py ──► BGE-M3 ──► ChromaDB (top-3) ──► umbral 0,40 ──► LLM ──► respuesta
```

1. `rag_vectorizar.py` carga los documentos de `docs/`, los divide en *chunks*, calcula sus vectores y los guarda en ChromaDB (`./chroma_rag`).
2. `rag_interfaz.py` abre un chat por consola. Cada pregunta se vectoriza con el mismo modelo y se buscan los 3 chunks más parecidos.
3. Se descartan los chunks con similitud menor que **0,40**. Si no queda ninguno, el sistema responde «No tengo esa información en los documentos» **sin llamar al LLM**.
4. Si quedan chunks, se envían al LLM con instrucciones que le obligan a responder solo con ellos, citar el id del chunk (por ejemplo `[camino_de_santiago:c1]`) y contestar en el idioma de la pregunta.
5. Tras cada pregunta, la consola muestra los chunks recuperados con su similitud (`✓` supera el umbral, `✗` no) para ver de dónde sale la respuesta.

## Estructura del proyecto

```
MOIA_rag1/
├── rag_chunking.py      # Carga de txt/md/pdf/docx, chunking recursivo y get_llm()
├── rag_vectorizar.py    # Embeddings BGE-M3 y almacenamiento en ChromaDB
├── rag_interfaz.py      # Chat por consola (programa principal)
├── rag_experimentos.py  # Comparativa de tamaños de chunk y solapamiento
├── requirements.txt     # Dependencias
├── .env                 # Tu clave y modelo (no se sube a git)
├── .env.example         # Plantilla de variables de entorno
├── .gitignore
├── docs/                # Colección de 10 documentos (txt, md, docx, pdf)
└── chroma_rag/          # Base vectorial (se crea sola al vectorizar)
```

## Requisitos

- Python 3.10 o superior.
- Espacio para el modelo de embeddings: BGE-M3 se descarga la primera vez (unos 2,3 GB).
- Una clave de Google Gemini (token del centro o clave personal de Google AI Studio) **o** [Ollama](https://ollama.com) con un modelo descargado (si usas el modo local).

## Instalación

Desde la carpeta del proyecto (Windows, PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración

Copia la plantilla y rellénala:

```powershell
copy .env.example .env
```

El archivo `.env` usa cuatro variables:

| Variable | Valores | Efecto |
|---|---|---|
| `GOOGLE_API_KEY` | Token del centro (`alu-…`) o clave de Google AI Studio (`AIza…`) | Obligatoria con `gemini`: si falta, `get_llm` lanza un error claro |
| `GOOGLE_GEMINI_BASE_URL` | URL del proxy del centro (solo en modo clase) | Se pasa como `base_url` al cliente y dirige las peticiones al proxy. Si no se define o está vacía, vale `None` y se usa la API pública |
| `LLM_PROVIDER` | `gemini` u `ollama` | Elige el proveedor. Si no se define, se usa `gemini` |
| `LLM_MODEL` | Nombre del modelo, p. ej. `gemini-3.5-flash-lite` o `llama3.1` | Elige el modelo concreto. Si no se define, se usa uno por defecto según el proveedor |

> No subas nunca el archivo `.env` a git ni lo compartas. Ya está incluido en `.gitignore`.

### Elegir el modelo

Para usar **Gemini en clase** (proxy del centro):

```dotenv
GOOGLE_API_KEY=alu-tu_token_de_clase
GOOGLE_GEMINI_BASE_URL=http://10.42.68.45:8338/google
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
```

Para usar **Gemini en casa** (clave personal, sin la línea del proxy):

```dotenv
GOOGLE_API_KEY=AIzaSyTuClavePersonal
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
```

Para usar **Ollama** en local (no necesita clave):

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
```

Después de editar el `.env`, reinicia el programa: el archivo solo se lee al arrancar.

Con Ollama, asegúrate de tener el servicio en marcha y el modelo descargado (`ollama pull llama3.1`).

Internamente, `rag_chunking.py` lee estas variables al importarse (`LLM_PROVIDER` se normaliza con `strip().lower()`; `GOOGLE_GEMINI_BASE_URL` vacía o ausente se convierte en `None`) y se las pasa a `get_llm(provider, model_name, base_url, max_output_tokens)`. Esta función devuelve el cliente de LangChain correspondiente (`ChatGoogleGenerativeAI` o `ChatOllama`) o lanza un error si falta la clave o el proveedor no es válido. Con Gemini se fija `temperature=0` y `max_output_tokens=1024` para respuestas deterministas y con poco gasto de tokens. El resto del programa no depende del proveedor elegido.

> Los nombres de los modelos de Gemini se retiran con el tiempo. Define siempre `LLM_MODEL` en el `.env` en lugar de fiarte del valor por defecto.

> Con `gemini-3.5-flash-lite` aparece el aviso `temperature will be ignored`: ese modelo usa valores de muestreo fijos, así que el `temperature=0` se ignora. Es inofensivo.

> Estado de las pruebas: el proyecto se ha probado con Gemini (`gemini-3.5-flash-lite`). El modo Ollama está implementado, pero no se ha probado.

## Uso

Los scripts se ejecutan por separado y en este orden:

```powershell
python .\rag_vectorizar.py      # 1) crea la base vectorial (repetir si cambian docs/ o el chunking)
python .\rag_interfaz.py        # 2) chat de preguntas (escribe 'salir' para terminar)
python .\rag_experimentos.py    # 3) comparativa de tamaños de chunk (opcional)
```

`rag_chunking.py` no se ejecuta directamente salvo para probar el chunking: solo imprime el número de chunks y los primeros cinco. El LLM solo se usa en `rag_interfaz.py` y `rag_experimentos.py`.

### Ejemplos de conversación

```
Tú: ¿Qué hay que hacer para visitar las Illas Cíes en verano?
Tú: Como se prepara o polbo á feira?
Tú: ¿Cuántos kilómetros tiene el Camino Primitivo y desde dónde sale?
Tú: ¿Qué subzona de la D.O. Rías Baixas es la cuna del albariño?
Tú: ¿Cuál es la capital de Australia?        (no está en los documentos)
```

## Parámetros de chunking

El chunking usa `RecursiveCharacterTextSplitter` con los separadores `["\n\n", "\n", ". ", " ", ""]`: corta primero por párrafos, luego por líneas y frases, y solo al final por palabras. La configuración elegida es **`chunk_size=600` y `chunk_overlap=100`**, que se fija en la llamada a `chunkenizar()` de `rag_vectorizar.py`.

Los documentos son artículos cortos (1.100 a 2.900 caracteres) con párrafos de 300 a 600 caracteres y alguna tabla pequeña. Se compararon cuatro configuraciones con cinco preguntas (cuatro documentadas, una de ellas en gallego, y una no documentada). Similitud del mejor chunk (coseno):

| Chunk/Overlap | Nº chunks | Camino Primitivo | Estatuto | Polbo (gallego) | Albariño | Australia (no doc.) |
|---|---|---|---|---|---|---|
| 300/50 | 91 | 0,63 | 0,77 | 0,78 | 0,65 | 0,38 |
| **600/100** | **39** | 0,54 | 0,71 | 0,77 | 0,67 | 0,37 |
| 900/150 | 27 | 0,53 | 0,77 | 0,78 | 0,66 | 0,38 |
| 2000/300 | 12 | 0,50 | 0,66 | 0,76 | 0,69 | 0,31 |

Las cuatro configuraciones respondieron bien a las preguntas documentadas y rechazaron la de Australia; la diferencia está en la calidad del contexto:

- **300/50** fragmenta demasiado: en la pregunta del polbo el chunk de mayor similitud (0,78) es solo el título «O polbo á feira», y la tabla del Camino se separa de su cabecera.
- **600/100** mantiene uno o dos párrafos completos por chunk; los tres chunks recuperados para el polbo son del documento correcto y la tabla llega con su cabecera.
- **900/150** deja pasar un chunk ajeno (el apartado «Qué comer» de la Costa da Morte, 0,50) que llega al LLM.
- **2000/300** mezcla varios temas en cada chunk y baja la similitud.

El análisis completo y las capturas de `rag_experimentos.py` con cada tamaño de chunk están en la sección 5 de `memoria_rag.pdf`.

## Respuesta «no hay información»

Se aplican dos capas:

1. **Umbral de similitud** (`UMBRAL = 0.40` en `rag_interfaz.py`): si ningún chunk lo supera, no se llama al LLM.
2. **Prompt**: el LLM debe responder solo con los documentos y decir «No tengo esa información en los documentos» si la respuesta no está.

En las pruebas, el mejor chunk de la pregunta no documentada dio entre 0,31 y 0,38, y el de las documentadas, 0,50 o más.

## Limitaciones conocidas

- `load_docx` añade las tablas al final del documento, no en su posición original, por lo que una tabla puede quedar separada de su texto.
- El umbral de 0,40 se ha ajustado con esta colección; el margen por debajo es pequeño (0,38 frente a 0,40) y habría que recalibrarlo con otros documentos.
- Los PDF escaneados (sin texto) no se pueden leer: no hay OCR.
- Si cambian los documentos o el chunking hay que volver a ejecutar `rag_vectorizar.py`, que borra y recrea la colección.
- El historial de la conversación no se conserva: cada pregunta se responde de forma independiente.
- Los datos de los documentos se redactaron como material de ejemplo y no están verificados con fuentes externas.