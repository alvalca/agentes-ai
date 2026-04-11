# ============================================================================
# backend/rag.py — Procesamiento de documentos y RAG
# Soporta: PDF, TXT, EPUB, XLSX, CSV, PPTX, DOCX, MD, HTML
# Cada usuario tiene su propia colección en Chroma
# ============================================================================

import logging
import shutil
from pathlib import Path
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LlamaIndex — chunking semántico y búsqueda híbrida
from llama_index.core.node_parser import SentenceSplitter as LlamaSentenceSplitter
from llama_index.core import Document as LlamaDocument
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CHROMA_DIR, UPLOADS_DIR,
    EMBED_MODEL_NAME, EMBED_DEVICE,
    RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_TOP_K,
    CHROMA_COLLECTION_PREFIX, SUPPORTED_EXTENSIONS,
    RERANKER_MODEL, RERANKER_DEVICE, RERANKER_ENABLED, RAG_CANDIDATES_K,
)

logger = logging.getLogger(__name__)

# ── Verificación de disponibilidad del reranker al importar el módulo ─────────
def _check_reranker_availability():
    if not RERANKER_ENABLED:
        logger.info("Reranker desactivado (RERANKER_ENABLED=false)")
        return
    try:
        from sentence_transformers import CrossEncoder  # noqa
        logger.info(
            f"Reranker disponible: {RERANKER_MODEL} en {RERANKER_DEVICE} "
            f"(se cargará en el primer uso de search_user_documents)"
        )
    except ImportError:
        logger.warning("sentence-transformers no instalado — reranker desactivado")

_check_reranker_availability()

# ── Embeddings (singleton para reutilizar en todas las llamadas) ───────────────
_embeddings: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info(f"Cargando modelo de embeddings: {EMBED_MODEL_NAME}")
        try:
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
                model_kwargs={"device": EMBED_DEVICE},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            logger.warning(f"Error cargando embeddings en {EMBED_DEVICE}: {e}")
            logger.warning("Fallback a CPU para embeddings")

            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

    return _embeddings

# ── Reranker (singleton) ──────────────────────────────────────────────────────
_reranker = None

def get_reranker():
    """Singleton lazy del cross-encoder para reranking.
    Se carga solo si RERANKER_ENABLED=true y sentence-transformers está instalado."""
    global _reranker
    if _reranker is None and RERANKER_ENABLED:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Cargando reranker en {RERANKER_DEVICE}: {RERANKER_MODEL}")
            _reranker = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)
            logger.info("Reranker cargado correctamente")
        except ImportError:
            logger.warning(
                "sentence-transformers no está instalado. "
                "El reranker está desactivado. "
                "Instala con: pip install sentence-transformers"
            )
            _reranker = None
        except Exception as e:
            logger.warning(f"Error cargando reranker en {RERANKER_DEVICE}: {e}")
            logger.warning("Fallback a CPU para reranker")
            try:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
                logger.info("Reranker cargado en CPU correctamente")
            except Exception as e2:
                logger.warning(f"No se pudo cargar reranker en CPU: {e2}")
                logger.warning("Reranker desactivado — usando solo búsqueda semántica")
                _reranker = None

    return _reranker

# ── Chroma client (singleton) ─────────────────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None

def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client

# ── Nombre de colección por usuario ──────────────────────────────────────────
def collection_name(user_id: str) -> str:
    # Chroma requiere nombres sin espacios ni caracteres especiales
    safe = user_id.replace(" ", "_").lower()
    return f"{CHROMA_COLLECTION_PREFIX}{safe}"

# ── Vectorstore por usuario ───────────────────────────────────────────────────
def get_vectorstore(user_id: str) -> Chroma:
    """Devuelve el vectorstore Chroma del usuario (crea si no existe)."""
    col_name = collection_name(user_id)
    return Chroma(
        collection_name=col_name,
        embedding_function=get_embeddings(),
        client=get_chroma_client(),
    )

# ── Extracción de texto por formato ──────────────────────────────────────────
def extract_text(file_path: Path) -> list[Document]:
    """Extrae texto de un archivo y devuelve lista de Documents."""
    ext = file_path.suffix.lower()
    docs = []

    try:
        # ── PDF ──────────────────────────────────────────────────────────────
        if ext == ".pdf":
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": file_path.name, "page": i + 1}
                        ))

        # ── TXT / MD ─────────────────────────────────────────────────────────
        elif ext in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(
                page_content=text,
                metadata={"source": file_path.name}
            ))

        # ── EPUB ─────────────────────────────────────────────────────────────
        elif ext == ".epub":
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            book = epub.read_epub(str(file_path))
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file_path.name, "chapter": item.get_name()}
                    ))

        # ── XLSX / XLS ───────────────────────────────────────────────────────
        elif ext in {".xlsx", ".xls"}:
            import pandas as pd
            xl = pd.ExcelFile(file_path)
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                text = f"Sheet: {sheet}\n{df.to_string(index=False)}"
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path.name, "sheet": sheet}
                ))

        # ── CSV ──────────────────────────────────────────────────────────────
        elif ext == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path, encoding="utf-8", errors="ignore")
            # Dividir en bloques de 100 filas para no saturar el chunk
            chunk_rows = 100
            for i in range(0, len(df), chunk_rows):
                chunk = df.iloc[i:i + chunk_rows]
                text = chunk.to_string(index=False)
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path.name, "rows": f"{i}-{i+chunk_rows}"}
                ))

        # ── PPTX ─────────────────────────────────────────────────────────────
        elif ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(str(file_path))
            for i, slide in enumerate(prs.slides):
                texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        texts.append(shape.text.strip())
                if texts:
                    docs.append(Document(
                        page_content="\n".join(texts),
                        metadata={"source": file_path.name, "slide": i + 1}
                    ))

        # ── DOCX ─────────────────────────────────────────────────────────────
        elif ext == ".docx":
            from docx import Document as DocxDocument
            docx = DocxDocument(str(file_path))
            text = "\n".join(
                p.text for p in docx.paragraphs if p.text.strip()
            )
            docs.append(Document(
                page_content=text,
                metadata={"source": file_path.name}
            ))

        # ── HTML ─────────────────────────────────────────────────────────────
        elif ext == ".html":
            from bs4 import BeautifulSoup
            html = file_path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            docs.append(Document(
                page_content=text,
                metadata={"source": file_path.name}
            ))

        else:
            logger.warning(f"Formato no soportado: {ext}")

    except Exception as e:
        logger.error(f"Error extrayendo texto de {file_path.name}: {e}")

    return docs

# ── Indexar documento ─────────────────────────────────────────────────────────
# ── Detección de tipo de documento para chunking adaptativo ──────────────────
_TRANSCRIPT_KEYWORDS = {
    "transcript", "transcripcion", "transcription",
    "youtube", "podcast", "interview", "entrevista",
    "subtitles", "subtitulos", "caption",
}

def _is_transcript(file_path: Path, raw_docs: list) -> bool:
    """
    Detecta si un documento es una transcripción por nombre de archivo
    o por características del contenido (texto continuo sin puntuación).
    """
    # 1. Detectar por nombre de archivo
    name_lower = file_path.stem.lower()
    if any(kw in name_lower for kw in _TRANSCRIPT_KEYWORDS):
        return True

    # 2. Detectar por contenido: transcripciones tienen pocas frases cortas
    # y alta proporción de texto continuo sin saltos de línea
    if raw_docs:
        sample = raw_docs[0].page_content[:2000]
        lines = [l for l in sample.split("\n") if l.strip()]
        if not lines:
            return False
        # Ratio de líneas largas vs totales — transcripciones suelen
        # tener pocas líneas pero muy largas (texto continuo)
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        # Densidad de puntuación — transcripciones tienen poca puntuación
        punct_count = sum(1 for c in sample if c in ".!?")
        punct_density = punct_count / max(len(sample), 1)
        if avg_line_len > 200 and punct_density < 0.02:
            return True

    return False


def index_document(file_path: Path, user_id: str) -> dict:
    """
    Extrae, trocea e indexa un documento en la colección del usuario.
    Usa chunking adaptativo según el tipo de documento:
    - Documentos estructurados: LlamaSentenceSplitter (512/64)
    - Transcripciones: RecursiveCharacterTextSplitter (256/128)
    Devuelve stats del proceso.
    """
    logger.info(f"Indexando {file_path.name} para usuario {user_id}")

    # 1. Extraer texto
    raw_docs = extract_text(file_path)
    if not raw_docs:
        return {"status": "error", "message": "No se pudo extraer texto del archivo"}

    # 2. Detectar tipo de documento para chunking adaptativo
    is_transcript = _is_transcript(file_path, raw_docs)

    if is_transcript:
        # ── Transcripciones: chunks pequeños con overlap alto ─────────────
        # Texto continuo sin estructura — necesita más superposición para
        # no perder ideas que cruzan límites de chunk
        logger.info(f"Transcripción detectada: {file_path.name} — usando chunking adaptativo (256/128)")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=128,
            separators=[". ", "? ", "! ", "\n", " ", ""],
            length_function=len,
        )
        chunks = []
        for doc in raw_docs:
            sub_chunks = splitter.create_documents(
                [doc.page_content],
                metadatas=[{**doc.metadata, "user_id": user_id,
                            "file_name": file_path.name, "doc_type": "transcript"}]
            )
            chunks.extend(sub_chunks)

    else:
        # ── Documentos estructurados: LlamaSentenceSplitter ───────────────
        # Respeta límites semánticos de frases y párrafos
        logger.info(f"Documento estructurado: {file_path.name} — usando LlamaSentenceSplitter ({RAG_CHUNK_SIZE}/{RAG_CHUNK_OVERLAP})")
        llama_docs = [
            LlamaDocument(
                text=doc.page_content,
                metadata={**doc.metadata, "user_id": user_id,
                          "file_name": file_path.name, "doc_type": "document"}
            )
            for doc in raw_docs
        ]
        llama_splitter = LlamaSentenceSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
        )
        llama_nodes = llama_splitter.get_nodes_from_documents(llama_docs)
        chunks = [
            Document(
                page_content=node.get_content(),
                metadata=node.metadata
            )
            for node in llama_nodes
        ]

    # 3. Indexar en Chroma
    vectorstore = get_vectorstore(user_id)
    vectorstore.add_documents(chunks)

    logger.info(
        f"Indexado: {file_path.name} → {len(raw_docs)} páginas, "
        f"{len(chunks)} chunks para {user_id}"
    )
    return {
        "status":    "ok",
        "file":      file_path.name,
        "pages":     len(raw_docs),
        "chunks":    len(chunks),
        "user_id":   user_id,
    }

# ── Guardar archivo subido ────────────────────────────────────────────────────
def save_upload(file_bytes: bytes, filename: str, user_id: str) -> Path:
    """Guarda el archivo subido en la carpeta del usuario."""
    user_dir = UPLOADS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / filename
    dest.write_bytes(file_bytes)
    return dest

# ── Listar documentos del usuario ─────────────────────────────────────────────
def list_user_documents(user_id: str) -> list[dict]:
    """Devuelve los documentos indexados del usuario."""
    user_dir = UPLOADS_DIR / user_id
    if not user_dir.exists():
        return []
    docs = []
    for f in sorted(user_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append({
                "name":     f.name,
                "size_kb":  round(f.stat().st_size / 1024, 1),
                "ext":      f.suffix.lower(),
            })
    return docs

# ── Eliminar documento del usuario ───────────────────────────────────────────
def delete_user_document(filename: str, user_id: str) -> bool:
    """Elimina el archivo y sus chunks del vectorstore."""
    file_path = UPLOADS_DIR / user_id / filename
    if not file_path.exists():
        return False

    # Eliminar chunks del vectorstore filtrando por metadata
    try:
        vectorstore = get_vectorstore(user_id)
        col = get_chroma_client().get_collection(collection_name(user_id))
        results = col.get(where={"file_name": filename})
        if results["ids"]:
            col.delete(ids=results["ids"])
        logger.info(f"Eliminados {len(results['ids'])} chunks de {filename}")
    except Exception as e:
        logger.warning(f"Error eliminando chunks de Chroma: {e}")

    file_path.unlink()
    return True

# ── Mini Search: snippets focalizados para guiar atención del modelo ─────────
import re as _re_snippets

def extract_relevant_snippets(docs, query, window_size=300):
    query_terms = set(query.lower().split())
    snippets = []

    for doc in docs:
        text = doc.page_content
        text_lower = text.lower()

        paragraphs = text.split("\n\n")

        for term in query_terms:
            for match in _re_snippets.finditer(_re_snippets.escape(term), text_lower):
                match_pos = match.start()

                current_pos = 0
                selected_paragraph = None

                for p in paragraphs:
                    start = current_pos
                    end   = current_pos + len(p)
                    if start <= match_pos <= end:
                        selected_paragraph = p.strip()
                        break
                    current_pos = end + 2  # por "\n\n"

                if selected_paragraph and len(selected_paragraph) > 50:
                    snippet = selected_paragraph
                else:
                    start = max(0, match_pos - window_size // 2)
                    end   = min(len(text), match_pos + window_size // 2)
                    snippet = text[start:end].strip()

                if len(snippet) < 30:
                    continue

                snippets.append(Document(
                    page_content=snippet,
                    metadata={**doc.metadata, "is_snippet": True}
                ))

    unique = {}
    for s in snippets:
        unique[s.page_content] = s

    return list(unique.values())

# ── Buscar en documentos del usuario ─────────────────────────────────────────
def search_documents(query: str, user_id: str, top_k: int = RAG_TOP_K, file_name: str = None) -> list[Document]:
    """Busca los chunks más relevantes para la query del usuario.
 
    Pipeline híbrido:
    1. Búsqueda semántica (Chroma) — encuentra conceptos similares
    2. Búsqueda BM25 (keywords exactas) — encuentra términos legales/técnicos
    3. Fusión de resultados eliminando duplicados
    4. Reranker cross-encoder — reordena por relevancia real
    5. Devuelve top_k más relevantes
 
    Si el reranker no está disponible usa solo búsqueda semántica.
    """
    vectorstore = get_vectorstore(user_id)
    reranker    = get_reranker()
 
    # ── Búsqueda semántica inicial ───────────────────────────────────────────
    # Primero recuperar candidatos semánticos para identificar el documento
    # más relevante y luego decidir la estrategia de recuperación
    semantic_candidates = vectorstore.similarity_search(query, k=RAG_CANDIDATES_K)
 
    if not semantic_candidates:
        return []
 
    # ── Detectar documento relevante y verificar si es pequeño ───────────────
    # El primer candidato semántico identifica el documento más relevante.
    # Si ese documento tiene pocos chunks (≤15), devolver todos sus chunks
    # directamente — el modelo tendrá todo el contexto sin depender de
    # similitud semántica (útil para referencias, conclusiones, anexos)
    SMALL_DOC_THRESHOLD = 15
    try:
        # Si el modelo especificó un archivo concreto, usarlo directamente
        # Si no, detectarlo a partir del primer candidato semántico
        if file_name:
            best_file = file_name
            logger.info(f"Archivo especificado: '{best_file}'")
        else:
            best_file = semantic_candidates[0].metadata.get("file_name", "")
            logger.info(f"Archivo detectado semánticamente: '{best_file}'")
        if best_file:
            col = vectorstore._collection
            file_results = col.get(
                where={"file_name": best_file},
                include=["documents", "metadatas"]
            )
            file_chunks_docs  = file_results.get("documents", [])
            file_chunks_metas = file_results.get("metadatas", [])
            file_total = len(file_chunks_docs)
            logger.info(
                f"Documento más relevante: '{best_file}' "
                f"({file_total} chunks)"
            )
            if file_total <= SMALL_DOC_THRESHOLD:
                all_chunks = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(file_chunks_docs, file_chunks_metas)
                ]
 
                # Extraer snippets focalizados para guiar atención del modelo
                snippets = []
                try:
                    snippets = extract_relevant_snippets(all_chunks, query)
                    if snippets:
                        logger.info(
                            f"Mini-search: {len(snippets)} snippets "
                            f"focalizados extraídos"
                        )
                except Exception as e:
                    logger.warning(f"Mini-search falló (ignorando): {e}")
 
                # Combinar chunks completos + snippets sin duplicar
                snippet_texts = {s.page_content for s in snippets}
                combined = [
                    c for c in all_chunks
                    if c.page_content not in snippet_texts
                ] + snippets
 
                logger.info(
                    f"Documento pequeño — devolviendo {len(combined)} "
                    f"elementos ({file_total} chunks + {len(snippets)} snippets)"
                )
                return combined
    except Exception as e:
        logger.warning(f"Detección documento pequeño falló ({e}), continuando")
 
    if reranker is None:
        # Sin reranker — solo búsqueda semántica
        return semantic_candidates[:top_k]

    # ── Búsqueda BM25 (keywords exactas) ─────────────────────────────────────
    # Convierte los candidatos semánticos a nodos LlamaIndex para BM25
    try:
        llama_nodes = [
            TextNode(text=doc.page_content, metadata=doc.metadata)
            for doc in semantic_candidates
        ]
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=llama_nodes,
            similarity_top_k=min(RAG_CANDIDATES_K, len(llama_nodes)),
        )
        bm25_results = bm25_retriever.retrieve(query)
        bm25_texts = {r.node.get_content() for r in bm25_results}
    except Exception as e:
        logger.warning(f"BM25 fallback a solo semántica: {e}")
        bm25_texts = set()

    # ── Fusión híbrida ────────────────────────────────────────────────────────
    # Prioriza candidatos que aparecen en ambas búsquedas (semántica + BM25)
    seen = set()
    candidates = []

    # Primero los que aparecen en BM25 (keywords exactas — alta precisión)
    for doc in semantic_candidates:
        if doc.page_content in bm25_texts and doc.page_content not in seen:
            candidates.append(doc)
            seen.add(doc.page_content)

    # Luego el resto de candidatos semánticos
    for doc in semantic_candidates:
        if doc.page_content not in seen:
            candidates.append(doc)
            seen.add(doc.page_content)

    logger.info(
        f"Búsqueda híbrida: {len(semantic_candidates)} semánticos + "
        f"{len(bm25_texts)} BM25 → {len(candidates)} fusionados"
    )

    # ── Reranker ──────────────────────────────────────────────────────────────
    pairs  = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top    = [doc for _, doc in ranked[:top_k]]

    logger.info(
        f"Reranker: {len(candidates)} candidatos → {len(top)} seleccionados "
        f"(scores: {[round(float(s),3) for s,_ in ranked[:top_k]]})"
    )
    return top

# ── Limpiar todos los datos del usuario ──────────────────────────────────────
def clear_user_data(user_id: str) -> bool:
    """Elimina todos los documentos y vectores del usuario."""
    try:
        # Eliminar colección Chroma
        client = get_chroma_client()
        col_name = collection_name(user_id)
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

        # Eliminar carpeta de uploads
        user_dir = UPLOADS_DIR / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir)

        logger.info(f"Datos eliminados para usuario {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error limpiando datos de {user_id}: {e}")
        return False
