# ============================================================================
# config.py — Configuración centralizada del proyecto
# Todos los parámetros del sistema en un único lugar
# ============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables del archivo .env si existe
load_dotenv(Path(__file__).parent / ".env")

# ── Rutas base ────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
UPLOADS_DIR     = DATA_DIR / "uploads"
CHROMA_DIR      = DATA_DIR / "chroma"
INDEXES_DIR     = DATA_DIR / "indexes"
CHAT_HISTORY_DIR= DATA_DIR / "chat_history"

# Crear carpetas si no existen
DOCUMENTS_DIR   = DATA_DIR / "documents"
for _dir in [UPLOADS_DIR, CHROMA_DIR, INDEXES_DIR, CHAT_HISTORY_DIR, DOCUMENTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── LM Studio ────────────────────────────────────────────────────────────────
LM_STUDIO_URL       = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY   = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL     = os.getenv("LM_STUDIO_MODEL", "local-model")
LLM_TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS      = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ── Embeddings (HuggingFace local) ───────────────────────────────────────────
EMBED_MODEL_NAME    = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBED_DEVICE        = os.getenv("EMBED_DEVICE", "cuda:1")  # "cuda:0"=5060Ti, "cuda:1"=1660Super, "cpu"

# ── Reranker (mejora la precisión del RAG reordenando candidatos) ─────────────
RERANKER_MODEL      = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANKER_DEVICE     = os.getenv("RERANKER_DEVICE", "cuda:1")  # "cuda:0"=5060Ti, "cuda:1"=1660Super, "cpu"
RERANKER_ENABLED    = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RAG_CANDIDATES_K    = int(os.getenv("RAG_CANDIDATES_K", "20"))  # candidatos para reranker

# ── Chroma ───────────────────────────────────────────────────────────────────
CHROMA_HOST         = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT         = int(os.getenv("CHROMA_PORT", "8001"))
# Colecciones separadas por usuario: f"{CHROMA_COLLECTION_PREFIX}{user_id}"
CHROMA_COLLECTION_PREFIX = "user_docs_"
# Colección global de historial de chat
CHROMA_CHAT_COLLECTION   = "chat_history"

# ── RAG ──────────────────────────────────────────────────────────────────────
RAG_CHUNK_SIZE      = int(os.getenv("RAG_CHUNK_SIZE", "512"))
RAG_CHUNK_OVERLAP   = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
RAG_TOP_K           = int(os.getenv("RAG_TOP_K", "5"))

# Formatos de archivo soportados para RAG
SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".epub",
    ".xlsx", ".xls", ".csv",
    ".pptx", ".docx",
    ".md", ".html",
}

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST            = os.getenv("API_HOST", "0.0.0.0")   # 0.0.0.0 = accesible en red local
API_PORT            = int(os.getenv("API_PORT", "8000"))
API_RELOAD          = os.getenv("API_RELOAD", "false").lower() == "true"

# ── Autenticación ─────────────────────────────────────────────────────────────
# Cambia SECRET_KEY por una cadena aleatoria segura en producción:
# python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY          = os.getenv(
    "SECRET_KEY",
    "cambia-esto-por-una-clave-segura-antes-de-usar"
)
JWT_ALGORITHM       = "HS256"
JWT_EXPIRE_MINUTES  = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 horas

# ── Streamlit ────────────────────────────────────────────────────────────────
STREAMLIT_HOST      = os.getenv("STREAMLIT_HOST", "0.0.0.0")
STREAMLIT_PORT      = int(os.getenv("STREAMLIT_PORT", "8501"))

# ── Tools de los agentes ─────────────────────────────────────────────────────
DUCKDUCKGO_MAX_RESULTS  = int(os.getenv("DUCKDUCKGO_MAX_RESULTS", "4"))

# ── Brave Search API ──────────────────────────────────────────────────────────
# Obtener API key gratuita en: https://api.search.brave.com/
# Plan gratuito: 2000 búsquedas/mes, sin tarjeta de crédito
BRAVE_API_KEY           = os.getenv("BRAVE_API_KEY", "")
BRAVE_MAX_RESULTS       = int(os.getenv("BRAVE_MAX_RESULTS", "5"))
CODE_INTERPRETER_TIMEOUT = int(os.getenv("CODE_INTERPRETER_TIMEOUT", "10"))  # segundos

# ── Modelos HuggingFace opcionales ───────────────────────────────────────────
# Solo se cargan si el usuario los solicita explícitamente
SENTIMENT_MODEL     = "siebert/sentiment-roberta-large-english"
SUMMARIZER_MODEL    = "facebook/bart-large-cnn"
CAPTION_MODEL       = "nlpconnect/vit-gpt2-image-captioning"
VIT_MODEL           = "google/vit-base-patch16-224"

# ── LangGraph ────────────────────────────────────────────────────────────────
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "25"))
AGENT_MAX_ITERATIONS  = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE            = BASE_DIR / "logs" / "agentes.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Usuarios predefinidos (se amplían vía API) ────────────────────────────────
# En producción esto va en base de datos; aquí es el archivo inicial
DEFAULT_ADMIN_USER  = os.getenv("ADMIN_USER", "admin")
DEFAULT_ADMIN_PASS  = os.getenv("ADMIN_PASS", "admin1234")  # cambia esto

# ── Resumen de configuración (para debug) ────────────────────────────────────
def print_config():
    print("="*60)
    print("CONFIGURACIÓN ACTIVA")
    print("="*60)
    print(f"  LM Studio URL  : {LM_STUDIO_URL}")
    print(f"  LM Studio Model: {LM_STUDIO_MODEL}")
    print(f"  Embed Model    : {EMBED_MODEL_NAME}")
    print(f"  Embed Device   : {EMBED_DEVICE}")
    print(f"  Chroma Dir     : {CHROMA_DIR}")
    print(f"  API            : {API_HOST}:{API_PORT}")
    print(f"  Streamlit      : {STREAMLIT_HOST}:{STREAMLIT_PORT}")
    print(f"  RAG chunk size : {RAG_CHUNK_SIZE} (overlap {RAG_CHUNK_OVERLAP})")
    print(f"  RAG top-k      : {RAG_TOP_K}")
    print("="*60)

if __name__ == "__main__":
    print_config()
