# ============================================================================
# backend/main.py — API FastAPI
# Endpoints: auth, chat, documentos, usuarios, historial
# ============================================================================

import logging
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File,
    Form, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    LOG_LEVEL, LOG_FILE,
    SUPPORTED_EXTENSIONS, DOCUMENTS_DIR,
)
from backend.auth import (
    authenticate_user, create_access_token,
    get_current_active_user, get_admin_user,
    create_user, list_users, delete_user, change_password,
    UserInDB, UserCreate, Token,
)
from backend.rag import (
    save_upload, index_document,
    list_user_documents, delete_user_document,
    clear_user_data,
)
from backend.memory import (
    get_history, clear_history, get_history_stats,
    create_conversation, list_conversations,
    rename_conversation, get_active_conversation,
)
from backend.agents import chat, simple_chat, stream_simple_chat, stream_chat, reset_llm, get_active_model

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentes AI — API Local",
    description="API multiusuario para agentes LLM locales con RAG",
    version="1.0.0",
)

# CORS: permite acceso desde Streamlit y otros clientes en la red local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En producción limitar a IPs conocidas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modelos de request/response ───────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:         str
    use_history:     bool = True
    simple_mode:     bool = False        # True = LLM puro sin tools
    agent_type:      Optional[str] = None  # None = auto_router, o nombre explícito
    system_prompt:   Optional[str] = None
    images:          list = []           # Lista de {data: base64, mime: "image/jpeg", name: "..."}
    conversation_id: str  = "default"

class ChatResponse(BaseModel):
    response:       str
    agent_used:     str  = "general"
    agent_name:     str  = "Asistente"
    tool_calls:     list = []
    sources:        list = []
    model_used:     str  = ""
    generated_docs: list = []

class PasswordChange(BaseModel):
    new_password: str

# ============================================================================
# AUTH
# ============================================================================

@app.post("/auth/token", response_model=Token, tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login — devuelve JWT token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return Token(
        access_token=access_token,
        token_type="bearer",
        username=user.username,
        is_admin=user.is_admin,
    )

@app.get("/auth/me", tags=["Auth"])
async def get_me(current_user: UserInDB = Depends(get_current_active_user)):
    """Devuelve información del usuario autenticado."""
    return {
        "username":  current_user.username,
        "email":     current_user.email,
        "full_name": current_user.full_name,
        "is_admin":  current_user.is_admin,
    }

# ============================================================================
# CHAT
# ============================================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request:      ChatRequest,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """
    Endpoint principal de chat.
    - simple_mode=False (default): usa agente con tools (RAG, web, calculator...)
    - simple_mode=True: chat directo con el LLM sin tools
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El mensaje no puede estar vacío"
        )

    if request.simple_mode:
        system = request.system_prompt or "You are a helpful AI assistant."
        response_text = await simple_chat(
            user_id=current_user.username,
            message=request.message,
            conversation_id=request.conversation_id or "default",
            system_prompt=system,
            use_history=request.use_history,
        )
        return ChatResponse(response=response_text)
    else:
        conv_id = request.conversation_id or "default"
        if conv_id == "None":
            conv_id = "default"
        result = await chat(
            user_id=current_user.username,
            message=request.message,
            agent_type=request.agent_type,
            use_history=request.use_history,
            images=request.images,
            conversation_id=conv_id,
        )
        return ChatResponse(**result)

# ============================================================================
# CHAT STREAMING
# ============================================================================

@app.post("/chat/stream")
async def chat_stream_endpoint(
    request:      ChatRequest,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """
    Endpoint de chat con streaming de tokens.
    Solo disponible en modo simple (simple_mode=True).
    Para modo agente usar /chat normal.
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El mensaje no puede estar vacío"
        )

    system = request.system_prompt or "You are a helpful AI assistant."

    async def token_generator():
        async for token in stream_simple_chat(
            user_id=current_user.username,
            message=request.message,
            conversation_id=request.conversation_id or "default",
            system_prompt=system,
            use_history=request.use_history,
        ):
            yield token

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
    )


# ============================================================================
# CHAT STREAMING AGENTE CON TOOLS
# ============================================================================

@app.post("/chat/stream/agent")
async def chat_stream_agent_endpoint(
    request:      ChatRequest,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """
    Endpoint de chat con streaming para modo agente con tools.
    Devuelve chunks con prefijos:
      TOKEN:<texto>  — fragmento de texto
      TOOL:<nombre>  — tool en ejecución
      META:<json>    — metadatos al finalizar
      DONE           — fin de stream
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Mensaje vacío")

    conv_id = request.conversation_id or "default"
    if conv_id == "None":
        conv_id = "default"

    async def event_generator():
        async for chunk in stream_chat(
            user_id=current_user.username,
            message=request.message,
            agent_type=request.agent_type,
            use_history=request.use_history,
            images=request.images or [],
            conversation_id=conv_id,
        ):
            yield chunk + "\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


# ============================================================================
# CONFIGURACIÓN DE AGENTES
# ============================================================================

@app.get("/agents/config", tags=["Agents"])
async def get_agent_config(
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Devuelve la configuración de agentes del usuario actual."""
    from backend.agent_profiles import get_user_config, AGENT_DISPLAY_NAMES
    config = get_user_config(current_user.username)
    return {
        "mode":     config["mode"],
        "agents":   [
            {"id": a, "name": AGENT_DISPLAY_NAMES.get(a, a)}
            for a in config["agents"]
        ],
        "default":  config["default_agent"],
    }

# ============================================================================
# DOCUMENTOS
# ============================================================================

@app.post("/documents/upload", tags=["Documents"])
async def upload_document(
    file:         UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_active_user),
):
    """
    Sube e indexa un documento en la base de conocimiento del usuario.
    Formatos soportados: PDF, TXT, EPUB, XLSX, CSV, PPTX, DOCX, MD, HTML
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no soportado: {ext}. "
                   f"Soportados: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Guardar archivo
    file_bytes = await file.read()
    file_path  = save_upload(file_bytes, file.filename, current_user.username)

    # Indexar
    result = index_document(file_path, current_user.username)
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )

    return result

@app.get("/documents", tags=["Documents"])
async def list_documents(
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Lista los documentos indexados del usuario."""
    return list_user_documents(current_user.username)

@app.delete("/documents/{filename}", tags=["Documents"])
async def remove_document(
    filename:     str,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Elimina un documento del índice del usuario."""
    ok = delete_user_document(filename, current_user.username)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documento '{filename}' no encontrado"
        )
    return {"status": "deleted", "file": filename}

@app.delete("/documents", tags=["Documents"])
async def clear_documents(
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Elimina todos los documentos del usuario."""
    clear_user_data(current_user.username)
    return {"status": "cleared"}

# ============================================================================
# HISTORIAL
# ============================================================================

@app.get("/history", tags=["History"])
async def get_chat_history(
    last_n:       int = 20,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Devuelve los últimos N mensajes de todas las conversaciones."""
    from backend.memory import get_all_messages
    all_msgs = get_all_messages(current_user.username)
    return all_msgs[-last_n:] if len(all_msgs) > last_n else all_msgs

@app.get("/history/recent", tags=["History"])
async def get_recent_history(
    conversation_id: str = "default",
    current_user:    UserInDB = Depends(get_current_active_user),
):
    """Devuelve el historial reciente de una conversación (legacy, mantener compatibilidad)."""
    from backend.memory import get_history_by_days
    from backend.agent_profiles import get_user_config
    cfg      = get_user_config(current_user.username)
    days     = cfg.get("history_days",    1)
    max_msgs = cfg.get("history_max_msg", 60)
    messages = get_history_by_days(
        current_user.username,
        days=days,
        max_msgs=max_msgs,
        conversation_id=conversation_id,
    )
    return {
        "messages":        messages,
        "days":            days,
        "max_msgs":        max_msgs,
        "count":           len(messages),
        "conversation_id": conversation_id,
    }


@app.get("/history/conversation", tags=["History"])
async def get_conversation_history(
    conversation_id: str = "default",
    last_n:          int = 40,
    current_user:    UserInDB = Depends(get_current_active_user),
):
    """
    Devuelve los últimos N mensajes de una conversación específica.
    Nuevo enfoque: sin límite de tiempo, carga la conversación completa
    hasta last_n mensajes.
    """
    from backend.memory import get_history
    messages = get_history(
        current_user.username,
        last_n=last_n,
        conversation_id=conversation_id,
    )
    return {
        "messages":        messages,
        "count":           len(messages),
        "conversation_id": conversation_id,
    }

@app.delete("/history", tags=["History"])
async def delete_history(
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Borra todo el historial del usuario."""
    clear_history(current_user.username)
    return {"status": "cleared"}


@app.get("/history/stats", tags=["History"])
async def history_stats(
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Estadísticas del historial y memoria semántica del usuario."""
    return get_history_stats(current_user.username)

@app.get("/memory/search", tags=["History"])
async def search_memory(
    query:        str,
    top_k:        int = 5,
    agent_filter: str = None,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """
    Busca en la memoria semántica del usuario.
    Útil para depuración y para que el usuario vea qué recuerda el sistema.
    """
    from backend.memory import search_semantic_memory
    results = search_semantic_memory(
        user_id=current_user.username,
        query=query,
        top_k=top_k,
        agent_filter=agent_filter,
    )
    return {
        "query":   query,
        "results": results,
        "count":   len(results),
    }

# ============================================================================
# USUARIOS (solo admin)
# ============================================================================

@app.post("/admin/users", tags=["Admin"])
async def create_new_user(
    user_data:    UserCreate,
    admin:        UserInDB = Depends(get_admin_user),
):
    """Crea un nuevo usuario (solo admin)."""
    return create_user(user_data)

@app.get("/admin/users", tags=["Admin"])
async def get_all_users(
    admin: UserInDB = Depends(get_admin_user),
):
    """Lista todos los usuarios (solo admin)."""
    return list_users()

@app.delete("/admin/users/{username}", tags=["Admin"])
async def remove_user(
    username: str,
    admin:    UserInDB = Depends(get_admin_user),
):
    """Elimina un usuario (solo admin)."""
    if username == admin.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No puedes eliminar tu propia cuenta"
        )
    ok = delete_user(username)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuario '{username}' no encontrado"
        )
    return {"status": "deleted", "username": username}

@app.put("/admin/users/{username}/password", tags=["Admin"])
async def reset_password(
    username:  str,
    body:      PasswordChange,
    admin:     UserInDB = Depends(get_admin_user),
):
    """Cambia la contraseña de un usuario (solo admin)."""
    ok = change_password(username, body.new_password)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuario '{username}' no encontrado"
        )
    return {"status": "password changed", "username": username}

# ============================================================================
# CONFIGURACIÓN DE USUARIO (agentes disponibles)
# ============================================================================

@app.get("/user/agent-config", tags=["Chat"])
async def get_agent_config(
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Devuelve la configuración de agentes del usuario actual."""
    from backend.agent_profiles import get_user_config, AGENT_DISPLAY_NAMES
    config = get_user_config(current_user.username)
    return {
        "mode":          config["mode"],
        "agents":        config["agents"],
        "default_agent": config["default_agent"],
        "agent_names":   {k: AGENT_DISPLAY_NAMES.get(k, k) for k in config["agents"]},
    }


# ============================================================================
# CONVERSACIONES
# ============================================================================

@app.get("/conversations", tags=["Conversations"])
async def get_conversations(
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Lista todas las conversaciones del usuario ordenadas por actividad."""
    return list_conversations(current_user.username)


@app.post("/conversations", tags=["Conversations"])
async def new_conversation(
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Crea una nueva conversación vacía."""
    conv = create_conversation(current_user.username, "Nueva conversación")
    return conv


class RenameRequest(BaseModel):
    name: str

@app.put("/conversations/{conversation_id}", tags=["Conversations"])
async def rename_conv(
    conversation_id: str,
    body:            RenameRequest,
    current_user:    UserInDB = Depends(get_current_active_user),
):
    """Renombra una conversación."""
    ok = rename_conversation(current_user.username, conversation_id, body.name)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    return {"status": "renamed", "id": conversation_id, "name": body.name}


@app.delete("/conversations/{conversation_id}", tags=["Conversations"])
async def delete_conv(
    conversation_id: str,
    current_user:    UserInDB = Depends(get_current_active_user),
):
    """Elimina una conversación y su historial."""
    ok = clear_history(current_user.username, conversation_id=conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    return {"status": "deleted", "id": conversation_id}


@app.get("/conversations/active", tags=["Conversations"])
async def get_active_conv(
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Devuelve el ID de la conversación más reciente."""
    conv_id = get_active_conversation(current_user.username)
    return {"conversation_id": conv_id}

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Comprueba que la API está activa."""
    return {
        "status":  "ok",
        "version": "1.0.0",
    }

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Agentes AI API — visita /docs para la documentación",
        "docs":    "/docs",
    }

# ── Modelo activo ─────────────────────────────────────────────────────────────
@app.get("/model/active")
async def get_model_active(current_user: dict = Depends(get_current_active_user)):
    """Devuelve el modelo actualmente cargado en LM Studio."""
    model = get_active_model()
    return {"model": model}


@app.post("/model/reset")
async def reset_model(current_user: dict = Depends(get_current_active_user)):
    """Resetea el singleton del LLM para que el próximo request use
    el modelo actualmente cargado en LM Studio."""
    reset_llm()
    model = get_active_model()
    return {"status": "reset", "model": model}

# ── Documentos generados ──────────────────────────────────────────────────────
@app.get("/documents/generated")
async def list_generated_documents(current_user: UserInDB = Depends(get_current_active_user)):
    """Lista los documentos generados por el usuario actual."""
    user_id  = current_user.username
    docs_dir = DOCUMENTS_DIR / user_id
    if not docs_dir.exists():
        return {"documents": []}
    docs = []
    for f in sorted(docs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.is_file():
            stat = f.stat()
            docs.append({
                "name":       f.name,
                "format":     f.suffix.lstrip("."),
                "size_kb":    round(stat.st_size / 1024, 1),
                "created_at": stat.st_mtime,
                "path":       str(f),
            })
    return {"documents": docs}


@app.delete("/documents/generated/{filename}")
async def delete_generated_document(
    filename: str,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Elimina un documento generado del usuario actual."""
    user_id  = current_user.username
    doc_path = DOCUMENTS_DIR / user_id / filename
    if not doc_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    # Verificar que el archivo pertenece al usuario (seguridad)
    if doc_path.parent.name != user_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Acceso denegado")
    doc_path.unlink()
    return {"status": "deleted", "filename": filename}


@app.get("/documents/generated/{filename}/download")
async def download_generated_document(
    filename: str,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Descarga un documento generado."""
    from fastapi.responses import FileResponse
    from fastapi import HTTPException
    user_id  = current_user.username
    doc_path = DOCUMENTS_DIR / user_id / filename
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return FileResponse(
        path=str(doc_path),
        filename=filename,
        media_type="application/octet-stream"
    )
