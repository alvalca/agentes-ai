# ============================================================================
# backend/memory.py — Historial de conversaciones + Memoria Semántica
#
# Dos capas de memoria:
#   1. Historial cronológico (JSON): últimos N mensajes de la sesión actual
#   2. Memoria semántica (Chroma): búsqueda por similitud en TODO el historial
#
# La memoria semántica permite recuperar conversaciones relevantes de hace
# semanas o meses basándose en el contenido, no solo en el orden temporal.
# ============================================================================

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CHAT_HISTORY_DIR,
    CHROMA_DIR,
    EMBED_MODEL_NAME,
    EMBED_DEVICE,
)

logger = logging.getLogger(__name__)

# ── Embeddings singleton ──────────────────────────────────────────────────────
_embeddings: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Cargando embeddings para memoria semántica...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": EMBED_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

# ── Chroma client para historial ──────────────────────────────────────────────
_memory_chroma: Optional[chromadb.PersistentClient] = None

def _get_memory_client() -> chromadb.PersistentClient:
    global _memory_chroma
    if _memory_chroma is None:
        memory_path = CHROMA_DIR / "memory"
        memory_path.mkdir(parents=True, exist_ok=True)
        _memory_chroma = chromadb.PersistentClient(path=str(memory_path))
    return _memory_chroma

def _get_memory_collection(user_id: str):
    """Colección Chroma de memoria semántica por usuario."""
    client   = _get_memory_client()
    col_name = f"memory_{user_id.replace(' ', '_').lower()}"
    return client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

# ── Conversaciones ───────────────────────────────────────────────────────────
def _conversations_file(user_id: str) -> Path:
    user_dir = CHAT_HISTORY_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / "conversations.json"

def _load_conversations(user_id: str) -> list[dict]:
    f = _conversations_file(user_id)
    if not f.exists():
        return []
    try:
        with open(f, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []

def _save_conversations(user_id: str, convs: list[dict]) -> None:
    f = _conversations_file(user_id)
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(convs, fh, indent=2, ensure_ascii=False)

def _ensure_default_conversation(user_id: str) -> str:
    """Garantiza que existe al menos una conversación. Devuelve su ID.
    También registra conversaciones huérfanas (conv_*.json sin entrada en el índice)."""
    convs    = _load_conversations(user_id)
    user_dir = CHAT_HISTORY_DIR / user_id
    now      = datetime.now().isoformat()

    # Migración: si existe history.json legacy, convertirlo a conv_default.json
    legacy = user_dir / "history.json"
    if legacy.exists():
        legacy.rename(user_dir / "conv_default.json")

    # Registrar archivos conv_*.json que no estén en el índice
    registered_ids = {c["id"] for c in convs}
    changed = False
    for conv_file in sorted(user_dir.glob("conv_*.json")):
        conv_id = conv_file.stem[5:]  # quitar "conv_"
        if conv_id not in registered_ids:
            # Intentar obtener fecha del primer mensaje
            try:
                with open(conv_file, "r", encoding="utf-8") as f:
                    msgs = json.load(f)
                created = msgs[0]["timestamp"] if msgs else now
                # Usar primeras palabras del primer mensaje usuario como nombre
                first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
                words = first_user.strip().split()[:6]
                name  = " ".join(words)[:60] if words else "Conversación recuperada"
            except Exception:
                created = now
                name    = "Conversación recuperada"
            convs.append({
                "id":          conv_id,
                "name":        name,
                "created":     created,
                "last_active": created,
            })
            registered_ids.add(conv_id)
            changed = True

    if changed or not convs:
        if not convs:
            # Sin ningún archivo — crear conversación inicial vacía
            conv_id = "default"
            convs = [{
                "id":          conv_id,
                "name":        "Conversación inicial",
                "created":     now,
                "last_active": now,
            }]
        _save_conversations(user_id, convs)

    return convs[-1]["id"] if convs else "default"

# ── Archivo JSON de historial cronológico ────────────────────────────────────
def _history_file(user_id: str, conversation_id: str = "default") -> Path:
    user_dir = CHAT_HISTORY_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / f"conv_{conversation_id}.json"

def _load_history(user_id: str, conversation_id: str = "default") -> list[dict]:
    f = _history_file(user_id, conversation_id)
    if not f.exists():
        return []
    try:
        with open(f, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []

def _save_history(user_id: str, history: list[dict], conversation_id: str = "default") -> None:
    f = _history_file(user_id, conversation_id)
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)

def _touch_conversation(user_id: str, conversation_id: str) -> None:
    """Actualiza last_active de la conversación."""
    convs = _load_conversations(user_id)
    for c in convs:
        if c["id"] == conversation_id:
            c["last_active"] = datetime.now().isoformat()
            break
    _save_conversations(user_id, convs)

# ── Vectorizar y guardar en memoria semántica ─────────────────────────────────
def _clean_content(content: str) -> str:
    """Elimina bloques <think>...</think> del contenido antes de indexar."""
    return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()


def _index_message(
    user_id:         str,
    role:            str,
    content:         str,
    timestamp:       str,
    msg_id:          str,
    agent:           str = "general",
    conversation_id: str = "default",
    conv_name:       str = "",
) -> None:
    """
    Vectoriza un mensaje y lo guarda en Chroma para búsqueda semántica.
    Añade prefijo de contexto [conv_name - fecha] para que el modelo
    distinga de qué conversación proviene cada recuerdo.
    Solo indexa mensajes con contenido suficiente (>20 chars).
    """
    content = _clean_content(content)
    if len(content.strip()) < 20:
        return
    try:
        collection = _get_memory_collection(user_id)
        # Prefijo de contexto: ayuda al modelo a distinguir entre conversaciones
        date_str = timestamp[:10] if timestamp else "unknown"
        prefix   = f"[{conv_name or conversation_id} - {date_str}] " if conv_name or conversation_id != "default" else ""
        indexed_content = prefix + content
        embedding  = get_embeddings().embed_query(indexed_content)
        collection.add(
            ids        = [msg_id],
            embeddings = [embedding],
            documents  = [indexed_content],
            metadatas  = [{
                "role":            role,
                "timestamp":       timestamp,
                "agent":           agent,
                "user_id":         user_id,
                "conversation_id": conversation_id,
                "conv_name":       conv_name,
                "preview":         indexed_content[:120],
            }],
        )
    except Exception as e:
        logger.warning(f"Error indexando mensaje en memoria semántica: {e}")

# ============================================================================
# API PÚBLICA
# ============================================================================

def add_message(
    user_id:         str,
    role:            str,
    content:         str,
    agent:           str = "general",
    conversation_id: str = "default",
) -> None:
    """
    Guarda un mensaje en:
    1. Historial cronológico JSON (por conversación)
    2. Memoria semántica Chroma (vectorizada, compartida entre conversaciones)
    role: 'user' | 'assistant'
    """
    timestamp = datetime.now().isoformat()
    msg_id    = str(uuid.uuid4())

    # Obtener nombre de la conversación para el prefijo semántico
    convs     = _load_conversations(user_id)
    conv_name = next((c["name"] for c in convs if c["id"] == conversation_id), conversation_id)

    # 1. Historial cronológico
    history = _load_history(user_id, conversation_id)
    history.append({
        "id":              msg_id,
        "role":            role,
        "content":         content,
        "timestamp":       timestamp,
        "agent":           agent,
        "conversation_id": conversation_id,
    })
    _save_history(user_id, history, conversation_id)
    _touch_conversation(user_id, conversation_id)

    # 2. Memoria semántica (async-like: si falla no bloquea)
    _index_message(user_id, role, content, timestamp, msg_id, agent, conversation_id, conv_name)

def get_history(
    user_id:         str,
    last_n:          int = 20,
    conversation_id: str = "default",
) -> list[dict]:
    """Devuelve los últimos N mensajes del historial cronológico."""
    history = _load_history(user_id, conversation_id)
    return history[-last_n:] if len(history) > last_n else history

def get_history_by_days(
    user_id:         str,
    days:            int = 1,
    max_msgs:        int = 60,
    conversation_id: str = "default",
) -> list[dict]:
    """
    Devuelve los mensajes de los últimos N días, con un límite máximo
    de mensajes para no sobrecargar la pantalla.

    Args:
        user_id:  ID del usuario
        days:     Número de días hacia atrás (1 = últimas 24h, 3 = últimos 3 días)
        max_msgs: Límite máximo de mensajes a devolver

    Returns:
        Lista de mensajes ordenados cronológicamente (más antiguo primero)
    """
    from datetime import timezone, timedelta
    history  = _load_history(user_id, conversation_id)
    if not history:
        return []

    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    # Filtrar por fecha
    recent = [
        m for m in history
        if m.get("timestamp", "") >= cutoff_str
    ]

    # Si hay más mensajes que el límite, quedarse con los más recientes
    if len(recent) > max_msgs:
        recent = recent[-max_msgs:]

    return recent

def get_langchain_messages(
    user_id:         str,
    last_n:          int = 10,
    conversation_id: str = "default",
) -> list[BaseMessage]:
    """
    Devuelve el historial reciente en formato LangChain.
    Usado para contexto de conversación inmediata.
    """
    history = get_history(user_id, last_n=last_n, conversation_id=conversation_id)
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages

def search_semantic_memory(
    user_id:    str,
    query:      str,
    top_k:      int = 5,
    min_score:  float = 0.3,
    role_filter: Optional[str] = None,  # 'user' | 'assistant' | None
    agent_filter: Optional[str] = None, # 'psicologo' | 'pedagogico' | None
) -> list[dict]:
    """
    Busca en la memoria semántica del usuario los fragmentos más relevantes
    para la query dada, independientemente de cuándo ocurrieron.

    Args:
        user_id:      ID del usuario
        query:        Texto para buscar por similitud semántica
        top_k:        Número máximo de resultados
        min_score:    Umbral mínimo de similitud (0-1, cosine)
        role_filter:  Filtrar por rol del mensaje
        agent_filter: Filtrar por agente que generó el mensaje

    Returns:
        Lista de dicts con content, role, timestamp, agent, score
    """
    try:
        collection = _get_memory_collection(user_id)

        # Verificar que hay documentos
        count = collection.count()
        if count == 0:
            return []

        # Construir filtro de metadata
        where = {}
        if role_filter and agent_filter:
            where = {"$and": [
                {"role":  {"$eq": role_filter}},
                {"agent": {"$eq": agent_filter}},
            ]}
        elif role_filter:
            where = {"role":  {"$eq": role_filter}}
        elif agent_filter:
            where = {"agent": {"$eq": agent_filter}}

        # Vectorizar query y buscar
        query_embedding = get_embeddings().embed_query(query)
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results":        min(top_k, count),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        # Procesar resultados
        memories = []
        docs      = results.get("documents",  [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            # Chroma cosine: distance = 1 - similarity
            similarity = 1.0 - dist
            if similarity >= min_score:
                memories.append({
                    "content":   doc,
                    "role":      meta.get("role", "unknown"),
                    "timestamp": meta.get("timestamp", ""),
                    "agent":     meta.get("agent", "general"),
                    "score":     round(similarity, 3),
                    "preview":   meta.get("preview", doc[:100]),
                })

        # Ordenar por similitud descendente
        memories.sort(key=lambda x: x["score"], reverse=True)
        return memories

    except Exception as e:
        logger.error(f"Error en búsqueda semántica para {user_id}: {e}")
        return []

# Agentes que comparten contexto semántico entre sí
# Si el agente activo está en este mapa, también se buscarán memorias
# de los agentes relacionados (búsqueda semántica cruzada)
_CROSS_AGENT_MEMORY = {
    "pedagogico": ["psicologo"],
    "psicologo":  ["pedagogico"],
}

def build_semantic_context(
    user_id:      str,
    query:        str,
    agent:        str = "general",
    top_k:        int = 4,
    min_score:    float = 0.35,
    cross_agents: bool = False,
) -> str:
    """
    Construye un bloque de contexto con memorias relevantes para inyectar
    al inicio de la conversación del agente.

    Si cross_agents=True y el agente tiene agentes relacionados definidos
    en _CROSS_AGENT_MEMORY, también busca memorias de esos agentes.
    Útil para usuarios con auto_router donde un tema puede haber sido
    tratado por diferentes agentes en conversaciones anteriores.

    Returns:
        String con el contexto semántico formateado, o "" si no hay memorias.
    """
    # Para psicólogo: buscar solo en conversaciones emocionales previas
    # Para pedagógico: buscar en conversaciones pedagógicas
    # Para otros: buscar en todo el historial
    agent_filter = agent if agent in {"psicologo", "pedagogico"} else None

    memories = search_semantic_memory(
        user_id=user_id,
        query=query,
        top_k=top_k,
        min_score=min_score,
        agent_filter=agent_filter,
    )

    # Búsqueda cruzada — buscar también en agentes relacionados
    if cross_agents and agent in _CROSS_AGENT_MEMORY:
        related_agents = _CROSS_AGENT_MEMORY[agent]
        for related_agent in related_agents:
            cross_memories = search_semantic_memory(
                user_id=user_id,
                query=query,
                top_k=max(2, top_k // 2),  # menos resultados para no saturar
                min_score=min_score + 0.05,  # umbral ligeramente más alto
                agent_filter=related_agent,
            )
            # Añadir solo memorias no duplicadas
            existing_ids = {m.get("id", m["content"][:50]) for m in memories}
            for cm in cross_memories:
                cm_id = cm.get("id", cm["content"][:50])
                if cm_id not in existing_ids:
                    cm["_cross_agent"] = related_agent  # marcar origen
                    memories.append(cm)

    if not memories:
        return ""

    lines = ["[RELEVANT MEMORIES FROM PREVIOUS CONVERSATIONS]"]
    for i, mem in enumerate(memories, 1):
        ts   = mem["timestamp"][:10] if mem["timestamp"] else "unknown date"
        role = "You said" if mem["role"] == "user" else "Assistant said"
        cross_label = f" — from {mem['_cross_agent']} context" if "_cross_agent" in mem else ""
        lines.append(
            f"[Memory {i} — {ts} — similarity {mem['score']:.2f}{cross_label}]"
            f"\n{role}: {mem['content'][:300]}"
        )
    lines.append("[END OF RELEVANT MEMORIES]\n")

    return "\n".join(lines)


def create_conversation(user_id: str, name: str = "") -> dict:
    """Crea una nueva conversación y la añade al índice."""
    conv_id = str(uuid.uuid4())[:8]
    now     = datetime.now().isoformat()
    conv    = {
        "id":          conv_id,
        "name":        name or "Nueva conversación",
        "created":     now,
        "last_active": now,
    }
    convs = _load_conversations(user_id)
    convs.append(conv)
    _save_conversations(user_id, convs)
    logger.info(f"Conversación creada: {conv_id} para {user_id}")
    return conv

def list_conversations(user_id: str) -> list[dict]:
    """Lista todas las conversaciones del usuario ordenadas por last_active."""
    _ensure_default_conversation(user_id)
    convs = _load_conversations(user_id)
    return sorted(convs, key=lambda c: c.get("last_active", ""), reverse=True)

def rename_conversation(user_id: str, conversation_id: str, new_name: str) -> bool:
    """Renombra una conversación."""
    convs = _load_conversations(user_id)
    for c in convs:
        if c["id"] == conversation_id:
            c["name"] = new_name.strip()[:80]  # máximo 80 chars
            _save_conversations(user_id, convs)
            return True
    return False

def auto_name_conversation(user_id: str, conversation_id: str, first_message: str) -> None:
    """Auto-nombra la conversación con las primeras palabras del primer mensaje."""
    convs = _load_conversations(user_id)
    auto_names = {"Nueva conversación", "Conversación inicial"}
    for c in convs:
        if c["id"] == conversation_id and c["name"] in auto_names:
            # Tomar primeras 6 palabras del mensaje
            words = first_message.strip().split()[:6]
            name  = " ".join(words)
            if len(name) > 60:
                name = name[:60] + "..."
            c["name"] = name
            _save_conversations(user_id, convs)
            break

def get_active_conversation(user_id: str) -> str:
    """Devuelve el ID de la conversación más reciente por last_active.
    list_conversations ya ordena por last_active descendente, así que
    convs[0] es siempre la más reciente."""
    convs = list_conversations(user_id)  # ordena por last_active desc
    if convs:
        return convs[0]["id"]
    return create_conversation(user_id, "Conversación inicial")["id"]

def clear_history(user_id: str, conversation_id: str = None) -> bool:
    """Borra historial cronológico Y memoria semántica.
    Si conversation_id es None, borra todo el usuario.
    Si se especifica, borra solo esa conversación."""
    try:
        if conversation_id:
            # Borrar solo esa conversación
            f = _history_file(user_id, conversation_id)
            if f.exists():
                f.unlink()
            convs = _load_conversations(user_id)
            convs = [c for c in convs if c["id"] != conversation_id]
            _save_conversations(user_id, convs)
            logger.info(f"Conversación {conversation_id} borrada para {user_id}")
            return True
        # JSON (borrar todos los archivos conv_*.json)
        user_dir = CHAT_HISTORY_DIR / user_id
        for f in user_dir.glob("conv_*.json"):
            f.unlink()
        conversations_f = _conversations_file(user_id)
        if conversations_f.exists():
            conversations_f.unlink()

        # Chroma
        try:
            client   = _get_memory_client()
            col_name = f"memory_{user_id.replace(' ', '_').lower()}"
            client.delete_collection(col_name)
        except Exception:
            pass

        logger.info(f"Historial y memoria semántica borrados para {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error borrando historial de {user_id}: {e}")
        return False

def get_all_messages(user_id: str) -> list[dict]:
    """Devuelve todos los mensajes del usuario de todas las conversaciones."""
    user_dir = CHAT_HISTORY_DIR / user_id
    all_msgs = []
    if not user_dir.exists():
        return []
    for conv_file in sorted(user_dir.glob("conv_*.json")):
        try:
            with open(conv_file, "r", encoding="utf-8") as f:
                msgs = json.load(f)
            all_msgs.extend(msgs)
        except Exception:
            pass
    # Ordenar por timestamp
    all_msgs.sort(key=lambda m: m.get("timestamp", ""))
    return all_msgs

def get_history_stats(user_id: str) -> dict:
    """Estadísticas del historial y memoria semántica (todas las conversaciones)."""
    history   = get_all_messages(user_id)
    user_msgs = [m for m in history if m["role"] == "user"]
    ai_msgs   = [m for m in history if m["role"] == "assistant"]

    # Contar vectores en Chroma
    semantic_count = 0
    try:
        col = _get_memory_collection(user_id)
        semantic_count = col.count()
    except Exception:
        pass

    return {
        "total_messages":     len(history),
        "user_messages":      len(user_msgs),
        "assistant_messages": len(ai_msgs),
        "semantic_memories":  semantic_count,
        "first_message":      history[0]["timestamp"]  if history else None,
        "last_message":       history[-1]["timestamp"] if history else None,
    }
