# ============================================================================
# backend/database.py — SQLite: esquema, conexión y migraciones
# ============================================================================
# Este módulo centraliza todo lo relacionado con SQLite:
#   - Creación de tablas (users, conversations, messages, agent_configs)
#   - Conexión thread-safe con check_same_thread=False
#   - Script de migración desde JSON legacy
#
# Todos los demás módulos importan get_db() de aquí.
# ============================================================================

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "agentes.db"

# ── Esquema ───────────────────────────────────────────────────────────────────
# Definido como string SQL para tener el DDL visible y versionado

_SCHEMA = """
-- ── Usuarios ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    username         TEXT PRIMARY KEY,
    email            TEXT,
    full_name        TEXT,
    disabled         INTEGER NOT NULL DEFAULT 0,   -- 0=activo, 1=desactivado
    is_admin         INTEGER NOT NULL DEFAULT 0,   -- 0=usuario, 1=admin
    hashed_password  TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Conversaciones ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT NOT NULL,          -- UUID de la conversación
    user_id      TEXT NOT NULL,
    name         TEXT NOT NULL DEFAULT 'Nueva conversación',
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    last_active  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (id, user_id),
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ── Mensajes ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,       -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    agent           TEXT NOT NULL DEFAULT 'general',
    timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, user_id)
        REFERENCES conversations(id, user_id) ON DELETE CASCADE
);

-- ── Configuración de agentes por usuario ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_configs (
    user_id          TEXT PRIMARY KEY,
    mode             TEXT NOT NULL DEFAULT 'manual_select',
    agents_json      TEXT NOT NULL DEFAULT '["general"]',  -- JSON array
    default_agent    TEXT NOT NULL DEFAULT 'general',
    history_days     INTEGER NOT NULL DEFAULT 1,
    history_max_msg  INTEGER NOT NULL DEFAULT 10,
    updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ── Índices para consultas frecuentes ────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_messages_user_conv
    ON messages(user_id, conversation_id);

CREATE INDEX IF NOT EXISTS idx_messages_timestamp
    ON messages(user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_user_active
    ON conversations(user_id, last_active DESC);
"""


# ── Conexión ──────────────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """
    Abre una conexión SQLite con opciones recomendadas:
    - WAL mode: permite lecturas concurrentes sin bloquear escrituras
    - FOREIGN KEYS: activa integridad referencial (SQLite la desactiva por defecto)
    - Row factory: devuelve filas como dict-like objects (row["campo"])
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager para obtener una conexión con commit/rollback automático.

    Uso:
        with get_db() as db:
            db.execute("INSERT INTO users ...")
            # commit automático al salir del bloque
            # rollback automático si hay excepción
    """
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Inicialización ────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Crea las tablas si no existen y aplica migraciones pendientes.
    Llamar una vez al arrancar la aplicación (en main.py lifespan).
    Es idempotente — seguro llamarlo múltiples veces.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as db:
        db.executescript(_SCHEMA)
    logger.info(f"SQLite inicializado: {DB_PATH}")


# ── Migración desde JSON ──────────────────────────────────────────────────────

def migrate_from_json(dry_run: bool = False) -> dict:
    """
    Migra datos existentes de los archivos JSON legacy a SQLite.

    Niveles migrados:
      1. users.json → tabla users
      2. chat_history/{user}/{conv_*.json} → tablas conversations + messages
      3. agent_configs.json → tabla agent_configs

    dry_run=True: solo cuenta registros sin escribir nada.
    Devuelve un dict con estadísticas de la migración.
    """
    stats = {
        "users": 0, "conversations": 0,
        "messages": 0, "agent_configs": 0, "errors": []
    }

    # ── Nivel 1: Usuarios ─────────────────────────────────────────────────────
    users_file = DATA_DIR / "users.json"
    if users_file.exists():
        try:
            users_data = json.loads(users_file.read_text(encoding="utf-8"))
            if not dry_run:
                with get_db() as db:
                    for username, u in users_data.items():
                        db.execute("""
                            INSERT OR IGNORE INTO users
                            (username, email, full_name, disabled,
                             is_admin, hashed_password)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            u["username"],
                            u.get("email"),
                            u.get("full_name"),
                            int(u.get("disabled", False)),
                            int(u.get("is_admin", False)),
                            u["hashed_password"],
                        ))
            stats["users"] = len(users_data)
            logger.info(f"Migración usuarios: {stats['users']}")
        except Exception as e:
            stats["errors"].append(f"users.json: {e}")
            logger.error(f"Error migrando usuarios: {e}")

    # ── Nivel 2: Conversaciones y mensajes ────────────────────────────────────
    chat_history_dir = DATA_DIR / "chat_history"
    if chat_history_dir.exists():
        for user_dir in sorted(chat_history_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            user_id = user_dir.name

            # Leer metadatos de conversaciones si existe el archivo
            convs_meta_file = user_dir / "conversations.json"
            convs_meta = {}
            if convs_meta_file.exists():
                try:
                    convs_list = json.loads(
                        convs_meta_file.read_text(encoding="utf-8")
                    )
                    convs_meta = {c["id"]: c for c in convs_list}
                except Exception as e:
                    stats["errors"].append(
                        f"conversations.json ({user_id}): {e}"
                    )

            # Procesar cada archivo de mensajes conv_*.json
            for conv_file in sorted(user_dir.glob("conv_*.json")):
                # Extraer conversation_id del nombre del archivo
                # Formato: conv_{conversation_id}.json
                conv_id = conv_file.stem.replace("conv_", "", 1)
                try:
                    messages = json.loads(
                        conv_file.read_text(encoding="utf-8")
                    )
                    if not messages:
                        continue

                    # Metadatos de la conversación
                    meta = convs_meta.get(conv_id, {})
                    conv_name = meta.get("name", "Conversación")
                    created_at = meta.get(
                        "created", messages[0].get("timestamp", "")
                    )
                    last_active = meta.get(
                        "last_active", messages[-1].get("timestamp", "")
                    )

                    if not dry_run:
                        with get_db() as db:
                            # Insertar conversación
                            db.execute("""
                                INSERT OR IGNORE INTO conversations
                                (id, user_id, name, created_at, last_active)
                                VALUES (?, ?, ?, ?, ?)
                            """, (conv_id, user_id, conv_name,
                                  created_at, last_active))

                            # Insertar mensajes
                            for msg in messages:
                                db.execute("""
                                    INSERT OR IGNORE INTO messages
                                    (user_id, conversation_id, role,
                                     content, agent, timestamp)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (
                                    user_id,
                                    conv_id,
                                    msg.get("role", "user"),
                                    msg.get("content", ""),
                                    msg.get("agent", "general"),
                                    msg.get("timestamp", ""),
                                ))

                    stats["conversations"] += 1
                    stats["messages"] += len(messages)

                except Exception as e:
                    stats["errors"].append(f"{conv_file.name} ({user_id}): {e}")
                    logger.error(f"Error migrando {conv_file}: {e}")

    logger.info(
        f"Migración historial: {stats['conversations']} conversaciones, "
        f"{stats['messages']} mensajes"
    )

    # ── Nivel 3: Configuración de agentes ─────────────────────────────────────
    agent_configs_file = DATA_DIR / "agent_configs.json"
    if agent_configs_file.exists():
        try:
            configs = json.loads(
                agent_configs_file.read_text(encoding="utf-8")
            )
            if not dry_run:
                with get_db() as db:
                    for user_id, cfg in configs.items():
                        if user_id.startswith("__"):
                            continue  # saltar __default__
                        agents = cfg.get("agents", ["general"])
                        # agents puede ser lista de strings o lista de dicts
                        if agents and isinstance(agents[0], dict):
                            agents = [a["id"] for a in agents]
                        db.execute("""
                            INSERT OR REPLACE INTO agent_configs
                            (user_id, mode, agents_json, default_agent,
                             history_days, history_max_msg)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            user_id,
                            cfg.get("mode", "manual_select"),
                            json.dumps(agents),
                            cfg.get("default_agent", "general"),
                            cfg.get("history_days", 1),
                            cfg.get("history_max_msg", 10),
                        ))
            stats["agent_configs"] = len(
                [k for k in configs if not k.startswith("__")]
            )
            logger.info(f"Migración agent_configs: {stats['agent_configs']}")
        except Exception as e:
            stats["errors"].append(f"agent_configs.json: {e}")
            logger.error(f"Error migrando agent_configs: {e}")

    return stats
