# tests/test_memory.py — Tests unitarios para el módulo de memoria
# ============================================================================
# Cubre: auto_name_conversation, add_message, get_langchain_messages,
#        formato de mensajes, gestión de conversaciones
# ============================================================================
import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAutoNameConversation:
    def test_renames_nueva_conversacion(self, setup_sqlite_db):
        """Renombra si el nombre es genérico."""
        from backend.memory import auto_name_conversation, get_db
        from backend.database import get_db as db_get_db

        # Crear conversación con nombre genérico en SQLite
        with db_get_db() as db:
            db.execute("""
                INSERT INTO conversations (id, user_id, name, created_at, last_active)
                VALUES (?, ?, ?, ?, ?)
            """, ("abc123", "user1", "Nueva conversación", "2026-04-10T10:00:00", "2026-04-10T10:00:00"))

        auto_name_conversation("user1", "abc123", "¿Cuánto es 2 más 2?")

        # Verificar en BD, no en mocks
        with db_get_db() as db:
            row = db.execute("SELECT name FROM conversations WHERE id=?", ("abc123",)).fetchone()
            assert row is not None
            assert "Cuánto" in row["name"] or "2" in row["name"]

    def test_does_not_rename_custom_name(self, setup_sqlite_db):
        from backend.memory import auto_name_conversation
        from backend.database import get_db as db_get_db

        with db_get_db() as db:
            db.execute("""
                INSERT INTO conversations (id, user_id, name, created_at, last_active)
                VALUES (?, ?, ?, ?, ?)
            """, ("abc123", "user1", "Mi conversación sobre Python", "2026-04-10T10:00:00", "2026-04-10T10:00:00"))

        auto_name_conversation("user1", "abc123", "nuevo mensaje")

        with db_get_db() as db:
            row = db.execute("SELECT name FROM conversations WHERE id=?", ("abc123",)).fetchone()
            assert row["name"] == "Mi conversación sobre Python"

    def test_name_truncated_to_60_chars(self, setup_sqlite_db):
        from backend.memory import auto_name_conversation
        from backend.database import get_db as db_get_db

        with db_get_db() as db:
            db.execute("""
                INSERT INTO conversations (id, user_id, name, created_at, last_active)
                VALUES (?, ?, ?, ?, ?)
            """, ("abc123", "user1", "Nueva conversación", "2026-04-10T10:00:00", "2026-04-10T10:00:00"))

        long_msg = "Esta es una pregunta muy muy larga con muchas palabras que supera los sesenta caracteres sin ninguna duda posible aquí"
        auto_name_conversation("user1", "abc123", long_msg)

        with db_get_db() as db:
            row = db.execute("SELECT name FROM conversations WHERE id=?", ("abc123",)).fetchone()
            assert len(row["name"]) <= 63

    def test_handles_nonexistent_conversation_id(self, setup_sqlite_db):
        from backend.memory import auto_name_conversation
        auto_name_conversation("user1", "nonexistent_id", "mensaje")  # No debe lanzar

class TestAddMessage:
    def test_message_structure(self, setup_sqlite_db):
        from backend.memory import add_message
        from backend.database import get_db as db_get_db

        add_message("user1", "user", "Hola, ¿cómo estás?", agent="general", conversation_id="default")

        with db_get_db() as db:
            msg = db.execute("SELECT role, content, agent FROM messages WHERE user_id=? ORDER BY id DESC LIMIT 1", ("user1",)).fetchone()
            assert msg["role"] == "user"
            assert msg["content"] == "Hola, ¿cómo estás?"
            assert msg["agent"] == "general"

    def test_multiple_messages_appended(self, setup_sqlite_db):
        from backend.memory import add_message
        from backend.database import get_db as db_get_db

        add_message("user1", "user", "Mensaje 1", agent="general", conversation_id="default")
        add_message("user1", "assistant", "Respuesta 1", agent="general", conversation_id="default")
        add_message("user1", "user", "Mensaje 2", agent="general", conversation_id="default")

        with db_get_db() as db:
            rows = db.execute("SELECT role FROM messages WHERE user_id=? ORDER BY id ASC", ("user1",)).fetchall()
            assert [r["role"] for r in rows] == ["user", "assistant", "user"]
            

class TestGetLangchainMessages:
    """Verifica la conversión de mensajes a formato LangChain."""

    def test_user_message_becomes_human_message(self, setup_sqlite_db):
        from backend.memory import get_langchain_messages
        from langchain_core.messages import HumanMessage, AIMessage
        from backend.database import get_db

        # Insertar conversación y mensajes en SQLite (no en archivo JSON)
        with get_db() as db:
            db.execute(
                "INSERT OR IGNORE INTO conversations (id, user_id, name, created_at, last_active) "
                "VALUES (?, ?, ?, datetime('now'), datetime('now'))",
                ("default", "user1", "Conversación")
            )
            db.execute(
                "INSERT INTO messages (user_id, conversation_id, role, content, agent, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("user1", "default", "user", "Hola", "general", "2026-04-10T10:00:00")
            )
            db.execute(
                "INSERT INTO messages (user_id, conversation_id, role, content, agent, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("user1", "default", "assistant", "Hola, ¿en qué puedo ayudarte?",
                 "general", "2026-04-10T10:00:01")
            )

        lc_messages = get_langchain_messages(
            "user1", last_n=10, conversation_id="default"
        )

        assert len(lc_messages) == 2
        assert isinstance(lc_messages[0], HumanMessage)
        assert isinstance(lc_messages[1], AIMessage)
        assert lc_messages[0].content == "Hola"

    def test_last_n_limits_messages(self, tmp_data_dir):
        """last_n debe limitar el número de mensajes devueltos."""
        messages = [
            {"role": "user", "content": f"Mensaje {i}",
             "agent": "general", "timestamp": f"2026-04-10T10:0{i}:00",
             "conversation_id": "default"}
            for i in range(20)
        ]
        history_file = tmp_data_dir / "user1_default.json"
        history_file.write_text(json.dumps(messages), encoding="utf-8")

        with patch("backend.memory.CHAT_HISTORY_DIR", tmp_data_dir):
            from backend.memory import get_langchain_messages
            lc_messages = get_langchain_messages(
                "user1", last_n=5, conversation_id="default"
            )
            assert len(lc_messages) <= 5

    def test_empty_history_returns_empty_list(self, tmp_data_dir):
        history_file = tmp_data_dir / "user1_default.json"
        history_file.write_text("[]", encoding="utf-8")

        with patch("backend.memory.CHAT_HISTORY_DIR", tmp_data_dir):
            from backend.memory import get_langchain_messages
            result = get_langchain_messages(
                "user1", last_n=10, conversation_id="default"
            )
            assert result == []

    def test_missing_file_returns_empty_list(self, tmp_data_dir):
        """Historial inexistente no debe lanzar excepción."""
        with patch("backend.memory.CHAT_HISTORY_DIR", tmp_data_dir):
            from backend.memory import get_langchain_messages
            result = get_langchain_messages(
                "user_sin_historial", last_n=10, conversation_id="default"
            )
            assert result == []


# ── Tests: _CROSS_AGENT_MEMORY ────────────────────────────────────────────────

class TestCrossAgentMemory:
    """Verifica la configuración de memoria semántica cruzada."""

    def test_cross_agent_map_is_symmetric(self):
        """pedagogico↔psicologo debe ser simétrico."""
        from backend.memory import _CROSS_AGENT_MEMORY
        for agent, related in _CROSS_AGENT_MEMORY.items():
            for other in related:
                assert other in _CROSS_AGENT_MEMORY, \
                    f"{other} no tiene entrada en _CROSS_AGENT_MEMORY"
                assert agent in _CROSS_AGENT_MEMORY[other], \
                    f"Relación {agent}↔{other} no es simétrica"

    def test_pedagogico_knows_psicologo(self):
        from backend.memory import _CROSS_AGENT_MEMORY
        assert "psicologo" in _CROSS_AGENT_MEMORY.get("pedagogico", [])

    def test_psicologo_knows_pedagogico(self):
        from backend.memory import _CROSS_AGENT_MEMORY
        assert "pedagogico" in _CROSS_AGENT_MEMORY.get("psicologo", [])
