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
    """Verifica el auto-nombrado de conversaciones."""

    def _make_conv(self, conv_id, name):
        return {"id": conv_id, "name": name,
                "created": "2026-04-10T10:00:00",
                "last_active": "2026-04-10T10:00:00"}

    def test_renames_nueva_conversacion(self, tmp_data_dir):
        conv_id = "abc123"
        convs = [self._make_conv(conv_id, "Nueva conversación")]
        message = "¿Cuánto es 2 más 2?"

        with patch("backend.memory.CHAT_HISTORY_DIR", tmp_data_dir), \
             patch("backend.memory._load_conversations",
                   return_value=convs), \
             patch("backend.memory._save_conversations") as mock_save:

            from backend.memory import auto_name_conversation
            auto_name_conversation("user1", conv_id, message)

            mock_save.assert_called_once()
            saved_convs = mock_save.call_args[0][1]
            conv = next(c for c in saved_convs if c["id"] == conv_id)
            # El nombre debe ser las primeras palabras del mensaje
            assert "Cuánto" in conv["name"] or "2" in conv["name"]

    def test_does_not_rename_custom_name(self, tmp_data_dir):
        conv_id = "abc123"
        custom_name = "Mi conversación sobre Python"
        convs = [self._make_conv(conv_id, custom_name)]

        with patch("backend.memory.CHAT_HISTORY_DIR", tmp_data_dir), \
             patch("backend.memory._load_conversations",
                   return_value=convs), \
             patch("backend.memory._save_conversations") as mock_save:

            from backend.memory import auto_name_conversation
            auto_name_conversation("user1", conv_id, "nuevo mensaje")

            # Si ya tiene nombre personalizado, NO debe renombrarse
            if mock_save.called:
                saved = mock_save.call_args[0][1]
                conv = next(c for c in saved if c["id"] == conv_id)
                assert conv["name"] == custom_name

    def test_name_truncated_to_60_chars(self, tmp_data_dir):
        conv_id = "abc123"
        convs = [self._make_conv(conv_id, "Nueva conversación")]
        long_message = "Esta es una pregunta muy muy larga con muchas palabras " \
                       "que supera los sesenta caracteres sin ninguna duda posible aquí"

        with patch("backend.memory.CHAT_HISTORY_DIR", tmp_data_dir), \
             patch("backend.memory._load_conversations",
                   return_value=convs), \
             patch("backend.memory._save_conversations") as mock_save:

            from backend.memory import auto_name_conversation
            auto_name_conversation("user1", conv_id, long_message)

            if mock_save.called:
                saved = mock_save.call_args[0][1]
                conv = next(c for c in saved if c["id"] == conv_id)
                assert len(conv["name"]) <= 63  # 60 + posible "..."

    def test_handles_nonexistent_conversation_id(self, tmp_data_dir):
        """No debe lanzar excepción si el conv_id no existe."""
        convs = [self._make_conv("other_id", "Nueva conversación")]

        with patch("backend.memory.CHAT_HISTORY_DIR", tmp_data_dir), \
             patch("backend.memory._load_conversations",
                   return_value=convs), \
             patch("backend.memory._save_conversations"):

            from backend.memory import auto_name_conversation
            # No debe lanzar excepción
            auto_name_conversation("user1", "nonexistent_id", "mensaje")


class TestAddMessage:
    """Verifica que add_message guarda correctamente los mensajes."""

    def test_message_structure(self, tmp_data_dir):
        """El mensaje guardado debe tener los campos obligatorios."""
        history_file = tmp_data_dir / "user1" / "conv_default.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)  # Crear la carpeta user1/
        history_file.write_text("[]", encoding="utf-8")

        with patch("backend.memory.CHAT_HISTORY_DIR", tmp_data_dir), \
             patch("backend.memory.get_embeddings", return_value=MagicMock()), \
             patch("backend.memory._get_memory_collection",
                   return_value=MagicMock()):

            from backend.memory import add_message
            add_message("user1", "user", "Hola, ¿cómo estás?",
                        agent="general", conversation_id="default")

            messages = json.loads(history_file.read_text())
            assert len(messages) == 1
            msg = messages[0]
            assert msg["role"] == "user"
            assert msg["content"] == "Hola, ¿cómo estás?"
            assert msg["agent"] == "general"
            assert "timestamp" in msg

    def test_multiple_messages_appended(self, tmp_data_dir):
        """Los mensajes se acumulan en orden."""
        history_file = tmp_data_dir / "user1" / "conv_default.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)  # Crear la carpeta user1/
        history_file.write_text("[]", encoding="utf-8")

        with patch("backend.memory.CHAT_HISTORY_DIR", tmp_data_dir), \
             patch("backend.memory.get_embeddings", return_value=MagicMock()), \
             patch("backend.memory._get_memory_collection",
                   return_value=MagicMock()):

            from backend.memory import add_message
            add_message("user1", "user", "Mensaje 1",
                        agent="general", conversation_id="default")
            add_message("user1", "assistant", "Respuesta 1",
                        agent="general", conversation_id="default")
            add_message("user1", "user", "Mensaje 2",
                        agent="general", conversation_id="default")

            messages = json.loads(history_file.read_text())
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"


class TestGetLangchainMessages:
    """Verifica la conversión de mensajes a formato LangChain."""

    def test_user_message_becomes_human_message(self, tmp_data_dir):
        messages = [
            {"role": "user", "content": "Hola",
             "agent": "general", "timestamp": "2026-04-10T10:00:00",
             "conversation_id": "default"},
            {"role": "assistant", "content": "Hola, ¿en qué puedo ayudarte?",
             "agent": "general", "timestamp": "2026-04-10T10:00:01",
             "conversation_id": "default"},
        ]
        history_file = tmp_data_dir / "user1" / "conv_default.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history_file.write_text(json.dumps(messages), encoding="utf-8")

        with patch("backend.memory.CHAT_HISTORY_DIR", tmp_data_dir):
            from backend.memory import get_langchain_messages
            from langchain_core.messages import HumanMessage, AIMessage

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
