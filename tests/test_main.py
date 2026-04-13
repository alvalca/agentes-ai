# ============================================================================
# tests/test_main.py — Tests para backend/main.py
# Cobertura: endpoints, dependencias, manejo de errores
# CORREGIDO: Usa app.dependency_overrides para mocks de autenticación
# ============================================================================

import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from fastapi import HTTPException

# Importar la app y el módulo main
from backend.main import app
import backend.main as main_module
from backend.auth import UserInDB

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_user():
    """Usuario de test estándar."""
    return UserInDB(
        username="testuser",
        email="test@test.com",
        full_name="Test User",
        disabled=False,
        is_admin=False,
        hashed_password="hashed"
    )


@pytest.fixture
def mock_admin():
    """Usuario admin de test."""
    return UserInDB(
        username="admin",
        email="admin@test.com",
        full_name="Admin",
        disabled=False,
        is_admin=True,
        hashed_password="hashed"
    )


@pytest.fixture
def client(mock_user, mock_admin, monkeypatch, tmp_path):
    """
    Cliente de test con autenticación mockeada vía dependency_overrides.

    Esta es la forma correcta de mockear dependencias en FastAPI.
    """
    # Configurar paths temporales
    monkeypatch.setattr(main_module, "DOCUMENTS_DIR", tmp_path / "documents")
    monkeypatch.setattr(main_module, "LOG_FILE", str(tmp_path / "test.log"))

    # Mock de dependencias de autenticación usando dependency_overrides
    async def override_get_current_active_user():
        return mock_user

    async def override_get_admin_user():
        return mock_admin

    # Aplicar overrides
    app.dependency_overrides[main_module.get_current_active_user] = override_get_current_active_user
    app.dependency_overrides[main_module.get_admin_user] = override_get_admin_user

    # Mock de funciones de RAG
    monkeypatch.setattr(main_module, "save_upload", MagicMock(return_value=Path("/tmp/test.txt")))
    monkeypatch.setattr(main_module, "index_document", MagicMock(return_value={"status": "ok", "chunks": 5}))
    monkeypatch.setattr(main_module, "list_user_documents", MagicMock(return_value=[{"name": "doc1.pdf"}]))
    monkeypatch.setattr(main_module, "delete_user_document", MagicMock(return_value=True))
    monkeypatch.setattr(main_module, "clear_user_data", MagicMock())

    # Mock de funciones de memory (las que están en main_module)
    monkeypatch.setattr(main_module, "get_history", MagicMock(return_value=[{"role": "user", "content": "hola"}]))
    monkeypatch.setattr(main_module, "clear_history", MagicMock(return_value=True))
    monkeypatch.setattr(main_module, "get_history_stats", MagicMock(return_value={"count": 10}))
    monkeypatch.setattr(main_module, "create_conversation", MagicMock(return_value={"id": "conv1", "name": "New"}))
    monkeypatch.setattr(main_module, "list_conversations", MagicMock(return_value=[{"id": "conv1"}]))
    monkeypatch.setattr(main_module, "rename_conversation", MagicMock(return_value=True))
    monkeypatch.setattr(main_module, "get_active_conversation", MagicMock(return_value="conv1"))

    # Mock de funciones importadas desde memory (mockear en el módulo memory)
    try:
        import backend.memory as memory_module
        monkeypatch.setattr(memory_module, "get_all_messages", MagicMock(return_value=[]))
        monkeypatch.setattr(memory_module, "get_history_by_days", MagicMock(return_value=[]))
        monkeypatch.setattr(memory_module, "search_semantic_memory", MagicMock(return_value=[]))
    except ImportError:
        pass

    # Mock de agentes - funciones async
    async def mock_chat(*args, **kwargs):
        return {
            "response": "Test response",
            "agent_used": "test",
            "agent_name": "Test Agent",
            "tool_calls": [],
            "sources": [],
            "model_used": "test-model",
            "generated_docs": []
        }

    async def mock_simple_chat(*args, **kwargs):
        return "Simple response"

    async def mock_stream_simple_chat(*args, **kwargs):
        yield "Hello"
        yield " World"

    async def mock_stream_chat(*args, **kwargs):
        yield "TOKEN:Hello"
        yield "DONE"

    monkeypatch.setattr(main_module, "chat", mock_chat)
    monkeypatch.setattr(main_module, "simple_chat", mock_simple_chat)
    monkeypatch.setattr(main_module, "stream_simple_chat", mock_stream_simple_chat)
    monkeypatch.setattr(main_module, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(main_module, "get_active_model", MagicMock(return_value="test-model"))
    monkeypatch.setattr(main_module, "reset_llm", MagicMock())

    # Crear cliente
    with TestClient(app) as c:
        yield c

    # Limpiar overrides después del test
    app.dependency_overrides.clear()


@pytest.fixture
def client_no_auth(monkeypatch, tmp_path):
    """Cliente sin mocks de auth (para tests de autenticación real)."""
    monkeypatch.setattr(main_module, "DOCUMENTS_DIR", tmp_path / "documents")

    with TestClient(app) as c:
        yield c


# ── Tests de sistema/health ─────────────────────────────────────────────────

class TestSystemEndpoints:
    def test_health_check(self, client):
        """Health check responde OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_root_endpoint(self, client):
        """Endpoint raíz responde correctamente."""
        response = client.get("/")
        assert response.status_code == 200
        assert "docs" in response.json()


# ── Tests de autenticación ──────────────────────────────────────────────────

class TestAuthEndpoints:
    def test_login_success(self, client_no_auth, monkeypatch):
        """Login con credenciales válidas retorna token."""
        mock_user = MagicMock()
        mock_user.username = "testuser"
        mock_user.is_admin = False
        mock_user.disabled = False

        monkeypatch.setattr(main_module, "authenticate_user", MagicMock(return_value=mock_user))
        monkeypatch.setattr(main_module, "create_access_token", MagicMock(return_value="fake-jwt-token"))

        response = client_no_auth.post(
            "/auth/token",
            data={"username": "test", "password": "test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "fake-jwt-token"
        assert data["token_type"] == "bearer"
        assert data["username"] == "testuser"

    def test_login_invalid_credentials(self, client_no_auth, monkeypatch):
        """Login con credenciales inválidas retorna 401."""
        monkeypatch.setattr(main_module, "authenticate_user", MagicMock(return_value=None))

        response = client_no_auth.post(
            "/auth/token",
            data={"username": "wrong", "password": "wrong"}
        )

        assert response.status_code == 401
        assert "incorrectos" in response.json()["detail"]

    def test_get_me(self, client):
        """Endpoint /auth/me retorna info del usuario autenticado."""
        response = client.get("/auth/me")
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["is_admin"] is False


# ── Tests de chat ───────────────────────────────────────────────────────────

class TestChatEndpoints:
    def test_chat_simple_mode(self, client):
        """Chat en modo simple funciona."""
        response = client.post(
            "/chat",
            json={
                "message": "Hola",
                "simple_mode": True,
                "use_history": True,
                "conversation_id": "default"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"] == "Simple response"

    def test_chat_agent_mode(self, client):
        """Chat en modo agente funciona."""
        response = client.post(
            "/chat",
            json={
                "message": "Hola",
                "simple_mode": False,
                "agent_type": "general",
                "conversation_id": "default"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert data["agent_used"] == "test"

    def test_chat_empty_message(self, client):
        """Chat con mensaje vacío retorna 400."""
        response = client.post(
            "/chat",
            json={"message": "   ", "simple_mode": True}
        )

        assert response.status_code == 400
        assert "vacío" in response.json()["detail"]

    def test_chat_stream(self, client):
        """Streaming de chat funciona."""
        response = client.post(
            "/chat/stream",
            json={"message": "Hola", "simple_mode": True}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_chat_stream_agent(self, client):
        """Streaming de agente funciona."""
        response = client.post(
            "/chat/stream/agent",
            json={"message": "Hola", "simple_mode": False}
        )

        assert response.status_code == 200


# ── Tests de documentos ────────────────────────────────────────────────────

class TestDocumentEndpoints:
    def test_list_documents(self, client):
        """Listar documentos funciona."""
        response = client.get("/documents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_upload_invalid_extension(self, client):
        """Subir archivo con extensión inválida retorna 400."""
        response = client.post(
            "/documents/upload",
            files={"file": ("virus.exe", b"malicious content", "application/octet-stream")}
        )

        assert response.status_code == 400
        assert "no soportado" in response.json()["detail"]

    def test_upload_success(self, client, monkeypatch):
        """Subir archivo válido funciona."""
        monkeypatch.setattr(main_module, "save_upload", MagicMock(return_value=Path("/tmp/test.txt")))

        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"content", "text/plain")}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_upload_index_error(self, client, monkeypatch):
        """Error en indexación retorna 500."""
        monkeypatch.setattr(
            main_module, "index_document", 
            MagicMock(return_value={"status": "error", "message": "Index failed"})
        )

        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"content", "text/plain")}
        )

        assert response.status_code == 500
        assert "Index failed" in response.json()["detail"]

    def test_delete_document_not_found(self, client, monkeypatch):
        """Eliminar documento inexistente retorna 404."""
        monkeypatch.setattr(main_module, "delete_user_document", MagicMock(return_value=False))

        response = client.delete("/documents/nonexistent.pdf")
        assert response.status_code == 404

    def test_clear_documents(self, client):
        """Limpiar todos los documentos funciona."""
        response = client.delete("/documents")
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"


# ── Tests de historial ─────────────────────────────────────────────────────

class TestHistoryEndpoints:
    def test_get_history(self, client):
        """Obtener historial funciona."""
        response = client.get("/history?last_n=10")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_recent_history(self, client):
        """Obtener historial reciente funciona."""
        response = client.get("/history/recent?conversation_id=default")
        assert response.status_code == 200
        assert "messages" in response.json()

    def test_get_conversation_history(self, client):
        """Obtener historial de conversación específica."""
        response = client.get("/history/conversation?conversation_id=test&last_n=20")
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert data["conversation_id"] == "test"

    def test_delete_history(self, client):
        """Borrar historial funciona."""
        response = client.delete("/history")
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"

    def test_history_stats(self, client):
        """Estadísticas de historial funcionan."""
        response = client.get("/history/stats")
        assert response.status_code == 200
        assert "count" in response.json()

    def test_search_memory(self, client):
        """Búsqueda en memoria semántica funciona."""
        response = client.get("/memory/search?query=test&top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test"
        assert "results" in data


# ── Tests de administración ─────────────────────────────────────────────────

class TestAdminEndpoints:
    def test_list_users_as_admin(self, client, monkeypatch):
        """Admin puede listar usuarios."""
        mock_users = [{"username": "user1"}, {"username": "user2"}]
        monkeypatch.setattr(main_module, "list_users", MagicMock(return_value=mock_users))

        response = client.get("/admin/users")
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_create_user_as_admin(self, client, monkeypatch):
        """Admin puede crear usuarios."""
        monkeypatch.setattr(
            main_module, "create_user",
            MagicMock(return_value={"username": "newuser", "is_admin": False})
        )

        response = client.post(
            "/admin/users",
            json={"username": "newuser", "password": "pass123", "is_admin": False}
        )

        assert response.status_code == 200
        assert response.json()["username"] == "newuser"

    def test_delete_own_account_forbidden(self, client):
        """No se puede eliminar la propia cuenta."""
        response = client.delete("/admin/users/admin")
        assert response.status_code == 400
        assert "propia cuenta" in response.json()["detail"]

    def test_reset_password(self, client, monkeypatch):
        """Admin puede resetear contraseñas."""
        monkeypatch.setattr(main_module, "change_password", MagicMock(return_value=True))

        response = client.put(
            "/admin/users/user1/password",
            json={"new_password": "newpass123"}
        )

        assert response.status_code == 200
        assert "password changed" in response.json()["status"]


# ── Tests de conversaciones ─────────────────────────────────────────────────

class TestConversationEndpoints:
    def test_list_conversations(self, client):
        """Listar conversaciones funciona."""
        response = client.get("/conversations")
        assert response.status_code == 200

    def test_create_conversation(self, client):
        """Crear conversación funciona."""
        response = client.post("/conversations")
        assert response.status_code == 200

    def test_rename_conversation(self, client, monkeypatch):
        """Renombrar conversación funciona."""
        monkeypatch.setattr(main_module, "rename_conversation", MagicMock(return_value=True))

        response = client.put(
            "/conversations/conv1",
            json={"name": "Nuevo nombre"}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "renamed"

    def test_rename_conversation_not_found(self, client, monkeypatch):
        """Renombrar conversación inexistente retorna 404."""
        monkeypatch.setattr(main_module, "rename_conversation", MagicMock(return_value=False))

        response = client.put(
            "/conversations/invalid",
            json={"name": "Nuevo nombre"}
        )

        assert response.status_code == 404

    def test_delete_conversation(self, client, monkeypatch):
        """Eliminar conversación funciona."""
        monkeypatch.setattr(main_module, "clear_history", MagicMock(return_value=True))

        response = client.delete("/conversations/conv1")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    def test_get_active_conversation(self, client):
        """Obtener conversación activa funciona."""
        response = client.get("/conversations/active")
        assert response.status_code == 200
        assert "conversation_id" in response.json()


# ── Tests de configuración ──────────────────────────────────────────────────

class TestConfigEndpoints:
    def test_get_agent_config(self, client, monkeypatch):
        """Obtener configuración de agentes funciona."""
        mock_config = {
            "mode": "auto",
            "agents": ["general", "code"],
            "default_agent": "general"
        }
        mock_names = {"general": "General", "code": "Programador"}

        try:
            import backend.agent_profiles as profiles
            monkeypatch.setattr(profiles, "get_user_config", MagicMock(return_value=mock_config))
            monkeypatch.setattr(profiles, "AGENT_DISPLAY_NAMES", mock_names)
        except ImportError:
            pytest.skip("agent_profiles no disponible")

        response = client.get("/agents/config")
        assert response.status_code == 200
        assert response.json()["mode"] == "auto"

    def test_get_user_agent_config(self, client, monkeypatch):
        """Endpoint alternativo de config funciona."""
        mock_config = {
            "mode": "auto",
            "agents": ["general"],
            "default_agent": "general"
        }

        try:
            import backend.agent_profiles as profiles
            monkeypatch.setattr(profiles, "get_user_config", MagicMock(return_value=mock_config))
            monkeypatch.setattr(profiles, "AGENT_DISPLAY_NAMES", {})
        except ImportError:
            pytest.skip("agent_profiles no disponible")

        response = client.get("/user/agent-config")
        assert response.status_code == 200
        assert "agents" in response.json()


# ── Tests de modelo ─────────────────────────────────────────────────────────

class TestModelEndpoints:
    def test_get_active_model(self, client):
        """Obtener modelo activo funciona."""
        response = client.get("/model/active")
        assert response.status_code == 200
        assert "model" in response.json()

    def test_reset_model(self, client):
        """Resetear modelo funciona."""
        response = client.post("/model/reset")
        assert response.status_code == 200
        assert response.json()["status"] == "reset"


# ── Tests de documentos generados ───────────────────────────────────────────

class TestGeneratedDocuments:
    def test_list_generated_documents(self, client, tmp_path, monkeypatch):
        """Listar documentos generados funciona."""
        docs_dir = tmp_path / "documents" / "testuser"
        docs_dir.mkdir(parents=True)

        test_file = docs_dir / "report.pdf"
        test_file.write_text("PDF content")

        monkeypatch.setattr(main_module, "DOCUMENTS_DIR", tmp_path / "documents")

        response = client.get("/documents/generated")
        assert response.status_code == 200
        assert "documents" in response.json()

    def test_delete_generated_document_not_found(self, client):
        """Eliminar documento generado inexistente retorna 404."""
        response = client.delete("/documents/generated/nonexistent.pdf")
        assert response.status_code == 404

    def test_download_generated_document(self, client, tmp_path, monkeypatch):
        """Descargar documento generado funciona."""
        docs_dir = tmp_path / "documents" / "testuser"
        docs_dir.mkdir(parents=True)

        test_file = docs_dir / "report.pdf"
        test_file.write_bytes(b"PDF binary content")

        monkeypatch.setattr(main_module, "DOCUMENTS_DIR", tmp_path / "documents")

        response = client.get("/documents/generated/report.pdf/download")
        assert response.status_code == 200
        assert response.content == b"PDF binary content"


# ── Tests de error handling y edge cases ────────────────────────────────────

class TestErrorHandling:
    def test_unauthorized_access(self, client_no_auth):
        """Acceso sin token retorna 401/403/422."""
        response = client_no_auth.get("/auth/me")
        assert response.status_code in [401, 403, 422]

    def test_invalid_json_body(self, client):
        """Body JSON inválido manejado correctamente."""
        response = client.post(
            "/chat",
            headers={"Content-Type": "application/json"},
            content=b"invalid json {{{"
        )
        assert response.status_code == 422

    def test_missing_required_field(self, client):
        """Falta campo requerido retorna 422."""
        response = client.post(
            "/chat",
            json={}  # Falta "message"
        )
        assert response.status_code == 422

    def test_conversation_id_none_handling(self, client):
        """Conversation_id 'None' string es convertido a 'default'."""
        response = client.post(
            "/chat",
            json={
                "message": "Hola",
                "conversation_id": "None"
            }
        )
        assert response.status_code == 200
