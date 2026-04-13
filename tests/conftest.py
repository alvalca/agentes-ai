# tests/conftest.py — Fixtures compartidos para todos los tests
# ============================================================================
import pytest
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.documents import Document

# ── Mock de dependencias pesadas para entornos sin GPU/ChromaDB ───────────────
# Estas dependencias se mockean para que los tests unitarios corran
# sin necesidad de instalar el stack completo del proyecto.
# En el servidor con el entorno agentes_py312 activo, los módulos reales
# se importarán y los mocks no tendrán efecto.

def _mock_if_missing(module_name: str):
    if module_name not in sys.modules:
        try:
            __import__(module_name)
        except ImportError:
            sys.modules[module_name] = MagicMock()
            # Mockear submódulos comunes
            parts = module_name.split(".")
            for i in range(1, len(parts)):
                sub = ".".join(parts[:i+1])
                if sub not in sys.modules:
                    sys.modules[sub] = MagicMock()

for _dep in [
    "ddgs", "chromadb", "chromadb.config",
    "sentence_transformers", "sentence_transformers.cross_encoder",
    "llama_index", "llama_index.core", "llama_index.core.node_parser",
    "llama_index.core.schema", "llama_index.retrievers.bm25",
    "pdfplumber", "ebooklib", "bs4",
    "langchain_chroma", "langchain_huggingface",
    "langchain_community", "langchain_community.embeddings",
    "langchain_text_splitters",
    "llama_index.core.node_parser.text.sentence",
]:
    _mock_if_missing(_dep)


# ── Fixtures de documentos ────────────────────────────────────────────────────

@pytest.fixture
def structured_doc():
    """Documento estructurado con párrafos, títulos y puntuación clara."""
    return Document(
        page_content=(
            "# La Revolución Francesa\n\n"
            "La Revolución Francesa fue un período de transformación política. "
            "Comenzó en 1789 y terminó en 1799. Sus causas fueron múltiples.\n\n"
            "## Causas principales\n\n"
            "La desigualdad social era extrema. El tercer estado pagaba impuestos. "
            "La nobleza estaba exenta de cargas fiscales.\n\n"
            "### Referencias\n\n"
            "- Furet, François. Interpreting the French Revolution. 1981.\n"
            "- Doyle, William. The Oxford History of the French Revolution. 2002."
        ),
        metadata={"source": "french_revolution.txt", "page": 1}
    )


@pytest.fixture
def transcript_doc():
    """Transcripción de YouTube — texto continuo sin estructura."""
    return Document(
        page_content=(
            "hola bienvenidos a este video donde vamos a hablar sobre inteligencia artificial "
            "y sus aplicaciones en el mundo moderno la inteligencia artificial ha avanzado "
            "muchísimo en los últimos años y cada vez vemos más aplicaciones en nuestra vida "
            "diaria desde los asistentes de voz hasta los sistemas de recomendación que usan "
            "plataformas como netflix o spotify todos estos sistemas usan algoritmos de "
            "machine learning que aprenden de los datos para hacer predicciones y "
            "recomendaciones personalizadas hoy vamos a ver tres casos de uso principales "
            "el primero es el procesamiento de lenguaje natural el segundo es la visión "
            "artificial y el tercero es el aprendizaje por refuerzo cada uno tiene sus "
            "características propias y sus aplicaciones específicas"
        ),
        metadata={"source": "ai_transcript.txt"}
    )


@pytest.fixture
def tmp_data_dir():
    """Directorio temporal para datos de test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_tasks():
    """Lista de tareas de ejemplo para tests de agenda."""
    return [
        {
            "id": 1,
            "content": "Llamar al padre de Miguel",
            "date": "2026-04-15",
            "priority": 1,
            "done": False,
            "created": "2026-04-10T10:00:00",
        },
        {
            "id": 2,
            "content": "Preparar examen de fracciones",
            "date": "2026-04-18",
            "priority": 2,
            "done": False,
            "created": "2026-04-10T10:05:00",
        },
        {
            "id": 3,
            "content": "Revisar RAG con transcripciones",
            "date": "",
            "priority": 3,
            "done": False,
            "created": "2026-04-10T10:10:00",
        },
        {
            "id": 4,
            "content": "Reunión con dirección",
            "date": "2026-04-10",  # fecha pasada — vencida
            "priority": 1,
            "done": False,
            "created": "2026-04-08T09:00:00",
        },
    ]


# ── NUEVAS FIXTURES PARA TESTS DE AUTH Y MAIN ───────────────────────────────
# Añadidas para soportar test_auth.py y test_main.py (cobertura de seguridad)
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Crear un event loop para tests async.

    Necesario para pytest-asyncio cuando se usan tests async a nivel de sesión.
    Esto evita errores de 'event loop is closed' entre tests.
    """
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars(monkeypatch, tmp_path):
    """Set de variables de entorno seguras para testing.

    Aisla los tests del entorno real y usa directorios temporales.
    Esto evita que los tests modifiquen tu users.json real.
    """
    # Variables de auth - CLAVE para que los tests JWT funcionen
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-12345-not-for-production")
    monkeypatch.setenv("JWT_EXPIRE_MINUTES", "30")
    monkeypatch.setenv("DEFAULT_ADMIN_USER", "admin")
    monkeypatch.setenv("DEFAULT_ADMIN_PASS", "adminpass")

    # Variables de paths (usar tmp_path para aislamiento)
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(test_data_dir))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "test.log"))

    return test_data_dir


@pytest.fixture
def temp_users_file(monkeypatch, tmp_path):
    """Crea un archivo de usuarios JSON temporal para tests de auth.

    Aísla los tests de autenticación para no modificar users.json real.
    Retorna el Path al archivo temporal.
    """
    users_file = tmp_path / "users.json"
    users_file.write_text("{}")  # Inicializar vacío

    # Parchear la ruta en auth.py si está disponible
    try:
        import backend.auth as auth_module
        original_users_file = auth_module.USERS_FILE
        monkeypatch.setattr(auth_module, "USERS_FILE", users_file)
        # También parchear en el momento de importación de otros módulos
        monkeypatch.setattr("backend.auth.USERS_FILE", users_file)
    except ImportError:
        pass  # auth.py no disponible en este entorno, el test lo manejará

    return users_file


@pytest.fixture
def sample_user_dict():
    """Datos de usuario de ejemplo para tests de auth.

    La contraseña hasheada corresponde a 'secret123'.
    """
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "disabled": False,
        "is_admin": False,
        "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiAYMyzJ/I3O"
    }


@pytest.fixture
def sample_admin_dict():
    """Datos de usuario admin de ejemplo para tests de auth.

    La contraseña hasheada corresponde a 'secret123'.
    """
    return {
        "username": "adminuser",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "disabled": False,
        "is_admin": True,
        "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiAYMyzJ/I3O"
    }


@pytest.fixture
def auth_test_client(mock_env_vars, temp_users_file):
    """Cliente de FastAPI configurado para tests de autenticación.

    Este fixture configura un entorno completamente aislado para tests de auth:
    - Usa archivos temporales para users.json
    - Variables de entorno de test
    - Base de datos limpia para cada test

    Uso:
        def test_algo(auth_test_client):
            client, auth_module = auth_test_client
            # Hacer requests...
    """
    try:
        from fastapi.testclient import TestClient
        from backend.main import app
        import backend.auth as auth_module

        # Reinicializar usuarios con el archivo temporal
        auth_module._init_users()

        client = TestClient(app)
        return client, auth_module
    except ImportError as e:
        pytest.skip(f"Dependencias no disponibles: {e}")
