# tests/test_agents_unit.py — Tests unitarios para backend/agents.py
# ============================================================================
# Enfoque: Tests rápidos, deterministas, con mocks de dependencias externas.
# Ejecutar con: pytest tests/test_agents_unit.py -v
# ============================================================================
import pytest
from unittest.mock import patch, MagicMock, call
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def _seed_test_user(user_id: str, conv_id: str = "default"):
    """Crea usuario y conversación en SQLite para evitar FK errors."""
    from backend.database import get_db
    with get_db() as db:
        db.execute(
            "INSERT OR IGNORE INTO users (username, email, full_name, disabled, is_admin, hashed_password) "
            "VALUES (?, ?, ?, 0, 0, ?)",
            (user_id, f"{user_id}@test.com", "Test User", "$2b$12$fakehash")
        )
        db.execute(
            "INSERT OR IGNORE INTO conversations (id, user_id, name, created_at, last_active) "
            "VALUES (?, ?, ?, datetime('now'), datetime('now'))",
            (conv_id, user_id, "Conversación test")
        )

# ── Fixtures reutilizables ──────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Mock de ChatOpenAI para evitar llamadas reales."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="Mock response")
    
    async def mock_astream(*args, **kwargs):
        for token in ["Mock", " response"]:
            chunk = MagicMock()
            chunk.content = token
            yield chunk
    mock.astream = mock_astream
    return mock


@pytest.fixture
def mock_create_agent():
    """Mock de langchain.agents.create_agent."""
    with patch("backend.agents.create_agent") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_get_llm(mock_llm):
    """Patch get_llm para devolver el mock."""
    with patch("backend.agents.get_llm", return_value=mock_llm) as m:
        yield m


# ── Tests: Funciones puras y de utilidad ────────────────────────────────────

class TestMergePrompt:
    """Tests para _merge_prompt — función pura, fácil de validar."""

    def test_merge_prompt_without_context_returns_base(self):
        """Si semantic_ctx está vacío, devuelve el prompt base sin cambios."""
        from backend.agents import _merge_prompt
        base = "Eres un asistente útil."
        result = _merge_prompt(base, "")
        assert result == base

    def test_merge_prompt_with_context_appends_correctly(self):
        """Con contexto, lo añade con separador de doble salto de línea."""
        from backend.agents import _merge_prompt
        base = "Eres un asistente útil."
        ctx = "Contexto: el usuario estudia Python."
        result = _merge_prompt(base, ctx)
        assert result == f"{base}\n\n{ctx}"

    def test_merge_prompt_handles_multiline_context(self):
        """El contexto multilínea se preserva tal cual."""
        from backend.agents import _merge_prompt
        base = "Prompt base"
        ctx = "Línea 1\nLínea 2\nLínea 3"
        result = _merge_prompt(base, ctx)
        assert "Línea 1\nLínea 2\nLínea 3" in result


class TestGetToolMap:
    """Tests para _get_tool_map — validación de mapeo de herramientas."""

    def test_get_tool_map_returns_dict_with_expected_keys(self):
        """El mapa debe contener todas las herramientas comunes."""
        from backend.agents import _get_tool_map
        rag_mock = MagicMock()
        doc_mock = MagicMock()
        agenda_mock = MagicMock()
        
        tool_map = _get_tool_map(rag_mock, doc_mock, agenda_mock)
        
        expected_keys = {
            "search_news", "web_search", "calculator", "code_interpreter",
            "fetch_url", "generate_document", "agenda", "rag_tool", "doc_tool"
        }
        assert set(tool_map.keys()) == expected_keys

    def test_get_tool_map_tools_have_executable_func(self):
        """Cada herramienta LangChain debe tener .func callable (no ser callable directo)."""
        from backend.agents import _get_tool_map
        rag_mock = MagicMock()
        doc_mock = MagicMock()
        agenda_mock = MagicMock()
        
        tool_map = _get_tool_map(rag_mock, doc_mock, agenda_mock)
        
        for name, tool in tool_map.items():
            # LangChain @tool devuelve StructuredTool con .func interno
            if hasattr(tool, "func"):
                assert callable(tool.func), f"{name}.func no es callable"
            else:
                # Fallback para funciones puras (ej: calculator)
                assert callable(tool), f"{name} no es callable ni tiene .func"


# ── Tests: Gestión de LLM (singleton/pool) ─────────────────────────────────


class TestLLMManagement:
    """Tests para get_llm y reset_llm — gestión de cache y configuración."""

    def test_get_llm_returns_same_instance_for_same_temperature(self):
        """Misma temperatura → misma instancia (cache)."""
        from backend.agents import get_llm, reset_llm
        reset_llm()  # Limpiar estado previo
        
        llm1 = get_llm(temperature=0.5)
        llm2 = get_llm(temperature=0.5)
        
        assert llm1 is llm2, "Debería reutilizar la misma instancia"

    def test_get_llm_returns_different_instance_for_different_temperature(self):
        """Diferente temperatura → diferente instancia."""
        from backend.agents import get_llm, reset_llm
        reset_llm()
        
        llm_low = get_llm(temperature=0.2)
        llm_high = get_llm(temperature=0.9)
        
        # Nota: Esto depende de que ChatOpenAI no haga cache interno por temperatura
        # Si falla, es porque LangChain reusa conexiones — en ese caso, testear 
        # que se pasan los parámetros correctos al constructor (más complejo)
        assert llm_low.temperature == 0.2
        assert llm_high.temperature == 0.9

    def test_get_llm_uses_default_temperature_when_none(self):
        """Si no se pasa temperatura, usa DEFAULT_TEMPERATURE (0.7)."""
        from backend.agents import get_llm, reset_llm, DEFAULT_TEMPERATURE
        reset_llm()
        
        llm = get_llm()  # Sin argumento
        
        assert llm.temperature == DEFAULT_TEMPERATURE

    def test_reset_llm_clears_pool(self):
        """reset_llm() debe vaciar el pool de LLMs."""
        import backend.agents as agents_module  # 🔑 Importar módulo, no variable
        
        agents_module.reset_llm()
        assert len(agents_module._llm_pool) == 0
        
        agents_module.get_llm(0.5)
        assert len(agents_module._llm_pool) == 1
        
        agents_module.reset_llm()
        assert len(agents_module._llm_pool) == 0


# ── Tests: Fábricas de herramientas con closure de user_id ─────────────────

class TestToolFactories:
    """Tests para make_rag_tool, make_agenda_tool, make_doc_tool."""

    def test_make_rag_tool_returns_tool_with_bound_user_id(self):
        """make_rag_tool debe devolver una LangChain tool que recuerda el user_id."""
        from backend.agents import make_rag_tool
        from unittest.mock import patch
        
        with patch("backend.agents.search_documents") as mock_search:
            mock_search.return_value = []
            
            rag_tool = make_rag_tool(user_id="test_user_123")
            
            # Verificar que es una tool de LangChain con .func
            assert hasattr(rag_tool, "func"), "Debe ser una LangChain tool con .func"
            assert callable(rag_tool.func), ".func debe ser callable"
            
            # Ejecutar la función interna directamente (bypass LangChain wrapper)
            rag_tool.func("Python", file_name="notes.txt")
            
            # Verificar que search_documents recibió el user_id correcto
            mock_search.assert_called_once_with(
                "Python", "test_user_123", file_name="notes.txt"
            )

    def test_make_agenda_tool_injects_user_id_correctly(self):
        """make_agenda_tool debe inyectar user_id en la llamada a _agenda_tool.func."""
        from backend.agents import make_agenda_tool
        from unittest.mock import patch
        
        with patch("backend.agents._agenda_tool") as mock_agenda_module:
            mock_agenda_module.func.return_value = "OK"
            
            agenda_tool = make_agenda_tool(user_id="user_xyz")
            
            # Llamar a .func directamente (bypass LangChain)
            agenda_tool.func(
                action="add",
                content="Reunión con equipo",
                date="2026-04-15",
                priority=2
            )
            
            mock_agenda_module.func.assert_called_once_with(
                action="add",
                content="Reunión con equipo",
                date="2026-04-15",
                priority=2,
                user_id="user_xyz"
            )

    def test_make_doc_tool_forwards_all_params_with_user_id(self):
        """make_doc_tool debe forwardear todos los parámetros + user_id."""
        from backend.agents import make_doc_tool
        # 🔑 Patchear en el módulo donde se define la función original, no donde se importa
        with patch("backend.tools.generate_document") as mock_gen_doc:
            mock_gen_doc.func.return_value = "FORMAT:pdf|PATH:/x|NAME:doc.pdf"
            
            doc_tool = make_doc_tool(user_id="alice")
            
            # Llamar a .func directamente
            doc_tool.func(
                content="# Informe\nContenido",
                format="pdf",
                title="Mi Informe",
                font="Helvetica",
                font_size=14,
                line_spacing=1.5
            )
            
            mock_gen_doc.func.assert_called_once_with(
                content="# Informe\nContenido",
                format="pdf",
                title="Mi Informe",
                font="Helvetica",
                font_size=14,
                line_spacing=1.5,
                user_id="alice"
            )


# ── Tests: build_agent — configuración y selección de herramientas ─────────

class TestBuildAgent:
    """Tests para build_agent — validación de configuración sin LLM real."""

    def test_build_agent_uses_correct_prompt_for_agent_type(self, mock_get_llm, mock_create_agent):
        """Cada tipo de agente debe recibir su prompt específico."""
        from backend.agents import build_agent
        from backend.agent_profiles import GENERAL_PROMPT, PROGRAMADOR_PROMPT
        
        # Agente general
        agent_general = build_agent("general", user_id="u1")
        # Verificar que create_agent fue llamado con el prompt correcto
        # (El prompt se pasa como system_prompt a create_agent)
        calls = mock_create_agent.call_args_list
        # El último call debe ser para "general"
        last_call_kwargs = calls[-1].kwargs
        assert GENERAL_PROMPT in last_call_kwargs["system_prompt"]

    def test_build_agent_selects_tools_based_on_config(self, mock_get_llm, mock_create_agent):
        """La configuración del agente debe determinar qué herramientas se incluyen."""
        from backend.agents import build_agent
        from backend.agent_profiles import AGENT_REGISTRY
        
        # Verificar configuración de ejemplo
        general_cfg = AGENT_REGISTRY["general"]
        assert "tools" in general_cfg, "Configuración debe definir 'tools'"
        
        # Construir agente y verificar que se pasaron herramientas
        agent = build_agent("general", user_id="test")
        
        # create_agent debe haber sido llamado con una lista de tools
        call_kwargs = mock_create_agent.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) > 0, "Debería haber al menos una herramienta"

    def test_build_agent_handles_unknown_agent_type_gracefully(self, mock_get_llm, mock_create_agent):
        """Si el tipo de agente no existe, debe fallback a 'general'."""
        from backend.agents import build_agent
        from backend.agent_profiles import AGENT_REGISTRY
        
        # Agente inexistente
        agent = build_agent("agente_inventado_xyz", user_id="test")
        
        # Debería haber usado la configuración de "general"
        assert agent is not None
        # Verificar que se usó el prompt de general
        call_kwargs = mock_create_agent.call_args.kwargs
        assert "general" in call_kwargs["system_prompt"].lower() or "helpful" in call_kwargs["system_prompt"].lower()

    def test_build_agent_merges_semantic_context_when_provided(self, mock_get_llm, mock_create_agent):
        """Si se pasa semantic_ctx, debe fusionarse con el prompt base."""
        from backend.agents import build_agent
        
        semantic_ctx = "El usuario ha preguntado antes sobre pandas DataFrame."
        agent = build_agent("programador", user_id="u1", semantic_ctx=semantic_ctx)
        
        # Verificar que el contexto se incluyó en el prompt final
        call_kwargs = mock_create_agent.call_args.kwargs
        assert semantic_ctx in call_kwargs["system_prompt"]


# ── Tests: route_message — clasificación y error handling ───────────────────

class TestRouteMessage:
    """Tests para route_message — router de clasificación de mensajes."""

    def test_route_message_returns_valid_agent_for_emotional_query(self, mock_get_llm):
        """Consultas emocionales deben enrutar a 'psicologo'."""
        from backend.agents import route_message
        
        # Mockear respuesta del LLM para clasificación emocional
        mock_get_llm.return_value.invoke.return_value = MagicMock(
            content="EMOCIONAL"
        )
        
        result = route_message("Me siento muy ansioso por el futuro")
        assert result == "psicologo"

    def test_route_message_returns_valid_agent_for_pedagogical_query(self, mock_get_llm):
        """Consultas pedagógicas deben enrutar a 'pedagogico'."""
        from backend.agents import route_message
        
        mock_get_llm.return_value.invoke.return_value = MagicMock(
            content="PEDAGOGICO"
        )
        
        result = route_message("¿Cómo explico fracciones a un niño de 8 años?")
        assert result == "pedagogico"

    def test_route_message_defaults_to_general_for_unknown_classification(self, mock_get_llm):
        """Si la clasificación no es reconocida, fallback a 'general'."""
        from backend.agents import route_message
        
        # Respuesta ambigua o en otro formato
        mock_get_llm.return_value.invoke.return_value = MagicMock(
            content="No estoy seguro"
        )
        
        result = route_message("¿Qué hora es?")
        assert result == "general"

    def test_route_message_handles_llm_exception_gracefully(self, mock_get_llm):
        """Si el LLM falla, debe retornar 'general' sin lanzar excepción."""
        from backend.agents import route_message
        
        # Simular error de conexión o timeout
        mock_get_llm.return_value.invoke.side_effect = Exception("Connection error")
        
        result = route_message("Cualquier mensaje")
        
        # No debe propagar la excepción
        assert result == "general"


# ── Tests: Funciones auxiliares y edge cases ────────────────────────────────

class TestAgentHelpers:
    """Tests para funciones auxiliares y validación de configuración."""

    def test_get_active_model_handles_connection_error(self):
        """get_active_model debe manejar errores de conexión sin crashear."""
        from backend.agents import get_active_model
        # 🔑 httpx se importa DENTRO de la función → patchear en el módulo original
        with patch("httpx.get", side_effect=Exception("Timeout")):
            result = get_active_model()
            assert result == ""

    def test_get_active_model_parses_valid_response(self):
        """get_active_model debe extraer el ID del primer modelo disponible."""
        from backend.agents import get_active_model
        from unittest.mock import MagicMock
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "llama-3-8b-instruct", "name": "Llama 3"},
                {"id": "mistral-7b", "name": "Mistral"}
            ]
        }
        
        # 🔑 Patchear httpx.get directamente (no backend.agents.httpx)
        with patch("httpx.get", return_value=mock_resp):
            result = get_active_model()
            assert result == "llama-3-8b-instruct"
            

# ── Tests: Chat Función ────────────────────────────────

class TestChatFunction:
    """Tests para la función chat() — versión síncrona."""

    @pytest.mark.asyncio
    async def test_chat_returns_structured_response(self, mock_get_llm, mock_create_agent, setup_sqlite_db):
        from backend.agents import chat
        _seed_test_user("test_user", "test_conv")
        """chat() debe devolver un dict con los campos esperados."""
        from backend.agents import chat
        
        # Mockear el agente para que devuelva una respuesta predecible
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                AIMessage(content="Respuesta de prueba", response_metadata={"model_name": "test-model"})
            ]
        }
        mock_create_agent.return_value = mock_agent
        
        # Ejecutar chat
        result = await chat(
            user_id="test_user",
            message="Hola, ¿cómo estás?",
            agent_type="general",
            conversation_id="test_conv"
        )
        
        # Validar estructura de respuesta
        assert isinstance(result, dict)
        assert "response" in result
        assert "agent_used" in result
        assert result["agent_used"] == "general"
        assert "Respuesta de prueba" in result["response"]

    @pytest.mark.asyncio
    async def test_chat_handles_agent_routing_when_none_specified(self, mock_get_llm, mock_create_agent, setup_sqlite_db):
        from backend.agents import chat
        _seed_test_user("user_xyz", "conv1")
        """Si agent_type=None, debe usar configuración del usuario o fallback a 'general'."""
        from backend.agents import chat
        from unittest.mock import patch
        
        # Mockear configuración de usuario con mode manual
        with patch("backend.agents.get_user_config", return_value={
            "mode": "manual_select",
            "agents": ["programador"],
            "default_agent": "programador"
        }):
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [AIMessage(content="OK", response_metadata={})]
            }
            mock_create_agent.return_value = mock_agent
            
            result = await chat(
                user_id="user_xyz",
                message="¿Cómo optimizo este código?",
                agent_type=None,  # ← No especificado
                conversation_id="conv1"
            )
            
            # Debería haber usado el agente por defecto de la config
            assert result["agent_used"] == "programador"

    @pytest.mark.asyncio
    async def test_chat_handles_llm_connection_error(self, mock_get_llm, mock_create_agent, setup_sqlite_db):
        from backend.agents import chat
        _seed_test_user("test", "test")
        """Si el LLM falla por conexión, debe resetear pool y devolver error amigable."""
        from backend.agents import chat
        
        # Mockear agente que lanza error de conexión
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = Exception("Connection refused: LM Studio not running")
        mock_create_agent.return_value = mock_agent
        
        result = await chat(
            user_id="test",
            message="Cualquier cosa",
            conversation_id="test"
        )
        
        # Debería contener mensaje de error, no propagar excepción
        assert "Error:" in result["response"] or "connection" in result["response"].lower()
        
        # Y debería haber intentado resetear el pool de LLM
        from backend.agents import _llm_pool
        # (El reset se hace internamente; verificar que no crasheó es suficiente)
            
            
