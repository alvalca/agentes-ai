# ============================================================================
# backend/agents.py — Sistema multiagente con router automático/manual
# ============================================================================

import logging
import re
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    LM_STUDIO_URL, LM_STUDIO_API_KEY, LM_STUDIO_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, AGENT_RECURSION_LIMIT,
)
from backend.tools import web_search, search_news, calculator, code_interpreter, fetch_url, generate_document, agenda as _agenda_tool, set_current_user
from backend.tools_programmer import PROGRAMMER_TOOLS
from backend.rag import search_documents
from backend.memory import (
    get_langchain_messages, add_message,
    auto_name_conversation, get_active_conversation,
    build_semantic_context,
)
from backend.agent_profiles import (
    get_user_config,
    GENERAL_PROMPT, PEDAGOGICO_PROMPT, PSICOLOGO_PROMPT,
    PROGRAMADOR_PROMPT, MATEMATICO_PROMPT,
    ROUTER_PROMPT, AGENT_DISPLAY_NAMES, AGENT_REGISTRY,
)

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.7
_llm_pool: dict = {}

def get_llm(temperature: Optional[float] = None) -> ChatOpenAI:
    global _llm_pool
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    if temp not in _llm_pool:
        _llm_pool[temp] = ChatOpenAI(
            model=LM_STUDIO_MODEL,
            api_key=LM_STUDIO_API_KEY,
            base_url=LM_STUDIO_URL,
            temperature=temp,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _llm_pool[temp]

def reset_llm() -> None:
    global _llm_pool
    _llm_pool = {}

def get_active_model() -> str:
    import httpx
    try:
        resp = httpx.get(
            f"{LM_STUDIO_URL.rstrip('/v1').rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {LM_STUDIO_API_KEY}"},
            timeout=3.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id", "")
    except Exception:
        pass
    return ""

def make_rag_tool(user_id: str):
    from langchain_core.tools import tool

    @tool
    def search_user_documents(query: str, file_name: str = "") -> str:
        """Search through the user's uploaded documents (PDFs, Excel, Word, etc.)
        to find relevant information.
        Args:
            query:     what you want to find (question or keywords).
            file_name: optional — name of the specific file to search in.
                       Use this when the user mentions a specific document by name
                       or when previous searches returned results from the wrong file.
                       Example: "french_revolution_analysis.txt", "informe.pdf"
        """
        docs = search_documents(query, user_id, file_name=file_name or None)
        if not docs:
            return "No relevant documents found in your knowledge base."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "unknown")
            page   = doc.metadata.get("page", "")
            page_str = f" (page {page})" if page else ""
            results.append(f"[{i}] {source}{page_str}\n{doc.page_content[:500]}")
        return "\n\n---\n\n".join(results)
 
    return search_user_documents

def _merge_prompt(base_prompt: str, semantic_ctx: str) -> str:
    if not semantic_ctx:
        return base_prompt
    return base_prompt + "\n\n" + semantic_ctx
    
def make_agenda_tool(user_id: str):
    """Crea la tool de agenda con user_id inyectado."""
    from langchain_core.tools import tool as _lc_tool
 
    @_lc_tool
    def agenda(
        action: str,
        content: str = "",
        date: str = "",
        priority: int = 0,
    ) -> str:
        """Personal agenda, task manager and priority tracker.
        Use proactively when user mentions tasks, reminders, deadlines,
        meetings, pending items, or asks what to focus on.
        Actions: add, list, done, delete, update, suggest, reorder.
        - add: content=description, date=YYYY-MM-DD or natural (mañana/viernes),
               priority=1(urgent)/2(this week)/3(backlog), 0=auto
        - list: content=filter (hoy/mañana/esta semana/urgente/sin fecha) or empty
        - done: content=task description or ID
        - delete: content=task description or ID
        - update: content=task ID or description, priority=1/2/3, date=new date
        - suggest: shows tasks ordered by urgency with recommendations
        - reorder: recalculates priorities based on dates automatically
        """
        return _agenda_tool.func(
            action=action, content=content,
            date=date, priority=priority, user_id=user_id,
        )
    return agenda

def make_doc_tool(user_id: str):
    from langchain_core.tools import tool as lc_tool
    from backend.tools import generate_document as _gen_doc

    @lc_tool
    def generate_document(
        content: str,
        format: str = "pdf",
        title: str = "",
        font: str = "Helvetica",
        font_size: int = 12,
        line_spacing: float = 1.5,
    ) -> str:
        """Generate and save a document from text content.

        Use this tool when the user asks to generate, save or download
        a file/document in any format (pdf, docx, markdown, latex, csv, excel).
        NEVER describe or simulate document creation without calling it.

        Args:
            content:      Full text content of the document. Use \n for line breaks.
                          For csv/excel: pass df.to_csv(index=False) as content.
            format:       Output format: "pdf", "docx", "markdown", "latex",
                          "csv" or "excel"
            title:        Optional document title shown at the top
            font:         Font name. For dyslexia use "Helvetica". Common: "Times-Roman", "Courier"
            font_size:    Font size in points (default 12, use 14-16 for accessibility)
            line_spacing: Line spacing multiplier (1.0=single, 1.5=default, 2.0=double)
        """
        return _gen_doc.func(
            content=content, format=format, title=title,
            font=font, font_size=font_size, line_spacing=line_spacing,
            user_id=user_id
        )
    return generate_document

_TOOL_MAP_FN = None

def _get_tool_map(rag_tool, doc_tool, agenda_tool=None):
    return {
        "search_news":           search_news,
        "web_search":            web_search,
        "calculator":            calculator,
        "code_interpreter":      code_interpreter,
        "fetch_url":             fetch_url,
        "generate_document":     generate_document,
        "agenda":                agenda_tool,
        "rag_tool":              rag_tool,
        "doc_tool":              doc_tool,
    }

def build_agent(agent_type: str, user_id: str, semantic_ctx: str = ""):

    set_current_user(user_id)
    rag_tool  = make_rag_tool(user_id)
    doc_tool  = make_doc_tool(user_id)
    agenda_tool = make_agenda_tool(user_id)
    common_tools = [search_news, web_search, calculator, code_interpreter, fetch_url, agenda_tool, rag_tool, doc_tool]
    tool_map     = _get_tool_map(rag_tool, doc_tool, agenda_tool)

    cfg  = AGENT_REGISTRY.get(agent_type, AGENT_REGISTRY["general"])
    temp = cfg.get("temperature", DEFAULT_TEMPERATURE)
    llm  = get_llm(temperature=temp)

    prompt_name = cfg.get("prompt", "GENERAL_PROMPT")
    prompt_obj  = globals().get(prompt_name, GENERAL_PROMPT)
    merged      = _merge_prompt(prompt_obj, semantic_ctx)

    tools_cfg = cfg.get("tools", "common")
    if tools_cfg == "common":
        tools = common_tools
    elif tools_cfg == "common+programmer":
        tools = common_tools + PROGRAMMER_TOOLS
    elif isinstance(tools_cfg, list):
        tools = [tool_map[t] for t in tools_cfg if t in tool_map]
    else:
        tools = common_tools

    return create_agent(model=llm, tools=tools, system_prompt=merged)

def route_message(message: str) -> str:
    try:
        prompt = f"{ROUTER_PROMPT}\n\n{message}"
        response = get_llm().invoke([HumanMessage(content=prompt)])
        classification = str(response.content).strip().upper()
        if "EMOCIONAL" in classification:
            return "psicologo"
        elif "PEDAGOGICO" in classification:
            return "pedagogico"
        else:
            return "general"
    except Exception as e:
        logger.error(f"Router error: {e}")
        return "general"

async def stream_chat(
    user_id:         str,
    message:         str,
    agent_type:      Optional[str] = None,
    use_history:     bool = True,
    images:          list = [],
    conversation_id: str  = "default",
):
    import json as _json

    # Auto-nombrar conversación con el primer mensaje antes de guardar
    auto_name_conversation(user_id, conversation_id, message)
    add_message(user_id, "user", message, agent="general", conversation_id=conversation_id)

    resolved_agent = agent_type
    if not resolved_agent:
        user_cfg = get_user_config(user_id)
        mode = user_cfg.get("mode", "manual_select")
        if mode == "auto_router":
            resolved_agent = route_message(message)
        else:
            agents_list = user_cfg.get("agents", ["general"])
            resolved_agent = agents_list[0] if agents_list else "general"

    if resolved_agent == "general":
        semantic_ctx = ""
        logger.info(f"Memoria semántica desactivada para agente general ({user_id})")
    else:
        # cross_agents=True para usuarios con auto_router — permite recuperar
        # memorias de agentes relacionados (ej: pedagogico↔psicologo)
        user_cfg = get_user_config(user_id)
        cross = user_cfg.get("mode") == "auto_router"
        semantic_ctx = build_semantic_context(
            user_id=user_id, query=message,
            agent=resolved_agent, top_k=4, min_score=0.35,
            cross_agents=cross,
        )

    agent = build_agent(resolved_agent, user_id, semantic_ctx=semantic_ctx)

    messages = []
    if use_history:
        history = get_langchain_messages(user_id, last_n=10, conversation_id=conversation_id)
        valid_history = []
        i = 0
        while i < len(history):
            if isinstance(history[i], HumanMessage):
                if i + 1 < len(history) and isinstance(history[i+1], AIMessage):
                    if history[i+1].content and "Error:" not in str(history[i+1].content):
                        valid_history.extend([history[i], history[i+1]])
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        messages.extend(valid_history)

    if images:
        content_parts = []
        for img in images:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": "data:" + img["mime"] + ";base64," + img["data"]},
            })
        content_parts.append({"type": "text", "text": message})
        messages.append(HumanMessage(content=content_parts))
    else:
        messages.append(HumanMessage(content=message))

    tool_calls_log = []
    sources        = []
    generated_docs = []
    full_response  = []
    model_used     = ""

    TOOL_NAMES = {
        "web_search":             "🔍 Buscando en la web",
        "search_news":            "📰 Buscando noticias",
        "search_user_documents":  "📄 Consultando documentos",
        "calculator":             "🧮 Calculando",
        "fetch_url":              "🌐 Accediendo a URL",
        "generate_document":      "📝 Generando documento",
        "code_interpreter":       "💻 Ejecutando código",
        "agenda":                 "📅 Consultando agenda",
    }

    try:
        async for event in agent.astream_events(
            {"messages": messages},
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    full_response.append(token)
                    safe_token = token.replace("\n", "\\n")
                    yield "TOKEN:" + safe_token

            elif kind == "on_tool_start":
                display = TOOL_NAMES.get(name, "⚙️ " + name)
                tool_calls_log.append({
                    "tool":  name,
                    "input": str(event.get("data", {}).get("input", ""))[:200],
                })
                yield "TOOL:" + display

            elif kind == "on_tool_end":
                raw_output = event.get("data", {}).get("output", "")
                if hasattr(raw_output, "content"):
                    output = str(raw_output.content)
                else:
                    output = str(raw_output)
                if name == "search_user_documents" and output:
                    sources.append(output[:300])
                if name == "generate_document" and "FORMAT:" in output:
                    m = re.search(r"FORMAT:(\w+)\|PATH:([^\|]+)\|NAME:(.+)", output)
                    if m:
                        generated_docs.append({
                            "format": m.group(1),
                            "path":   m.group(2).strip(),
                            "name":   m.group(3).strip(),
                        })
                if name == "code_interpreter" and "PLOT:" in output:
                    for line in output.split("\n"):
                        if line.startswith("PLOT:"):
                            plot_path = line[5:].strip()
                            generated_docs.append({
                                "format": "plot",
                                "path":   plot_path,
                                "name":   __import__("pathlib").Path(plot_path).name,
                            })
                if name == "code_interpreter" and "FILE:" in output:
                    for line in output.split("\n"):
                        if line.startswith("FILE:"):
                            file_path = line[5:].strip()
                            _p = __import__("pathlib").Path(file_path)
                            generated_docs.append({
                                "format": _p.suffix.lstrip("."),
                                "path":   file_path,
                                "name":   _p.name,
                            })

            elif kind == "on_chat_model_end":
                meta = event.get("data", {}).get("output")
                if meta and hasattr(meta, "response_metadata"):
                    rm = meta.response_metadata or {}
                    model_used = rm.get("model_name") or rm.get("system_fingerprint") or ""

    except Exception as e:
        logger.error("stream_chat error for %s: %s", user_id, e)
        yield "TOKEN:Error: " + str(e)
        full_response.append("Error: " + str(e))

    response_text = "".join(full_response)
    response_clean = re.sub("<think>.*?</think>", "", response_text, flags=re.DOTALL).strip() or response_text
    add_message(user_id, "assistant", response_clean, agent=resolved_agent or "general", conversation_id=conversation_id)

    meta_payload = _json.dumps({
        "tool_calls":     tool_calls_log,
        "sources":        sources,
        "generated_docs": generated_docs,
        "agent_used":     resolved_agent,
        "agent_name":     AGENT_DISPLAY_NAMES.get(resolved_agent, resolved_agent),
        "model_used":     model_used,
    }, ensure_ascii=False)
    yield "META:" + meta_payload
    yield "DONE"


async def stream_simple_chat(
    user_id:         str,
    message:         str,
    conversation_id: str  = "default",
    system_prompt:   str  = (
        "You are a helpful AI assistant. Today's date is 2026. "
        "Your training data has a cutoff of early 2025. "
        "For questions about recent events, remind the user to use agent mode."
    ),
    use_history:     bool = True,
):
    add_message(user_id, "user", message, agent="none", conversation_id=conversation_id)
    messages = [SystemMessage(content=system_prompt)]
    if use_history:
        history = get_langchain_messages(user_id, last_n=10, conversation_id=conversation_id)
        if history and isinstance(history[-1], HumanMessage):
            messages += history[:-1]
        else:
            messages += history
    messages.append(HumanMessage(content=message))

    full_response = []
    try:
        async for chunk in get_llm().astream(messages):
            token = chunk.content
            if token:
                full_response.append(token)
                yield token
    except Exception as e:
        error_msg = f"Error: {e}"
        yield error_msg
        full_response.append(error_msg)

    add_message(
        user_id, "assistant",
        "".join(full_response),
        agent="none",
        conversation_id=conversation_id
    )


async def chat(
    user_id:         str,
    message:         str,
    agent_type:      Optional[str] = None,
    use_history:     bool = True,
    images:          list = [],
    conversation_id: str  = "default",
) -> dict:
    logger.info(f"Chat - user: {user_id}, agent: {agent_type}, msg: {message[:50]}...")

    user_config    = get_user_config(user_id)
    resolved_agent = agent_type

    if resolved_agent is None or resolved_agent == "auto":
        if user_config["mode"] == "auto_router":
            resolved_agent = route_message(message)
            logger.info(f"Router → {resolved_agent} para usuario {user_id}")
        else:
            resolved_agent = user_config.get("default_agent", "general")
            if resolved_agent == "auto":
                resolved_agent = "general"

    auto_name_conversation(user_id, conversation_id, message)
    add_message(user_id, "user", message, agent=resolved_agent, conversation_id=conversation_id)

    if resolved_agent == "general":
        semantic_ctx = ""
        logger.info(f"Memoria semántica desactivada para agente general ({user_id})")
    else:
        semantic_ctx = build_semantic_context(
            user_id=user_id,
            query=message,
            agent=resolved_agent,
            top_k=4,
            min_score=0.35,
        )
        if semantic_ctx:
            logger.info(
                f"Memoria semántica inyectada para {user_id}: "
                f"{len(semantic_ctx)} chars, agente {resolved_agent}"
            )

    agent = build_agent(resolved_agent, user_id, semantic_ctx=semantic_ctx)

    messages = []
    if use_history:
        history = get_langchain_messages(user_id, last_n=10, conversation_id=conversation_id)
        from langchain_core.messages import ToolMessage
        error_indices = set()
        for i, m in enumerate(history):
            if isinstance(m, AIMessage) and str(m.content).startswith("Error:"):
                error_indices.add(i)
                if i > 0 and isinstance(history[i-1], HumanMessage):
                    error_indices.add(i-1)
            if isinstance(m, ToolMessage):
                error_indices.add(i)
            if isinstance(m, AIMessage):
                has_tool_calls = bool(getattr(m, "tool_calls", None))
                content_empty = not m.content or str(m.content).strip() == "" or str(m.content) == "None"
                if has_tool_calls or content_empty:
                    error_indices.add(i)
        history = [m for i, m in enumerate(history) if i not in error_indices]
        first_human = next((i for i, m in enumerate(history) if isinstance(m, HumanMessage)), None)
        if first_human is not None and first_human > 0:
            history = history[first_human:]
        if history and isinstance(history[-1], HumanMessage):
            messages = history[:-1]
        else:
            messages = history

    if images:
        content_parts = []
        for img in images:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['mime']};base64,{img['data']}"
                },
            })
        content_parts.append({"type": "text", "text": message})
        messages.append(HumanMessage(content=content_parts))
        logger.info(f"Mensaje vision: {len(images)} imagen(es) adjuntas")
    else:
        messages.append(HumanMessage(content=message))

    tool_calls_log = []
    sources        = []
    response_text  = ""
    model_used     = ""
    generated_docs = []

    try:
        result      = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )
        all_messages = result.get("messages", [])

        for msg in all_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_log.append({
                        "tool":  tc.get("name", "unknown"),
                        "input": str(tc.get("args", ""))[:200],
                    })
            if hasattr(msg, "name") and msg.name == "generate_document":
                if hasattr(msg, "content") and "FORMAT:" in str(msg.content):
                    import re as _re
                    m = _re.search(r"FORMAT:(\w+)\|PATH:([^\|]+)\|NAME:(.+)", str(msg.content))
                    if m:
                        generated_docs.append({
                            "format": m.group(1),
                            "path":   m.group(2).strip(),
                            "name":   m.group(3).strip(),
                        })
            if hasattr(msg, "name") and msg.name == "search_user_documents":
                if hasattr(msg, "content"):
                    sources.append(msg.content[:300])

        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content:
                response_text = str(msg.content)
                if not model_used:
                    meta = getattr(msg, "response_metadata", {}) or {}
                    model_used = (
                        meta.get("model_name")
                        or meta.get("system_fingerprint")
                        or ""
                    )
                if response_text:
                    break

        if not response_text:
            response_text = "No response generated."

    except Exception as e:
        logger.error(f"Agent error for {user_id}: {e}")
        err_str = str(e)
        if "Connection refused" in err_str or "ConnectError" in err_str or "connect" in err_str.lower():
            reset_llm()
            logger.info("LLM pool reseteado por error de conexión")
        if "Recursion limit" in err_str:
            partial_text = ""
            try:
                if "result" in dir():
                    for m in reversed(result.get("messages", [])):
                        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                            partial_text = re.sub(
                                r"<think>.*?</think>\s*", "", str(m.content),
                                flags=re.DOTALL
                            ).strip()
                            if partial_text:
                                break
            except Exception:
                pass

            if partial_text:
                response_text = partial_text
                logger.warning(f"Recursion limit alcanzado para {user_id} — usando respuesta parcial ({len(partial_text)} chars)")
            else:
                response_text = (
                    "He realizado el máximo de búsquedas permitidas. "
                    "Con la información obtenida no pude completar una respuesta "
                    "completa. Te recomiendo reformular la pregunta de forma más "
                    "específica, por ejemplo indicando el país, fecha o tema concreto."
                )
        else:
            response_text = f"Error: {e}"

    response_text_clean = re.sub(
        r"<think>.*?</think>\s*", "", response_text, flags=re.DOTALL
    ).strip() or response_text

    add_message(user_id, "assistant", response_text_clean, agent=resolved_agent or "general", conversation_id=conversation_id)

    return {
        "response":       response_text_clean,
        "tool_calls":     tool_calls_log,
        "sources":        sources,
        "agent_used":     resolved_agent,
        "agent_name":     AGENT_DISPLAY_NAMES.get(resolved_agent, resolved_agent),
        "model_used":     model_used,
        "generated_docs": generated_docs,
    }

async def simple_chat(
    user_id:         str,
    message:         str,
    conversation_id: str  = "default",
    system_prompt:   str  = (
        "You are a helpful AI assistant. Today's date is 2026. "
        "Your training data has a cutoff of early 2025. "
        "For questions about recent events, remind the user to use agent mode."
    ),
    use_history:     bool = True,
) -> str:
    add_message(user_id, "user", message, agent="none", conversation_id=conversation_id)
    messages = [SystemMessage(content=system_prompt)]
    if use_history:
        history = get_langchain_messages(user_id, last_n=10, conversation_id=conversation_id)
        if history and isinstance(history[-1], HumanMessage):
            messages += history[:-1]
        else:
            messages += history
    messages.append(HumanMessage(content=message))
    try:
        response      = get_llm().invoke(messages)
        response_text = str(response.content)
    except Exception as e:
        response_text = f"Error: {e}"
    add_message(user_id, "assistant", response_text, agent="none")
    return response_text
