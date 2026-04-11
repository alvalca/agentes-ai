# ============================================================================
# frontend/app.py — Interfaz Streamlit multiusuario con agentes especializados
# ============================================================================

import streamlit as st
import requests
from pathlib import Path

st.set_page_config(
    page_title="Agentes AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

st.markdown("""
<style>
    /* ── Reducir espaciado en sidebar ────────────────────────────────── */
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] .stButton {
        margin-bottom: -8px !important;
    }
    [data-testid="stSidebar"] .stButton button {
        padding: 0.2rem 0.5rem !important;
        font-size: 0.82em !important;
        min-height: 0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        margin-bottom: -4px !important;
    }
    [data-testid="stSidebar"] hr {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }
    [data-testid="stSidebar"] .stToggle {
        margin-bottom: -6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        margin-bottom: -8px !important;
    }
    /* Compactar file uploader en sidebar */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        padding: 0.2rem 0.5rem !important;
        min-height: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section > div {
        padding: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        padding: 0.2rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        padding: 0.2rem 0.5rem !important;
        font-size: 0.82em !important;
        min-height: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span {
        font-size: 0.82em !important;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] {
        margin-bottom: -8px !important;
    }

    .chat-user {
        background-color: #d6e8f7;
        padding: 10px 15px; border-radius: 10px;
        margin: 5px 0; color: #1a1a1a;
    }
    /* Colores para st.chat_message */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #d6e8f7 !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #f5efe6 !important;
        border-radius: 10px !important;
    }

    .agent-badge {
        display: inline-block;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.8em; margin-bottom: 4px;
        background-color: #0d6efd; color: white;
    }
    .agent-badge.psicologo  { background-color: #6f42c1; }
    .agent-badge.pedagogico { background-color: #198754; }
    .agent-badge.programador{ background-color: #dc3545; }
    .agent-badge.general    { background-color: #0d6efd; }
    .agent-badge.matematico { background-color: #e67e00; }
    .tool-badge {
        background-color: #495057; color: #adb5bd;
        padding: 1px 7px; border-radius: 8px;
        font-size: 0.75em; margin-right: 3px;
    }
    /* Tablas HTML generadas por pandas */
    .dataframe {
        border-collapse: collapse;
        font-size: 0.85em;
        margin: 8px 0;
        width: 100%;
    }
    .dataframe th {
        background-color: #2d5a8e;
        color: white;
        padding: 6px 10px;
        text-align: left;
    }
    .dataframe td {
        padding: 5px 10px;
        border-bottom: 1px solid #3d3d3d;
    }
    .dataframe tr:nth-child(even) {
        background-color: #1e1e2e;
    }
    .source-box {
        background-color: #252525; border-left: 3px solid #4caf7d;
        padding: 8px; margin: 4px 0;
        font-size: 0.83em; border-radius: 4px;
        color: #b0c4b0;
    }

    /* ── Padding inferior para que el chat no quede tapado ──────────── */
    .main .block-container {
        padding-bottom: 80px !important;
    }
</style>
""", unsafe_allow_html=True)

# Mapeo de agent_id → nombre legible (espejo de agent_profiles.py)
# Importado dinámicamente de agent_profiles para no requerir actualización manual
try:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.agent_profiles import AGENT_DISPLAY_NAMES
except Exception:
    # Fallback por si el backend no está disponible
    AGENT_DISPLAY_NAMES = {
        "general":     "🤖 Asistente General",
        "pedagogico":  "📚 Profesor / Psicopedagogo",
        "psicologo":   "💙 Apoyo Psicológico",
        "programador": "💻 Programador",
        "matematico":  "🔢 Matemático / Filósofo",
    }

# ── Estado de sesión ──────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "token":          None,
        "username":       None,
        "is_admin":       False,
        "messages":       [],
        "page":           "chat",
        "agent_config":   None,
        "selected_agent": None,
        "history_loaded":    False,
        "active_images":     [],
        "active_model":      "",
        "conversation_id":   None,
        "conversations":     [],
        "renaming_conv_id":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── Helpers API ───────────────────────────────────────────────────────────────
def headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

def api_get(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", headers=headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar con la API.")
        return None
    except Exception as e:
        detail = ""
        try: detail = e.response.json().get("detail","")
        except: pass
        st.error(f"❌ {detail or e}")
        return None

def api_post(endpoint, data=None, files=None, method="POST"):
    try:
        if files:
            r = requests.post(f"{API_URL}{endpoint}", headers=headers(), files=files, timeout=300)
        elif method == "PUT":
            r = requests.put(f"{API_URL}{endpoint}", headers=headers(), json=data, timeout=30)
        else:
            r = requests.post(f"{API_URL}{endpoint}", headers=headers(), json=data, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar con la API.")
        return None
    except Exception as e:
        detail = ""
        try: detail = e.response.json().get("detail","")
        except: pass
        st.error(f"❌ {detail or e}")
        return None

def api_delete(endpoint):
    try:
        r = requests.delete(f"{API_URL}{endpoint}", headers=headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ {e}")
        return None

def load_agent_config():
    config = api_get("/agents/config")
    if config:
        st.session_state.agent_config = config
    model_data = api_get("/model/active")
    if model_data:
        st.session_state.active_model = model_data.get("model", "")
        if not st.session_state.selected_agent:
            st.session_state.selected_agent = config["default"]

def load_recent_history():
    conv_id = st.session_state.get("conversation_id") or "default"
    result  = api_get(f"/history/conversation?conversation_id={conv_id}")
    if not result or not result.get("messages"):
        st.session_state.history_loaded = True
        return 0

    loaded = []
    for msg in result["messages"]:
        agent_id = msg.get("agent", "general")
        if msg["role"] == "user":
            loaded.append({
                "role":    "user",
                "content": msg["content"],
            })
        else:
            loaded.append({
                "role":       "assistant",
                "content":    msg["content"],
                "agent_used": agent_id,
                "agent_name": "⚡ Sin agente" if agent_id == "none" else AGENT_DISPLAY_NAMES.get(agent_id, "🤖 Agente"),
                "tool_calls":     [],
                "sources":        [],
                "model_used":     msg.get("model_used", ""),
                "generated_docs": [],
            })

    st.session_state.messages       = loaded
    st.session_state.history_loaded = True

    count = len(loaded)
    if count > 0:
        st.toast(f"💬 {count} mensajes cargados", icon="🕐")
    return count

def load_conversations():
    result = api_get("/conversations")
    if result is not None:
        st.session_state.conversations = result if isinstance(result, list) else []

def ensure_active_conversation():
    if not st.session_state.get("conversation_id"):
        load_conversations()
        convs = st.session_state.get("conversations", [])
        if convs:
            st.session_state.conversation_id = convs[0]["id"]
        else:
            result = api_get("/conversations/active")
            conv_id = "default"
            if result:
                conv_id = result.get("conversation_id") or "default"
            st.session_state.conversation_id = conv_id
            load_conversations()

def switch_conversation(conv_id: str):
    st.session_state.conversation_id  = conv_id
    st.session_state.messages         = []
    st.session_state.history_loaded   = False
    st.session_state.active_images    = []

def page_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🤖 Agentes AI")
        st.markdown("### Iniciar sesión")
        with st.form("login_form"):
            username  = st.text_input("Usuario")
            password  = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Entrar", use_container_width=True)

        if submitted:
            try:
                r = requests.post(
                    f"{API_URL}/auth/token",
                    data={"username": username, "password": password},
                    timeout=10,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.token          = data["access_token"]
                    st.session_state.username       = data["username"]
                    st.session_state.is_admin       = data["is_admin"]
                    st.session_state.history_loaded = False
                    st.success(f"✅ Bienvenido, {data['username']}!")
                    st.rerun()
                else:
                    st.error(f"❌ {r.json().get('detail', 'Error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ No se puede conectar con la API en localhost:8000")

def render_sidebar():
    with st.sidebar:
        admin_label = " 🔑" if st.session_state.is_admin else ""
        st.caption(f"👤 **{st.session_state.username}**{admin_label}")

        pages = {"💬 Chat": "chat", "📁 Documentos": "documents", "📂 Generados": "generated", "🕐 Historial": "history"}
        if st.session_state.is_admin:
            pages["⚙️ Admin"] = "admin"

        for label, key in pages.items():
            btn_type = "primary" if st.session_state.page == key else "secondary"
            if st.button(label, use_container_width=True, type=btn_type):
                st.session_state.page = key
                st.rerun()

        st.divider()

        if st.session_state.page == "chat":
            config = st.session_state.agent_config

            st.markdown("📎 **Adjuntar imagen**")
            img_files = st.file_uploader(
                "Imagen", type=["jpg", "jpeg", "png", "gif", "webp"],
                label_visibility="collapsed",
                accept_multiple_files=True,
                key="img_uploader",
            )
            if img_files:
                import base64
                for img_file in img_files:
                    nombres_activos = [i["name"] for i in st.session_state.active_images]
                    if img_file.name not in nombres_activos:
                        img_data = base64.b64encode(img_file.read()).decode("utf-8")
                        st.session_state.active_images.append({
                            "data": img_data,
                            "mime": img_file.type or "image/jpeg",
                            "name": img_file.name,
                        })

            st.divider()

            if config:
                mode   = config.get("mode")
                agents = config.get("agents", [])

                if mode == "manual_select" and len(agents) > 1:
                    st.markdown("**Agente activo**")
                    agent_options = {a["name"]: a["id"] for a in agents}
                    current_name  = next(
                        (a["name"] for a in agents
                         if a["id"] == st.session_state.selected_agent),
                        agents[0]["name"]
                    )
                    selected_name = st.selectbox(
                        "Selecciona agente",
                        options=list(agent_options.keys()),
                        index=list(agent_options.keys()).index(current_name),
                        label_visibility="collapsed",
                    )
                    st.session_state.selected_agent = agent_options[selected_name]

                elif mode == "auto_router":
                    st.markdown("**Modo:** 🔄 Router automático")
                    st.caption("El agente se selecciona según tu mensaje")

            st.divider()
            st.markdown("**Opciones**")
            st.session_state.simple_mode = st.toggle(
                "Modo simple (sin tools)",
                value=st.session_state.get("simple_mode", False)
            )
            st.session_state.use_history = st.toggle(
                "Usar historial",
                value=st.session_state.get("use_history", True)
            )

            if st.button("🗑️ Limpiar chat", use_container_width=True):
                st.session_state.messages       = []
                st.session_state.history_loaded = True
                st.rerun()

            st.divider()

            st.markdown("**💬 Conversaciones**")
            if st.button("➕ Nueva conversación", use_container_width=True):
                result = api_post("/conversations", {})
                if result:
                    switch_conversation(result["id"])
                    load_conversations()
                    st.rerun()

            convs = st.session_state.get("conversations", [])
            active_id = st.session_state.get("conversation_id")
            renaming_id = st.session_state.get("renaming_conv_id")

            for conv in convs:
                cid   = conv["id"]
                cname = conv["name"]
                is_active = cid == active_id

                if renaming_id == cid:
                    new_name = st.text_input(
                        "Nombre", value=cname, key=f"rename_input_{cid}",
                        label_visibility="collapsed"
                    )
                    col_ok, col_cancel = st.columns(2)
                    with col_ok:
                        if st.button("✓", key=f"ok_{cid}", use_container_width=True):
                            api_post(f"/conversations/{cid}", {"name": new_name}, method="PUT")
                            st.session_state.renaming_conv_id = None
                            load_conversations()
                            st.rerun()
                    with col_cancel:
                        if st.button("✕", key=f"cancel_{cid}", use_container_width=True):
                            st.session_state.renaming_conv_id = None
                            st.rerun()
                else:
                    col_btn, col_edit, col_del = st.columns([6, 1, 1])
                    with col_btn:
                        btn_type = "primary" if is_active else "secondary"
                        label = f"{'▶ ' if is_active else ''}{cname[:30]}"
                        if st.button(label, key=f"conv_{cid}",
                                     use_container_width=True, type=btn_type):
                            if not is_active:
                                switch_conversation(cid)
                                st.rerun()
                    with col_edit:
                        if st.button("✏️", key=f"edit_{cid}"):
                            st.session_state.renaming_conv_id = cid
                            st.rerun()
                    with col_del:
                        if st.button("🗑", key=f"del_{cid}"):
                            api_delete(f"/conversations/{cid}")
                            if is_active:
                                st.session_state.conversation_id = None
                                st.session_state.messages = []
                                st.session_state.history_loaded = False
                            load_conversations()
                            st.rerun()

        st.divider()
        if st.button("🚪 Cerrar sesión", use_container_width=True):
            for k in ["token","username","is_admin","messages",
                      "agent_config","selected_agent"]:
                st.session_state[k] = None if k != "messages" else []
            st.session_state.is_admin          = False
            st.session_state.history_loaded    = False
            st.session_state.conversation_id   = None
            st.session_state.conversations     = []
            st.session_state.renaming_conv_id  = None
            st.session_state.active_images     = []
            st.session_state["_convs_initialized"] = False
            st.rerun()

def _stream_response(message: str, use_history: bool, conversation_id: str, token: str):
    try:
        with requests.post(
            f"{API_URL}/chat/stream",
            json={
                "message":         message,
                "use_history":     use_history,
                "simple_mode":     True,
                "conversation_id": conversation_id,
            },
            headers={"Authorization": f"Bearer {token}"},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except Exception as e:
        yield "Error de conexión: " + str(e)


def _stream_agent_response(
    message: str, use_history: bool, conversation_id: str,
    token: str, agent_type: str, images: list,
    tool_placeholder, meta_container
):
    import json as _json
    buffer = ""
    active_tools = []

    try:
        with requests.post(
            f"{API_URL}/chat/stream/agent",
            json={
                "message":         message,
                "use_history":     use_history,
                "simple_mode":     False,
                "agent_type":      agent_type,
                "images":          images,
                "conversation_id": conversation_id,
            },
            headers={"Authorization": f"Bearer {token}"},
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("TOKEN:"):
                    token_text = raw[6:].replace("\\n", "\n")
                    buffer += token_text
                    yield token_text
                elif raw.startswith("TOOL:"):
                    tool_name = raw[5:]
                    active_tools.append(tool_name)
                    tool_placeholder.markdown(
                        " · ".join(f"`{t}`" for t in active_tools)
                    )
                elif raw.startswith("META:"):
                    try:
                        meta = _json.loads(raw[5:])
                        st.session_state["_last_stream_meta"] = meta
                    except Exception:
                        pass
                elif raw == "DONE":
                    tool_placeholder.empty()
                    break
    except Exception as e:
        yield "Error: " + str(e)


def page_chat():
    st.markdown("## 💬 Chat")

    if not st.session_state.get("_convs_initialized"):
        if not st.session_state.get("conversation_id"):
            ensure_active_conversation()
        if not st.session_state.get("conversations"):
            load_conversations()
        st.session_state["_convs_initialized"] = True
        st.rerun()

    if not st.session_state.get("history_loaded", False):
        load_recent_history()

    config = st.session_state.agent_config
    mode   = config.get("mode") if config else "manual_select"

    if st.session_state.get("simple_mode"):
        st.caption("Modo: 💬 LLM directo (sin tools)")
    elif mode == "auto_router":
        st.caption("Modo: 🔄 Router automático — el agente se elige según tu mensaje")
    else:
        agent_id   = st.session_state.selected_agent or "general"
        agent_name = next(
            (a["name"] for a in (config.get("agents", []) if config else [])
             if a["id"] == agent_id),
            AGENT_DISPLAY_NAMES.get(agent_id, agent_id)
        )
        st.caption(f"Agente activo: {agent_name}")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            agent_id  = msg.get("agent_used", "general")
            badge_cls = agent_id if agent_id in ["psicologo","pedagogico","programador","matematico"] else "general"
            name      = msg.get("agent_name", AGENT_DISPLAY_NAMES.get(agent_id, "🤖 Agente"))
            model_label = ""
            if msg.get("model_used"):
                model_short = msg["model_used"].split("/")[-1]
                model_label = f' <small style="color:#888;font-size:0.72em">🧠 {model_short}</small>'
            with st.chat_message("assistant"):
                st.markdown(
                    f'<span class="agent-badge {badge_cls}">{name}</span>{model_label}',
                    unsafe_allow_html=True
                )
                st.markdown(msg["content"])
            if msg.get("tool_calls"):
                tools_html = " ".join(
                    f'<span class="tool-badge">🔧 {tc["tool"]}</span>'
                    for tc in msg["tool_calls"]
                )
                st.markdown(f"<small>{tools_html}</small>", unsafe_allow_html=True)

            # Botones de descarga, plots y tablas HTML
            mime_map = {
                "pdf":      "application/pdf",
                "docx":     "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "markdown": "text/markdown",
                "latex":    "text/x-tex",
                "csv":      "text/csv",
                "xlsx":     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }
            for doc in msg.get("generated_docs", []):
                from pathlib import Path as _Path
                doc_path = _Path(doc["path"])
                if doc.get("format") == "plot":
                    if doc_path.exists():
                        st.image(str(doc_path), width=600)
                elif doc.get("format") == "table":
                    st.markdown(doc.get("html", ""), unsafe_allow_html=True)
                else:
                    if doc_path.exists():
                        with open(doc_path, "rb") as _f:
                            st.download_button(
                                label=f"⬇️ Descargar {doc['name']}",
                                data=_f.read(),
                                file_name=doc["name"],
                                mime=mime_map.get(doc["format"], "application/octet-stream"),
                                key=f"dl_{doc['name']}",
                            )
            if msg.get("sources"):
                with st.expander("📚 Fuentes de documentos"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-box">{src[:300]}...</div>',
                            unsafe_allow_html=True
                        )

    active_images = st.session_state.get("active_images", [])
    if active_images:
        img_cols = st.columns(len(active_images))
        for i, img in enumerate(active_images):
            with img_cols[i]:
                st.image(
                    f"data:{img['mime']};base64,{img['data']}",
                    caption=img["name"], width=120,
                )
                if st.button(f"✕ Descartar", key=f"discard_{i}"):
                    st.session_state.active_images.pop(i)
                    st.rerun()

    user_input = st.chat_input("Escribe tu mensaje...")

    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})

        agent_type = None
        if not st.session_state.get("simple_mode") and mode == "manual_select":
            agent_type = st.session_state.selected_agent

        simple_mode = st.session_state.get("simple_mode", False)

        if simple_mode:
            with st.chat_message("user"):
                st.markdown(
                    f'<div class="chat-user">{user_input.strip()}</div>',
                    unsafe_allow_html=True
                )
            agent_header = st.empty()
            model_short = st.session_state.get("active_model", "").split("/")[-1]
            agent_header.markdown(
                f'<div style="font-size:0.78em;color:#888;padding:2px 8px;">'
                f'<span class="agent-badge general">⚡ Sin agente</span>'
                f'&nbsp;<small>🧠 {model_short}</small></div>',
                unsafe_allow_html=True
            )
            with st.chat_message("assistant"):
                streamed_text = st.write_stream(_stream_response(
                    message=user_input.strip(),
                    use_history=st.session_state.get("use_history", True),
                    conversation_id=st.session_state.get("conversation_id") or "default",
                    token=st.session_state.token,
                ))
            st.session_state.messages.append({
                "role":           "assistant",
                "content":        streamed_text or "",
                "agent_used":     "general",
                "agent_name":     "⚡ Sin agente",
                "tool_calls":     [],
                "sources":        [],
                "model_used":     st.session_state.get("active_model", ""),
                "generated_docs": [],
            })
            st.rerun()
        else:
            with st.chat_message("user"):
                st.markdown(
                    f'<div class="chat-user">👤 <b>Tú:</b> {user_input.strip()}</div>',
                    unsafe_allow_html=True
                )
            agent_header = st.empty()
            _model_short = st.session_state.get("active_model", "").split("/")[-1]
            _agent_name = AGENT_DISPLAY_NAMES.get(agent_type or "general", "🤖 Agente")
            _badge_cls = agent_type if agent_type in ["psicologo","pedagogico","programador"] else "general"
            agent_header.markdown(
                f'<div style="font-size:0.78em;color:#888;padding:2px 8px;">'
                f'<span class="agent-badge {_badge_cls}">{_agent_name}</span>'
                f'&nbsp;<small>🧠 {_model_short}</small></div>',
                unsafe_allow_html=True
            )
            with st.chat_message("assistant"):
                tool_placeholder = st.empty()
                st.session_state["_last_stream_meta"] = {}
                streamed_text = st.write_stream(_stream_agent_response(
                    message=user_input.strip(),
                    use_history=st.session_state.get("use_history", True),
                    conversation_id=st.session_state.get("conversation_id") or "default",
                    token=st.session_state.token,
                    agent_type=agent_type,
                    images=st.session_state.get("active_images", []),
                    tool_placeholder=tool_placeholder,
                    meta_container=None,
                ))

            meta = st.session_state.get("_last_stream_meta", {})
            agent_id = meta.get("agent_used", "general")
            if not meta.get("model_used"):
                meta["model_used"] = st.session_state.get("active_model", "")
            badge_cls = agent_id if agent_id in ["psicologo","pedagogico","programador","matematico"] else "general"
            agent_name = meta.get("agent_name", AGENT_DISPLAY_NAMES.get(agent_id, "🤖 Agente"))
            model_short = meta.get("model_used", "").split("/")[-1]
            agent_header.markdown(
                f'<div style="font-size:0.78em;color:#888;padding:2px 8px;">'
                f'<span class="agent-badge {badge_cls}">{agent_name}</span>'
                f'&nbsp;<small>🧠 {model_short}</small></div>',
                unsafe_allow_html=True
            )
            import re as _re
            clean_text = _re.sub("<think>.*?</think>", "", streamed_text or "", flags=_re.DOTALL).strip()
            st.session_state.messages.append({
                "role":           "assistant",
                "content":        clean_text or "",
                "agent_used":     agent_id,
                "agent_name":     meta.get("agent_name", AGENT_DISPLAY_NAMES.get(agent_id, "🤖 Agente")),
                "tool_calls":     meta.get("tool_calls", []),
                "sources":        meta.get("sources", []),
                "model_used":     meta.get("model_used", ""),
                "generated_docs": meta.get("generated_docs", []),
            })
            st.rerun()


def page_documents():
    st.markdown("## 📁 Documentos")
    with st.expander("➕ Subir nuevo documento", expanded=True):
        uploaded = st.file_uploader(
            "Selecciona un archivo",
            type=["pdf","txt","epub","xlsx","xls","csv","pptx","docx","md","html"],
        )
        if uploaded and st.button("📤 Indexar documento"):
            with st.spinner(f"Indexando {uploaded.name}..."):
                result = api_post(
                    "/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )
            if result and result.get("status") == "ok":
                st.success(
                    f"✅ **{result['file']}** — "
                    f"{result['pages']} páginas, {result['chunks']} chunks"
                )

    st.divider()
    st.markdown("### 📋 Documentos indexados")
    docs = api_get("/documents")
    if docs is None:
        return
    if not docs:
        st.info("No tienes documentos indexados.")
        return

    icons = {
        ".pdf":"📄",".txt":"📝",".epub":"📚",".xlsx":"📊",
        ".xls":"📊",".csv":"📊",".pptx":"📑",".docx":"📃",
        ".md":"📝",".html":"🌐"
    }
    for doc in docs:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"{icons.get(doc['ext'],'📄')} **{doc['name']}** — {doc['size_kb']} KB")
        with col2:
            if st.button("🗑️", key=f"del_{doc['name']}"):
                if api_delete(f"/documents/{doc['name']}"):
                    st.rerun()

    st.divider()
    if st.button("🗑️ Eliminar todos", type="secondary"):
        api_delete("/documents")
        st.success("Todos los documentos eliminados")
        st.rerun()


def page_generated_documents():
    import datetime
    st.markdown("## 📂 Documentos Generados")
    st.markdown("Archivos generados por los agentes durante las conversaciones.")

    docs_data = api_get("/documents/generated")
    if docs_data is None:
        st.error("Error al cargar los documentos generados.")
        return

    docs = docs_data.get("documents", [])

    if not docs:
        st.info("No tienes documentos generados aún. Pide a un agente que genere un documento.")
        return

    total_size = sum(d["size_kb"] for d in docs)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total documentos", len(docs))
    with col2:
        st.metric("Espacio usado", f"{round(total_size/1024, 2)} MB" if total_size > 1024 else f"{round(total_size, 1)} KB")

    st.divider()

    fmt_icons = {
        "pdf": "📄", "docx": "📃", "md": "📝",
        "markdown": "📝", "tex": "🔬", "latex": "🔬",
        "csv": "📊", "xlsx": "📊",
    }

    for doc in docs:
        fmt   = doc["format"]
        icon  = fmt_icons.get(fmt, "📄")
        fecha = datetime.datetime.fromtimestamp(doc["created_at"]).strftime("%d/%m/%Y %H:%M")

        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            st.markdown(f"{icon} **{doc['name']}** — {fecha} · {doc['size_kb']} KB · `{fmt.upper()}`")
        with col2:
            try:
                resp = requests.get(
                    f"{API_URL}/documents/generated/{doc['name']}/download",
                    headers={"Authorization": f"Bearer {st.session_state.token}"},
                    timeout=30
                )
                if resp.status_code == 200:
                    mime_map = {
                        "pdf": "application/pdf",
                        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "md": "text/markdown",
                        "markdown": "text/markdown",
                        "tex": "text/x-tex",
                        "latex": "text/x-tex",
                        "csv": "text/csv",
                        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    }
                    st.download_button(
                        "⬇️",
                        data=resp.content,
                        file_name=doc["name"],
                        mime=mime_map.get(fmt, "application/octet-stream"),
                        key=f"dl_{doc['name']}",
                    )
            except Exception:
                st.markdown("⬇️")
        with col3:
            if st.button("🗑️", key=f"del_gen_{doc['name']}"):
                result = api_delete(f"/documents/generated/{doc['name']}")
                if result is not None:
                    st.rerun()

    st.divider()
    if st.button("🗑️ Eliminar todos los documentos generados", type="secondary"):
        deleted = 0
        for doc in docs:
            if api_delete(f"/documents/generated/{doc['name']}") is not None:
                deleted += 1
        st.success(f"✅ {deleted} documentos eliminados")
        st.rerun()


def page_history():
    st.markdown("## 🕐 Historial")
    stats = api_get("/history/stats")
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total",         stats["total_messages"])
        c2.metric("Tus mensajes",  stats["user_messages"])
        c3.metric("Respuestas AI", stats["assistant_messages"])
        if stats.get("semantic_memories", 0) > 0:
            st.info(
                f"🧠 **Memoria semántica:** "
                f"{stats['semantic_memories']} fragmentos vectorizados"
            )
        if stats["first_message"]:
            st.caption(
                f"Desde: {stats['first_message'][:19].replace('T',' ')} | "
                f"Último: {stats['last_message'][:19].replace('T',' ')}"
            )

    st.divider()
    n = st.slider("Últimos N mensajes", 10, 100, 20, step=10)
    history = api_get(f"/history?last_n={n}")
    if not history:
        st.info("Sin historial.")
        return

    for msg in reversed(history):
        ts   = msg.get("timestamp","")[:19].replace("T"," ")
        icon = "👤" if msg["role"] == "user" else "🤖"
        agent_label = ""
        if msg["role"] == "assistant" and msg.get("agent"):
            _agent_display = {
                "none":       "⚡ Sin agente",
                "general":    "🤖 Asistente General",
                "pedagogico": "📚 Profesor / Psicopedagogo",
                "psicologo":  "💙 Apoyo Psicológico",
                "programador":"💻 Programador",
                "matematico": "🔢 Matemático / Filósofo",
            }
            agent_label = f" [{_agent_display.get(msg['agent'], msg['agent'])}]"
        st.markdown(f"**{icon}** `{ts}`{agent_label}")
        st.markdown(f"> {msg['content'][:500]}")
        st.divider()

    st.divider()
    st.markdown("### 🧠 Buscar en memoria semántica")
    st.caption("Busca en todas tus conversaciones pasadas por contenido similar")
    with st.form("semantic_search_form"):
        sem_query = st.text_input(
            "Consulta",
            placeholder="Ej: problemas con alumno TDAH, día difícil en clase..."
        )
        col1, col2 = st.columns(2)
        with col1:
            sem_top_k = st.slider("Resultados", 1, 10, 5)
        with col2:
            sem_agent = st.selectbox(
                "Filtrar por agente",
                ["Todos","pedagogico","psicologo","general","programador"]
            )
        sem_submit = st.form_submit_button("🔍 Buscar en memoria")

    if sem_submit and sem_query:
        agent_param = None if sem_agent == "Todos" else sem_agent
        url = f"/memory/search?query={sem_query}&top_k={sem_top_k}"
        if agent_param:
            url += f"&agent_filter={agent_param}"
        results = api_get(url)
        if results and results.get("results"):
            st.markdown(f"**{results['count']} memorias encontradas:**")
            for mem in results["results"]:
                ts    = mem.get("timestamp","")[:10]
                score = mem.get("score", 0)
                role  = "👤 Tú" if mem["role"] == "user" else "🤖 Agente"
                agent = AGENT_DISPLAY_NAMES.get(mem.get("agent",""), mem.get("agent",""))
                with st.expander(f"{role} — {ts} — {score:.0%} — {agent}"):
                    st.markdown(mem["content"][:500])
        else:
            st.info("No se encontraron memorias relevantes.")

    st.divider()
    if st.button("🗑️ Borrar historial", type="secondary"):
        api_delete("/history")
        st.success("Historial y memoria semántica borrados")
        st.rerun()


def page_admin():
    st.markdown("## ⚙️ Administración")
    st.markdown("### 👥 Usuarios")
    users = api_get("/admin/users")
    if users:
        for user in users:
            c1, c2 = st.columns([5, 1])
            with c1:
                role   = "🔑 Admin" if user["is_admin"] else "👤 Usuario"
                status = "✅" if not user["disabled"] else "❌"
                full_name = user.get("full_name") or ""
                name_label = f" — {full_name}" if full_name else ""
                st.markdown(f"{status} **{user['username']}**{name_label} {role}")
                if user.get("email"):
                    st.caption(f"  {user['email']}")
            with c2:
                if user["username"] != st.session_state.username:
                    if st.button("🗑️", key=f"du_{user['username']}"):
                        if api_delete(f"/admin/users/{user['username']}"):
                            st.rerun()

    st.divider()
    st.markdown("### ➕ Crear usuario")
    with st.form("create_user"):
        c1, c2 = st.columns(2)
        with c1:
            nu = st.text_input("Usuario")
            np = st.text_input("Contraseña", type="password")
        with c2:
            ne = st.text_input("Email (opcional)")
            nf = st.text_input("Nombre completo (opcional)")
        na = st.checkbox("Es administrador")
        if st.form_submit_button("Crear usuario"):
            if nu and np:
                result = api_post("/admin/users", {
                    "username": nu, "password": np,
                    "email": ne or None, "full_name": nf or None,
                    "is_admin": na,
                })
                if result:
                    st.success(f"✅ Usuario '{nu}' creado")
                    st.rerun()

    st.divider()
    st.markdown("### 💡 Configurar agentes por usuario")
    st.markdown(
        "ℹ️ Edita `backend/agent_profiles.py` en el servidor y reinicia el backend."
    )


def main():
    if not st.session_state.token:
        page_login()
        return

    try:
        r = requests.get(f"{API_URL}/auth/me", headers=headers(), timeout=30)
        if r.status_code == 401:
            st.session_state.token = None
            st.rerun()
            return
    except requests.exceptions.ReadTimeout:
        pass
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar con la API en localhost:8000")
        st.info("Arranca el backend: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`")
        return

    if not st.session_state.agent_config:
        load_agent_config()

    render_sidebar()

    page = st.session_state.page
    if page == "chat":
        page_chat()
    elif page == "documents":
        page_documents()
    elif page == "generated":
        page_generated_documents()
    elif page == "history":
        page_history()
    elif page == "admin":
        if st.session_state.is_admin:
            page_admin()
        else:
            st.error("Acceso denegado")

if __name__ == "__main__":
    main()
