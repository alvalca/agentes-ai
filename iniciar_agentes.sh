#!/bin/bash
# ============================================================================
# iniciar_agentes.sh — Configuración Inteligente de GPU por Tipo de Modelo
# ============================================================================

# ── 1. Cargar .env (para overrides de rutas privadas) ────────────────────────
if [ -f .env ]; then
    source .env
fi
 
# ── 2. Configuración con valores por defecto ──────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MAMBA_DIR="${MAMBA_DIR:-$HOME/mamba}"
ENV_NAME="${ENV_NAME:-agentes_py312}"
LMS="${LMS:-$HOME/.lmstudio/bin/lms}"
 
# ── 3. Detección automática de LM Studio si no está en .env ──────────────────
# Añade LM_STUDIO_APP a tu .env para evitar esta búsqueda automática.
# Ejemplo: LM_STUDIO_APP=/home/usuario/Applications/lm-studio.AppImage
if [ -z "$LM_STUDIO_APP" ]; then
    CANDIDATES=(
        "$HOME/Applications/lm-studio.AppImage"        # AppImage en ~/Applications
        "$HOME/Desktop/lm-studio.AppImage"              # AppImage en escritorio
        "/opt/lm-studio/lm-studio"                     # instalación .deb / sistema
        "/usr/bin/lm-studio"                            # paquete deb estándar
        "/usr/local/bin/lm-studio"                      # instalación manual
        "$HOME/.local/bin/lm-studio"                    # instalación de usuario
        "/var/lib/flatpak/app/ai.lmstudio.LMStudio/current/active/export/bin/lm-studio"  # Flatpak
        "$HOME/.local/share/flatpak/app/ai.lmstudio.LMStudio/current/active/export/bin/lm-studio"
    )
    for candidate in "${CANDIDATES[@]}"; do
        if [ -f "$candidate" ]; then
            LM_STUDIO_APP="$candidate"
            break
        fi
    done
fi

# Parámetros de carga
CONTEXT_LENGTH=65536
GPU_OFFLOAD="max"

# Puertos
BACKEND_PORT=8000
FRONTEND_PORT=8501

# ── COLORES ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log()   { echo -e "${GREEN}[Agentes]${NC} $1"; }
warn()  { echo -e "${YELLOW}[Agentes]${NC} $1"; }
err()   { echo -e "${RED}[Agentes]${NC} $1"; }
info()  { echo -e "${BLUE}[Agentes]${NC} $1"; }
debug() { echo -e "${CYAN}[Agentes]${NC} $1"; }

# ── SELECCIÓN DE MODELO ──────────────────────────────────────────────────────
MODELS_FILE="$PROJECT_DIR/modelos.conf"
LAST_MODEL_FILE="$PROJECT_DIR/.last_model"
# Etiqueta del modelo por defecto — debe coincidir exactamente con modelos.conf
DEFAULT_MODEL_LABEL="Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-i1 MoE · Q5_K_M"

# ── MODO SIMPLE vs MODO AVANZADO ─────────────────────────────────────────────
# Modo simple:   sin modelos.conf — LM Studio arranca sin modelo precargado.
#                JIT loading cargará el último modelo usado automáticamente.
# Modo avanzado: con modelos.conf — menú de selección con carga automática.

LLM_MODEL=""
LLM_LABEL="(ninguno — carga manual desde LM Studio)"
SKIP_MODEL_LOAD=false

if [ ! -f "$MODELS_FILE" ]; then
    warn "No existe modelos.conf — arrancando en modo simple."
    warn "LM Studio usará JIT loading con el último modelo usado."
    warn "Para el menú de modelos crea modelos.conf (ver modelos.conf.example)"
    SKIP_MODEL_LOAD=true
else
    MODELS_ID=(); MODELS_LABEL=()
    while IFS='|' read -r id label; do
        [[ "$id" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${id// }" ]] && continue
        MODELS_ID+=("${id// /}")
        MODELS_LABEL+=("${label# }")
    done < "$MODELS_FILE"

    if [ ${#MODELS_ID[@]} -eq 0 ]; then
        warn "modelos.conf vacío — arrancando en modo simple."
        SKIP_MODEL_LOAD=true
    else
        echo ""
        echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
        echo -e "${CYAN}            Selecciona el modelo a cargar                 ${NC}"
        echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
        echo ""
        for i in "${!MODELS_LABEL[@]}"; do
            echo -e "  ${GREEN}[$((i+1))]${NC} ${MODELS_LABEL[$i]}"
        done
        echo ""
        echo -e "${YELLOW}⏳ Selección automática en 30 segundos...${NC}"
        read -t 30 -p "Elige modelo (1-${#MODELS_ID[@]}): " choice
        echo ""

        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#MODELS_ID[@]}" ]; then
            SEL_IDX=$((choice-1))
            log "Modelo seleccionado: ${MODELS_LABEL[$SEL_IDX]}"
        elif [ -f "$LAST_MODEL_FILE" ]; then
            last_label=$(cat "$LAST_MODEL_FILE")
            SEL_IDX=0
            for i in "${!MODELS_LABEL[@]}"; do
                if [ "${MODELS_LABEL[$i]}" = "$last_label" ]; then SEL_IDX=$i; break; fi
            done
            warn "Timeout — usando último modelo: ${MODELS_LABEL[$SEL_IDX]}"
        else
            SEL_IDX=0
            for i in "${!MODELS_LABEL[@]}"; do
                if [ "${MODELS_LABEL[$i]}" = "$DEFAULT_MODEL_LABEL" ]; then SEL_IDX=$i; break; fi
            done
            warn "Timeout — usando modelo por defecto: ${MODELS_LABEL[$SEL_IDX]}"
        fi

        LLM_MODEL="${MODELS_ID[$SEL_IDX]}"
        LLM_LABEL="${MODELS_LABEL[$SEL_IDX]}"
        echo "$LLM_LABEL" > "$LAST_MODEL_FILE"
    fi
fi
# ── CONFIGURACIÓN INTELIGENTE DE GPU ─────────────────────────────────────────
MOE_PATTERNS=(
    "qwen3.5-35b-a3b"
    "qwen3.5-32b-a3b"
    "qwen3.5-30b-a3b"
    "qwen3-30b-a3b"
    "mixtral"
    "deepseek-moe"
    "qwen2-moe"
)

is_moe_model() {
    local model_name=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    for pattern in "${MOE_PATTERNS[@]}"; do
        if [[ "$model_name" == *"$pattern"* ]]; then
            return 0
        fi
    done
    return 1
}

if is_moe_model "$LLM_LABEL"; then
    LM_CUDA_DEVICES="0"
    GPU_STRATEGY="single"
    info "🧠 Modelo MoE detectado: $LLM_LABEL"
    info "⚡ Estrategia: Solo GPU 0 (RTX 5060 Ti) — máxima velocidad"
    info "📝 Embeddings en GPU 1 (GTX 1660 SUPER)"
else
    LM_CUDA_DEVICES="0,1"
    GPU_STRATEGY="dual"
    info "🧮 Modelo Denso detectado: $LLM_LABEL"
    info "⚡ Estrategia: Dual GPU — máximo rendimiento"
fi

# Override manual (descomenta para forzar):
# LM_CUDA_DEVICES="0"    # Forzar solo GPU 0
# LM_CUDA_DEVICES="0,1"  # Forzar ambas GPUs

# ── FUNCIÓN DE LIMPIEZA ──────────────────────────────────────────────────────
cleanup() {
    log "Deteniendo servicios..."
    kill $UVICORN_PID $STREAMLIT_PID 2>/dev/null
    log "Servicios detenidos."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── 1.5. Toggle: Detener Docker si está corriendo (evita conflictos de puertos)
if [ -f "$PROJECT_DIR/docker-compose.yml" ]; then
    if (cd "$PROJECT_DIR" && docker compose ps -q agentes_backend agentes_frontend 2>/dev/null | grep -q .); then
        warn "⚠️  Docker está activo para este proyecto. Deteniendo contenedores..."
        (cd "$PROJECT_DIR" && docker compose down >/dev/null 2>&1)
        echo -e "✅ ${GREEN}Contenedores detenidos. Puerto liberado.${NC}"
    fi
fi

# ── VERIFICAR INSTALACIONES ───────────────────────────────────────────────────
if [ ! -f "$LMS" ] || [ ! -f "$LM_STUDIO_APP" ]; then
    err "No se encontró lms o LM Studio"
    exit 1
fi

# ── INICIAR LM STUDIO ────────────────────────────────────────────────────────
if $LMS status 2>/dev/null | grep -q "Server: ON"; then
    warn "LM Studio ya está corriendo."
else
    log "Lanzando LM Studio (CUDA_VISIBLE_DEVICES=$LM_CUDA_DEVICES)..."
    CUDA_VISIBLE_DEVICES=$LM_CUDA_DEVICES "$LM_STUDIO_APP" --no-sandbox > /dev/null 2>&1 &

    log "Esperando servidor..."
    SERVER_UP=false
    for i in $(seq 1 30); do
        if $LMS status 2>/dev/null | grep -q "Server: ON"; then
            SERVER_UP=true
            log "Servidor listo."
            break
        fi
        sleep 1
    done

    if [ "$SERVER_UP" = false ]; then
        warn "No arrancó automáticamente. Iniciando explícitamente..."
        $LMS server start
        for i in $(seq 1 15); do
            if $LMS status 2>/dev/null | grep -q "Server: ON"; then
                SERVER_UP=true
                log "Servidor iniciado."
                break
            fi
            sleep 1
        done
    fi

    if [ "$SERVER_UP" = false ]; then
        err "No se pudo iniciar el servidor de LM Studio."
        exit 1
    fi
fi

# ── CARGAR MODELO ─────────────────────────────────────────────────────────────
if [ "$SKIP_MODEL_LOAD" = false ]; then
    log "Cargando: $LLM_LABEL"
    log "Identificador: $LLM_MODEL"
    $LMS load "$LLM_MODEL" \
        --context-length $CONTEXT_LENGTH \
        --gpu $GPU_OFFLOAD \
        --yes

    if [ $? -ne 0 ]; then
        err "Error cargando el modelo"
        exit 1
    fi
    log "Modelo cargado correctamente."
else
    warn "Modo simple — sin carga automática de modelo."
    warn "Carga un modelo desde LM Studio GUI o via: lms load <modelo>"
fi

# Verificar distribución de VRAM
sleep 2
log "Distribución de VRAM:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -F',' '{printf "  GPU %s: %s | Usada: %s MB / %s MB\n", $1, $2, $3, $4}'

# ── ACTIVAR ENTORNO MAMBA ────────────────────────────────────────────────────
log "Activando entorno mamba '$ENV_NAME'..."
export MAMBA_ROOT_PREFIX="$MAMBA_DIR"
source "$MAMBA_DIR/etc/profile.d/mamba.sh"
mamba activate "$ENV_NAME"

if [ $? -ne 0 ]; then
    err "Error activando el entorno mamba '$ENV_NAME'."
    exit 1
fi

# Verificar GPUs visibles para el backend
python3 -c "
import torch
gpus = torch.cuda.device_count()
print(f'[Backend] GPUs visibles: {gpus}')
for i in range(gpus):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || true

# ── LANZAR UVICORN ────────────────────────────────────────────────────────────
log "Iniciando backend (uvicorn) en puerto $BACKEND_PORT..."
cd "$PROJECT_DIR"
# Suprimir warnings internos de librerías grpc/abseil
export GRPC_VERBOSITY=ERROR
export GLOG_minloglevel=2
export TF_CPP_MIN_LOG_LEVEL=3
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port $BACKEND_PORT \
    --log-level info \
    --reload &
UVICORN_PID=$!
sleep 3

if ! kill -0 $UVICORN_PID 2>/dev/null; then
    err "Error iniciando uvicorn."
    exit 1
fi
log "Backend iniciado (PID: $UVICORN_PID)"

# ── LANZAR STREAMLIT ──────────────────────────────────────────────────────────
log "Iniciando frontend (streamlit) en puerto $FRONTEND_PORT..."
streamlit run frontend/app.py \
    --server.port $FRONTEND_PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
STREAMLIT_PID=$!
sleep 3

if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
    err "Error iniciando streamlit."
    kill $UVICORN_PID 2>/dev/null
    exit 1
fi
log "Frontend iniciado (PID: $STREAMLIT_PID)"

# ── ABRIR NAVEGADOR ───────────────────────────────────────────────────────────
log "Abriendo navegador..."
sleep 2
xdg-open "http://localhost:$FRONTEND_PORT" 2>/dev/null &

# ── RESUMEN ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        AGENTES AI — Sistema Inteligente Activo            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$GPU_STRATEGY" == "single" ]; then
    echo -e "  🧠 LLM (MoE):     ${CYAN}GPU 0 (RTX 5060 Ti)${NC} — Optimizado tokens/seg"
    echo -e "  📊 Embeddings:    ${CYAN}GPU 1 (GTX 1660)${NC} — RAG/Rerank"
    echo -e "  🎯 Estrategia:    ${YELLOW}Single GPU para MoE${NC}"
else
    echo -e "  🧠 LLM (Denso):   ${CYAN}GPU 0+1 (Ambas)${NC} — Máximo rendimiento"
    echo -e "  📊 Embeddings:    ${CYAN}GPU 1 (GTX 1660)${NC} — RAG/Rerank"
    echo -e "  🎯 Estrategia:    ${YELLOW}Dual GPU para modelo denso${NC}"
fi

echo ""
echo -e "  🔧 Modelo:        ${YELLOW}$LLM_LABEL${NC}"
echo -e "  🌐 Backend:       http://localhost:$BACKEND_PORT"
echo -e "  🎨 Frontend:      http://localhost:$FRONTEND_PORT"
echo -e "  📚 API Docs:      http://localhost:$BACKEND_PORT/docs"
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
log "Presiona Ctrl+C para detener todos los servicios."

wait $UVICORN_PID $STREAMLIT_PID
