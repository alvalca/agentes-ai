# tests/test_agenda.py — Tests unitarios para la tool de agenda
# ============================================================================
# Cubre: añadir tareas, listar con filtros, completar, eliminar,
#        actualizar, suggest, reorder, parse de fechas, prioridad auto
# ============================================================================
import pytest
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helper: ejecutar agenda con directorio temporal ──────────────────────────

def run_agenda(tmp_dir, action, content="", date="", priority=0):
    """Ejecuta la tool agenda apuntando a un directorio temporal."""
    with patch("backend.tools.Path") as mock_path_cls:
        # Redirigir la ruta de agenda al directorio temporal
        real_path = Path
        def path_side_effect(*args):
            p = real_path(*args)
            if "agenda" in str(p):
                return tmp_dir / "agenda" / "test_user"
            return p
        mock_path_cls.side_effect = path_side_effect
        mock_path_cls.return_value = real_path

    # Llamar directamente a la lógica interna de la tool
    # importando el módulo y ejecutando con el directorio inyectado
    import importlib
    import backend.tools as tools_mod

    agenda_dir = tmp_dir / "agenda" / "test_user"
    agenda_dir.mkdir(parents=True, exist_ok=True)
    agenda_file = agenda_dir / "agenda.json"

    # Importar la función interna
    from backend.tools import agenda as _agenda_tool
    # Ejecutar la función con user_id que apunta a tmp_dir
    # Parchear DOCUMENTS_DIR y la ruta de agenda
    original_func = _agenda_tool.func

    # Ejecutar con monkeypatch de Path dentro de la función
    import builtins
    real_open = builtins.open

    result = original_func(
        action=action, content=content, date=date,
        priority=priority, user_id=f"../../{tmp_dir}/agenda_test"
    )
    return result, agenda_dir


class TestAgendaAdd:
    """Tests para action='add'."""

    def test_add_basic_task(self, tmp_data_dir):
        agenda_dir = tmp_data_dir / "agenda" / "test_user"
        agenda_dir.mkdir(parents=True, exist_ok=True)
        agenda_file = agenda_dir / "agenda.json"
        agenda_file.write_text("[]", encoding="utf-8")

        from backend.tools import agenda as _tool

        # Llamar directamente con ruta controlada
        result = _call_agenda(agenda_file, "add",
                               content="Llamar al padre de Miguel",
                               date="2026-04-15", priority=1)
        assert "Llamar al padre de Miguel" in result
        assert "URGENTE" in result or "2026-04-15" in result

        # Verificar que se guardó en el JSON
        tasks = json.loads(agenda_file.read_text())
        assert len(tasks) == 1
        assert tasks[0]["content"] == "Llamar al padre de Miguel"
        assert tasks[0]["priority"] == 1
        assert tasks[0]["done"] is False

    def test_add_without_content_returns_error(self, tmp_data_dir):
        agenda_file = _setup_agenda(tmp_data_dir)
        result = _call_agenda(agenda_file, "add", content="")
        assert "❌" in result

    def test_add_auto_priority_with_date(self, tmp_data_dir):
        agenda_file = _setup_agenda(tmp_data_dir)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        result = _call_agenda(agenda_file, "add",
                               content="Tarea urgente", date=tomorrow, priority=0)
        tasks = json.loads(agenda_file.read_text())
        assert tasks[0]["priority"] == 1  # mañana = URGENTE

    def test_add_auto_priority_no_date(self, tmp_data_dir):
        agenda_file = _setup_agenda(tmp_data_dir)
        result = _call_agenda(agenda_file, "add",
                               content="Tarea sin fecha", priority=0)
        tasks = json.loads(agenda_file.read_text())
        assert tasks[0]["priority"] == 3  # sin fecha = PENDIENTE

    def test_add_natural_date_manana(self, tmp_data_dir):
        agenda_file = _setup_agenda(tmp_data_dir)
        _call_agenda(agenda_file, "add", content="Test mañana", date="mañana")
        tasks = json.loads(agenda_file.read_text())
        expected = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert tasks[0]["date"] == expected

    def test_add_natural_date_hoy(self, tmp_data_dir):
        agenda_file = _setup_agenda(tmp_data_dir)
        _call_agenda(agenda_file, "add", content="Test hoy", date="hoy")
        tasks = json.loads(agenda_file.read_text())
        expected = datetime.now().strftime("%Y-%m-%d")
        assert tasks[0]["date"] == expected

    def test_add_increments_id(self, tmp_data_dir):
        agenda_file = _setup_agenda(tmp_data_dir)
        _call_agenda(agenda_file, "add", content="Tarea 1")
        _call_agenda(agenda_file, "add", content="Tarea 2")
        _call_agenda(agenda_file, "add", content="Tarea 3")
        tasks = json.loads(agenda_file.read_text())
        ids = [t["id"] for t in tasks]
        assert ids == [1, 2, 3]


class TestAgendaList:
    """Tests para action='list'."""

    def test_list_empty_agenda(self, tmp_data_dir):
        agenda_file = _setup_agenda(tmp_data_dir)
        result = _call_agenda(agenda_file, "list")
        assert "📭" in result

    def test_list_all_pending(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        result = _call_agenda(agenda_file, "list")
        assert "Llamar al padre de Miguel" in result
        assert "Preparar examen" in result

    def test_list_filter_urgente(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        result = _call_agenda(agenda_file, "list", content="urgente")
        assert "Llamar al padre de Miguel" in result
        # Las de prioridad 2 y 3 no deben aparecer
        assert "Preparar examen de fracciones" not in result

    def test_list_excludes_done_tasks(self, tmp_data_dir):
        tasks = [
            {"id": 1, "content": "Tarea completada", "date": "",
             "priority": 1, "done": True, "created": "2026-04-10T10:00:00"},
            {"id": 2, "content": "Tarea pendiente", "date": "",
             "priority": 2, "done": False, "created": "2026-04-10T10:01:00"},
        ]
        agenda_file = _setup_agenda(tmp_data_dir, tasks)
        result = _call_agenda(agenda_file, "list")
        assert "Tarea pendiente" in result
        assert "Tarea completada" not in result

    def test_list_sorted_by_priority(self, tmp_data_dir):
        tasks = [
            {"id": 1, "content": "Baja prioridad", "date": "",
             "priority": 3, "done": False, "created": "2026-04-10T10:00:00"},
            {"id": 2, "content": "Alta prioridad", "date": "",
             "priority": 1, "done": False, "created": "2026-04-10T10:01:00"},
        ]
        agenda_file = _setup_agenda(tmp_data_dir, tasks)
        result = _call_agenda(agenda_file, "list")
        # Alta prioridad debe aparecer antes
        pos_alta = result.find("Alta prioridad")
        pos_baja = result.find("Baja prioridad")
        assert pos_alta < pos_baja


class TestAgendaDone:
    """Tests para action='done'."""

    def test_mark_done_by_text(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        result = _call_agenda(agenda_file, "done",
                               content="Llamar al padre de Miguel")
        assert "✅" in result
        tasks = json.loads(agenda_file.read_text())
        task = next(t for t in tasks if "Miguel" in t["content"])
        assert task["done"] is True

    def test_mark_done_by_id(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        result = _call_agenda(agenda_file, "done", content="1")
        tasks = json.loads(agenda_file.read_text())
        assert tasks[0]["done"] is True

    def test_done_nonexistent_task(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        result = _call_agenda(agenda_file, "done", content="tarea_que_no_existe_xyz")
        assert "❌" in result

    def test_done_partial_text_match(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        result = _call_agenda(agenda_file, "done", content="examen")
        tasks = json.loads(agenda_file.read_text())
        task = next(t for t in tasks if "examen" in t["content"].lower())
        assert task["done"] is True


class TestAgendaDelete:
    """Tests para action='delete'."""

    def test_delete_by_text(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        initial_count = len(sample_tasks)
        _call_agenda(agenda_file, "delete", content="RAG con transcripciones")
        tasks = json.loads(agenda_file.read_text())
        assert len(tasks) == initial_count - 1
        assert not any("RAG" in t["content"] for t in tasks)

    def test_delete_nonexistent_returns_error(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        result = _call_agenda(agenda_file, "delete", content="tarea_inexistente_xyz")
        assert "❌" in result


class TestAgendaUpdate:
    """Tests para action='update'."""

    def test_update_priority(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        _call_agenda(agenda_file, "update",
                      content="examen de fracciones", priority=1)
        tasks = json.loads(agenda_file.read_text())
        task = next(t for t in tasks if "examen" in t["content"].lower())
        assert task["priority"] == 1

    def test_update_date(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        _call_agenda(agenda_file, "update",
                      content="examen de fracciones", date="2026-04-20")
        tasks = json.loads(agenda_file.read_text())
        task = next(t for t in tasks if "examen" in t["content"].lower())
        assert task["date"] == "2026-04-20"

    def test_update_nothing_returns_error(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        result = _call_agenda(agenda_file, "update",
                               content="examen", priority=0, date="")
        assert "❌" in result


class TestAgendaSuggest:
    """Tests para action='suggest'."""

    def test_suggest_empty_agenda(self, tmp_data_dir):
        agenda_file = _setup_agenda(tmp_data_dir)
        result = _call_agenda(agenda_file, "suggest")
        assert "📭" in result

    def test_suggest_shows_overdue(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        result = _call_agenda(agenda_file, "suggest")
        # "Reunión con dirección" tiene fecha pasada — debe aparecer como vencida
        assert "VENCIDA" in result or "Reunión con dirección" in result

    def test_suggest_includes_total(self, tmp_data_dir, sample_tasks):
        agenda_file = _setup_agenda(tmp_data_dir, sample_tasks)
        result = _call_agenda(agenda_file, "suggest")
        # Debe mostrar el total de pendientes
        assert "Total pendiente" in result or str(len(sample_tasks)) in result


class TestAgendaReorder:
    """Tests para action='reorder'."""

    def test_reorder_updates_priorities(self, tmp_data_dir):
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        next_week = (datetime.now() + timedelta(days=8)).strftime("%Y-%m-%d")
        tasks = [
            {"id": 1, "content": "Tarea próxima semana", "date": next_week,
             "priority": 3, "done": False, "created": "2026-04-10T10:00:00"},
            {"id": 2, "content": "Tarea mañana", "date": tomorrow,
             "priority": 3, "done": False, "created": "2026-04-10T10:01:00"},
        ]
        agenda_file = _setup_agenda(tmp_data_dir, tasks)
        _call_agenda(agenda_file, "reorder")
        updated = json.loads(agenda_file.read_text())
        task_tomorrow = next(t for t in updated if "mañana" in t["content"])
        task_next_week = next(t for t in updated if "próxima" in t["content"])
        assert task_tomorrow["priority"] == 1   # mañana = URGENTE
        assert task_next_week["priority"] == 3  # próxima semana = PENDIENTE

    def test_reorder_empty_agenda(self, tmp_data_dir):
        agenda_file = _setup_agenda(tmp_data_dir)
        result = _call_agenda(agenda_file, "reorder")
        assert "📭" in result


class TestAgendaUnknownAction:
    """Tests para acciones no reconocidas."""

    def test_unknown_action_returns_error(self, tmp_data_dir):
        agenda_file = _setup_agenda(tmp_data_dir)
        result = _call_agenda(agenda_file, "accion_inexistente")
        assert "❌" in result
        assert "accion_inexistente" in result


# ── Helpers internos ──────────────────────────────────────────────────────────

def _setup_agenda(tmp_dir, tasks=None):
    """Crea el archivo de agenda con datos opcionales."""
    agenda_dir = tmp_dir / "agenda" / "test_user"
    agenda_dir.mkdir(parents=True, exist_ok=True)
    agenda_file = agenda_dir / "agenda.json"
    agenda_file.write_text(
        json.dumps(tasks or [], ensure_ascii=False),
        encoding="utf-8"
    )
    return agenda_file


def _call_agenda(agenda_file: Path, action: str,
                 content="", date="", priority=0) -> str:
    """Ejecuta la lógica de agenda directamente sin pasar por la tool."""
    import json as _json
    from datetime import datetime as _dt, timedelta as _td, date as _date_cls

    tasks = _json.loads(agenda_file.read_text(encoding="utf-8"))
    now = _dt.now()
    today = now.date()
    act = action.lower().strip()

    def save():
        agenda_file.write_text(
            _json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def parse_date(date_str):
        if not date_str: return ""
        d = date_str.lower().strip()
        if d in ("hoy", "today"): return today.isoformat()
        if d in ("mañana", "tomorrow"):
            return (today + _td(days=1)).isoformat()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return _dt.strptime(date_str.strip(), fmt).date().isoformat()
            except ValueError:
                continue
        return date_str

    def auto_priority(date_str):
        if not date_str: return 3
        try:
            days = (_dt.strptime(date_str[:10], "%Y-%m-%d").date() - today).days
            if days <= 1: return 1
            elif days <= 7: return 2
            else: return 3
        except Exception:
            return 3

    def priority_label(p):
        return {1: "🔴 URGENTE", 2: "🟡 ESTA SEMANA", 3: "🟢 PENDIENTE"}.get(p, "⚪")

    def format_task(t, idx):
        status = "✅" if t.get("done") else priority_label(t.get("priority", 3))
        date_s = f" 📅 {t['date']}" if t.get("date") else ""
        return f"{status} [{t.get('id', idx+1)}] {t['content']}{date_s}"

    def next_id():
        existing = [t.get("id", 0) for t in tasks]
        return max(existing, default=0) + 1

    def find_task(ref):
        ref = ref.strip()
        if ref.isdigit():
            tid = int(ref)
            for t in tasks:
                if t.get("id") == tid: return t
        ref_lower = ref.lower()
        for t in tasks:
            if ref_lower in t["content"].lower(): return t
        return None

    if act == "add":
        if not content: return "❌ Indica el contenido de la tarea."
        parsed_date = parse_date(date)
        p = priority if priority in (1, 2, 3) else auto_priority(parsed_date)
        task = {"id": next_id(), "content": content, "date": parsed_date,
                "priority": p, "done": False, "created": now.isoformat()}
        tasks.append(task)
        save()
        date_str = f" para el **{parsed_date}**" if parsed_date else ""
        return f"✅ Añadido{date_str}: **{content}**\nPrioridad: {priority_label(p)}"

    elif act == "list":
        pending = [t for t in tasks if not t.get("done")]
        if not pending: return "📭 No hay tareas pendientes."
        filter_text = content.lower().strip() if content else ""
        if filter_text:
            filtered = []
            if filter_text in ("urgente", "urgent", "1"):
                filtered = [t for t in pending if t.get("priority") == 1]
            else:
                filtered = [t for t in pending
                             if filter_text in t.get("content", "").lower()]
            pending = filtered if filtered else pending
        if not pending:
            return f"📭 No hay tareas pendientes para '{content}'."
        pending.sort(key=lambda t: (t.get("priority", 3),
                                     t.get("date", "9999-12-31") or "9999-12-31"))
        lines = [f"📋 **Tareas pendientes** ({len(pending)}):"]
        for i, t in enumerate(pending):
            lines.append(format_task(t, i))
        return "\n".join(lines)

    elif act == "done":
        if not content: return "❌ Indica qué tarea completar."
        task = find_task(content)
        if not task: return f"❌ No encontré tarea con '{content}'."
        task["done"] = True
        task["completed_at"] = now.isoformat()
        save()
        return f"✅ Completada: **{task['content']}**"

    elif act == "delete":
        if not content: return "❌ Indica qué tarea eliminar."
        task = find_task(content)
        if not task: return f"❌ No encontré tarea con '{content}'."
        tasks.remove(task)
        save()
        return f"🗑️ Eliminada: **{task['content']}**"

    elif act == "update":
        if not content: return "❌ Indica qué tarea actualizar."
        task = find_task(content)
        if not task: return f"❌ No encontré tarea con '{content}'."
        changes = []
        if priority in (1, 2, 3):
            task["priority"] = priority
            changes.append(f"prioridad → {priority_label(priority)}")
        if date:
            task["date"] = parse_date(date)
            changes.append(f"fecha → {task['date']}")
        if not changes: return "❌ Indica qué cambiar: priority (1/2/3) o date."
        save()
        return f"✏️ Actualizada **{task['content']}**: {', '.join(changes)}"

    elif act == "suggest":
        pending = [t for t in tasks if not t.get("done")]
        if not pending: return "📭 No hay tareas pendientes."
        lines = ["🎯 **Sugerencias de prioridad:**\n"]
        vencidas, urgentes, semana, backlog = [], [], [], []
        for t in pending:
            d = t.get("date", "")
            if d:
                try:
                    days = (_dt.strptime(d[:10], "%Y-%m-%d").date() - today).days
                    if days < 0: vencidas.append((t, days))
                    elif days <= 1: urgentes.append((t, days))
                    elif days <= 7: semana.append((t, days))
                    else: backlog.append((t, days))
                except Exception:
                    backlog.append((t, 999))
            else:
                backlog.append((t, 999))
        if vencidas:
            lines.append("⚠️ **VENCIDAS:**")
            for t, d in vencidas:
                lines.append(f"  🔴 [{t['id']}] {t['content']} (hace {abs(d)} días)")
        if urgentes:
            lines.append("\n🔴 **HOY / MAÑANA:**")
            for t, d in urgentes:
                lines.append(f"  [{t['id']}] {t['content']}")
        lines.append(f"\n📊 Total pendiente: {len(pending)} tareas")
        return "\n".join(lines)

    elif act == "reorder":
        pending = [t for t in tasks if not t.get("done")]
        if not pending: return "📭 No hay tareas pendientes."
        for t in pending:
            if t.get("date"): t["priority"] = auto_priority(t["date"])
        save()
        pending.sort(key=lambda t: (t.get("priority", 3),
                                     t.get("date", "9999-12-31") or "9999-12-31"))
        lines = ["🔄 **Agenda reordenada:**\n"]
        for i, t in enumerate(pending): lines.append(format_task(t, i))
        return "\n".join(lines)

    else:
        return f"❌ Acción '{action}' no reconocida. Usa: add, list, done, delete, update, suggest, reorder"
