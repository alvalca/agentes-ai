# tests/test_database.py — Tests para backend/database.py
# Cobertura: init_db, get_db (commit/rollback), migrate_from_json (ramas y edge cases)
import pytest
import json
import sqlite3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import init_db, get_db, migrate_from_json, DB_PATH
from backend.auth import get_password_hash

# ── Fixtures de aislamiento ──────────────────────────────────────────────────
@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch):
    """Redirige DB_PATH y DATA_DIR a un directorio temporal limpio."""
    monkeypatch.setattr("backend.database.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("backend.database.DATA_DIR", tmp_path)
    return tmp_path

# ── Tests de inicialización y conexión ───────────────────────────────────────
class TestDatabaseInit:
    def test_init_db_creates_tables(self, temp_data_dir):
        init_db()
        assert (temp_data_dir / "test.db").exists()
        with get_db() as db:
            tables = [r["name"] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for expected in ["users", "conversations", "messages", "agent_configs"]:
                assert expected in tables

    def test_init_db_is_idempotent(self, temp_data_dir):
        init_db()
        init_db()  # No debe fallar ni duplicar tablas
        with get_db() as db:
            assert db.execute("SELECT count(*) FROM users").fetchone()[0] == 0

class TestGetDbContextManager:
    def test_commits_on_success(self, temp_data_dir):
        init_db()
        with get_db() as db:
            db.execute("INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
                       ("test", "t@t.com", "hash"))
        # Verificar que persistió fuera del bloque
        with get_db() as db:
            row = db.execute("SELECT username FROM users WHERE username=?", ("test",)).fetchone()
            assert row["username"] == "test"

    def test_rolls_back_on_error(self, temp_data_dir):
        init_db()
        with pytest.raises(sqlite3.IntegrityError):
            with get_db() as db:
                db.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", ("u1", "h1"))
                # Violación de PK duplicada fuerza rollback
                db.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", ("u1", "h2"))
        with get_db() as db:
            assert db.execute("SELECT count(*) FROM users").fetchone()[0] == 0

# ── Tests de migración desde JSON ────────────────────────────────────────────
class TestMigrationUsers:
    def test_dry_run_counts_users_without_writing(self, temp_data_dir):
        (temp_data_dir / "users.json").write_text(json.dumps({
            "user1": {"username": "user1", "email": "u1@x.com", "full_name": "U1", "hashed_password": "h1"},
            "user2": {"username": "user2", "email": "u2@x.com", "full_name": "U2", "hashed_password": "h2"}
        }))
        init_db()
        stats = migrate_from_json(dry_run=True)
        assert stats["users"] == 2
        with get_db() as db:
            assert db.execute("SELECT count(*) FROM users").fetchone()[0] == 0

    def test_real_migration_inserts_users(self, temp_data_dir):
        (temp_data_dir / "users.json").write_text(json.dumps({
            "admin": {"username": "admin", "email": "a@x.com", "full_name": "Admin", 
                      "hashed_password": get_password_hash("pass"), "is_admin": True}
        }))
        init_db()
        stats = migrate_from_json(dry_run=False)
        assert stats["users"] == 1
        with get_db() as db:
            row = db.execute("SELECT username, is_admin FROM users WHERE username='admin'").fetchone()
            assert row["is_admin"] == 1

    def test_handles_missing_users_json(self, temp_data_dir):
        init_db()
        stats = migrate_from_json(dry_run=False)
        assert stats["users"] == 0
        assert "users.json" not in str(stats["errors"])

    def test_handles_malformed_users_json(self, temp_data_dir):
        (temp_data_dir / "users.json").write_text("{ invalid json }}}")
        init_db()
        stats = migrate_from_json(dry_run=False)
        assert stats["users"] == 0
        assert len(stats["errors"]) == 1
        assert "users.json" in stats["errors"][0]

class TestMigrationHistory:
    def test_migrates_conversations_and_messages(self, temp_data_dir):
        # Preparar estructura chat_history/
        user_dir = temp_data_dir / "chat_history" / "user_xyz"
        user_dir.mkdir(parents=True)
        
        # Conversación con 2 mensajes
        conv_file = user_dir / "conv_abc123.json"
        conv_file.write_text(json.dumps([
            {"role": "user", "content": "Hola", "agent": "general", "timestamp": "2026-01-01T10:00:00"},
            {"role": "assistant", "content": "Hi", "agent": "general", "timestamp": "2026-01-01T10:00:01"}
        ]))
        
        # Usuario debe existir para FK
        (temp_data_dir / "users.json").write_text(json.dumps({
            "user_xyz": {"username": "user_xyz", "hashed_password": "h", "email": "u@x.com"}
        }))
        
        init_db()
        stats = migrate_from_json(dry_run=False)
        assert stats["conversations"] == 1
        assert stats["messages"] == 2
        
        with get_db() as db:
            conv = db.execute("SELECT id FROM conversations WHERE user_id='user_xyz'").fetchone()
            assert conv["id"] == "abc123"
            msgs = db.execute("SELECT count(*) as n FROM messages WHERE user_id='user_xyz'").fetchone()
            assert msgs["n"] == 2

    def test_skips_empty_conversation_files(self, temp_data_dir):
        user_dir = temp_data_dir / "chat_history" / "skip_user"
        user_dir.mkdir(parents=True)
        (user_dir / "conv_empty.json").write_text("[]")
        (temp_data_dir / "users.json").write_text(json.dumps({
            "skip_user": {"username": "skip_user", "hashed_password": "h", "email": "s@x.com"}
        }))
        init_db()
        stats = migrate_from_json(dry_run=False)
        assert stats["conversations"] == 0
        assert stats["messages"] == 0

    def test_logs_fk_constraint_errors_gracefully(self, temp_data_dir):
        user_dir = temp_data_dir / "chat_history" / "ghost_user"
        user_dir.mkdir(parents=True)
        (user_dir / "conv_no_fk.json").write_text(json.dumps([
            {"role": "user", "content": "Hola", "agent": "general", "timestamp": "2026-01-01"}
        ]))
        # NO creamos users.json → FK fallará
        init_db()
        stats = migrate_from_json(dry_run=False)
        assert stats["conversations"] == 0
        assert stats["messages"] == 0
        assert len(stats["errors"]) >= 1
        assert any("FOREIGN KEY" in str(e) for e in stats["errors"])

class TestMigrationConfigs:
    def test_migrates_agent_configs_skipping_defaults(self, temp_data_dir):
        (temp_data_dir / "agent_configs.json").write_text(json.dumps({
            "__default__": {"mode": "auto", "agents": ["general"]},
            "alice": {"mode": "manual", "agents": ["general", "pedagogico"], "default_agent": "general"}
        }))
        (temp_data_dir / "users.json").write_text(json.dumps({
            "alice": {"username": "alice", "hashed_password": "h", "email": "a@x.com"}
        }))
        init_db()
        stats = migrate_from_json(dry_run=False)
        assert stats["agent_configs"] == 1  # __default__ se ignora
        
        with get_db() as db:
            cfg = db.execute("SELECT mode, agents_json FROM agent_configs WHERE user_id='alice'").fetchone()
            assert cfg["mode"] == "manual"
            assert json.loads(cfg["agents_json"]) == ["general", "pedagogico"]

    def test_handles_missing_or_malformed_configs(self, temp_data_dir):
        init_db()
        assert migrate_from_json(dry_run=False)["agent_configs"] == 0
        
        (temp_data_dir / "agent_configs.json").write_text("not json")
        stats = migrate_from_json(dry_run=False)
        assert stats["agent_configs"] == 0
        assert len(stats["errors"]) == 1
