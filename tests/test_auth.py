# tests/test_auth.py — Tests para backend/auth.py (SQLite)
import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta, UTC
from jose import jwt, JWTError
from fastapi import HTTPException
from backend.database import get_db

try:
    import backend.auth as auth_module
    from backend.auth import (
        verify_password,
        get_password_hash,
        create_access_token,
        authenticate_user,
        get_user,
        create_user,
        list_users,
        delete_user,
        change_password,
        get_current_user,
        get_current_active_user,
        get_admin_user,
        UserCreate,
        UserInDB,
        TokenData,
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    pytest.skip("backend.auth no disponible", allow_module_level=True)

# ── Tests de hashing de contraseñas ──────────────────────────────────────────
class TestPasswordHashing:
    def test_get_password_hash_generates_bcrypt(self):
        password = "mysecret"
        hashed = get_password_hash(password)
        assert hashed != password
        assert hashed.startswith("$2")

    def test_verify_password_correct(self):
        password = "testpass"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        password = "testpass"
        wrong_password = "wrongpass"
        hashed = get_password_hash(password)
        assert verify_password(wrong_password, hashed) is False

    def test_verify_password_empty(self):
        hashed = get_password_hash("")
        assert verify_password("", hashed) is True
        assert verify_password("notempty", hashed) is False

# ── Tests de JWT ────────────────────────────────────────────────────────────
class TestJWTToken:
    def test_create_access_token_with_default_expiration(self, monkeypatch):
        monkeypatch.setattr(auth_module, "JWT_EXPIRE_MINUTES", 30)
        data = {"sub": "testuser"}
        token = create_access_token(data)
        payload = jwt.decode(token, auth_module.SECRET_KEY, algorithms=[auth_module.JWT_ALGORITHM], options={"verify_exp": False})
        assert payload["sub"] == "testuser"
        assert "exp" in payload

    def test_create_access_token_with_custom_delta(self, monkeypatch):
        data = {"sub": "testuser"}
        delta = timedelta(minutes=5)
        token = create_access_token(data, expires_delta=delta)
        payload = jwt.decode(token, auth_module.SECRET_KEY, algorithms=[auth_module.JWT_ALGORITHM], options={"verify_exp": False})
        assert payload["sub"] == "testuser"

    def test_token_expires_correctly(self, monkeypatch):
        expired_time = datetime.now(UTC) - timedelta(seconds=1)
        expired_token = jwt.encode(
            {"sub": "testuser", "exp": expired_time},
            auth_module.SECRET_KEY,
            algorithm=auth_module.JWT_ALGORITHM
        )
        with pytest.raises(JWTError):
            jwt.decode(expired_token, auth_module.SECRET_KEY, algorithms=[auth_module.JWT_ALGORITHM])

        valid_token = create_access_token({"sub": "testuser"}, expires_delta=timedelta(minutes=5))
        payload = jwt.decode(valid_token, auth_module.SECRET_KEY, algorithms=[auth_module.JWT_ALGORITHM])
        assert payload["sub"] == "testuser"

# ── Tests de gestión de usuarios (CRUD) ─────────────────────────────────────
class TestUserCRUD:
    def test_create_user_success(self, setup_sqlite_db):
        user_data = UserCreate(username="newuser", password="newpass123", email="new@example.com", full_name="New User", is_admin=False)
        result = create_user(user_data)
        assert result.username == "newuser"
        assert result.email == "new@example.com"
        assert result.is_admin is False

        with get_db() as db:
            row = db.execute("SELECT * FROM users WHERE username=?", ("newuser",)).fetchone()
            assert row is not None
            assert row["email"] == "new@example.com"
            assert row["hashed_password"] != "newpass123"

    def test_create_user_duplicate_raises_error(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("""INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password)
                          VALUES (?, ?, ?, 0, 0, ?)""",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], sample_user_dict["hashed_password"]))

        user_data = UserCreate(username="testuser", password="password", email="other@example.com")
        with pytest.raises(HTTPException) as exc_info:
            create_user(user_data)
        assert exc_info.value.status_code == 400
        assert "ya existe" in exc_info.value.detail

    def test_get_user_exists(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("""INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password)
                          VALUES (?, ?, ?, 0, 0, ?)""",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], sample_user_dict["hashed_password"]))

        user = get_user("testuser")
        assert user is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.hashed_password is not None

    def test_get_user_not_exists(self, setup_sqlite_db):
        assert get_user("nonexistent") is None

    def test_list_users(self, setup_sqlite_db, sample_user_dict, sample_admin_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 0, ?)",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], sample_user_dict["hashed_password"]))
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 1, ?)",
                       (sample_admin_dict["username"], sample_admin_dict["email"], sample_admin_dict["full_name"], sample_admin_dict["hashed_password"]))

        users = list_users()
        usernames = [u.username for u in users]
        assert "testuser" in usernames
        assert "adminuser" in usernames

    def test_list_users_empty(self, setup_sqlite_db):
        with get_db() as db:
            db.execute("DELETE FROM users WHERE username NOT IN ('testuser', 'admin', 'user1')")
        users = list_users()
        assert len(users) == 3  # Los que crea el fixture setup_sqlite_db

    def test_delete_user_success(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 0, ?)",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], sample_user_dict["hashed_password"]))

        result = delete_user("testuser")
        assert result is True

        with get_db() as db:
            row = db.execute("SELECT * FROM users WHERE username=?", ("testuser",)).fetchone()
            assert row is None

    def test_delete_user_not_exists(self, setup_sqlite_db):
        assert delete_user("nonexistent") is False

    def test_change_password_success(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 0, ?)",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], sample_user_dict["hashed_password"]))

        new_pass = "newsecret456"
        result = change_password("testuser", new_pass)
        assert result is True

        with get_db() as db:
            row = db.execute("SELECT hashed_password FROM users WHERE username=?", ("testuser",)).fetchone()
            assert verify_password(new_pass, row["hashed_password"])
            assert not verify_password("secret123", row["hashed_password"])

    def test_change_password_user_not_exists(self, setup_sqlite_db):
        assert change_password("nonexistent", "newpass") is False

# ── Tests de autenticación ─────────────────────────────────────────────────
class TestAuthentication:
    def test_authenticate_user_success(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 0, ?)",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], get_password_hash("secret123")))
        user = authenticate_user("testuser", "secret123")
        assert user is not None
        assert user.username == "testuser"

    def test_authenticate_user_wrong_password(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 0, ?)",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], get_password_hash("secret123")))
        assert authenticate_user("testuser", "wrongpassword") is None

    def test_authenticate_user_wrong_username(self, setup_sqlite_db):
        assert authenticate_user("nonexistent", "anypass") is None

    def test_authenticate_user_empty_password(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 0, ?)",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], get_password_hash("notempty")))
        assert authenticate_user("testuser", "") is None

# ── Tests de dependencias FastAPI ───────────────────────────────────────────
@pytest.mark.asyncio
class TestFastAPIDependencies:
    async def test_get_current_user_valid_token(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 0, ?)",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], sample_user_dict["hashed_password"]))
        token = create_access_token({"sub": "testuser"})
        user = await get_current_user(token)
        assert user.username == "testuser"

    async def test_get_current_user_invalid_token(self):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("invalid.token.here")
        assert exc_info.value.status_code == 401
        assert "No se pudo validar" in exc_info.value.detail

    async def test_get_current_user_missing_sub(self):
        token = jwt.encode({"exp": 1234567890}, auth_module.SECRET_KEY, algorithm=auth_module.JWT_ALGORITHM)
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token)
        assert exc_info.value.status_code == 401

    async def test_get_current_user_not_found(self, setup_sqlite_db):
        token = create_access_token({"sub": "ghostuser"})
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token)
        assert exc_info.value.status_code == 401

    async def test_get_current_active_user_active(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 0, ?)",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], sample_user_dict["hashed_password"]))
        user_in_db = get_user("testuser")
        result = await get_current_active_user(user_in_db)
        assert result.username == "testuser"

    async def test_get_current_active_user_disabled(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 1, 0, ?)",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], sample_user_dict["hashed_password"]))
        user_in_db = get_user("testuser")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(user_in_db)
        assert exc_info.value.status_code == 400
        assert "desactivado" in exc_info.value.detail.lower()

    async def test_get_admin_user_success(self, setup_sqlite_db, sample_admin_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 1, ?)",
                       (sample_admin_dict["username"], sample_admin_dict["email"], sample_admin_dict["full_name"], sample_admin_dict["hashed_password"]))
        admin = get_user("adminuser")
        result = await get_admin_user(admin)
        assert result.is_admin is True

    async def test_get_admin_user_forbidden(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("INSERT OR REPLACE INTO users (username, email, full_name, disabled, is_admin, hashed_password) VALUES (?, ?, ?, 0, 0, ?)",
                       (sample_user_dict["username"], sample_user_dict["email"], sample_user_dict["full_name"], sample_user_dict["hashed_password"]))
        user = get_user("testuser")
        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user(user)
        assert exc_info.value.status_code == 403
        assert "permisos de administrador" in exc_info.value.detail.lower()

# ── Tests de inicialización ─────────────────────────────────────────────────
class TestInitialization:
    def test_init_users_creates_default_admin(self, setup_sqlite_db, monkeypatch):
        monkeypatch.setattr(auth_module, "DEFAULT_ADMIN_USER", "admin_test")
        monkeypatch.setattr(auth_module, "DEFAULT_ADMIN_PASS", "adminpass")
        
        with get_db() as db:
            db.execute("DELETE FROM users")
            
        auth_module._init_users()
        
        with get_db() as db:
            row = db.execute("SELECT * FROM users WHERE username='admin_test'").fetchone()
        assert row is not None
        assert row["is_admin"] == 1
        assert verify_password("adminpass", row["hashed_password"])

    def test_init_users_skips_if_users_exist(self, setup_sqlite_db, sample_user_dict):
        with get_db() as db:
            db.execute("""INSERT OR REPLACE INTO users 
                (username, email, full_name, disabled, is_admin, hashed_password)
                VALUES (?, ?, ?, 0, 0, ?)""",
                (sample_user_dict["username"], sample_user_dict["email"], 
                 sample_user_dict["full_name"], sample_user_dict["hashed_password"]))
            count_before = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            
        auth_module._init_users()
        
        with get_db() as db:
            count_after = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        assert count_after == count_before

# ── Tests de edge cases y seguridad ─────────────────────────────────────────
class TestSecurityEdgeCases:
    def test_sql_injection_in_username(self, setup_sqlite_db):
        malicious_user = UserCreate(username="'; DROP TABLE users; --", password="pass", email="test@test.com")
        result = create_user(malicious_user)
        assert result.username == "'; DROP TABLE users; --"
        with get_db() as db:
            row = db.execute("SELECT * FROM users WHERE username=?", ("'; DROP TABLE users; --",)).fetchone()
            assert row is not None

    def test_very_long_password_handling(self):
        long_pass = "a" * 1000
        hashed = get_password_hash(long_pass)
        assert verify_password(long_pass, hashed)

    def test_unicode_in_password(self):
        unicode_pass = "ñáéíóú🔐日本語"
        hashed = get_password_hash(unicode_pass)
        assert verify_password(unicode_pass, hashed)

    def test_xss_attempt_in_user_data(self, setup_sqlite_db):
        xss_email = "<script>alert('xss')</script>@test.com"
        user = UserCreate(username="xssuser", password="pass", email=xss_email)
        result = create_user(user)
        assert result.email == xss_email
