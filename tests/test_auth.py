# ============================================================================
# tests/test_auth.py — Tests para backend/auth.py
# Cobertura: hashing, JWT, dependencias, CRUD usuarios
# REQUIERE: conftest.py con fixtures temp_users_file, sample_user_dict, etc.
# ============================================================================

import pytest
import json
from pathlib import Path
from datetime import timedelta, UTC
from jose import jwt, JWTError
from fastapi import HTTPException

# Importar módulo bajo test
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
        _load_users,
        _save_users,
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    pytest.skip("backend.auth no disponible", allow_module_level=True)


# ── Tests de hashing de contraseñas ──────────────────────────────────────────

class TestPasswordHashing:
    def test_get_password_hash_generates_bcrypt(self):
        """Verifica que el hash se genera y es distinto del texto plano."""
        password = "mysecret"
        hashed = get_password_hash(password)
        assert hashed != password
        assert hashed.startswith("$2")  # Prefijo bcrypt

    def test_verify_password_correct(self):
        """Verifica contraseña correcta."""
        password = "testpass"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Rechaza contraseña incorrecta."""
        password = "testpass"
        wrong_password = "wrongpass"
        hashed = get_password_hash(password)
        assert verify_password(wrong_password, hashed) is False

    def test_verify_password_empty(self):
        """Maneja contraseñas vacías correctamente."""
        hashed = get_password_hash("")
        assert verify_password("", hashed) is True
        assert verify_password("notempty", hashed) is False


# ── Tests de JWT ────────────────────────────────────────────────────────────

class TestJWTToken:
    def test_create_access_token_with_default_expiration(self, monkeypatch):
        """Token se crea con expiración por defecto."""
        monkeypatch.setattr(auth_module, "JWT_EXPIRE_MINUTES", 30)
        data = {"sub": "testuser"}
        token = create_access_token(data)

        # Decodificar sin verificar exp (solo estructura)
        payload = jwt.decode(token, auth_module.SECRET_KEY, algorithms=[auth_module.JWT_ALGORITHM], options={"verify_exp": False})
        assert payload["sub"] == "testuser"
        assert "exp" in payload

    def test_create_access_token_with_custom_delta(self, monkeypatch):
        """Token se crea con expiración personalizada."""
        data = {"sub": "testuser"}
        delta = timedelta(minutes=5)
        token = create_access_token(data, expires_delta=delta)

        payload = jwt.decode(token, auth_module.SECRET_KEY, algorithms=[auth_module.JWT_ALGORITHM], options={"verify_exp": False})
        assert payload["sub"] == "testuser"

    def test_token_expires_correctly(self, monkeypatch):
        """Token expira correctamente después del tiempo definido."""
        from datetime import datetime, timedelta

        # Crear token que YA expiró (1 segundo en el pasado)
        # Esto es más confiable que time.sleep() que puede ser inestable
        expired_time = datetime.now(UTC) - timedelta(seconds=1)
        expired_token = jwt.encode(
            {"sub": "testuser", "exp": expired_time},
            auth_module.SECRET_KEY,
            algorithm=auth_module.JWT_ALGORITHM
        )

        # Debería fallar inmediatamente al decodificar
        with pytest.raises(JWTError):
            jwt.decode(expired_token, auth_module.SECRET_KEY, algorithms=[auth_module.JWT_ALGORITHM])

        # Verificar que un token válido (sin expirar) funciona
        valid_token = create_access_token({"sub": "testuser"}, expires_delta=timedelta(minutes=5))
        payload = jwt.decode(valid_token, auth_module.SECRET_KEY, algorithms=[auth_module.JWT_ALGORITHM])
        assert payload["sub"] == "testuser"


# ── Tests de gestión de usuarios (CRUD) ─────────────────────────────────────

class TestUserCRUD:
    def test_create_user_success(self, temp_users_file):
        """Crear usuario nuevo funciona."""
        user_data = UserCreate(
            username="newuser",
            password="newpass123",
            email="new@example.com",
            full_name="New User",
            is_admin=False
        )

        result = create_user(user_data)

        assert result.username == "newuser"
        assert result.email == "new@example.com"
        assert result.is_admin is False

        # Verificar que se guardó en archivo
        users = _load_users()
        assert "newuser" in users
        assert users["newuser"]["email"] == "new@example.com"
        # Verificar que la contraseña está hasheada
        assert users["newuser"]["hashed_password"] != "newpass123"

    def test_create_user_duplicate_raises_error(self, temp_users_file, sample_user_dict):
        """No permite crear usuario duplicado."""
        # Crear usuario primero
        _save_users({sample_user_dict["username"]: sample_user_dict})

        user_data = UserCreate(
            username="testuser",  # Ya existe
            password="password",
            email="other@example.com"
        )

        with pytest.raises(HTTPException) as exc_info:
            create_user(user_data)

        assert exc_info.value.status_code == 400
        assert "ya existe" in exc_info.value.detail

    def test_get_user_exists(self, temp_users_file, sample_user_dict):
        """Obtener usuario existente."""
        _save_users({sample_user_dict["username"]: sample_user_dict})

        user = get_user("testuser")
        assert user is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.hashed_password is not None

    def test_get_user_not_exists(self, temp_users_file):
        """Obtener usuario inexistente retorna None."""
        user = get_user("nonexistent")
        assert user is None

    def test_list_users(self, temp_users_file, sample_user_dict, sample_admin_dict):
        """Listar usuarios retorna todos."""
        _save_users({
            sample_user_dict["username"]: sample_user_dict,
            sample_admin_dict["username"]: sample_admin_dict
        })

        users = list_users()
        usernames = [u.username for u in users]
        assert "testuser" in usernames
        assert "adminuser" in usernames

    def test_list_users_empty(self, temp_users_file):
        """Listar usuarios vacío retorna lista vacía."""
        users = list_users()
        assert users == []

    def test_delete_user_success(self, temp_users_file, sample_user_dict):
        """Eliminar usuario existente."""
        _save_users({sample_user_dict["username"]: sample_user_dict})

        result = delete_user("testuser")
        assert result is True

        users = _load_users()
        assert "testuser" not in users

    def test_delete_user_not_exists(self, temp_users_file):
        """Eliminar usuario inexistente retorna False."""
        result = delete_user("nonexistent")
        assert result is False

    def test_change_password_success(self, temp_users_file, sample_user_dict):
        """Cambiar contraseña funciona."""
        _save_users({sample_user_dict["username"]: sample_user_dict})

        new_pass = "newsecret456"
        result = change_password("testuser", new_pass)
        assert result is True

        # Verificar que el hash cambió
        users = _load_users()
        new_hash = users["testuser"]["hashed_password"]
        assert verify_password(new_pass, new_hash)
        assert not verify_password("secret123", new_hash)

    def test_change_password_user_not_exists(self, temp_users_file):
        """Cambiar contraseña de usuario inexistente retorna False."""
        result = change_password("nonexistent", "newpass")
        assert result is False


# ── Tests de autenticación ─────────────────────────────────────────────────

class TestAuthentication:
    def test_authenticate_user_success(self, temp_users_file, sample_user_dict):
        """Autenticación con credenciales correctas."""
        # Crear usuario con contraseña conocida
        sample_user_dict["hashed_password"] = get_password_hash("secret123")
        _save_users({sample_user_dict["username"]: sample_user_dict})

        user = authenticate_user("testuser", "secret123")
        assert user is not None
        assert user.username == "testuser"

    def test_authenticate_user_wrong_password(self, temp_users_file, sample_user_dict):
        """Autenticación rechaza contraseña incorrecta."""
        sample_user_dict["hashed_password"] = get_password_hash("secret123")
        _save_users({sample_user_dict["username"]: sample_user_dict})

        user = authenticate_user("testuser", "wrongpassword")
        assert user is None

    def test_authenticate_user_wrong_username(self, temp_users_file):
        """Autenticación rechaza usuario inexistente."""
        user = authenticate_user("nonexistent", "anypass")
        assert user is None

    def test_authenticate_user_empty_password(self, temp_users_file, sample_user_dict):
        """Autenticación rechaza contraseña vacía si no coincide."""
        sample_user_dict["hashed_password"] = get_password_hash("notempty")
        _save_users({sample_user_dict["username"]: sample_user_dict})

        user = authenticate_user("testuser", "")
        assert user is None


# ── Tests de dependencias FastAPI ───────────────────────────────────────────

@pytest.mark.asyncio
class TestFastAPIDependencies:
    async def test_get_current_user_valid_token(self, temp_users_file, sample_user_dict):
        """Dependencia retorna usuario con token válido."""
        _save_users({sample_user_dict["username"]: sample_user_dict})

        # Crear token válido
        token = create_access_token({"sub": "testuser"})

        user = await get_current_user(token)
        assert user.username == "testuser"

    async def test_get_current_user_invalid_token(self):
        """Token inválido lanza 401."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("invalid.token.here")

        assert exc_info.value.status_code == 401
        assert "No se pudo validar" in exc_info.value.detail

    async def test_get_current_user_missing_sub(self):
        """Token sin 'sub' lanza 401."""
        token = jwt.encode({"exp": 1234567890}, auth_module.SECRET_KEY, algorithm=auth_module.JWT_ALGORITHM)

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token)

        assert exc_info.value.status_code == 401

    async def test_get_current_user_not_found(self, temp_users_file):
        """Token válido pero usuario no existe lanza 401."""
        token = create_access_token({"sub": "ghostuser"})

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token)

        assert exc_info.value.status_code == 401

    async def test_get_current_active_user_active(self, temp_users_file, sample_user_dict):
        """Usuario activo pasa el check."""
        _save_users({sample_user_dict["username"]: sample_user_dict})
        user_in_db = get_user("testuser")

        result = await get_current_active_user(user_in_db)
        assert result.username == "testuser"

    async def test_get_current_active_user_disabled(self, temp_users_file, sample_user_dict):
        """Usuario desactivado lanza 400."""
        sample_user_dict["disabled"] = True
        _save_users({sample_user_dict["username"]: sample_user_dict})

        user_in_db = get_user("testuser")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(user_in_db)

        assert exc_info.value.status_code == 400
        assert "desactivado" in exc_info.value.detail

    async def test_get_admin_user_success(self, temp_users_file, sample_admin_dict):
        """Usuario admin pasa el check."""
        _save_users({sample_admin_dict["username"]: sample_admin_dict})
        admin = get_user("adminuser")

        result = await get_admin_user(admin)
        assert result.is_admin is True

    async def test_get_admin_user_forbidden(self, temp_users_file, sample_user_dict):
        """Usuario no admin es rechazado."""
        _save_users({sample_user_dict["username"]: sample_user_dict})
        user = get_user("testuser")

        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user(user)

        assert exc_info.value.status_code == 403
        assert "permisos de administrador" in exc_info.value.detail


# ── Tests de inicialización ─────────────────────────────────────────────────

class TestInitialization:
    def test_init_users_creates_default_admin(self, temp_users_file, monkeypatch):
        """Si no hay usuarios, crea admin por defecto."""
        monkeypatch.setattr(auth_module, "DEFAULT_ADMIN_USER", "admin")
        monkeypatch.setattr(auth_module, "DEFAULT_ADMIN_PASS", "adminpass")

        # Simular importación limpia llamando a _init_users directamente
        auth_module._init_users()

        users = _load_users()
        assert "admin" in users
        assert users["admin"]["is_admin"] is True
        assert verify_password("adminpass", users["admin"]["hashed_password"])

    def test_init_users_skips_if_users_exist(self, temp_users_file, sample_user_dict):
        """Si ya hay usuarios, no crea admin por defecto."""
        _save_users({sample_user_dict["username"]: sample_user_dict})

        auth_module._init_users()

        users = _load_users()
        # No debería haber creado admin por defecto porque ya existe testuser
        assert len(users) == 1


# ── Tests de edge cases y seguridad ─────────────────────────────────────────

class TestSecurityEdgeCases:
    def test_sql_injection_in_username(self, temp_users_file):
        """Intento de SQL injection en username es tratado como string literal."""
        malicious_user = UserCreate(
            username="'; DROP TABLE users; --",
            password="pass",
            email="test@test.com"
        )

        result = create_user(malicious_user)
        assert result.username == "'; DROP TABLE users; --"

        # Verificar que existe como usuario normal
        users = _load_users()
        assert "'; DROP TABLE users; --" in users

    def test_very_long_password_handling(self):
        """Contraseñas muy largas son manejadas correctamente."""
        long_pass = "a" * 1000
        hashed = get_password_hash(long_pass)
        assert verify_password(long_pass, hashed)

    def test_unicode_in_password(self):
        """Unicode en contraseñas funciona."""
        unicode_pass = "ñáéíóú🔐日本語"
        hashed = get_password_hash(unicode_pass)
        assert verify_password(unicode_pass, hashed)

    def test_xss_attempt_in_user_data(self, temp_users_file):
        """XSS en campos de usuario es almacenado (el escape debe hacerse en frontend)."""
        xss_email = "<script>alert('xss')</script>@test.com"
        user = UserCreate(
            username="xssuser",
            password="pass",
            email=xss_email
        )

        result = create_user(user)
        # El sistema almacena el email tal cual (la sanitización es responsabilidad del cliente)
        assert result.email == xss_email
