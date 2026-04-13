# ============================================================================
# backend/auth.py — Autenticación y gestión de usuarios
# JWT tokens + almacenamiento de usuarios en JSON local
# ============================================================================

import json
import logging
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_MINUTES,
    DATA_DIR, DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASS
)

logger = logging.getLogger(__name__)

# ── Configuración de seguridad ────────────────────────────────────────────────
pwd_context    = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme  = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ── Archivo de usuarios (JSON local) ─────────────────────────────────────────
USERS_FILE = DATA_DIR / "users.json"

# ── Modelos Pydantic ──────────────────────────────────────────────────────────
class User(BaseModel):
    username:    str
    email:       Optional[str] = None
    full_name:   Optional[str] = None
    disabled:    bool = False
    is_admin:    bool = False

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username:   str
    password:   str
    email:      Optional[str] = None
    full_name:  Optional[str] = None
    is_admin:   bool = False

class Token(BaseModel):
    access_token: str
    token_type:   str
    username:     str
    is_admin:     bool

class TokenData(BaseModel):
    username: Optional[str] = None

# ── Gestión del archivo de usuarios ──────────────────────────────────────────
def _load_users() -> dict:
    """Carga usuarios desde el archivo JSON."""
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def _save_users(users: dict) -> None:
    """Guarda usuarios en el archivo JSON."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def _init_users() -> None:
    """Crea el usuario admin por defecto si no existe ningún usuario."""
    users = _load_users()
    if not users:
        logger.info("Creando usuario admin por defecto...")
        hashed = pwd_context.hash(DEFAULT_ADMIN_PASS)
        users[DEFAULT_ADMIN_USER] = {
            "username":        DEFAULT_ADMIN_USER,
            "email":           "admin@local",
            "full_name":       "Administrador",
            "disabled":        False,
            "is_admin":        True,
            "hashed_password": hashed,
        }
        _save_users(users)
        logger.info(f"Usuario admin creado: {DEFAULT_ADMIN_USER}")

# Inicializar al importar el módulo
_init_users()

# ── Funciones de autenticación ────────────────────────────────────────────────
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[UserInDB]:
    users = _load_users()
    if username in users:
        return UserInDB(**users[username])
    return None

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta if expires_delta
        else timedelta(minutes=JWT_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=JWT_ALGORITHM)

# ── Dependencias FastAPI ──────────────────────────────────────────────────────
async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Usuario desactivado"
        )
    return current_user

async def get_admin_user(
    current_user: UserInDB = Depends(get_current_active_user)
) -> UserInDB:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Se requieren permisos de administrador"
        )
    return current_user

# ── CRUD de usuarios ──────────────────────────────────────────────────────────
def create_user(user_data: UserCreate) -> User:
    users = _load_users()
    if user_data.username in users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"El usuario '{user_data.username}' ya existe"
        )
    hashed = get_password_hash(user_data.password)
    users[user_data.username] = {
        "username":        user_data.username,
        "email":           user_data.email,
        "full_name":       user_data.full_name,
        "disabled":        False,
        "is_admin":        user_data.is_admin,
        "hashed_password": hashed,
    }
    _save_users(users)
    logger.info(f"Usuario creado: {user_data.username}")
    return User(**users[user_data.username])

def list_users() -> list[User]:
    users = _load_users()
    return [User(**u) for u in users.values()]

def delete_user(username: str) -> bool:
    users = _load_users()
    if username not in users:
        return False
    del users[username]
    _save_users(users)
    logger.info(f"Usuario eliminado: {username}")
    return True

def change_password(username: str, new_password: str) -> bool:
    users = _load_users()
    if username not in users:
        return False
    users[username]["hashed_password"] = get_password_hash(new_password)
    _save_users(users)
    logger.info(f"Contraseña cambiada para: {username}")
    return True
