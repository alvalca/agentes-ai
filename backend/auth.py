# ============================================================================
# backend/auth.py — Autenticación y gestión de usuarios
# JWT tokens + almacenamiento en SQLite
# ============================================================================

import logging
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Optional

from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_MINUTES,
    DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASS
)
from backend.database import get_db

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ── Modelos Pydantic ──────────────────────────────────────────────────────────
class User(BaseModel):
    username:  str
    email:     Optional[str] = None
    full_name: Optional[str] = None
    disabled:  bool = False
    is_admin:  bool = False

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username:  str
    password:  str
    email:     Optional[str] = None
    full_name: Optional[str] = None
    is_admin:  bool = False

class Token(BaseModel):
    access_token: str
    token_type:   str
    username:     str
    is_admin:     bool

class TokenData(BaseModel):
    username: Optional[str] = None

# ── Inicialización ────────────────────────────────────────────────────────────
def _init_users() -> None:
    """Crea el usuario admin por defecto si la tabla está vacía."""
    with get_db() as db:
        row = db.execute("SELECT COUNT(*) as n FROM users").fetchone()
        if row["n"] == 0:
            logger.info("Creando usuario admin por defecto...")
            hashed = get_password_hash(DEFAULT_ADMIN_PASS)
            db.execute("""
                INSERT INTO users (username, email, full_name,
                                   disabled, is_admin, hashed_password)
                VALUES (?, ?, ?, 0, 1, ?)
            """, (DEFAULT_ADMIN_USER, "admin@local", "Administrador", hashed))
            logger.info(f"Usuario admin creado: {DEFAULT_ADMIN_USER}")

# ── Contraseñas ───────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

# ── CRUD ──────────────────────────────────────────────────────────────────────
def get_user(username: str) -> Optional[UserInDB]:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return UserInDB(**dict(row)) if row else None

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_user(user_data: UserCreate) -> User:
    if get_user(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"El usuario '{user_data.username}' ya existe"
        )
    hashed = get_password_hash(user_data.password)
    with get_db() as db:
        db.execute("""
            INSERT INTO users (username, email, full_name,
                               disabled, is_admin, hashed_password)
            VALUES (?, ?, ?, 0, ?, ?)
        """, (
            user_data.username, user_data.email, user_data.full_name,
            int(user_data.is_admin), hashed
        ))
    logger.info(f"Usuario creado: {user_data.username}")
    return get_user(user_data.username)

def list_users() -> list[User]:
    with get_db() as db:
        rows = db.execute("SELECT * FROM users").fetchall()
    return [User(**dict(r)) for r in rows]

def delete_user(username: str) -> bool:
    with get_db() as db:
        cur = db.execute("DELETE FROM users WHERE username = ?", (username,))
    if cur.rowcount:
        logger.info(f"Usuario eliminado: {username}")
    return cur.rowcount > 0

def change_password(username: str, new_password: str) -> bool:
    hashed = get_password_hash(new_password)
    with get_db() as db:
        cur = db.execute(
            "UPDATE users SET hashed_password = ? WHERE username = ?",
            (hashed, username)
        )
    if cur.rowcount:
        logger.info(f"Contraseña cambiada: {username}")
    return cur.rowcount > 0

# ── JWT ───────────────────────────────────────────────────────────────────────
def create_access_token(data: dict,
                        expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=JWT_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=JWT_ALGORITHM)

# ── Dependencias FastAPI ──────────────────────────────────────────────────────
async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> UserInDB:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise exc
    except JWTError:
        raise exc
    user = get_user(username)
    if user is None:
        raise exc
    return user

async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Usuario desactivado")
    return current_user

async def get_admin_user(
    current_user: UserInDB = Depends(get_current_active_user)
) -> UserInDB:
    if not current_user.is_admin:
        raise HTTPException(status_code=403,
                            detail="Se requieren permisos de administrador")
    return current_user
