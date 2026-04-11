# ============================================================================
# backend/tools_programmer.py — Tools con acceso real al sistema
# Solo disponibles para el agente programador (admin)
# ============================================================================

import os
import subprocess
import logging
import tempfile
from pathlib import Path

from langchain_core.tools import tool

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import BASE_DIR

logger = logging.getLogger(__name__)

PROJECT_ROOT = BASE_DIR.resolve()

def _safe_path(relative_path: str) -> Path:
    target = (PROJECT_ROOT / relative_path).resolve()
    if not str(target).startswith(str(PROJECT_ROOT)):
        raise PermissionError(
            f"Acceso denegado: '{relative_path}' está fuera del proyecto."
        )
    return target

@tool
def read_file(path: str) -> str:
    """Read a file from the project directory.
    Input: relative path from project root (e.g. 'backend/agents.py').
    Returns the full file content with line numbers."""
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"Error: File '{path}' does not exist."
        content = target.read_text(encoding="utf-8", errors="replace")
        lines   = content.split("\n")
        numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        return f"File: {path} ({len(lines)} lines)\n\n{numbered}"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error reading '{path}': {e}"

@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file in the project directory.
    Input: path (relative), content (full file text).
    Creates intermediate directories if needed. Makes .bak backup if file exists."""
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        backup_info = ""
        if target.exists():
            backup = target.with_suffix(target.suffix + ".bak")
            backup.write_bytes(target.read_bytes())
            backup_info = f" (backup: {backup.name})"
        target.write_text(content, encoding="utf-8")
        lines = len(content.split("\n"))
        return f"✅ '{path}' written ({lines} lines){backup_info}"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error writing '{path}': {e}"

@tool
def list_directory(path: str = ".") -> str:
    """List files and directories in a project path.
    Input: relative path (default '.' = project root)."""
    try:
        target = _safe_path(path)
        if not target.is_dir():
            return f"Error: '{path}' is not a directory."
        lines = [f"📁 {path}/"]
        for item in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name)):
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            if item.is_dir():
                lines.append(f"  📁 {item.name}/")
            else:
                size = item.stat().st_size
                size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
                lines.append(f"  📄 {item.name} ({size_str})")
        return "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error: {e}"

@tool
def execute_python(code: str, timeout: int = 30) -> str:
    """Execute real Python code in the project environment.
    Input: code (Python string), timeout (seconds, max 120).
    Working directory is project root. Returns stdout + stderr."""
    timeout = min(int(timeout), 120)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True,
            timeout=timeout, cwd=str(PROJECT_ROOT),
        )
        output = []
        if result.stdout: output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:  output.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0: output.append(f"Exit code: {result.returncode}")
        return "\n".join(output) if output else "✅ Executed (no output)"
    except subprocess.TimeoutExpired:
        return f"❌ Timeout after {timeout}s"
    except Exception as e:
        return f"❌ Error: {e}"
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass

@tool
def execute_bash(command: str, timeout: int = 30) -> str:
    """Execute a bash command in the project directory.
    Input: command (bash string), timeout (seconds, max 120).
    Working directory is project root. Returns stdout + stderr.
    Blocked: rm -rf /, shutdown, reboot, passwd, mkfs, dd."""
    dangerous = [
        "rm -rf /", "shutdown", "reboot", "poweroff",
        "mkfs", "fdisk", ":(){:|:&};:", "passwd",
        "useradd", "userdel", "chmod 777 /",
    ]
    for d in dangerous:
        if d in command.lower():
            return f"❌ Blocked: '{d}' is not allowed."
    timeout = min(int(timeout), 120)
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=timeout, cwd=str(PROJECT_ROOT),
        )
        output = []
        if result.stdout: output.append(result.stdout)
        if result.stderr:  output.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0: output.append(f"Exit code: {result.returncode}")
        return "\n".join(output) if output else "✅ Command executed (no output)"
    except subprocess.TimeoutExpired:
        return f"❌ Timeout after {timeout}s"
    except Exception as e:
        return f"❌ Error: {e}"

PROGRAMMER_TOOLS = [read_file, write_file, list_directory, execute_python, execute_bash]
