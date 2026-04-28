# ============================================================================
# backend/tools.py — Herramientas disponibles para los agentes
# web_search, calculator, code_interpreter, summarizer
# ============================================================================

import math
import logging

# Variable global para el user_id actual — seteada por build_agent antes de cada llamada
_current_user_id: str = "default"

def set_current_user(user_id: str) -> None:
    """Setea el user_id para que code_interpreter sepa dónde guardar plots."""
    global _current_user_id
    _current_user_id = user_id
import requests
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from ddgs import DDGS

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DUCKDUCKGO_MAX_RESULTS,
    CODE_INTERPRETER_TIMEOUT,
    BRAVE_API_KEY,
    BRAVE_MAX_RESULTS,
    DOCUMENTS_DIR,
)

logger = logging.getLogger(__name__)

# ── Brave Search helper ──────────────────────────────────────────────────────
def _brave_search(query: str, search_type: str = "web", count: int = None) -> list:
    """Llama a la API de Brave Search. search_type: 'web' o 'news'."""
    if not BRAVE_API_KEY:
        logger.info("Brave API no configurada — usando DDGS fallback")
        return []
    logger.info(f"Brave API activa — buscando [{search_type}]: {query}")
    count = count or BRAVE_MAX_RESULTS
    url = "https://api.search.brave.com/res/v1/news/search" if search_type == "news"           else "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": query, "count": count, "search_lang": "en"}
    if search_type == "news":
        params["freshness"] = "pw"  # última semana
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if search_type == "news":
            return data.get("results", [])
        return data.get("web", {}).get("results", [])
    except Exception as e:
        logger.error(f"Brave API error: {e}")
        return []

def _ddgs_fallback_web(query: str) -> list:
    """Fallback a DDGS si Brave no está configurado."""
    try:
        return list(DDGS().text(query, max_results=DUCKDUCKGO_MAX_RESULTS))
    except Exception:
        return []

def _ddgs_fallback_news(query: str) -> list:
    """Fallback a DDGS news si Brave no está configurado."""
    try:
        return list(DDGS().news(query, max_results=BRAVE_MAX_RESULTS))
    except Exception:
        return []

# ── Tool 1: Web Search ────────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """
    Search the internet for REAL and CURRENT information from the web.
    Use this for ANY question about events, news, or data from 2025-2026.
    Results are real and up-to-date — trust them even if they describe
    events you have no internal knowledge of.
    Input: a specific search query string (keep it concise, 3-8 words).
    Returns: real search results with titles, URLs and content snippets.
    """
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    try:
        # Usar Brave si está configurado, sino fallback a DDGS
        raw = _brave_search(query, "web") if BRAVE_API_KEY else _ddgs_fallback_web(query)
        if raw:
            parts = []
            for r in raw:
                if isinstance(r, dict):
                    # Brave devuelve 'url' y 'description'; DDGS devuelve 'href' y 'body'
                    title = r.get("title", "No title")
                    url   = r.get("url", r.get("href", ""))
                    body  = r.get("description", r.get("body", ""))[:400]
                    if url and body:
                        parts.append(
                            f"RESULT:\n"
                            f"  Title: {title}\n"
                            f"  URL:   {url}\n"
                            f"  Content: {body}"
                        )
            if parts:
                header = (
                    f"[Web search results for: '{query}']\n"
                    f"[Search performed on: {current_date}]\n"
                    f"[These results are REAL and current — use them confidently]\n"
                    f"[IMPORTANT: Only use URLs listed above. Never invent URLs.]\n"
                    f"{'='*60}\n\n"
                )
                return header + "\n\n".join(parts)
        return (
            f"No results found for '{query}'. "
            "Switch to search_news tool instead. "
            "Do NOT invent or hallucinate any URLs or article links. "
            "Do not keep searching with minor variations."
        )
    except Exception as e:
        logger.error(f"web_search error: {e}")
        return f"Search error: {e}. Try search_news instead."

# ── Tool 1b: News Search ─────────────────────────────────────────────────────
@tool
def search_news(query: str) -> str:
    """
    Search specifically for RECENT NEWS articles (last days/weeks/months).
    Use this when the user asks about today's news, this week's events,
    breaking news, or anything very recent.
    Better than web_search for current events and news articles.
    Input: news search query (e.g. 'AI news today', 'Spain economy 2026').
    Returns: recent news articles with dates, titles and URLs.
    """
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    try:
        # Usar Brave News si está configurado, sino fallback a DDGS news
        raw = _brave_search(query, "news") if BRAVE_API_KEY else _ddgs_fallback_news(query)
        if raw:
            parts = []
            for r in raw:
                if isinstance(r, dict):
                    title  = r.get("title", "No title")
                    # Brave: 'url'; DDGS: 'url' o 'href'
                    url    = r.get("url", r.get("href", ""))
                    body   = r.get("description", r.get("body", r.get("excerpt", "")))[:400]
                    # Brave: 'age' o 'published'; DDGS: 'date'
                    date   = r.get("age", r.get("published", r.get("date", "recent")))
                    # Brave: dentro de 'meta_url.netloc' o 'source'; DDGS: 'source'
                    source = r.get("source", r.get("meta_url", {}).get("netloc", ""))
                    if title:
                        url_line = f"  URL:    {url}" if url else "  URL:    Not available"
                        parts.append(
                            f"NEWS ARTICLE:\n"
                            f"  Title:  {title}\n"
                            f"  Date:   {date}\n"
                            f"  Source: {source}\n"
                            f"{url_line}\n"
                            f"  Summary: {body}"
                        )
            if parts:
                header = (
                    f"[News search results for: '{query}']\n"
                    f"[Searched on: {current_date}]\n"
                    f"[These are REAL recent news articles]\n"
                    f"{'='*60}\n\n"
                )
                return header + "\n\n".join(parts)
        # Fallback a web_search si news no da resultados
        return web_search.invoke(query)
    except Exception as e:
        logger.error(f"search_news error: {e}")
        try:
            return web_search.invoke(query)
        except Exception:
            return f"News search error: {e}"

# ── Tool 2: Calculator ────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a safe mathematical expression with precision.
    Supports: +, -, *, /, **, sqrt(), log(), log10(), sin(), cos(),
              tan(), ceil(), floor(), round(), abs(), pow(), pi, e
    Input MUST be a valid Python math expression using only numbers and operators.
    Examples:
      - 'round(28e12 / 335e6 / 12, 2)'
      - 'sqrt(144) + 1000 * 2'
      - '2024 - 1856'
      - '[x**2 for x in range(5)]'
    Do NOT include text, variables or units — only the numeric expression.
    """
    allowed_builtins = {
        "abs": abs, "round": round, "pow": pow,
        "min": min, "max": max, "sum": sum,
        "len": len, "range": range, "list": list,
        "int": int, "float": float, "str": str,
        "True": True, "False": False, "None": None,
        "Exception": Exception, "ValueError": ValueError,
        "TypeError": TypeError, "KeyError": KeyError,
        "IndexError": IndexError, "RuntimeError": RuntimeError,
    }
    allowed_math = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith("_")
    }
    safe_globals = {"__builtins__": {}, **allowed_builtins, **allowed_math}
    try:
        result = eval(expression, safe_globals)
        return str(result)
    except Exception as e:
        return (
            f"Error evaluating '{expression}': {e}. "
            "Provide a valid Python math expression with only numbers and operators."
        )

# ── Tool 3: Code Interpreter (Python seguro) ──────────────────────────────────
@tool
def code_interpreter(code: str) -> str:
    """
    Execute safe Python code for data analysis and manipulation.
    Supports: arithmetic, math functions, list/dict operations,
              string manipulation, pandas-style data processing on literals,
              sorting, filtering, statistics (mean, median, stdev).
    Input: valid Python code as a string.
    Examples:
      - 'sorted([3,1,4,1,5,9,2,6])'
      - 'sum([x**2 for x in range(10)])'
      - 'max([1500*16*5, 1200*8*3])'
    Also supports: numpy (np), scipy, sympy (sp), pandas (pd), matplotlib (plt).
    For tables: return DataFrame directly (HTML) or print(df) for console format.
    For charts: use plt.figure(), plt.plot(), plt.show().
    For symbolic math: symbols, diff, integrate, solve, simplify.
    To save CSV/Excel use SAVE_DIR (pre-configured, no import needed):
      df.to_excel(SAVE_DIR + "/file.xlsx", index=False)
      df.to_csv(SAVE_DIR + "/file.csv", index=False)
    IMPORTANT: os, sys, pathlib are NOT available. Use SAVE_DIR directly.
    """
    import statistics

    import numpy as _np
    import scipy as _scipy
    import sympy as _sympy
    import pandas as _pd
    import matplotlib as _mpl
    import matplotlib.pyplot as _plt
    _mpl.use("Agg")  # backend no-interactivo, no necesita pantalla

    allowed_builtins = {
        "abs": abs, "round": round, "pow": pow,
        "min": min, "max": max, "sum": sum,
        "len": len, "range": range,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "int": int, "float": float, "str": str, "bool": bool,
        "sorted": sorted, "enumerate": enumerate, "zip": zip,
        "map": map, "filter": filter,
        "print": print,
        "True": True, "False": False, "None": None,
        "Exception": Exception, "ValueError": ValueError,
        "TypeError": TypeError, "KeyError": KeyError,
        "IndexError": IndexError, "RuntimeError": RuntimeError,
    }
    allowed_math = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith("_")
    }
    allowed_stats = {
        "mean":   statistics.mean,
        "median": statistics.median,
        "stdev":  statistics.stdev,
        "mode":   statistics.mode,
    }
    # Módulos permitidos para import explícito dentro del código
    _ALLOWED_MODULES = {
        "numpy", "np", "scipy", "scipy.optimize", "scipy.stats",
        "scipy.linalg", "sympy", "pandas", "pd", "math", "statistics",
        "itertools", "functools", "collections",
        "matplotlib", "matplotlib.pyplot", "matplotlib.figure",
        "matplotlib.patches", "matplotlib.colors",
    }

    def _safe_import(name, *args, **kwargs):
        base = name.split(".")[0]
        if base not in _ALLOWED_MODULES and name not in _ALLOWED_MODULES:
            raise ImportError(f"Import de '{name}' no permitido en este entorno")
        return __builtins__["__import__"](name, *args, **kwargs) if isinstance(__builtins__, dict) else __import__(name, *args, **kwargs)

    safe_globals = {
        "__builtins__": {"__import__": _safe_import, "print": print},
        **allowed_builtins,
        **allowed_math,
        **allowed_stats,
        # ── Librerías científicas pre-importadas ───────────────────────
        "numpy":  _np,   "np":      _np,
        "scipy":  _scipy,
        "sympy":  _sympy, "sp":     _sympy,
        "pandas": _pd,   "pd":      _pd,
        # Accesos directos numpy
        "array":    _np.array,
        "linspace": _np.linspace,
        "arange":   _np.arange,
        "zeros":    _np.zeros,
        "ones":     _np.ones,
        "dot":      _np.dot,
        "linalg":   _np.linalg,
        # Accesos directos sympy
        "symbols":   _sympy.symbols,
        "diff":      _sympy.diff,
        "integrate": _sympy.integrate,
        "solve":     _sympy.solve,
        "simplify":  _sympy.simplify,
        "Matrix":    _sympy.Matrix,
        "latex":     _sympy.latex,
        # Accesos directos pandas
        "DataFrame": _pd.DataFrame,
        "Series":    _pd.Series,
        # Matplotlib
        "matplotlib": _mpl,
        "plt":        _plt,
    }

    # Capturar output con timeout compatible con uvicorn (ThreadPoolExecutor)
    import io
    from pathlib import Path as _Path
    from contextlib import redirect_stdout
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    # Necesitamos el user_id para guardar plots — lo capturamos del closure
    # tools.py no tiene user_id directamente, se pasa via safe_globals
    from backend.tools import _current_user_id
    _user_id = _current_user_id
    _user_dir = _Path(__file__).parent.parent / "data" / "documents" / _user_id
    _user_dir.mkdir(parents=True, exist_ok=True)

    # Interceptar to_csv y to_excel para redirigir a _user_dir con nombre único
    # El modelo llama df.to_csv('nombre.csv') normalmente — nosotros
    # interceptamos silenciosamente y guardamos en la carpeta correcta
    _saved_files = []  # lista de archivos guardados durante la ejecución

    _orig_to_csv   = _pd.DataFrame.to_csv
    _orig_to_excel = _pd.DataFrame.to_excel

    def _intercept_to_csv(self, path_or_buf=None, *args, **kwargs):
        import uuid as _uuid2
        if isinstance(path_or_buf, str):
            # Interceptar tanto rutas relativas como absolutas
            stem = _Path(path_or_buf).stem
            dest = _user_dir / (stem + "_" + _uuid2.uuid4().hex[:8] + ".csv")
            kwargs.pop("path_or_buf", None)
            _orig_to_csv(self, str(dest), *args, **kwargs)
            _saved_files.append(str(dest))
            return None
        return _orig_to_csv(self, path_or_buf, *args, **kwargs)

    def _intercept_to_excel(self, excel_writer=None, *args, **kwargs):
        import uuid as _uuid2
        if isinstance(excel_writer, str):
            # Interceptar tanto rutas relativas como absolutas
            stem = _Path(excel_writer).stem
            dest = _user_dir / (stem + "_" + _uuid2.uuid4().hex[:8] + ".xlsx")
            _orig_to_excel(self, str(dest), *args, **kwargs)
            _saved_files.append(str(dest))
            return None
        return _orig_to_excel(self, excel_writer, *args, **kwargs)

    _pd.DataFrame.to_csv   = _intercept_to_csv
    _pd.DataFrame.to_excel = _intercept_to_excel

    def _run_code():
        import uuid as _uuid
        buf = io.StringIO()
        _plt.close("all")  # limpiar figuras previas

        try:
            with redirect_stdout(buf):
                result = eval(code, safe_globals)
            printed = buf.getvalue()
        except SyntaxError:
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                exec(code, safe_globals)
            printed = buf2.getvalue()
            result = None

        # Capturar figura si matplotlib generó algo
        plot_output = ""
        if _plt.get_fignums():
            plot_path = _user_dir / f"plot_{_uuid.uuid4().hex[:8]}.png"
            _plt.savefig(str(plot_path), bbox_inches="tight", dpi=100)
            _plt.close("all")
            plot_output = "\nPLOT:" + str(plot_path)

        # Capturar tabla HTML si el resultado es un DataFrame
        table_output = ""
        if isinstance(result, _pd.DataFrame):
            html = result.to_html(classes="dataframe", border=0, index=True)
            html_inline = html.replace("\n", "")
            table_output = "\nTABLE:" + html_inline
            result = None

        # Archivos CSV/Excel interceptados durante la ejecución
        file_output = ""
        for fpath in _saved_files:
            file_output += "\nFILE:" + fpath

        text_output = printed.strip() if printed else (str(result) if result is not None else "Code executed successfully.")
        return text_output + plot_output + table_output + file_output

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_code)
            result_val = future.result(timeout=15)
        return result_val
    except FuturesTimeout:
        return "Timeout: code execution exceeded 15 seconds. Simplify or split into smaller parts."
    except SyntaxError as e:
        return f"SyntaxError: {e}. Use valid Python syntax: ** for powers (x**2), no LaTeX (**(y-3)2 is invalid, use (y-3)**2)."
    except MemoryError:
        return "Error: memoria insuficiente — simplifica el cálculo"
    except Exception as e:
        return f"Error: {e}"
    finally:
        # Restaurar métodos originales de pandas siempre
        _pd.DataFrame.to_csv   = _orig_to_csv
        _pd.DataFrame.to_excel = _orig_to_excel



# ── Tool 4: Web Content Extractor + Summarizer ────────────────────────────────
# Dominios conocidos que bloquean scrapers — no perder tiempo con ellos
_BLOCKED_DOMAINS = {
    # Muros de pago o bloqueo activo a scrapers
    "elpais.com"
}

# Fuentes que sí permiten scraping y dan buen contenido
_GOOD_SOURCES = {
    "apnews.com", "axios.com", "politico.com",
    "thehill.com", "npr.org", "abc.net.au",
    "dw.com", "france24.com", "rt.com",
    "euronews.com", "spaceweather.com",
}

@tool
def fetch_url(url: str) -> str:
    """
    Fetch and extract text content from a web URL.
    Useful for reading articles, documentation or web pages.
    Input: a valid URL string starting with http:// or https://
    Returns the first 3000 characters of the page content.
    NOTE: Some sites (Reuters, Bloomberg, BBC, NYT, WSJ) block automated
    access. If fetch fails, use the web_search snippets instead.
    """
    # Detectar dominios bloqueados antes de intentar
    from urllib.parse import urlparse
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if any(blocked in domain for blocked in _BLOCKED_DOMAINS):
            return (
                f"BLOCKED: {domain} does not allow automated access. "
                "DO NOT try other URLs. "
                "Compose your final answer NOW using the search snippets already obtained."
            )
    except Exception:
        pass

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                text = response.text
        else:
            text = response.text

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        clean_text = "\n".join(lines)
        return clean_text[:3000] + ("..." if len(clean_text) > 3000 else "")

    except Exception as e:
        logger.error(f"fetch_url error for {url}: {e}")
        return (
            f"FETCH FAILED for {url}. "
            "STOP trying to fetch URLs and compose your answer now "
            "using only the search result snippets already obtained. "
            "Do not call fetch_url again."
        )


# ── Tool: Generador de documentos ────────────────────────────────────────────
@tool
def generate_document(
    content: str,
    format: str = "pdf",
    title: str = "",
    font: str = "Helvetica",
    font_size: int = 12,
    line_spacing: float = 1.5,
    user_id: str = "default",
) -> str:
    """Generate and save a document from text content.

    Use this tool when the user asks to create, generate, save or download
    a document, file, report, letter, worksheet or any written content.
    
    CRITICAL: Math expressions MUST use LaTeX format:
    - Fractions: Use $\\frac{numerador}{denominador}$ NEVER "1/2" or "1/2"
    - Variables: Use $x$, $y$, $z$ with dollar signs
    - Operations: $+$, $-$, $\\times$, $\\div$, $=$

    Args:
        content:      Full text content of the document. Use \\n for line breaks.                      
                      For DOCX or PDF math expressions ALWAYS use LaTeX syntax:                        
                        Inline: $\\frac{1}{2}$  or  $x^2 + y^2$
                        Block:  $$\\int_0^1 x^2 dx$$
                      Math expressions are automatically rendered as native equations.
                      For DOCX tables use markdown: | Col1 | Col2 |\\n|---|---|\\n| val | val |
                      Tables are automatically converted to real Word tables.
        format:       Output format: "pdf", "docx", "markdown", "latex", "csv" or "excel"
        title:        Optional document title shown at the top
        font:         Font name. For dyslexia use "Helvetica". Common: "Times-Roman", "Courier"
        font_size:    Font size in points (default 12, use 14-16 for accessibility)
        line_spacing: Line spacing multiplier (1.0=single, 1.5=default, 2.0=double)

    Returns:
        Download path or error message
    """
    import uuid
    from pathlib import Path

    fmt      = format.lower().strip()
    out_dir  = DOCUMENTS_DIR / user_id
    out_dir.mkdir(parents=True, exist_ok=True)
    file_id  = str(uuid.uuid4())[:8]
    safe_title = title[:40].replace(" ", "_") if title else "documento"

    try:
        # ── PDF ──────────────────────────────────────────────────────────────
        if fmt == "pdf":
            import re as _re
            import subprocess as _sp
            import tempfile as _tmp

            out_path = out_dir / f"{safe_title}_{file_id}.pdf"

            # Detectar contenido markdown "rico"
            has_rich_md = bool(_re.search(r'(^#|\*|- |\|)', content, _re.MULTILINE))

            if has_rich_md:
                # ── Pandoc + Tectonic: Markdown → PDF ────────────────────────

                # Evitar título duplicado
                clean_content_start = content.strip().lstrip('#').strip()
                if title and not clean_content_start.startswith(title):
                    md_content = f"# {title}\n\n{content}"
                else:
                    md_content = content

                with _tmp.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, encoding="utf-8"
                ) as tmp_md:
                    tmp_md.write(md_content)
                    tmp_md_path = tmp_md.name

                try:
                    # ── Pandoc → PDF usando tectonic ─────────────────────────
                    result = _sp.run(
                        [
                            "pandoc",
                            tmp_md_path,
                            "-o", str(out_path),
                            "--from", "markdown+pipe_tables",
                            "--pdf-engine=tectonic",
                            "-V", "documentclass=article",
                            "-V", "geometry:margin=1.5cm",
                            "-V", "linestretch=1.2",
                            "-V", f"fontsize={font_size}pt",
                            "-V", "tables=true",                                                               
                            "-V", r"""header-includes=\usepackage[sfdefault]{inter}
                            \usepackage{amsmath,amssymb,array,longtable,xcolor,colortbl,booktabs,enumitem}
                            \renewcommand{\arraystretch}{1.3}
                            \setlist{itemsep=1.2em, parsep=0.4em, topsep=0.3em}
                            """,
                        ],
                        capture_output=True, text=True, timeout=60
                    )

                    if result.returncode != 0:
                        raise RuntimeError(f"Pandoc error: {result.stderr}")

                finally:
                    import os as _os
                    if _os.path.exists(tmp_md_path):
                        _os.unlink(tmp_md_path)

            else:
                # ── ReportLab fallback: texto simple ─────────────────────────
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.styles import ParagraphStyle
                from reportlab.lib.units import cm
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY

                doc = SimpleDocTemplate(
                    str(out_path), pagesize=A4,
                    leftMargin=1.5*cm, rightMargin=1.5*cm,
                    topMargin=1.5*cm, bottomMargin=1.5*cm,
                )

                style = ParagraphStyle(
                    "custom",
                    fontName=font,
                    fontSize=font_size,
                    leading=font_size * line_spacing * 1.2,
                    alignment=TA_JUSTIFY,
                    spaceAfter=6,
                    justifyBreaks=1,
                )

                title_style = ParagraphStyle(
                    "title",
                    fontName=font + "-Bold" if font == "Helvetica" else font,
                    fontSize=font_size + 4,
                    leading=(font_size + 4) * 1.4,
                    alignment=TA_LEFT,
                    spaceAfter=12,
                )

                story = []

                if title:
                    story.append(Paragraph(title, title_style))
                    story.append(Spacer(1, 0.3*cm))

                for para in content.split("\n"):
                    para = para.strip()
                    if para:
                        story.append(Paragraph(para, style))
                    else:
                        story.append(Spacer(1, 0.2*cm))

                doc.build(story)

        # ── DOCX ─────────────────────────────────────────────────────────────
        elif fmt == "docx":
            import re as _re
            import subprocess as _sp
            import tempfile as _tmp

            out_path = out_dir / f"{safe_title}_{file_id}.docx"

            # Detectar si hay expresiones matemáticas LaTeX ($...$ o $$...$$)
            has_math = bool(_re.search(r'\$\$?.+?\$\$?', content, _re.DOTALL))

            if has_math:
                # ── Pandoc path: LaTeX math → DOCX nativo ────────────────
                # Evitar título duplicado
                clean_content_start = content.strip().lstrip('#').strip()
                if title and not clean_content_start.startswith(title):
                    md_content = f"# {title}\n\n{content}"
                else:
                    md_content = content

                # Escribir markdown temporal
                with _tmp.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False,
                    encoding="utf-8"
                ) as tmp_md:
                    tmp_md.write(md_content)
                    tmp_md_path = tmp_md.name
                    
                # Archivo temporal para DOCX de Pandoc (no el final)
                with _tmp.NamedTemporaryFile(
                    suffix=".docx", 
                    delete=False
                ) as tmp_docx: 
                    tmp_docx_path = tmp_docx.name                   

                try:
                    # ── PASO 1: Pandoc genera DOCX en temporal ─────────────────
                    result = _sp.run(
                        [
                            "pandoc",
                            tmp_md_path,
                            "-o", tmp_docx_path,
                            "--from", "markdown+tex_math_dollars+tex_math_single_backslash",
                            "--to", "docx",
                            "--mathml",
                        ],
                        capture_output=True, text=True, timeout=30
                        )

                    if result.returncode != 0:
                        raise RuntimeError(f"Pandoc error: {result.stderr}")

                    # ── PASO 2: Post-proceso con python-docx ───────────────────
                    try:
                        from docx import Document
                        from docx.shared import Pt, Cm
                        from docx.enum.text import WD_ALIGN_PARAGRAPH
                        
                        doc = Document(tmp_docx_path)
                        
                        for section in doc.sections:
                            section.top_margin    = Cm(1.5)
                            section.bottom_margin = Cm(1.5)
                            section.left_margin   = Cm(1.5)
                            section.right_margin  = Cm(1.5)
                        
                        # Justificación condicional para párrafos
                        for para in doc.paragraphs:
                            text_len = len(para.text.strip())
                            # Evitar párrafos vacíos
                            if text_len > 0:
                                para.alignment = (
                                    WD_ALIGN_PARAGRAPH.JUSTIFY 
                                    if text_len > 80 
                                    else WD_ALIGN_PARAGRAPH.LEFT
                                )
                        
                        # Justificar tablas (umbral más bajo por espacio limitado)
                        for table in doc.tables:
                            for row in table.rows:
                                for cell in row.cells:
                                    for para in cell.paragraphs:
                                        text_len = len(para.text.strip())
                                        if text_len > 0:
                                            para.alignment = (
                                                WD_ALIGN_PARAGRAPH.JUSTIFY 
                                                if text_len > 50 
                                                else WD_ALIGN_PARAGRAPH.LEFT
                                            )
                        
                        # Guardar en destino final (única escritura en out_path)
                        doc.save(str(out_path))
                        
                    except Exception as post_error:
                        # Si falla post-proceso, copiar el DOCX de Pandoc sin modificaciones
                        logger.warning(f"Justificación falló ({post_error}), usando DOCX original de Pandoc")
                        try:
                            import shutil as _shutil
                            _shutil.copy2(tmp_docx_path, str(out_path))
                        except Exception as copy_error:
                            logger.error(f"Fallo al copiar documento: {copy_error}")
                            raise

                finally:
                    # Limpieza garantizada de ambos temporales
                    import os as _os
                    _os.unlink(tmp_md_path)
                    if _os.path.exists(tmp_docx_path):
                        _os.unlink(tmp_docx_path)

            else:
                # ── python-docx path: texto plano + tablas markdown ───────
                from docx import Document
                from docx.shared import Pt, Cm
                from docx.enum.text import WD_ALIGN_PARAGRAPH

                document = Document()

                for section in document.sections:
                    section.top_margin    = Cm(1.5)
                    section.bottom_margin = Cm(1.5)
                    section.left_margin   = Cm(1.5)
                    section.right_margin  = Cm(1.5)

                if title:
                    h = document.add_heading(title, level=1)
                    for run in h.runs:
                        run.font.name = font
                        run.font.size = Pt(font_size + 4)

                def _is_table_line(line):
                    s = line.strip()
                    return s.startswith("|") and s.endswith("|")

                def _is_separator_line(line):
                    s = line.strip()
                    if not (s.startswith("|") and s.endswith("|")):
                        return False
                    return all(c in "-| :" for c in s[1:-1])

                def _parse_table_block(lines):
                    rows = []
                    for line in lines:
                        if _is_separator_line(line):
                            continue
                        cells = [c.strip() for c in line.strip()[1:-1].split("|")]
                        rows.append(cells)
                    return rows

                def _add_table(doc, rows):
                    if not rows:
                        return
                    ncols = max(len(r) for r in rows)
                    rows  = [r + [""] * (ncols - len(r)) for r in rows]
                    table = doc.add_table(rows=len(rows), cols=ncols)
                    table.style = "Table Grid"
                    for i, row in enumerate(rows):
                        for j, cell_text in enumerate(row):
                            cell = table.cell(i, j)
                            cell.text = cell_text
                            if i == 0:
                                runs = cell.paragraphs[0].runs
                                if runs:
                                    runs[0].bold = True
                                from docx.oxml.ns import qn as _qn
                                from docx.oxml import OxmlElement
                                tc   = cell._tc
                                tcPr = tc.get_or_add_tcPr()
                                shd  = OxmlElement("w:shd")
                                shd.set(_qn("w:val"),   "clear")
                                shd.set(_qn("w:color"), "auto")
                                shd.set(_qn("w:fill"),  "D9D9D9")
                                tcPr.append(shd)
                            for para in cell.paragraphs:
                                para.paragraph_format.space_after = Pt(2)
                                for run in para.runs:
                                    run.font.name = font
                                    run.font.size = Pt(font_size - 1)
                    doc.add_paragraph("")

                lines = content.split("\n")
                i = 0
                while i < len(lines):
                    line = lines[i]
                    if _is_table_line(line):
                        table_lines = []
                        while i < len(lines) and _is_table_line(lines[i]):
                            table_lines.append(lines[i])
                            i += 1
                        _add_table(document, _parse_table_block(table_lines))
                    else:
                        para_text = line.strip()
                        p  = document.add_paragraph(para_text if para_text else "")
                        pf = p.paragraph_format
                        pf.line_spacing = Pt(font_size * line_spacing)
                        pf.space_after  = Pt(4)
                        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                        for run in p.runs:
                            run.font.name = font
                            run.font.size = Pt(font_size)
                        i += 1

                document.save(str(out_path))                

        # ── MARKDOWN ─────────────────────────────────────────────────────────
        elif fmt == "markdown":
            out_path = out_dir / f"{safe_title}_{file_id}.md"
            md_content = f"# {title}\n\n" if title else ""
            md_content += content
            out_path.write_text(md_content, encoding="utf-8")

        # ── LATEX ────────────────────────────────────────────────────────────
        elif fmt == "latex":
            out_path = out_dir / f"{safe_title}_{file_id}.tex"
            body = content.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
            paragraphs = "\n\n".join(
                p.strip() for p in body.split("\n") if p.strip()
            )
            latex = (
                "\\documentclass[{pt}pt,a4paper]{{article}}\n"
                "\\usepackage[utf8]{{inputenc}}\n"
                "\\usepackage[T1]{{fontenc}}\n"
                "\\usepackage[margin=2.5cm]{{geometry}}\n"
                "\\usepackage{{setspace}}\n"
                "\\setstretch{{{ls}}}\n"
                "\\begin{{document}}\n"
                "{title_block}"
                "{body}\n"
                "\\end{{document}}"
            ).format(
                pt=font_size,
                ls=line_spacing,
                title_block=f"\\title{{{title}}}\n\\maketitle\n\n" if title else "",
                body=paragraphs,
            )
            out_path.write_text(latex, encoding="utf-8")

        # ── CSV ──────────────────────────────────────────────────────────────
        elif fmt == "csv":
            import io as _io
            out_path = out_dir / f"{safe_title}_{file_id}.csv"
            out_path.write_text(content, encoding="utf-8-sig")  # utf-8-sig para Excel

        # ── EXCEL (XLSX) ──────────────────────────────────────────────────────
        elif fmt in ("excel", "xlsx"):
            import pandas as _pd
            import io as _io
            fmt = "xlsx"
            out_path = out_dir / f"{safe_title}_{file_id}.xlsx"
            # Intentar parsear el contenido como CSV para convertir a Excel
            try:
                df = _pd.read_csv(_io.StringIO(content))
                df.to_excel(str(out_path), index=False, sheet_name=title or "Datos")
            except Exception:
                # Si no es CSV válido, guardar como texto en Excel
                df = _pd.DataFrame({"Contenido": content.split("\n")})
                df.to_excel(str(out_path), index=False)

        else:
            return f"Formato '{fmt}' no soportado. Usa: pdf, docx, markdown, latex, csv, excel"

        logger.info(f"Documento generado: {out_path.name}")
        return (
            f"✅ Documento generado correctamente.\n"
            f"📄 Archivo: {out_path.name}\n"
            f"📁 Ruta: {str(out_path)}\n"
            f"FORMAT:{fmt}|PATH:{str(out_path)}|NAME:{out_path.name}"
        )

    except Exception as e:
        logger.error(f"generate_document error: {e}")
        return f"Error generando documento: {e}"
        
# ── Tool: Agenda / Secretario personal ───────────────────────────────────────
@tool
def agenda(
    action: str,
    content: str = "",
    date: str = "",
    priority: int = 0,
    user_id: str = "default",
) -> str:
    """Personal agenda, task manager and priority tracker.
 
    Use this tool proactively whenever the user mentions:
    - Tasks, reminders, pending items, things to do
    - Deadlines, meetings, calls, emails to send
    - Priority reordering or what to focus on next
    - Asking what they have pending, planned or urgent
 
    Actions:
      add      — Add task/reminder. Required: content. Optional: date (YYYY-MM-DD
                 or natural like "mañana", "el viernes"), priority (1=high, 2=medium,
                 3=low, 0=auto-assign based on existing tasks)
      list     — List pending tasks. Optional: content = filter ("hoy", "esta semana",
                 "lunes", "urgente", "sin fecha") or empty for all pending
      done     — Mark task complete. content = task description or ID number
      delete   — Delete task. content = task description or ID number
      update   — Update priority or date of existing task. content = task description
                 or ID, priority = new priority (1/2/3), date = new date
      suggest  — Show tasks ordered by urgency with suggestions on what to tackle first
      reorder  — Reorder all pending tasks by priority+date automatically
 
    Priority levels:
      1 = URGENTE (hoy o mañana, crítico)
      2 = ESTA SEMANA (importante pero no inmediato)
      3 = PENDIENTE (sin prisa, backlog)
 
    Examples:
      action="add", content="Llamar al padre de Miguel", date="2026-04-10", priority=1
      action="add", content="Preparar examen de fracciones", priority=2
      action="add", content="Revisar RAG con transcripciones"  (sin fecha, prioridad auto)
      action="list", content="hoy"
      action="list"  (todas las pendientes)
      action="done", content="Llamar al padre de Miguel"
      action="update", content="Preparar examen", priority=1, date="2026-04-11"
      action="suggest"
      action="reorder"
    """
    import json
    from datetime import datetime, timedelta, date as _date
    from pathlib import Path as _Path
 
    agenda_dir  = _Path(__file__).parent.parent / "data" / "agenda" / user_id
    agenda_dir.mkdir(parents=True, exist_ok=True)
    agenda_file = agenda_dir / "agenda.json"
 
    # ── Cargar agenda ─────────────────────────────────────────────────────────
    if agenda_file.exists():
        try:
            tasks = json.loads(agenda_file.read_text(encoding="utf-8"))
        except Exception:
            tasks = []
    else:
        tasks = []
 
    now      = datetime.now()
    today    = now.date()
    act      = action.lower().strip()
 
    def save():
        agenda_file.write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
 
    def parse_date(date_str: str):
        """Convierte texto de fecha a string YYYY-MM-DD."""
        if not date_str:
            return ""
        d = date_str.lower().strip()
        if d in ("hoy", "today"):
            return today.isoformat()
        if d in ("mañana", "tomorrow"):
            return (today + timedelta(days=1)).isoformat()
        if d in ("pasado mañana",):
            return (today + timedelta(days=2)).isoformat()
        day_map = {
            "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
            "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6,
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }
        for name, weekday in day_map.items():
            if name in d:
                days_ahead = (weekday - today.weekday()) % 7 or 7
                return (today + timedelta(days=days_ahead)).isoformat()
        # Intentar parsear formato YYYY-MM-DD o DD/MM/YYYY
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(date_str.strip(), fmt).date().isoformat()
            except ValueError:
                continue
        return date_str  # devolver tal cual si no se reconoce
 
    def auto_priority(date_str: str) -> int:
        """Asigna prioridad automática según fecha."""
        if not date_str:
            return 3
        try:
            task_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            days = (task_date - today).days
            if days <= 1:
                return 1
            elif days <= 7:
                return 2
            else:
                return 3
        except Exception:
            return 3
 
    def priority_label(p: int) -> str:
        return {1: "🔴 URGENTE", 2: "🟡 ESTA SEMANA", 3: "🟢 PENDIENTE"}.get(p, "⚪")
 
    def format_task(t: dict, idx: int) -> str:
        status  = "✅" if t.get("done") else priority_label(t.get("priority", 3))
        date_s  = f" 📅 {t['date']}" if t.get("date") else ""
        return f"{status} [{t.get('id', idx+1)}] {t['content']}{date_s}"
 
    def next_id() -> int:
        existing = [t.get("id", 0) for t in tasks]
        return max(existing, default=0) + 1
 
    def find_task(ref: str):
        """Busca tarea por ID numérico o texto parcial."""
        ref = ref.strip()
        # Por ID
        if ref.isdigit():
            tid = int(ref)
            for t in tasks:
                if t.get("id") == tid:
                    return t
        # Por texto parcial (insensible a mayúsculas)
        ref_lower = ref.lower()
        for t in tasks:
            if ref_lower in t["content"].lower():
                return t
        return None
 
    # ── ADD ───────────────────────────────────────────────────────────────────
    if act == "add":
        if not content:
            return "❌ Indica el contenido de la tarea."
        parsed_date = parse_date(date)
        p = priority if priority in (1, 2, 3) else auto_priority(parsed_date)
        task = {
            "id":       next_id(),
            "content":  content,
            "date":     parsed_date,
            "priority": p,
            "done":     False,
            "created":  now.isoformat(),
        }
        tasks.append(task)
        save()
        date_str = f" para el **{parsed_date}**" if parsed_date else ""
        return (
            f"✅ Añadido{date_str}: **{content}**\n"
            f"Prioridad: {priority_label(p)}"
        )
 
    # ── LIST ──────────────────────────────────────────────────────────────────
    elif act == "list":
        pending = [t for t in tasks if not t.get("done")]
        if not pending:
            return "📭 No hay tareas pendientes. ¡Agenda limpia!"
 
        filter_text = content.lower().strip() if content else ""
 
        if filter_text:
            filtered = []
            # Filtros por periodo
            if filter_text in ("hoy", "today"):
                for t in pending:
                    if t.get("date") == today.isoformat():
                        filtered.append(t)
            elif filter_text in ("mañana", "tomorrow"):
                tmr = (today + timedelta(days=1)).isoformat()
                for t in pending:
                    if t.get("date") == tmr:
                        filtered.append(t)
            elif "semana" in filter_text or "week" in filter_text:
                end_week = today + timedelta(days=6 - today.weekday())
                for t in pending:
                    if t.get("date"):
                        try:
                            td = datetime.strptime(t["date"][:10], "%Y-%m-%d").date()
                            if today <= td <= end_week:
                                filtered.append(t)
                        except Exception:
                            pass
            elif filter_text in ("urgente", "urgent", "1"):
                filtered = [t for t in pending if t.get("priority") == 1]
            elif filter_text in ("sin fecha", "no date"):
                filtered = [t for t in pending if not t.get("date")]
            else:
                # Filtro por día de semana
                parsed = parse_date(filter_text)
                if parsed and len(parsed) == 10:
                    filtered = [t for t in pending if t.get("date") == parsed]
                else:
                    # Búsqueda de texto en contenido
                    filtered = [t for t in pending
                                 if filter_text in t["content"].lower()]
            pending = filtered if filtered else pending
 
        if not pending:
            return f"📭 No hay tareas pendientes para '{content}'."
 
        # Ordenar por prioridad y fecha
        def sort_key(t):
            p = t.get("priority", 3)
            d = t.get("date", "9999-12-31") or "9999-12-31"
            return (p, d)
 
        pending.sort(key=sort_key)
        lines = [f"📋 **Tareas pendientes** ({len(pending)}):"]
        for i, t in enumerate(pending):
            lines.append(format_task(t, i))
        return "\n".join(lines)
 
    # ── DONE ──────────────────────────────────────────────────────────────────
    elif act == "done":
        if not content:
            return "❌ Indica qué tarea marcar como completada."
        task = find_task(content)
        if not task:
            return f"❌ No encontré tarea con '{content}'. Usa action='list' para ver las tareas."
        task["done"] = True
        task["completed_at"] = now.isoformat()
        save()
        return f"✅ Completada: **{task['content']}**"
 
    # ── DELETE ────────────────────────────────────────────────────────────────
    elif act == "delete":
        if not content:
            return "❌ Indica qué tarea eliminar."
        task = find_task(content)
        if not task:
            return f"❌ No encontré tarea con '{content}'."
        tasks.remove(task)
        save()
        return f"🗑️ Eliminada: **{task['content']}**"
 
    # ── UPDATE ────────────────────────────────────────────────────────────────
    elif act == "update":
        if not content:
            return "❌ Indica qué tarea actualizar."
        task = find_task(content)
        if not task:
            return f"❌ No encontré tarea con '{content}'."
        changes = []
        if priority in (1, 2, 3):
            task["priority"] = priority
            changes.append(f"prioridad → {priority_label(priority)}")
        if date:
            parsed_date = parse_date(date)
            task["date"] = parsed_date
            changes.append(f"fecha → {parsed_date}")
        if not changes:
            return "❌ Indica qué cambiar: priority (1/2/3) o date."
        save()
        return f"✏️ Actualizada **{task['content']}**: {', '.join(changes)}"
 
    # ── SUGGEST ───────────────────────────────────────────────────────────────
    elif act == "suggest":
        pending = [t for t in tasks if not t.get("done")]
        if not pending:
            return "📭 No hay tareas pendientes. ¡Agenda limpia!"
 
        # Calcular urgencia real de cada tarea
        def urgency_score(t):
            p = t.get("priority", 3)
            d = t.get("date", "")
            if d:
                try:
                    days = (datetime.strptime(d[:10], "%Y-%m-%d").date() - today).days
                    if days < 0:
                        return (-1, p)   # vencida — máxima urgencia
                    return (days, p)
                except Exception:
                    pass
            return (999, p)  # sin fecha — al final
 
        pending.sort(key=urgency_score)
 
        lines = ["🎯 **Sugerencias de prioridad:**\n"]
        vencidas = []
        urgentes = []
        semana   = []
        backlog  = []
 
        for t in pending:
            d = t.get("date", "")
            if d:
                try:
                    days = (datetime.strptime(d[:10], "%Y-%m-%d").date() - today).days
                    if days < 0:
                        vencidas.append((t, days))
                    elif days <= 1:
                        urgentes.append((t, days))
                    elif days <= 7:
                        semana.append((t, days))
                    else:
                        backlog.append((t, days))
                except Exception:
                    backlog.append((t, 999))
            else:
                backlog.append((t, 999))
 
        if vencidas:
            lines.append("⚠️ **VENCIDAS — actuar de inmediato:**")
            for t, days in vencidas:
                lines.append(f"  🔴 [{t['id']}] {t['content']} (hace {abs(days)} días)")
        if urgentes:
            lines.append("\n🔴 **HOY / MAÑANA:**")
            for t, days in urgentes:
                when = "HOY" if days == 0 else "MAÑANA"
                lines.append(f"  [{t['id']}] {t['content']} — {when}")
        if semana:
            lines.append("\n🟡 **ESTA SEMANA:**")
            for t, days in semana:
                lines.append(f"  [{t['id']}] {t['content']} — en {days} días ({t.get('date','')})")
        if backlog:
            lines.append("\n🟢 **PENDIENTE / SIN FECHA:**")
            for t, _ in backlog:
                date_info = f" ({t['date']})" if t.get("date") else ""
                lines.append(f"  [{t['id']}] {t['content']}{date_info}")
 
        lines.append(f"\n📊 Total pendiente: {len(pending)} tareas")
        return "\n".join(lines)
 
    # ── REORDER ───────────────────────────────────────────────────────────────
    elif act == "reorder":
        pending = [t for t in tasks if not t.get("done")]
        if not pending:
            return "📭 No hay tareas pendientes para reordenar."
 
        # Recalcular prioridad automáticamente según fecha
        for t in pending:
            if t.get("date"):
                t["priority"] = auto_priority(t["date"])
 
        save()
        lines = ["🔄 **Agenda reordenada:**\n"]
 
        def sort_key(t):
            return (t.get("priority", 3), t.get("date", "9999-12-31") or "9999-12-31")
 
        pending.sort(key=sort_key)
        for i, t in enumerate(pending):
            lines.append(format_task(t, i))
        return "\n".join(lines)
 
    else:
        return (
            f"❌ Acción '{action}' no reconocida. "
            "Usa: add, list, done, delete, update, suggest, reorder"
        )

# ── Registro de todas las tools disponibles ───────────────────────────────────
ALL_TOOLS = [
    web_search,
    search_news,
    calculator,
    code_interpreter,
    fetch_url,
    generate_document,
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

def get_tools(names: Optional[list[str]] = None):
    """
    Devuelve lista de tools por nombre.
    Si names es None devuelve todas.
    """
    if names is None:
        return ALL_TOOLS
    return [TOOLS_BY_NAME[n] for n in names if n in TOOLS_BY_NAME]
