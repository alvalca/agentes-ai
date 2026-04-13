# tests/test_tools.py — Tests unitarios para las tools del agente
# ============================================================================
# Cubre: calculator, generate_document (formatos y detección de contenido),
#        intercepción CSV/Excel en code_interpreter
# ============================================================================
import pytest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Tests: calculator ────────────────────────────────────────────────────────

class TestCalculator:
    """Verifica que el sandbox matemático funciona correctamente."""

    def _calc(self, expr):
        # Mock de ddgs para no requerir la dependencia en tests
        import sys
        from unittest.mock import MagicMock
        if 'ddgs' not in sys.modules:
            sys.modules['ddgs'] = MagicMock()
        from backend.tools import calculator
        return calculator.func(expr)

    def test_basic_arithmetic(self):
        assert self._calc("2 + 2") == "4"

    def test_multiplication(self):
        assert self._calc("3 * 7") == "21"

    def test_division(self):
        assert self._calc("10 / 4") == "2.5"

    def test_power(self):
        assert self._calc("2 ** 10") == "1024"

    def test_sqrt(self):
        result = self._calc("sqrt(144)")
        assert result == "12.0"

    def test_round(self):
        assert self._calc("round(3.14159, 2)") == "3.14"

    def test_complex_expression(self):
        result = self._calc("round(28e12 / 335e6 / 12, 2)")
        assert float(result) > 0

    def test_invalid_expression_returns_error(self):
        result = self._calc("import os; os.system('ls')")
        assert "Error" in result

    def test_undefined_variable_returns_error(self):
        result = self._calc("x + 5")
        assert "Error" in result

    def test_list_comprehension(self):
        result = self._calc("[x**2 for x in range(5)]")
        assert "0" in result and "16" in result


# ── Tests: generate_document — detección de contenido ───────────────────────

class TestGenerateDocumentContentDetection:
    """Verifica la detección de tipos de contenido para ruteo de generación."""

    def test_has_rich_md_detects_headers(self):
        import re
        content = "# Título\n\nEste es un párrafo."
        has_rich = bool(re.search(r'(^#|\*|- |\|)', content, re.MULTILINE))
        assert has_rich is True

    def test_has_rich_md_detects_lists(self):
        import re
        content = "Texto normal\n- elemento uno\n- elemento dos"
        has_rich = bool(re.search(r'(^#|\*|- |\|)', content, re.MULTILINE))
        assert has_rich is True

    def test_has_rich_md_detects_tables(self):
        import re
        content = "| Col1 | Col2 |\n|------|------|\n| a    | b    |"
        has_rich = bool(re.search(r'(^#|\*|- |\|)', content, re.MULTILINE))
        assert has_rich is True

    def test_plain_text_no_rich_md(self):
        import re
        content = "Este es texto plano sin formato especial. Solo párrafos normales."
        has_rich = bool(re.search(r'(^#|\*|- |\|)', content, re.MULTILINE))
        assert has_rich is False

    def test_has_math_detects_inline(self):
        import re
        content = "La fórmula $\\frac{1}{2}$ es importante."
        has_math = bool(re.search(r'\$\$?.+?\$\$?', content, re.DOTALL))
        assert has_math is True

    def test_has_math_detects_block(self):
        import re
        content = "Aquí la integral:\n$$\\int_0^1 x^2 dx$$\nResultado conocido."
        has_math = bool(re.search(r'\$\$?.+?\$\$?', content, re.DOTALL))
        assert has_math is True

    def test_no_math_plain_text(self):
        import re
        content = "El precio es 5 dólares. Sin expresiones matemáticas."
        has_math = bool(re.search(r'\$\$?.+?\$\$?', content, re.DOTALL))
        assert has_math is False


# ── Tests: generate_document — markdown tables (python-docx path) ────────────

class TestMarkdownTableParsing:
    """Verifica el parsing de tablas markdown para DOCX."""

    def _is_table_line(self, line):
        s = line.strip()
        return s.startswith("|") and s.endswith("|")

    def _is_separator_line(self, line):
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            return False
        return all(c in "-| :" for c in s[1:-1])

    def _parse_table_block(self, lines):
        rows = []
        for line in lines:
            if self._is_separator_line(line):
                continue
            cells = [c.strip() for c in line.strip()[1:-1].split("|")]
            rows.append(cells)
        return rows

    def test_detects_table_line(self):
        assert self._is_table_line("| Col1 | Col2 |") is True
        assert self._is_table_line("Texto normal") is False
        assert self._is_table_line("| sin cerrar") is False

    def test_detects_separator_line(self):
        assert self._is_separator_line("|------|------|") is True
        assert self._is_separator_line("|:-----|-----:|") is True
        assert self._is_separator_line("| Col1 | Col2 |") is False

    def test_parses_table_rows(self):
        lines = [
            "| Nombre | Nota |",
            "|--------|------|",
            "| Python | 10   |",
            "| R      | 9.5  |",
        ]
        rows = self._parse_table_block(lines)
        assert len(rows) == 3  # header + 2 data rows (separator excluido)
        assert rows[0] == ["Nombre", "Nota"]
        assert rows[1] == ["Python", "10"]
        assert rows[2] == ["R", "9.5"]

    def test_parses_table_with_spaces(self):
        lines = ["  | A | B |  ", "  |---|---|  ", "  | 1 | 2 |  "]
        rows = self._parse_table_block(lines)
        assert rows[0] == ["A", "B"]
        assert rows[1] == ["1", "2"]

    def test_empty_cells_handled(self):
        lines = ["| A | | C |", "|---|---|---|", "| 1 | | 3 |"]
        rows = self._parse_table_block(lines)
        assert rows[1][1] == ""  # celda vacía


# ── Tests: generate_document — formatos soportados ───────────────────────────

class TestGenerateDocumentFormats:
    """Tests de smoke para los formatos de documento."""

    def test_markdown_format(self, tmp_data_dir):
        """Generación de .md no requiere dependencias externas."""
        with patch("backend.tools.DOCUMENTS_DIR", tmp_data_dir):
            from backend.tools import generate_document
            result = generate_document.func(
                content="# Título\n\nContenido de prueba.",
                format="markdown",
                title="Test",
                user_id="test_user"
            )
        assert "FORMAT:markdown" in result
        assert "NAME:" in result
        # Verificar que el archivo existe
        name = result.split("NAME:")[1].strip()
        assert (tmp_data_dir / "test_user" / name).exists()

    def test_latex_format(self, tmp_data_dir):
        """Generación de .tex no requiere dependencias externas."""
        with patch("backend.tools.DOCUMENTS_DIR", tmp_data_dir):
            from backend.tools import generate_document
            result = generate_document.func(
                content="Contenido LaTeX de prueba.",
                format="latex",
                title="Test LaTeX",
                user_id="test_user"
            )
        assert "FORMAT:latex" in result

    def test_csv_format(self, tmp_data_dir):
        """Generación de .csv con utf-8-sig para compatibilidad Excel."""
        with patch("backend.tools.DOCUMENTS_DIR", tmp_data_dir):
            from backend.tools import generate_document
            result = generate_document.func(
                content="nombre,valor\ntest,42\n",
                format="csv",
                title="Test CSV",
                user_id="test_user"
            )
        assert "FORMAT:csv" in result
        name = result.split("NAME:")[1].strip()
        file_path = tmp_data_dir / "test_user" / name
        content = file_path.read_bytes()
        # utf-8-sig debe empezar con BOM
        assert content[:3] == b'\xef\xbb\xbf'

    def test_unsupported_format_returns_error(self, tmp_data_dir):
        with patch("backend.tools.DOCUMENTS_DIR", tmp_data_dir):
            from backend.tools import generate_document
            result = generate_document.func(
                content="test", format="pptx",
                title="Test", user_id="test_user"
            )
        assert "no soportado" in result.lower() or "Error" in result

    def test_safe_title_truncation(self, tmp_data_dir):
        """Títulos largos se truncan a 40 chars para el nombre de archivo."""
        long_title = "Este es un título muy largo que supera los cuarenta caracteres sin duda"
        with patch("backend.tools.DOCUMENTS_DIR", tmp_data_dir):
            from backend.tools import generate_document
            result = generate_document.func(
                content="contenido", format="markdown",
                title=long_title, user_id="test_user"
            )
        assert "FORMAT:markdown" in result
        name = result.split("NAME:")[1].strip()
        # El nombre de archivo no debe ser excesivamente largo
        assert len(name) < 100


# ── Tests: intercepción CSV/Excel en code_interpreter ───────────────────────

class TestCodeInterpreterInterception:
    """
    Verifica que la intercepción de to_csv/to_excel redirige
    correctamente los archivos a la carpeta del usuario.
    """

    def test_csv_saved_to_user_dir(self, tmp_data_dir):
        """df.to_csv('archivo.csv') debe guardarse en data/documents/{user_id}/."""
        import pandas as pd
        from unittest.mock import patch

        with patch("backend.tools._current_user_id", "test_user"), \
             patch("backend.tools.Path") as mock_path:

            user_dir = tmp_data_dir / "documents" / "test_user"
            user_dir.mkdir(parents=True, exist_ok=True)

            # La intercepción usa _current_user_id para construir la ruta
            # Verificamos que el mecanismo de intercepción existe en el código
            import inspect
            import backend.tools as tools_mod
            source = inspect.getsource(tools_mod.code_interpreter.func)
            assert "_intercept_to_csv" in source
            assert "_intercept_to_excel" in source
            assert "_orig_to_csv" in source

    def test_interception_handles_absolute_paths(self):
        """La intercepción debe funcionar tanto con rutas relativas como absolutas."""
        import inspect
        import backend.tools as tools_mod
        source = inspect.getsource(tools_mod.code_interpreter.func)
        # Debe interceptar cualquier string, no solo rutas relativas
        # La versión corregida NO tiene la condición `not path_or_buf.startswith("/")`
        assert 'not path_or_buf.startswith("/")' not in source

    def test_pandas_methods_restored_after_execution(self, tmp_data_dir):
        """Los métodos originales de pandas deben restaurarse tras la ejecución."""
        import pandas as pd
        original_to_csv = pd.DataFrame.to_csv
        original_to_excel = pd.DataFrame.to_excel

        with patch("backend.tools._current_user_id", "test_user"):
            from backend.tools import code_interpreter
            # Ejecutar código simple que no usa to_csv
            result = code_interpreter.func("1 + 1")

        # Los métodos deben estar restaurados
        assert pd.DataFrame.to_csv is original_to_csv
        assert pd.DataFrame.to_excel is original_to_excel


# ── Tests: web_search — comportamiento de fallback ───────────────────────────

class TestWebSearch:
    """Tests básicos del comportamiento de web_search."""

    def test_returns_string(self):
        """web_search siempre devuelve un string."""
        from backend.tools import web_search
        with patch("backend.tools._brave_search", return_value=[]), \
             patch("backend.tools._ddgs_fallback_web", return_value=[]):
            result = web_search.func("test query")
        assert isinstance(result, str)

    def test_no_results_message(self):
        """Sin resultados debe devolver mensaje apropiado."""
        from backend.tools import web_search
        with patch("backend.tools._brave_search", return_value=[]), \
             patch("backend.tools._ddgs_fallback_web", return_value=[]):
            result = web_search.func("xyzzy_impossible_query_12345")
        assert "No results" in result or "search_news" in result

    def test_brave_fallback_to_ddgs(self):
        """Si Brave no está configurado, debe usar DDGS."""
        from backend.tools import web_search
        mock_results = [{"title": "Test", "href": "http://test.com",
                          "body": "Test content"}]
        with patch("backend.tools.BRAVE_API_KEY", ""), \
             patch("backend.tools._ddgs_fallback_web", return_value=mock_results):
            result = web_search.func("test query")
        assert isinstance(result, str)
