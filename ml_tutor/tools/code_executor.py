#!/usr/bin/env python
"""
Run Code Tool - Code execution tool
Execute Python code in isolated workspace. Every execution is persisted
under task-scoped ``code_runs/`` directories (or the detached runtime workspace) with its source code, output log, and any
generated artifacts (images, data files, etc.).
"""

import ast
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ml_tutor.services.path_service import get_path_service

RUN_CODE_WORKSPACE_ENV = "RUN_CODE_WORKSPACE"
RUN_CODE_ALLOWED_ROOTS_ENV = "RUN_CODE_ALLOWED_ROOTS"
DEFAULT_WORKSPACE_NAME = "_detached_code_execution"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SAFE_IMPORTS = [
    "math",
    "numpy",
    "pandas",
    "matplotlib",
    "plt",
    "seaborn",
    "scipy",
    "statsmodels",
    "json",
    "datetime",
    "re",
    "collections",
    "itertools",
    "functools",
    "random",
    "time",
    "statistics",
    "sympy",
]
DISALLOWED_CALL_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
}
DISALLOWED_ATTRIBUTE_BASES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "importlib",
    "builtins",
    "__builtins__",
}

logger = logging.getLogger(__name__)

# Files managed by the executor itself (excluded from user-artifact lists)
_META_FILES = frozenset({"code.py", "output.log", ".gitkeep"})


def _load_config() -> dict[str, Any]:
    """Load run_code configuration from main.yaml."""
    from ml_tutor.services.config import load_config_with_main

    config = load_config_with_main("main.yaml", PROJECT_ROOT)
    run_code_config = config.get("tools", {}).get("run_code", {})
    if run_code_config:
        logger.debug("Loaded run_code config from main.yaml")
    return run_code_config


def _save_output_log(
    execution_dir: Path,
    stdout: str,
    stderr: str,
    exit_code: int,
    elapsed_ms: float,
) -> Path:
    """Persist execution output to ``output.log`` inside *execution_dir*."""
    log_file = execution_dir / "output.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Exit Code: {exit_code}\n")
        f.write(f"Elapsed: {elapsed_ms:.1f}ms\n")
        f.write(f"{'=' * 50}\n")
        if stdout:
            f.write(f"[STDOUT]\n{stdout}\n")
        if stderr:
            f.write(f"[STDERR]\n{stderr}\n")
    return log_file


class CodeExecutionError(Exception):
    """Code execution error"""


@dataclass
class OperationEntry:
    action: str
    details: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class OperationLogger:
    """Simple operation history logger."""

    def __init__(self, max_entries: int = 200):
        self._history: list[OperationEntry] = []
        self._max_entries = max_entries

    def log(self, action: str, details: dict[str, Any]):
        entry = OperationEntry(action=action, details=details)
        self._history.append(entry)
        if len(self._history) > self._max_entries:
            self._history.pop(0)
        logger.debug(f"Operation logged: {action} | details={details.get('status')}")

    @property
    def history(self) -> list[OperationEntry]:
        return list(self._history)


class WorkspaceManager:
    """Manages detached code-execution workspaces for explicit non-task runs."""

    def __init__(self):
        env_path = os.getenv(RUN_CODE_WORKSPACE_ENV)
        if env_path:
            self.base_dir = Path(env_path).expanduser().resolve()
        else:
            path_service = get_path_service()
            self.base_dir = path_service.get_run_code_workspace_dir().resolve()

        self._initialized = False

    def initialize(self):
        if not self._initialized:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            logger.info(f"Run-code workspace initialized at {self.base_dir}")

    def ensure_initialized(self):
        if not self._initialized:
            self.initialize()

    def create_execution_dir(self, prefix: str = "exec") -> Path:
        """Create a persistent, timestamped execution directory under the workspace."""
        self.ensure_initialized()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        exec_dir = self.base_dir / f"{prefix}_{timestamp}"
        exec_dir.mkdir(parents=True, exist_ok=True)
        return exec_dir

    def collect_artifacts(self, exec_dir: Path | None) -> tuple[list[str], list[str]]:
        """Return user-generated files (excluding code.py / output.log)."""
        artifacts: list[str] = []
        artifact_paths: list[str] = []
        if not exec_dir or not exec_dir.exists():
            return artifacts, artifact_paths

        for file_path in exec_dir.iterdir():
            if file_path.is_file() and file_path.name not in _META_FILES:
                artifacts.append(file_path.name)
                artifact_paths.append(str(file_path.resolve()))
        return artifacts, artifact_paths


class ImportGuard:
    """Parse AST, restrict import modules."""

    @staticmethod
    def _extract_all_names(node: ast.AST) -> set[str]:
        """Recursively extract all Name nodes from an AST subtree."""
        names: set[str] = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
        return names

    @staticmethod
    def _has_blocked_subscript_access(tree: ast.AST) -> bool:
        """Check for __builtins__['xxx'] or getattr(__builtins__, ...) patterns."""
        for node in ast.walk(tree):
            # __builtins__['open'](...)
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == '__builtins__':
                    return True
                if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name) and node.value.value.id == '__builtins__':
                    return True
            # getattr(__builtins__, ...)(...)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ('getattr', 'vars', 'setattr', 'hasattr'):
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id == '__builtins__':
                            return True
                        if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and arg.value.id == '__builtins__':
                            return True
        return False

    @staticmethod
    def _has_type_confusion_escape(tree: ast.AST) -> bool:
        """Detect ()__class__.__base__.__subclasses__ type confusion pattern."""
        for node in ast.walk(tree):
            # ().__class__ or [].__class__ or "".__class__
            if isinstance(node, ast.Attribute) and node.attr == '__class__':
                if isinstance(node.value, (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict)):
                    if isinstance(node.value, ast.Constant) and node.value.value in (None, True, False):
                        continue
                    return True
                if isinstance(node.value, ast.Name) and node.value.id in ('None', 'True', 'False'):
                    continue
                # Also catch (lambda:0).__class__
                if isinstance(node.value, ast.Lambda):
                    return True
            # Check for .__base__ or .__mro__ 
            if isinstance(node, ast.Attribute) and node.attr in ('__base__', '__bases__', '__mro__'):
                return True
            # Check for .__subclasses__()
            if isinstance(node, ast.Attribute) and node.attr == '__subclasses__':
                return True
        return False

    @staticmethod
    def _has_dangerous_import_aliases(tree: ast.AST, allowed: set[str]) -> set[str]:
        """Detect from blocked-module import Name patterns like 'from pathlib import Path'."""
        blocked_top = {'os', 'sys', 'subprocess', 'socket', 'pathlib', 'shutil', 'importlib', 'builtins'}
        dangerous_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split('.')[0]
                    if top in blocked_top:
                        dangerous_names.add(top)
        return dangerous_names

    @staticmethod
    def validate(code: str, allowed_imports: list[str] | None):
        if not allowed_imports:
            return

        allowed = set(allowed_imports)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise CodeExecutionError(f"Code syntax error: {exc}") from exc

        # --- 1. Import allowlist ---
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    imported.append(top)

        unauthorized = sorted({name for name in imported if name not in allowed})
        if unauthorized:
            raise CodeExecutionError(
                f"The following modules are not in the allowed list: {', '.join(unauthorized)}"
            )

        # --- 2. Blocked call names ---
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALL_NAMES:
                    raise CodeExecutionError(
                        f"Use of unsafe builtin is not allowed: {node.func.id}"
                    )
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in DISALLOWED_ATTRIBUTE_BASES
                ):
                    raise CodeExecutionError(
                        f"Use of unsafe module access is not allowed: "
                        f"{node.func.value.id}.{node.func.attr}"
                    )

        # --- 3. Block __builtins__ subscript access ---
        if ImportGuard._has_blocked_subscript_access(tree):
            raise CodeExecutionError(
                "Access to __builtins__ is not allowed for security reasons."
            )

        # --- 4. Block type confusion escape ---
        if ImportGuard._has_type_confusion_escape(tree):
            raise CodeExecutionError(
                "Type confusion escape pattern detected and blocked."
            )

        # --- 5. Block from blocked-module import ---
        dangerous = ImportGuard._has_dangerous_import_aliases(tree, allowed)
        if dangerous:
            raise CodeExecutionError(
                f"Import from blocked module is not allowed: {', '.join(sorted(dangerous))}"
            )


class CodeExecutionEnvironment:
    """Run Python code inside a persistent execution directory."""

    @staticmethod
    def _build_isolated_env() -> dict[str, str]:
        env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        for key in ("PATH", "SYSTEMROOT", "HOME", "USERPROFILE", "TMPDIR", "TEMP", "TMP"):
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def run_python(
        self,
        code: str,
        timeout: int,
        execution_dir: Path,
    ) -> tuple[str, str, int, float]:
        """Write *code* to ``execution_dir/code.py``, execute it, and return
        (stdout, stderr, exit_code, elapsed_ms).  The source file is kept on
        disk for later inspection."""
        env = self._build_isolated_env()
        env["MPLCONFIGDIR"] = str(execution_dir / ".mpl_config")

        mpl_dir = str(execution_dir / ".mpl_config").replace("\\", "/")
        preamble = (
            f"import os; os.environ.setdefault('MPLCONFIGDIR', \"{mpl_dir}\")\n"
            "import matplotlib; matplotlib.use('Agg')\n"
            "matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']\n"
            "matplotlib.rcParams['axes.unicode_minus'] = False\n"
        )
        code_file = execution_dir / "code.py"
        code_file.write_text(preamble + code, encoding="utf-8")

        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(code_file)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(execution_dir),
            env=env,
        )
        elapsed_ms = (time.time() - start_time) * 1000
        return result.stdout, result.stderr, result.returncode, elapsed_ms


WORKSPACE_MANAGER = WorkspaceManager()
OPERATION_LOGGER = OperationLogger()
EXECUTION_ENV = CodeExecutionEnvironment()


async def run_code(
    language: str,
    code: str,
    timeout: int = 10,
    allowed_imports: list[str] | None = None,
    workspace_dir: str | Path | None = None,
    feature: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> dict[str, Any]:
    """Execute code in a restricted, persistent directory under task-scoped ``code_runs``.

    This is a best-effort restricted runner, not a true operating-system sandbox.

    Each invocation creates a new timestamped directory containing:
    - ``code.py``    – the executed source code
    - ``output.log`` – captured stdout / stderr and exit info
    - any files the code itself generates (images, data, etc.)

    Returns a dict with stdout, stderr, exit_code, elapsed_ms, execution_dir,
    source_file, output_log, artifacts, and artifact_paths.
    """
    if language.lower() != "python":
        raise ValueError(f"Unsupported language: {language}, currently only Python is supported")

    if workspace_dir is None:
        workspace_dir = _resolve_task_workspace(
            feature=feature,
            task_id=task_id,
            session_id=session_id,
            turn_id=turn_id,
        )

    if workspace_dir is not None:
        custom_workspace = Path(workspace_dir).expanduser().resolve()
        custom_workspace.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        execution_dir = custom_workspace / f"exec_{timestamp}"
        execution_dir.mkdir(parents=True, exist_ok=True)
    else:
        WORKSPACE_MANAGER.ensure_initialized()
        execution_dir = WORKSPACE_MANAGER.create_execution_dir()
    stdout, stderr, exit_code, elapsed_ms = "", "", -1, 0.0
    status = "error"

    try:
        if allowed_imports is None:
            allowed_imports = DEFAULT_SAFE_IMPORTS
        ImportGuard.validate(code, allowed_imports)

        loop = asyncio.get_running_loop()

        def _execute():
            return EXECUTION_ENV.run_python(code, timeout, execution_dir)

        stdout, stderr, exit_code, elapsed_ms = await loop.run_in_executor(None, _execute)
        status = "success"

    except subprocess.TimeoutExpired:
        elapsed_ms = timeout * 1000
        stderr = f"Code execution timeout ({timeout} seconds)"
        status = "timeout"
        logger.warning(f"Code execution timeout after {timeout}s")
        # Ensure source file is written even if timeout happened before write
        src = execution_dir / "code.py"
        if not src.exists():
            src.write_text(code, encoding="utf-8")

    except CodeExecutionError as exc:
        stderr = str(exc)
        status = "validation_error"
        # Source code was invalid; still save it for diagnosis
        src = execution_dir / "code.py"
        if not src.exists():
            src.write_text(code, encoding="utf-8")

    except Exception as exc:  # pylint: disable=broad-except
        stderr = f"Code execution failed: {exc}"
        logger.error(f"Code execution error: {exc}", exc_info=True)
        src = execution_dir / "code.py"
        if not src.exists():
            src.write_text(code, encoding="utf-8")

    # Always persist the output log
    _save_output_log(execution_dir, stdout, stderr, exit_code, elapsed_ms)

    artifacts, artifact_paths = WORKSPACE_MANAGER.collect_artifacts(execution_dir)

    OPERATION_LOGGER.log(
        "execute_python",
        {
            "status": status,
            "language": language,
            "timeout": timeout,
            "execution_dir": str(execution_dir),
            "exit_code": exit_code,
            "elapsed_ms": elapsed_ms,
            "code_size": len(code),
        },
    )

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "elapsed_ms": elapsed_ms,
        "execution_dir": str(execution_dir),
        "source_file": str(execution_dir / "code.py"),
        "output_log": str(execution_dir / "output.log"),
        "artifacts": artifacts,
        "artifact_paths": artifact_paths,
    }


def _resolve_task_workspace(
    *,
    feature: str | None,
    task_id: str | None,
    session_id: str | None,
    turn_id: str | None,
) -> Path | None:
    """Resolve a task-scoped code workspace from runtime identifiers."""
    feature_name = str(feature or "").strip()
    if not feature_name:
        return None

    identifier = (
        str(task_id or "").strip() or str(turn_id or "").strip() or str(session_id or "").strip()
    )
    if not identifier:
        return None

    path_service = get_path_service()
    task_root = path_service.get_task_workspace(feature_name, identifier)
    return task_root / "code_runs"


def run_code_sync(
    language: str,
    code: str,
    timeout: int = 10,
) -> dict[str, Any]:
    """Synchronous version of code execution (for non-async environments)."""
    return asyncio.run(run_code(language, code, timeout))


if __name__ == "__main__":
    import textwrap

    async def _demo():
        print("==== 1. Test normal output ====")
        sample1 = "print('Hello from run_code workspace!')"
        result1 = await run_code("python", sample1, timeout=5)
        print("stdout:", result1["stdout"])
        print("stderr:", result1["stderr"])
        print("execution_dir:", result1["execution_dir"])
        print("source_file:", result1["source_file"])
        print("output_log:", result1["output_log"])
        print("artifacts:", result1["artifacts"])
        print("exit_code:", result1["exit_code"])
        print("-" * 40)

        print("==== 2. Test exception case ====")
        sample2 = "raise ValueError('Test error from run_code!')"
        result2 = await run_code("python", sample2, timeout=5)
        print("stdout:", result2["stdout"])
        print("stderr:", result2["stderr"])
        print("execution_dir:", result2["execution_dir"])
        print("exit_code:", result2["exit_code"])
        print("-" * 40)

        print("==== 3. Test code timeout ====")
        sample3 = textwrap.dedent("""\
            import time
            time.sleep(10)
            print("Timeout should occur before this prints.")
        """)
        result3 = await run_code("python", sample3, timeout=2)
        print("stdout:", result3["stdout"])
        print("stderr:", result3["stderr"])
        print("execution_dir:", result3["execution_dir"])
        print("exit_code:", result3["exit_code"])
        print("-" * 40)

        print("==== 4. Test plotting functionality (matplotlib) ====")
        sample4 = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot([1, 2, 3], [4, 2, 5])
            plt.title('Simple Plot')
            plt.savefig('test_plot.png')
            print('Plot created!')
        """)
        result4 = await run_code("python", sample4, timeout=5)
        print("stdout:", result4["stdout"])
        print("stderr:", result4["stderr"])
        print("execution_dir:", result4["execution_dir"])
        print("artifacts:", result4["artifacts"])
        print("artifact_paths:", result4["artifact_paths"])
        print("exit_code:", result4["exit_code"])
        print("-" * 40)

        print("==== 5. Test file read/write ====")
        sample5 = textwrap.dedent("""\
            with open('test_file.txt', 'w', encoding='utf-8') as f:
                f.write('Fake data for test!\\nAnother line.')
            with open('test_file.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            print('File content:', content)
        """)
        result5 = await run_code("python", sample5, timeout=5)
        print("stdout:", result5["stdout"])
        print("stderr:", result5["stderr"])
        print("execution_dir:", result5["execution_dir"])
        print("artifacts:", result5["artifacts"])
        print("exit_code:", result5["exit_code"])
        print("-" * 40)

    asyncio.run(_demo())
