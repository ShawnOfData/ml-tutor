import base64
import logging
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ml_tutor.api.routers.auth import require_auth
from ml_tutor.tools.code_executor import run_code

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_auth)])

COURSE_ALLOWED_IMPORTS = [
    "math", "numpy", "pandas", "matplotlib", "seaborn",
    "scipy", "statsmodels", "json", "datetime", "re", "collections",
    "itertools", "functools", "random", "time", "statistics", "sympy",
    "sklearn", "torch", "transformers", "datasets",
    "PIL", "io", "base64", "typing", "dataclasses", "copy",
    "collections.abc", "warnings",
    "clip", "whisper", "openai",
    "segment_anything",
    "langchain", "langchain_community",
    "peft",
]


class ExecuteCodeRequest(BaseModel):
    code: str
    timeout: int = 15
    allowed_imports: list[str] | None = None


class ImageData(BaseModel):
    filename: str
    data: str  # base64-encoded
    mime: str


class ExecuteCodeResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    elapsed_ms: float
    images: list[ImageData] = []


@router.post("/execute-code", response_model=ExecuteCodeResponse)
async def execute_code_endpoint(req: ExecuteCodeRequest):
    result = await run_code(
        language="python",
        code=req.code,
        timeout=req.timeout,
        allowed_imports=req.allowed_imports or COURSE_ALLOWED_IMPORTS,
    )

    images: list[ImageData] = []
    for path_str in result.get("artifact_paths", []):
        p = Path(path_str)
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".svg"):
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".svg": "image/svg+xml",
            }.get(p.suffix.lower(), "image/png")
            try:
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                images.append(ImageData(filename=p.name, data=b64, mime=mime))
            except Exception as e:
                logger.warning("Failed to read artifact %s: %s", p, e)

    return ExecuteCodeResponse(
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        exit_code=result.get("exit_code", -1),
        elapsed_ms=result.get("elapsed_ms", 0.0),
        images=images,
    )
