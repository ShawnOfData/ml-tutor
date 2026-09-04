"""Per-user project points stored as JSON files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .identity import get_user_by_id
from .paths import SYSTEM_ROOT

POINTS_DIR = SYSTEM_ROOT / "points"
_POINTS_WRITE_LOCK = Lock()
PROJECT_POINTS = 100


def _empty_points(user_id: str) -> dict[str, Any]:
    return {
        "version": 1,
        "user_id": user_id,
        "total": 0,
        "projects": {},
    }


def _points_path(user_id: str) -> Path:
    POINTS_DIR.mkdir(parents=True, exist_ok=True)
    return POINTS_DIR / f"{user_id}.json"


def load_points(user_id: str) -> dict[str, Any]:
    path = _points_path(user_id)
    if not path.exists():
        return _empty_points(user_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_points(user_id)
        return {
            "version": int(data.get("version") or 1),
            "user_id": user_id,
            "total": int(data.get("total") or 0),
            "projects": dict(data.get("projects") or {}),
        }
    except Exception:
        return _empty_points(user_id)


def award_points(user_id: str, project_slug: str) -> dict[str, Any]:
    user_record = get_user_by_id(user_id)
    if user_record is None:
        raise ValueError(f"Unknown user id: {user_id}")

    with _POINTS_WRITE_LOCK:
        data = load_points(user_id)
        if project_slug in data["projects"]:
            return {
                "newly_completed": False,
                "points_earned": 0,
                "total": data["total"],
            }

        data["projects"][project_slug] = {
            "points": PROJECT_POINTS,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        data["total"] += PROJECT_POINTS

        path = _points_path(user_id)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "newly_completed": True,
            "points_earned": PROJECT_POINTS,
            "total": data["total"],
        }
