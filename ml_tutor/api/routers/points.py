"""API endpoints for user project points."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ml_tutor.multi_user.context import get_current_user
from ml_tutor.multi_user.points import award_points, load_points

router = APIRouter()


class AwardRequest(BaseModel):
    project_slug: str


@router.get("")
async def get_points():
    user = get_current_user()
    data = load_points(user.id)
    return {
        "total": data["total"],
        "projects": list(data["projects"].keys()),
    }


@router.post("/award")
async def award(req: AwardRequest):
    user = get_current_user()
    try:
        result = award_points(user.id, req.project_slug)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return result
