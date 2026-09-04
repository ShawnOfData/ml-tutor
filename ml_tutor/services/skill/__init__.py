"""Skill service: load user-authored SKILL.md files and inject them into the chat system prompt."""

from ml_tutor.services.skill.service import SkillService, get_skill_service

__all__ = ["SkillService", "get_skill_service"]
