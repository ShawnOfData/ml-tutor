"""Built-in capability class paths."""

BUILTIN_CAPABILITY_CLASSES: dict[str, str] = {
    "chat": "ml_tutor.capabilities.chat:ChatCapability",
    "deep_solve": "ml_tutor.capabilities.deep_solve:DeepSolveCapability",
    "deep_question": "ml_tutor.capabilities.deep_question:DeepQuestionCapability",
    "deep_research": "ml_tutor.capabilities.deep_research:DeepResearchCapability",
}
