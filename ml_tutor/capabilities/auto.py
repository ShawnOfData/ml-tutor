"""Auto capability — agent loop that routes between other capabilities.

Thin wrapper that hands off to ``AutoPipeline``. See
``ml_tutor/agents/auto/auto_pipeline.py`` for the loop body.
"""

from __future__ import annotations

from ml_tutor.capabilities.request_contracts import get_capability_request_schema
from ml_tutor.core.capability_protocol import BaseCapability, CapabilityManifest
from ml_tutor.core.context import UnifiedContext
from ml_tutor.core.stream_bus import StreamBus


class AutoCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="auto",
        description=(
            "Agentic router: analyzes intent, autonomously delegates to the best "
            "matching capability (deep_solve, deep_question, deep_research, "
            "math_animator, visualize), and synthesizes the final response. Use "
            "when you want the system to pick the right mode for you."
        ),
        stages=["analyzing", "delegating", "synthesizing"],
        tools_used=[],
        cli_aliases=["auto"],
        request_schema=get_capability_request_schema("auto"),
        trigger_keywords=["auto"],
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        # Local import keeps capability registration cheap (registry boot only
        # imports the class; the heavy pipeline imports happen at first run).
        from ml_tutor.agents.auto.auto_pipeline import AutoPipeline

        pipeline = AutoPipeline(language=context.language)
        await pipeline.run(context, stream)
