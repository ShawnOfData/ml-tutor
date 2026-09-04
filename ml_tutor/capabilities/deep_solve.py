"""Deep Solve capability — agentic-engine-based multi-step problem solving.

Thin shim that delegates to :class:`SolvePipeline`. All orchestration (the
pre-retrieve sub-DAG, per-step agentic loops with ``THINK`` / ``TOOL`` /
``FINISH`` / ``REPLAN``, replan back-edge, synthesize) lives in the pipeline
module; the capability just wires the manifest.
"""

from __future__ import annotations

from ml_tutor.agents.solve.pipeline import SolvePipeline
from ml_tutor.capabilities.request_contracts import get_capability_request_schema
from ml_tutor.core.capability_protocol import BaseCapability, CapabilityManifest
from ml_tutor.core.context import UnifiedContext
from ml_tutor.core.stream_bus import StreamBus


class DeepSolveCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="deep_solve",
        description="Multi-agent problem solving (Plan -> ReAct -> Write).",
        stages=["planning", "reasoning", "writing"],
        tools_used=["rag", "web_search", "code_execution", "reason"],
        cli_aliases=["solve"],
        request_schema=get_capability_request_schema("deep_solve"),
        trigger_keywords=["解题", "求解", "解方程", "计算", "deep solve", "deep_solve"],
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        # Knowledge bases are the single source of truth for whether ``rag``
        # is available. There is no separate "enable rag" toggle.
        kb_name = context.knowledge_bases[0] if context.knowledge_bases else None
        requested = list(
            self.manifest.tools_used if context.enabled_tools is None else context.enabled_tools
        )
        # Drop ``rag`` from the user-visible toggle list — the pipeline mounts
        # it itself when a KB is attached, and never when none is.
        enabled_tools = [tool for tool in requested if tool != "rag"]

        pipeline = SolvePipeline(
            language=context.language,
            kb_name=kb_name,
            enabled_tools=enabled_tools,
        )
        await pipeline.run(
            context=context,
            question=context.user_message,
            attachments=context.attachments,
            conversation_context=str(
                context.metadata.get("conversation_context_text", "") or ""
            ).strip(),
            memory_context=str(context.memory_context or "").strip(),
            stream=stream,
        )
