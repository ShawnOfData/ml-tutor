"""Agentic chat capability."""

from __future__ import annotations

from ml_tutor.agents.chat.agentic_pipeline import CHAT_OPTIONAL_TOOLS, AgenticChatPipeline
from ml_tutor.capabilities.request_contracts import get_capability_request_schema
from ml_tutor.core.capability_protocol import BaseCapability, CapabilityManifest
from ml_tutor.core.context import UnifiedContext
from ml_tutor.core.stream_bus import StreamBus


class ChatCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="chat",
        description="Agentic chat with autonomous tool selection across enabled tools.",
        stages=["thinking", "acting", "observing", "responding"],
        tools_used=CHAT_OPTIONAL_TOOLS,
        cli_aliases=["chat"],
        request_schema=get_capability_request_schema("chat"),
        trigger_keywords=["chat"],
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        pipeline = AgenticChatPipeline(language=context.language)
        await pipeline.run(context, stream)
