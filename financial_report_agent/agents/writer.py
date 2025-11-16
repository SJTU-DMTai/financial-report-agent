from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter
from agentscope.tool import Toolkit
from agentscope.model import DashScopeChatModel

from ..config.prompts import WRITER_SYS_PROMPT


def create_writer_agent(
    model: DashScopeChatModel,
    toolkit: Toolkit,
) -> ReActAgent:
    return ReActAgent(
        name="Writer",
        sys_prompt=WRITER_SYS_PROMPT,
        model=model,
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        parallel_tool_calls=True,
        max_iters=50
    )
