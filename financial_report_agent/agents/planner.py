from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter
from agentscope.tool import Toolkit
from agentscope.model import DashScopeChatModel

from ..config.prompts import PLANNER_SYS_PROMPT


def create_planner_agent(
    model: DashScopeChatModel,
    toolkit: Toolkit,
) -> ReActAgent:
    return ReActAgent(
        name="Planner",
        sys_prompt=PLANNER_SYS_PROMPT,
        model=model,
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        parallel_tool_calls=True,
    )
