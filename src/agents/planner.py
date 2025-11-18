from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter
from agentscope.tool import Toolkit
from agentscope.model import DashScopeChatModel

from ..prompt import prompt_dict


def create_planner_agent(
    model,
    formatter,
    toolkit: Toolkit,
) -> ReActAgent:
    return ReActAgent(
        name="Planner",
        sys_prompt=prompt_dict['planner_sys_prompt'],
        model=model,
        memory=InMemoryMemory(),
        formatter=formatter,
        toolkit=toolkit,
        parallel_tool_calls=True,
    )
