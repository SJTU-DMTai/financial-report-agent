from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter
from agentscope.tool import Toolkit
from agentscope.model import DashScopeChatModel

from ..config.prompts import SEARCHER_SYS_PROMPT


def create_searcher_agent(
    model: DashScopeChatModel,
    toolkit: Toolkit,
) -> ReActAgent:
    """Searcher 使用 ReActAgent 实现。
    """
    return ReActAgent(
        name="Searcher",
        sys_prompt=SEARCHER_SYS_PROMPT,
        model=model,
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        parallel_tool_calls=True,
    )
