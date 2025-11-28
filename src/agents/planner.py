from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit
from agentscope.model import DashScopeChatModel

from ..tools.outline_tools import *
from ..tools.search_tools import *
from ..memory.short_term import ShortTermMemoryStore

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

    

# ---- Toolkit Builder ----
def build_planner_toolkit(
    short_term: ShortTermMemoryStore,
    searcher: ReActAgent,
) -> Toolkit:
    """创建 Planner 专用 Toolkit。"""
    toolkit = Toolkit()

    outline_tools = OutlineTools(short_term=short_term)
    # toolkit.register_tool_function(outline_tools.read_outline)
    toolkit.register_tool_function(outline_tools.read_demonstration)
    toolkit.register_tool_function(outline_tools.replace_outline)
    toolkit.register_tool_function(searcher_tool(searcher))
    return toolkit
