# -*- coding: utf-8 -*-
from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit
from agentscope.model import DashScopeChatModel

from ..memory.working import Section
from ..tools.outline_tools import *
from ..tools.search_tools import *
from ..memory.short_term import ShortTermMemoryStore
from ..memory.working_memory import SlidingWindowMemory
from ..prompt import prompt_dict

import config
cfg = config.Config()
planner_cfg = cfg.get_planner_cfg()
use_demo = planner_cfg.get("use_demonstration", False)

def create_planner_agent(
    model,
    formatter,
    toolkit: Toolkit,
) -> ReActAgent:
    sys_prompt_key = 'plan_outline' if use_demo else 'planner_sys_prompt'
    return ReActAgent(
        name="Planner",
        sys_prompt=prompt_dict[sys_prompt_key],
        model=model,
        # memory=SlidingWindowMemory(),
        formatter=formatter,
        toolkit=toolkit,
        parallel_tool_calls=False,
        print_hint_msg=True,
        max_iters=100,
    )

    

# ---- Toolkit Builder ----
def build_planner_toolkit(
    manuscript: Section,
    searcher: ReActAgent,
) -> Toolkit:
    """创建 Planner 专用 Toolkit。"""
    toolkit = Toolkit()

    outline_tools = OutlineTools(manuscript=manuscript)
    # toolkit.register_tool_function(outline_tools.read_outline)
    if use_demo:
        toolkit.register_tool_function(outline_tools.read_demonstration)
    toolkit.register_tool_function(outline_tools.replace_outline)

    search_tools = SearchTools(manuscript=manuscript)
    toolkit.register_tool_function(search_tools.searcher_tool(searcher))
    return toolkit
