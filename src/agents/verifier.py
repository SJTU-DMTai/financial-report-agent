# -*- coding: utf-8 -*-
from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter
from agentscope.tool import Toolkit

from ..tools.material_tools import MaterialTools
from ..tools.search_tools import SearchTools
from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import LongTermMemoryStore
from ..prompt import prompt_dict


def create_verifier_agent(
    model,
    formatter,
    toolkit: Toolkit,
    verifier_type: str = "numeric",  # 默认创建数值一致性Verifier
) -> ReActAgent:
    """
    verifier_type 可选值：
    - fact: 事实核查
    - numeric: 数值核查
    - temporal: 时间核查
    """
    sys_prompt_map = {
        "fact": prompt_dict['verifier_fact_prompt'],
        "numeric": prompt_dict['verifier_numeric_prompt'],
        "temporal": prompt_dict['verifier_temporal_prompt'],
    }

    sys_prompt = sys_prompt_map.get(verifier_type)
    if sys_prompt is None:
        raise ValueError(f"未知 verifier_type: {verifier_type}")

    return ReActAgent(
        name=f"Verifier-{verifier_type}",
        sys_prompt=sys_prompt,
        model=model,
        memory=InMemoryMemory(),
        formatter=formatter,
        toolkit=toolkit,
        parallel_tool_calls=False,
        max_iters=10,
    )


# ---- Toolkit Builder ----
def build_verifier_toolkit(
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
) -> Toolkit:
    toolkit = Toolkit()

    # manuscript_tools = ManuscriptTools(short_term=short_term)
    # toolkit.register_tool_function(manuscript_tools.read_manuscript_section)
    # toolkit.register_tool_function(manuscript_tools.count_manuscript_words)
    material_tools = MaterialTools(short_term=short_term, long_term=long_term)
    search_tools = SearchTools(short_term=short_term, long_term=long_term)
    toolkit.register_tool_function(material_tools.read_material)
    return toolkit


# ---- 工厂函数：创建三路 Verifier ----
def create_three_verifiers(model, formatter, short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore):
    verifiers = {}
    for name in ["fact", "numeric", "temporal"]:
        toolkit = build_verifier_toolkit(short_term, long_term)
        verifiers[name] = create_verifier_agent(
            model, formatter, toolkit, verifier_type=name
        )
    return verifiers