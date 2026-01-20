# -*- coding: utf-8 -*-
from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter
from agentscope.tool import Toolkit

from ..tools.material_tools import MaterialTools
from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import LongTermMemoryStore
from ..prompt import prompt_dict


def create_verifier_agent(
    model,
    formatter,
    toolkit: Toolkit,
    multi_source_verification : bool = False,
    verifier_type: str = "numeric",  # 默认创建数值一致性Verifier
) -> ReActAgent:
    """
    verifier_type 可选值：
    - numeric: 数值一致性
    - reference: 引用正确性
    - logic: 逻辑与语言
    - quality: 写作水平与参考对比
    - final: 最终质量审核
    """
    sys_prompt_map = {
        "numeric": prompt_dict['verifier_numeric_prompt'],
        "reference": prompt_dict['verifier_reference_prompt'],
        "logic": prompt_dict['verifier_logic_prompt'],
        "quality": prompt_dict['verifier_quality_prompt'],
        "final": prompt_dict['verifier_final_check'],
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
        max_iters=15,
    )


# ---- Toolkit Builder ----
def build_verifier_toolkit(
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    multi_source_verification : bool = False,
) -> Toolkit:
    toolkit = Toolkit()

    # manuscript_tools = ManuscriptTools(short_term=short_term)

    # toolkit.register_tool_function(manuscript_tools.read_manuscript_section)
    # toolkit.register_tool_function(manuscript_tools.count_manuscript_words)
    material_tools = MaterialTools(short_term=short_term, long_term=long_term)
    toolkit.register_tool_function(material_tools.read_material)

    if multi_source_verification:
        toolkit.create_tool_group(
            group_name="multi_source_search",
            description="多源交叉验证֤",
            active=False,
        )
        toolkit.register_tool_function(material_tools.fetch_stock_news_material, group_name="multi_source_search")
        toolkit.register_tool_function(material_tools.fetch_disclosure_material, group_name="multi_source_search")
        # toolkit.create_tool_group(
        #     group_name="numeric_consistency",
        #     description="校验数值一致性",
        #     active=False,
        # )
    return toolkit


# ---- 工厂函数：创建四个环节 Verifier ----
def create_all_verifiers(model, formatter, short_term):
    verifiers = {}
    for verifier_name in ["numeric", "reference", "logic", "quality"]:
        # 每个 verifier 用独立 toolkit
        toolkit = build_verifier_toolkit(short_term)
        verifiers[verifier_name] = create_verifier_agent(model, formatter, toolkit, verifier_name)

    return verifiers


# ---- 创建最终审核 Verifier ----
def create_final_verifier(model, formatter, short_term: ShortTermMemoryStore):
    toolkit = build_verifier_toolkit(short_term)
    return create_verifier_agent(model, formatter, toolkit, "final")