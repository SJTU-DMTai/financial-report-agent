# src/evaluation/segment_scorer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, TYPE_CHECKING

from agentscope.formatter import FormatterBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase

from prompt import prompt_dict
from utils.call_with_retry import call_chatbot_with_retry
from src.memory.working import Segment

WRITING_DIMENSIONS = {
    "comprehensiveness": "信息覆盖的广度和深度",
    "insight": "分析见解的深度和价值",
    "readability": "结构的清晰度、语言的流畅度、数据呈现的效果以及整体的易理解性",
    "relevance": "各论据、数据和陈述与论点主题的相关性",
    "sufficiency": "所有论据的充分性和支撑力"
}

async def get_content_score(model: ChatModelBase, formatter: FormatterBase, content: str, topic: str) -> Dict[str, int]:
    """
    评估给定的内容字符串。

    Args:
        content: 待评估的内容字符串

    Returns:
        Dict[str, int]: 包含四个维度的评分
    """

    # 准备评估提示
    user_prompt = f"""
    # 评估任务
    **研报片段主题:** {topic}
    **研报片段内容:**
    {content}
    """
    scores = await call_chatbot_with_retry(model, formatter,
                                           prompt_dict['eval_content'], user_prompt,
                                           hook=_extract_score,
                                           handle_hook_exceptions=(AssertionError,))
    return scores

def _extract_score(text: str) -> Dict[str, int]:
    scores_dict = {}
    for dimension in ["insight", "readability", "relevance", "sufficiency"]:
        # 提取该维度的分数
        score_match = re.search(
            rf"<{dimension}>.*?<score>(\d{1,2})</score>.*?</{dimension}>",
            text,
            re.DOTALL | re.IGNORECASE
        )
        assert score_match is not None, f"格式错误"
        scores_dict[dimension] = int(score_match.group(1))
    return scores_dict

async def evaluate_segment(model: ChatModelBase, formatter: FormatterBase, segment: Segment) -> Tuple[Dict[str, int], str]:
    """
    评估给定的 Segment。

    Args:
        segment: Segment 对象，包含 topic, requirements, reference, content 字段

    Returns:
        Dict[str, int]: 包含五个维度的评分
    """
    # 准备评估提示
    user_prompt = f"""
# 评估任务

**核心主题:** {segment.topic or '未指定'}

**人类研报片段（参考）:**
{segment.reference}

**根据参考片段总结的部分写作要求:**
{segment.requirements}

**AI Agent生成的研报片段（待评估）:**
{segment.content}
"""
    scores, suggestions = await call_chatbot_with_retry(model, formatter,
                                                        prompt_dict['compare_content_with_ref'], user_prompt,
                                                        hook=_extract_score_suggestion,
                                                        handle_hook_exceptions=(AssertionError, ))
    return scores, suggestions

def _extract_score_suggestion(text: str) -> Tuple[Dict[str, int], str | None]:
    scores_dict = {}
    suggestions_dict = {}

    dimensions = [
        "comprehensiveness",
        "insight",
        "readability",
        "relevance",
        "sufficiency"
    ]

    for dimension in dimensions:
        # 提取该维度的分数
        score_match = re.search(
            rf"<{dimension}>.*?<score>(-?1|0|1)</score>.*?</{dimension}>",
            text,
            re.DOTALL | re.IGNORECASE
        )

        scores_dict[dimension] = int(score_match.group(1)) if score_match else 0

        # 提取该维度的建议（仅当分数为 -1 时）
        if scores_dict[dimension] < 0:
            suggestion_match = re.search(
                rf"<{dimension}>.*?<suggestion>(.*?)</suggestion>.*?</{dimension}>",
                text,
                re.DOTALL | re.IGNORECASE
            )
            assert suggestion_match, '未按格式输出'
            suggestion_text = suggestion_match.group(1).strip()
            if suggestion_text:
                suggestions_dict[dimension] = suggestion_text

    suggestions = suggestions_dict if any(suggestions_dict.values()) and len(suggestions_dict) > 0 else None
    if suggestions:
        suggestions = "\n".join([f"- {WRITING_DIMENSIONS.get(dim, dim)}不足，建议: {suggestion}"
                                 for dim, suggestion in suggestions.items() if suggestions])
    return scores_dict, suggestions
