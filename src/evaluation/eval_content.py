# src/evaluation/segment_scorer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import json
import traceback
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, TYPE_CHECKING

from agentscope.formatter import FormatterBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase

from src.prompt import prompt_dict
from src.utils.call_with_retry import call_chatbot_with_retry
from src.memory.working import Segment, Section
from pydantic import BaseModel

WRITING_DIMENSIONS = {
    "comprehensiveness": "信息覆盖的广度和深度",
    "insight": "分析见解的深度和价值",
    "readability": "结构的清晰度、语言的流畅度、数据呈现的效果以及整体的易理解性",
    "relevance": "各论据、数据和陈述与论点主题的相关性",
    "sufficiency": "所有论据的充分性和支撑力"
}

class ContentScore(BaseModel):
    insightfulness: int | float
    readability: int | float
    relevance: int | float
    sufficiency: int | float

def _build_content_user_prompt(content: str, topic: str) -> str:
    return f"""# 评估任务
**研报片段主题:** {topic}
**研报片段内容:**
{content}

请你仔细思考分析后进行打分。"""


async def get_content_score(
    model: ChatModelBase,
    formatter: FormatterBase,
    content: str,
    topic: str,
    label: str = "",
) -> ContentScore:
    """
    评估给定的内容字符串。

    Args:
        content: 待评估的内容字符串

    Returns:
        Dict[str, int]: 包含四个维度的评分
    """
    user_prompt = _build_content_user_prompt(content, topic)
    scores = await call_chatbot_with_retry(model, formatter,
                                           prompt_dict['eval_content'], user_prompt,
                                           structured_model=ContentScoreWithReasons)
    _print_content_score_io(label, prompt_dict['eval_content'], user_prompt, scores)
    return _extract_content_scores(scores)


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

class Comparison(BaseModel):
    analysis: str
    score: int
    suggestion: Optional[str] = None

class SegmentScore(BaseModel):
    comprehensiveness: Comparison
    insightfulness: Comparison
    readability: Comparison
    relevance: Comparison
    sufficiency: Comparison

async def evaluate_segment(model: ChatModelBase, formatter: FormatterBase, segment: Segment) -> str:
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
    return await call_chatbot_with_retry(model, formatter,
                                         prompt_dict['compare_content_with_ref'], user_prompt,
                                         hook=_extract_score_suggestion,
                                         handle_hook_exceptions=(AssertionError, KeyError))

def _extract_score_suggestion(text: str) -> str:
    scores_dict = {}
    suggestions_dict = {}

    dimensions = [
        "comprehensiveness",
        "insightfulness",
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
    return suggestions

# add ContentScore reson; report-level & section level evaluation 

class ContentDimensionScore(BaseModel):
    score: int | float
    reason: str


class ContentScoreWithReasons(BaseModel):
    insightfulness: ContentDimensionScore
    readability: ContentDimensionScore
    relevance: ContentDimensionScore
    sufficiency: ContentDimensionScore


def _content_score_output_to_text(scores) -> str:
    if hasattr(scores, "model_dump"):
        return json.dumps(scores.model_dump(), ensure_ascii=False, indent=2)
    if hasattr(scores, "dict"):
        return json.dumps(scores.dict(), ensure_ascii=False, indent=2)
    return str(scores)


def _print_content_score_io(label: str, sys_prompt: str, user_prompt: str, scores) -> None:
    output_text = _content_score_output_to_text(scores)
    print(
        f"\n====== Content Score LLM {label} ======\n"
        f"[system]\n{sys_prompt}\n"
        f"[user]\n{user_prompt}\n"
        f"[output]\n{output_text}\n"
        f"====== End Content Score LLM {label} ======",
        flush=True,
    )


def _extract_content_scores(scores: ContentScoreWithReasons) -> ContentScore:
    return ContentScore(
        insightfulness=scores.insightfulness.score,
        readability=scores.readability.score,
        relevance=scores.relevance.score,
        sufficiency=scores.sufficiency.score,
    )


async def get_content_score_with_reasons(
    model: ChatModelBase,
    formatter: FormatterBase,
    content: str,
    topic: str,
    label: str = "",
) -> ContentScoreWithReasons:
    user_prompt = _build_content_user_prompt(content, topic)
    scores = await call_chatbot_with_retry(model, formatter,
                                           prompt_dict['eval_content'], user_prompt,
                                           structured_model=ContentScoreWithReasons)
    _print_content_score_io(label, prompt_dict['eval_content'], user_prompt, scores)
    return scores


class SectionContentScore(BaseModel):
    section_id: int
    level: int
    title: str
    content: ContentScoreWithReasons


def _build_report_level_content_user_prompt(report_text: str, report_title: str = "全文") -> str:
    return f"""# 整篇研报content评估任务
**研报标题:** {report_title}
**研报全文:**
{report_text}

请你从整篇研报层面仔细思考分析后进行打分。"""


def _build_section_level_content_user_prompt(section_text: str, section_title: str) -> str:
    return f"""# 章节content评估任务
**章节标题:** {section_title}
**章节内容:**
{section_text}

请你从该章节层面仔细思考分析后进行打分。"""


def _append_section_content_lines(section: Section, lines: list[str], read_subsections: bool) -> None:
    title = (section.title or "").strip()
    if title:
        lines.append(f"{'#' * max(section.level, 1)} {title}")

    if section.content and section.content.strip():
        lines.append(section.content.strip())
    else:
        for segment in section.segments or []:
            segment_text = segment.content or segment.reference
            if segment_text and segment_text.strip():
                lines.append(segment_text.strip())

    if read_subsections:
        for subsection in section.subsections or []:
            _append_section_content_lines(subsection, lines, read_subsections=True)


def _section_to_content_text(section: Section, read_subsections: bool) -> str:
    lines: list[str] = []
    _append_section_content_lines(section, lines, read_subsections)
    return "\n\n".join(lines).strip()


def _section_has_content_for_scoring(section: Section) -> bool:
    if section.content and section.content.strip():
        return True
    if any((segment.content or segment.reference or "").strip() for segment in section.segments or []):
        return True
    return any(_section_has_content_for_scoring(subsection) for subsection in section.subsections or [])


def _flatten_sections_for_content(section: Section) -> list[Section]:
    sections: list[Section] = []
    pending = list(reversed(section.subsections or []))

    while pending:
        current = pending.pop()
        sections.append(current)
        for subsection in reversed(current.subsections or []):
            pending.append(subsection)

    return sections


def _select_sections_for_content(section: Section, include_root: bool = False) -> list[Section]:
    candidates = [
        current
        for current in _flatten_sections_for_content(section)
        if _section_has_content_for_scoring(current)
    ]
    level_2_sections = [current for current in candidates if current.level == 2]
    if len(level_2_sections) >= 3:
        selected = level_2_sections
    else:
        selected = [current for current in candidates if current.level == 3]

    if not selected:
        selected = candidates
    if include_root and _section_has_content_for_scoring(section):
        return [section] + selected
    return selected


def _report_input_to_title_and_text(report: Section | str, report_title: str = "全文") -> tuple[str, str]:
    if isinstance(report, Section):
        return report.title or report_title, _section_to_content_text(report, read_subsections=True)
    return report_title, str(report).strip()


def _collect_sections_for_content(section: Section, include_root: bool = False) -> list[tuple[Section, str]]:
    tasks: list[tuple[Section, str]] = []
    for current in _select_sections_for_content(section, include_root=include_root):
        section_text = _section_to_content_text(current, read_subsections=True)
        if section_text:
            tasks.append((current, section_text))

    return tasks


async def get_report_level_content_score(
    model: ChatModelBase,
    formatter: FormatterBase,
    report: Section | str,
    report_title: str = "全文",
    label: str = "[report-level]",
) -> ContentScoreWithReasons:
    title, report_text = _report_input_to_title_and_text(report, report_title)
    user_prompt = _build_report_level_content_user_prompt(report_text, title)
    scores = await call_chatbot_with_retry(
        model,
        formatter,
        prompt_dict["eval_content_report_level"],
        user_prompt,
        structured_model=ContentScoreWithReasons,
    )
    _print_content_score_io(label, prompt_dict["eval_content_report_level"], user_prompt, scores)
    return scores


async def get_section_level_content_score(
    model: ChatModelBase,
    formatter: FormatterBase,
    section_title: str,
    section_text: str,
    label: str = "[section-level]",
) -> ContentScoreWithReasons:
    user_prompt = _build_section_level_content_user_prompt(section_text, section_title)
    scores = await call_chatbot_with_retry(
        model,
        formatter,
        prompt_dict["eval_content_section_level"],
        user_prompt,
        structured_model=ContentScoreWithReasons,
    )
    _print_content_score_io(label, prompt_dict["eval_content_section_level"], user_prompt, scores)
    return scores


async def get_section_level_content_scores(
    model: ChatModelBase,
    formatter: FormatterBase,
    report: Section,
    include_root: bool = False,
) -> list[SectionContentScore]:
    section_tasks = _collect_sections_for_content(report, include_root=include_root)
    results: list[SectionContentScore] = []
    total = len(section_tasks)

    for idx, (section, section_text) in enumerate(section_tasks, 1):
        scores = await get_section_level_content_score(
            model,
            formatter,
            section.title,
            section_text,
            label=f"[section-level {idx}/{total}]",
        )
        results.append(
            SectionContentScore(
                section_id=section.section_id,
                level=section.level,
                title=section.title,
                content=scores,
            )
        )

    return results
