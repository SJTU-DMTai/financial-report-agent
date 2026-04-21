from typing import Tuple

from pydantic import BaseModel

from src.memory.working import Section
from src.prompt import prompt_dict
from src.utils.call_with_retry import call_chatbot_with_retry

from src.utils.instance import llm_reasoning, formatter


def num_of_segment(report: Section) -> Tuple[int, float]:
    """
    统计一篇研报中所有segment数量，以及每个section平均segment数量。

    Args:
        report: 代表整篇研报的根 Section 对象。

    Returns:
        一个元组 (total_segments, avg_segments_per_section)。
    """
    total_segments = 0
    total_sections = 0
    pending_sections = [report] if report else []

    while pending_sections:
        section = pending_sections.pop()
        total_sections += 1

        if section.segments:
            total_segments += len(section.segments)

        if section.subsections:
            for subsection in section.subsections:
                pending_sections.append(subsection)

    if total_sections == 0:
        avg_segments_per_section = 0.0
    else:
        avg_segments_per_section = round(total_segments / total_sections, 1)

    return total_segments, avg_segments_per_section


class StructureScore(BaseModel):
    comprehensiveness: int
    logicality: int

async def structure_score(report: Section, reference_report: Section) -> Tuple[int, int]:
    """
    调用LLM基于参考研报评估新研报结构的完整性和逻辑性。

    Args:
        report: 待评估研报的根 Section 对象。
        reference_report: 参考研报的根 Section 对象。

    Returns:
        一个元组 (comprehensiveness_score, logicality_score)。
    """
    outline = report.read(
        with_requirements=False,
        with_content=False,
        with_evidence=False,
        with_reference=False,
        read_subsections=True,
    )
    reference_outline = reference_report.read(
        with_requirements=False,
        with_content=False,
        with_evidence=False,
        with_reference=False,
        read_subsections=True,
    )

    user_prompt = (
        f"**参考的专家研报大纲（human_report）：**\n---\n{reference_outline}\n---\n\n"
        f"**待评估的研报大纲：**\n---\n{outline}\n---"
    )

    scores = await call_chatbot_with_retry(
        llm_reasoning,
        formatter,
        prompt_dict["eval_structure"],
        user_prompt,
        structured_model=StructureScore
    )
    comprehensiveness = scores.get("comprehensiveness")
    logicality = scores.get("logicality")
    return comprehensiveness, logicality
