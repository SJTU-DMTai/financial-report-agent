import json
from typing import Tuple

from agentscope.formatter import FormatterBase
from agentscope.model import ChatModelBase
from pydantic import BaseModel

from src.memory.working import Section
from src.prompt import prompt_dict
from src.utils.call_with_retry import call_chatbot_with_retry


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


class StructureDimensionScore(BaseModel):
    score: int | float
    reason: str


class StructureScore(BaseModel):
    comprehensiveness: StructureDimensionScore
    logicality: StructureDimensionScore


def _structure_score_output_to_text(scores: StructureScore) -> str:
    if hasattr(scores, "model_dump"):
        return json.dumps(scores.model_dump(), ensure_ascii=False, indent=2)
    if hasattr(scores, "dict"):
        return json.dumps(scores.dict(), ensure_ascii=False, indent=2)
    return str(scores)


def _print_structure_score_io(label: str, sys_prompt: str, user_prompt: str, scores: StructureScore) -> None:
    print(
        f"\n====== Structure Score LLM {label} ======\n"
        f"[system]\n{sys_prompt}\n"
        f"[user]\n{user_prompt}\n"
        f"[output]\n{_structure_score_output_to_text(scores)}\n"
        f"====== End Structure Score LLM {label} ======",
        flush=True,
    )


async def structure_score(
    report: Section,
    reference_report: Section,
    model: ChatModelBase,
    formatter: FormatterBase,
    label: str = "",
) -> Tuple[int | float, int | float]:
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
        model,
        formatter,
        prompt_dict["eval_structure"],
        user_prompt,
        structured_model=StructureScore
    )
    if isinstance(scores, dict):
        scores = StructureScore(**scores)
    _print_structure_score_io(label, prompt_dict["eval_structure"], user_prompt, scores)
    comprehensiveness = scores.comprehensiveness.score
    logicality = scores.logicality.score
    return comprehensiveness, logicality
