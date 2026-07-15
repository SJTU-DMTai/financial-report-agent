# -*- coding: utf-8 -*-
"""
Benchmark direct评估流程。

与 evaluation.py 保持相同评估流程，但structure/content评分不要求模型输出reason，
只要求输出score。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from agentscope.formatter import FormatterBase
from agentscope.model import ChatModelBase
from pydantic import BaseModel

from src.evaluation.eval_content import (
    ContentScore,
    _build_content_user_prompt,
    _build_report_level_content_user_prompt,
)
from src.memory.working import Section
from src.pipelines import evaluation as base
from src.prompt import prompt_dict
from src.utils.call_with_retry import call_chatbot_with_retry


class StructureScoreDirect(BaseModel):
    comprehensiveness: int | float
    logicality: int | float


def _score_output_to_text(scores) -> str:
    if hasattr(scores, "model_dump"):
        return json.dumps(scores.model_dump(), ensure_ascii=False, indent=2)
    if hasattr(scores, "dict"):
        return json.dumps(scores.dict(), ensure_ascii=False, indent=2)
    return str(scores)


def _print_score_io(kind: str, label: str, sys_prompt: str, user_prompt: str, scores) -> None:
    print(
        f"\n====== {kind} Score LLM {label} ======\n"
        f"[system]\n{sys_prompt}\n"
        f"[user]\n{user_prompt}\n"
        f"[output]\n{_score_output_to_text(scores)}\n"
        f"====== End {kind} Score LLM {label} ======",
        flush=True,
    )


async def structure_score_direct(
    report: Section,
    reference_report: Section,
    model: ChatModelBase,
    formatter: FormatterBase,
    label: str = "",
) -> Tuple[int | float, int | float]:
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
        prompt_dict["eval_structure_direct"],
        user_prompt,
        structured_model=StructureScoreDirect,
    )
    if isinstance(scores, dict):
        scores = StructureScoreDirect(**scores)
    _print_score_io("Structure", label, prompt_dict["eval_structure_direct"], user_prompt, scores)
    return scores.comprehensiveness, scores.logicality


async def get_content_score_direct(
    model: ChatModelBase,
    formatter: FormatterBase,
    content: str,
    topic: str,
    label: str = "",
) -> ContentScore:
    user_prompt = _build_content_user_prompt(content, topic)
    scores = await call_chatbot_with_retry(
        model,
        formatter,
        prompt_dict["eval_content_direct"],
        user_prompt,
        structured_model=ContentScore,
    )
    if isinstance(scores, dict):
        scores = ContentScore(**scores)
    _print_score_io("Content", label, prompt_dict["eval_content_direct"], user_prompt, scores)
    return scores


async def get_report_level_content_score_direct(
    model: ChatModelBase,
    formatter: FormatterBase,
    report_text: str,
    report_title: str = "全文",
    label: str = "[report-level]",
) -> ContentScore:
    user_prompt = _build_report_level_content_user_prompt(report_text, report_title)
    scores = await call_chatbot_with_retry(
        model,
        formatter,
        prompt_dict["eval_content_report_level_direct"],
        user_prompt,
        structured_model=ContentScore,
    )
    if isinstance(scores, dict):
        scores = ContentScore(**scores)
    _print_score_io("Content", label, prompt_dict["eval_content_report_level_direct"], user_prompt, scores)
    return scores


async def evaluate_structure(
    new_section: Section,
    human_section: Section,
    text_units: int,
    label: str = "",
) -> base.StructureMetrics:
    print(f"    - 正在评估structure指标...")
    total_segments, avg_segments_per_section = base.num_of_segment(new_section)
    segment_density = total_segments / text_units if text_units > 0 else 0.0
    comprehensiveness, logicality = await structure_score_direct(
        new_section,
        human_section,
        base.evaluation_judge_llm,
        base.evaluation_judge_formatter,
        label=label,
    )
    return base.StructureMetrics(
        total_segments=total_segments,
        avg_segments_per_section=avg_segments_per_section,
        segment_density=segment_density,
        comprehensiveness=comprehensiveness,
        logicality=logicality,
    )


async def evaluate_content(new_section: Section, report_text: str, report_title: str = "全文") -> base.ContentMetrics:
    print(f"    - 正在评估content指标...")
    segment_tasks = base._collect_segment_tasks_for_content(new_section)
    if not segment_tasks:
        raise RuntimeError(
            "新报告缺少可用于content评估的segment内容，请检查评估输入或outline解析结果。"
        )

    print(f"      - segment-level 共需评估 {len(segment_tasks)} 个segment")
    segment_score_results = await asyncio.gather(
        *[
            get_content_score_direct(
                base.evaluation_judge_llm,
                base.evaluation_judge_formatter,
                content,
                topic,
                label=f"[segment {idx}/{len(segment_tasks)}]",
            )
            for idx, (content, topic) in enumerate(segment_tasks, 1)
        ],
        return_exceptions=True,
    )
    failed_count = 0
    score_dicts = []

    for scores in segment_score_results:
        if isinstance(scores, Exception):
            failed_count += 1
            continue
        normalized_scores = base._content_score_to_dict(scores)
        if normalized_scores is None:
            failed_count += 1
            continue
        score_dicts.append(normalized_scores)

    if not score_dicts:
        raise RuntimeError("segment content评分失败，未获得任何有效的评分结果。")

    avg_scores = base._average_content_scores(score_dicts)
    segment_level = ContentScore(
        insightfulness=avg_scores["insightfulness"],
        readability=avg_scores["readability"],
        relevance=avg_scores["relevance"],
        sufficiency=avg_scores["sufficiency"],
    )

    print(f"      - segment-level 成功评估 {len(score_dicts)}/{len(segment_tasks)} 个segment")
    if failed_count > 0:
        print(f"      - segment-level 评估失败 {failed_count} 个segment")
    print(f"      - segment-level: {base._format_content_score(segment_level)}")

    print(f"      - 正在评估report-level content指标...")
    report_level_scores = await get_report_level_content_score_direct(
        base.evaluation_judge_llm,
        base.evaluation_judge_formatter,
        report_text,
        report_title=report_title,
        label=f"[report-level {report_title}]",
    )
    report_level = base._content_score_from_any(report_level_scores)
    print(f"      - report-level: {base._format_content_score(report_level)}")

    return base.ContentMetrics(segment_level=segment_level, report_level=report_level)


async def run_benchmark(
    benchmark_json_path: Path,
    new_reports_dir: Path,
    long_term_dir: Path,
    output_path: Optional[Path] = None,
):
    original_evaluate_structure = base.evaluate_structure
    original_evaluate_content = base.evaluate_content
    base.evaluate_structure = evaluate_structure
    base.evaluate_content = evaluate_content
    try:
        return await base.run_benchmark(
            benchmark_json_path=benchmark_json_path,
            new_reports_dir=new_reports_dir,
            long_term_dir=long_term_dir,
            output_path=output_path,
        )
    finally:
        base.evaluate_structure = original_evaluate_structure
        base.evaluate_content = original_evaluate_content


async def main(method_name: str, new_reports_dir: Path):
    project_root = Path(__file__).resolve().parent.parent.parent
    benchmark_json = project_root / "benchmark.json"
    long_term = project_root / "data" / "memory" / "long_term"
    output_dir = project_root / "output"
    evaluator_llm_name = base.MODEL_CONFIG.llm_name
    safe_method_name = re.sub(r'[<>:"/\\|?*\s]+', "_", method_name.strip())
    safe_evaluator_llm_name = re.sub(r'[<>:"/\\|?*\s]+', "_", evaluator_llm_name.strip())
    output = output_dir / f"{safe_method_name}_{safe_evaluator_llm_name}_benchmark_results.json"

    summary = await run_benchmark(
        benchmark_json_path=benchmark_json,
        new_reports_dir=new_reports_dir,
        long_term_dir=long_term,
        output_path=output,
    )

    if summary:
        base.print_benchmark_summary(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 direct benchmark 测试")
    parser.add_argument(
        "--new_reports_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="qwen3-32b",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    new_reports_path = (
        Path(args.new_reports_path)
        if args.new_reports_path
        else project_root / "output" / "reports" / args.method_name
    )
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(args.method_name, new_reports_path))
