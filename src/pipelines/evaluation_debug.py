# -*- coding: utf-8 -*-
"""
Content-only benchmark debug evaluation.

- 读取 benchmark.json 配置
- 配对新研报与参考研报
- 只评估 content，不评估 structure/evidence
- 同时输出 segment-level、section-level、report-level content 评分
"""
import argparse
import asyncio
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.eval_content import (
    ContentScoreWithReasons,
    SectionContentScore,
    get_content_score_with_reasons,
    get_report_level_content_score,
    get_section_level_content_scores,
)
from src.pipelines.evaluation import (
    BenchmarkItem,
    _build_pair_cache_key,
    _collect_segment_tasks_for_content,
    _load_json_file,
    _resolve_report_paths,
    _sanitize_section_for_scoring,
    _write_summary,
)
from src.pipelines.planning import process_pdf_to_outline
from src.utils.instance import cfg as MODEL_CONFIG, formatter, llm_instruct, llm_reasoning


def _build_debug_results_path(output_dir: Path, method_name: str, evaluator_llm_name: str) -> Path:
    safe_method_name = re.sub(r'[<>:"/\\|?*\s]+', "_", method_name.strip())
    safe_evaluator_llm_name = re.sub(r'[<>:"/\\|?*\s]+', "_", evaluator_llm_name.strip())
    return output_dir / "debug" / f"{safe_method_name}_{safe_evaluator_llm_name}_content_debug_results.json"


def _score_with_reasons_to_dict(score: ContentScoreWithReasons) -> Dict[str, Dict[str, Any]]:
    if hasattr(score, "model_dump"):
        return score.model_dump()
    return score.dict()


def _score_with_reasons_to_plain_dict(score: ContentScoreWithReasons) -> Dict[str, float]:
    return {
        "insightfulness": float(score.insightfulness.score),
        "readability": float(score.readability.score),
        "relevance": float(score.relevance.score),
        "sufficiency": float(score.sufficiency.score),
    }


def _average_score_dicts(score_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    avg_scores = {
        "insightfulness": 0.0,
        "readability": 0.0,
        "relevance": 0.0,
        "sufficiency": 0.0,
    }
    if not score_dicts:
        return avg_scores

    for dim in avg_scores:
        values = [score[dim] for score in score_dicts if dim in score]
        avg_scores[dim] = sum(values) / len(values) if values else 0.0
    return avg_scores


def _get_content_average(content: Dict[str, float]) -> float:
    values = [
        content.get("insightfulness", 0.0),
        content.get("readability", 0.0),
        content.get("relevance", 0.0),
        content.get("sufficiency", 0.0),
    ]
    return sum(values) / len(values) if values else 0.0


def _section_scores_to_dicts(section_scores: List[SectionContentScore]) -> List[Dict[str, Any]]:
    results = []
    for section_score in section_scores:
        results.append(
            {
                "section_id": section_score.section_id,
                "level": section_score.level,
                "title": section_score.title,
                "content": _score_with_reasons_to_dict(section_score.content),
            }
        )
    return results


def _section_scores_to_plain_dicts(section_scores: List[SectionContentScore]) -> List[Dict[str, float]]:
    return [
        _score_with_reasons_to_plain_dict(section_score.content)
        for section_score in section_scores
    ]


def _segment_scores_to_dicts(
    segment_tasks: List[Tuple[str, str]],
    segment_scores: List[ContentScoreWithReasons],
) -> List[Dict[str, Any]]:
    results = []
    for idx, ((content, topic), score) in enumerate(zip(segment_tasks, segment_scores), 1):
        results.append(
            {
                "segment_index": idx,
                "topic": topic,
                "text_length": len(content),
                "content": _score_with_reasons_to_dict(score),
            }
        )
    return results


async def _evaluate_segment_level_content_with_reasons(
    segment_tasks: List[Tuple[str, str]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], int]:
    segment_score_results = await asyncio.gather(
        *[
            get_content_score_with_reasons(
                llm_reasoning,
                formatter,
                content,
                topic,
                label=f"[segment {idx}/{len(segment_tasks)}]",
            )
            for idx, (content, topic) in enumerate(segment_tasks, 1)
        ],
        return_exceptions=True,
    )

    failed_count = 0
    valid_scores: List[ContentScoreWithReasons] = []
    plain_scores: List[Dict[str, float]] = []
    valid_tasks: List[Tuple[str, str]] = []

    for task, score in zip(segment_tasks, segment_score_results):
        if isinstance(score, Exception):
            failed_count += 1
            continue
        valid_scores.append(score)
        plain_scores.append(_score_with_reasons_to_plain_dict(score))
        valid_tasks.append(task)

    return _average_score_dicts(plain_scores), _segment_scores_to_dicts(valid_tasks, valid_scores), failed_count


def _build_summary(
    benchmark_items: List[BenchmarkItem],
    results: List[Dict[str, Any]],
    successful: int,
    failed: int,
    none_count: int,
    failure_reasons: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "total": len(benchmark_items),
        "evaluator_llm_name": MODEL_CONFIG.llm_name,
        "successful": successful,
        "failed": failed,
        "NONE": none_count,
        "results": results,
        "failure_reasons": failure_reasons,
        "timestamp": datetime.now().isoformat(),
    }


def _load_existing_result_cache(output_path: Path) -> Dict[str, Dict[str, Any]]:
    existing_result_cache: Dict[str, Dict[str, Any]] = {}
    cached_summary = _load_json_file(output_path)
    if not isinstance(cached_summary, dict) or not isinstance(cached_summary.get("results"), list):
        return existing_result_cache

    for result in cached_summary["results"]:
        cache_key = _build_pair_cache_key(
            result["stock_code"],
            result["date"],
            result["human_report_name"],
            result["new_report_name"],
        )
        if cache_key not in existing_result_cache:
            existing_result_cache[cache_key] = result
    print(f"✓ 已加载当前评测LLM content debug缓存: {output_path}")
    return existing_result_cache


async def benchmark_single_pair_content_debug(
    stock_code: str,
    date: str,
    new_report_path: Path,
    human_report_path: Path,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"正在评估content debug: {stock_code} ({date})")
    print(f"{'='*60}")

    print("[1/3] 处理报告...")
    new_section = await process_pdf_to_outline(
        new_report_path,
        new_report_path.parent,
        llm_reasoning,
        llm_instruct,
        formatter,
        only_evidence=True,
        reuse_other_model_cache=True,
    )
    sanitized_new_section = _sanitize_section_for_scoring(new_section)

    print("[2/3] 评估content指标...")
    segment_tasks = _collect_segment_tasks_for_content(sanitized_new_section)
    if not segment_tasks:
        raise RuntimeError("新报告缺少可用于content评估的segment内容，请检查评估输入或outline解析结果。")
    print(f"    - segment-level 共需评估 {len(segment_tasks)} 个segment")
    segment_level_dict, segment_level_details, segment_failed_count = await _evaluate_segment_level_content_with_reasons(segment_tasks)
    print(f"    - segment-level 成功评估 {len(segment_level_details)}/{len(segment_tasks)} 个segment")
    if segment_failed_count > 0:
        print(f"    - segment-level 评估失败 {segment_failed_count} 个segment")

    print("    - 正在评估report-level content指标...")
    report_text = new_report_path.read_text(encoding="utf-8")
    report_level = await get_report_level_content_score(
        llm_reasoning,
        formatter,
        report_text,
        report_title=sanitized_new_section.title or new_report_path.stem,
        label=f"[report-level {stock_code} {date}]",
    )

    print("    - 正在评估section-level content指标...")
    section_level = await get_section_level_content_scores(
        llm_reasoning,
        formatter,
        sanitized_new_section,
        include_root=False,
    )

    report_level_dict = _score_with_reasons_to_dict(report_level)
    report_level_plain = _score_with_reasons_to_plain_dict(report_level)
    section_level_dicts = _section_scores_to_dicts(section_level)
    section_level_plain = _section_scores_to_plain_dicts(section_level)
    section_level_average = _average_score_dicts(section_level_plain)

    print("[3/3] 汇总结果...")
    return {
        "stock_code": stock_code,
        "date": date,
        "human_report_name": human_report_path.name,
        "new_report_name": new_report_path.name,
        "content": {
            "segment_level": {
                "average": segment_level_dict,
                "overall": _get_content_average(segment_level_dict),
                "segment_count": len(segment_tasks),
                "failed_count": segment_failed_count,
                "segments": segment_level_details,
            },
            "report_level": {
                "score": report_level_dict,
                "overall": _get_content_average(report_level_plain),
            },
            "section_level": {
                "average": section_level_average,
                "overall": _get_content_average(section_level_average),
                "section_count": len(section_level_dicts),
                "sections": section_level_dicts,
            },
        },
        "timestamp": datetime.now().isoformat(),
    }


async def run_content_debug_benchmark(
    benchmark_json_path: Path,
    new_reports_dir: Path,
    output_path: Path,
) -> Optional[Dict[str, Any]]:
    if not benchmark_json_path.exists():
        print(f"✗ 错误: benchmark.json 不存在: {benchmark_json_path}")
        return None

    if not new_reports_dir.exists():
        print(f"✗ 错误: 新研报目录不存在: {new_reports_dir}")
        return None

    print(f"\n读取benchmark配置: {benchmark_json_path}")
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    if not isinstance(benchmark_data, list):
        print(f"✗ 错误: benchmark.json应为数组，实际为 {type(benchmark_data)}")
        return None

    benchmark_items = [BenchmarkItem(**item) for item in benchmark_data]
    print(f"✓ 读取了 {len(benchmark_items)} 个benchmark条目\n")

    results: List[Dict[str, Any]] = []
    successful = 0
    failed = 0
    none_count = 0
    failure_reasons: Dict[str, str] = {}
    existing_result_cache = _load_existing_result_cache(output_path)
    pending_pairs = []

    for idx, item in enumerate(benchmark_items, 1):
        ref_path, new_report_path, missing_type, missing_detail = _resolve_report_paths(item, new_reports_dir)

        if missing_type == "human_report":
            print(f"[{idx}/{len(benchmark_items)}] 处理 {item.stock_code} ({item.date})")
            print(f"  ✗ 参考报告不存在: {item.human_report}")
            none_count += 1
            failure_reasons[f"human_report_NONE_{item.stock_code}_{item.date}"] = item.human_report
            continue

        if missing_type == "new_report":
            print(f"[{idx}/{len(benchmark_items)}] 处理 {item.stock_code} ({item.date})")
            print(f"  ✗ 未找到新报告（搜索模式: {missing_detail}）")
            none_count += 1
            failure_reasons[f"new_report_NONE_{item.stock_code}_{item.date}"] = missing_detail
            continue

        cache_key = _build_pair_cache_key(item.stock_code, item.date, ref_path.name, new_report_path.name)
        cached_result = existing_result_cache.get(cache_key)
        if cached_result is not None:
            print(f"[{idx}/{len(benchmark_items)}] 处理 {item.stock_code} ({item.date})")
            print(f"  ✓ 复用content debug缓存: {new_report_path.name} vs {ref_path.name}")
            results.append(cached_result)
            successful += 1
            continue

        pending_pairs.append((idx, item, ref_path, new_report_path))

    _write_summary(output_path, _build_summary(benchmark_items, results, successful, failed, none_count, failure_reasons))

    for idx, item, ref_path, new_report_path in pending_pairs:
        print(f"[{idx}/{len(benchmark_items)}] 处理 {item.stock_code} ({item.date})")
        try:
            result = await benchmark_single_pair_content_debug(
                stock_code=item.stock_code,
                date=item.date,
                new_report_path=new_report_path,
                human_report_path=ref_path,
            )
            results.append(result)
            successful += 1
        except Exception as e:
            traceback.print_exc()
            failed += 1
            failure_reasons[f"content_debug_failed_{item.stock_code}_{item.date}"] = str(e)
        finally:
            _write_summary(output_path, _build_summary(benchmark_items, results, successful, failed, none_count, failure_reasons))

    summary = _build_summary(benchmark_items, results, successful, failed, none_count, failure_reasons)
    _write_summary(output_path, summary)
    print(f"\n✓ content debug结果已保存到: {output_path}")
    return summary


async def main(method_name: str, new_reports_dir: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    benchmark_json = project_root / "benchmark.json"
    output_path = _build_debug_results_path(project_root / "output", method_name, MODEL_CONFIG.llm_name)
    await run_content_debug_benchmark(
        benchmark_json_path=benchmark_json,
        new_reports_dir=new_reports_dir,
        output_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 content-only benchmark debug 评估")
    parser.add_argument("--new_reports_path", type=str, default="")
    parser.add_argument("--method_name", type=str, default="qwen3-32b")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    reports_path = Path(args.new_reports_path) if args.new_reports_path else root / "output" / "reports" / args.method_name

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(args.method_name, reports_path))
