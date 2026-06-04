# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DEFAULT_JSON_DIR = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_PATH = DEFAULT_JSON_DIR / "benchmark_json_summary2.txt"

METRIC_GROUPS: tuple[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], ...] = (
    (
        "structure",
        (
            ("avg_segments_per_section", ("structure", "avg_segments_per_section")),
            ("comprehensiveness", ("structure", "comprehensiveness")),
            ("logicality", ("structure", "logicality")),
        ),
    ),
    (
        "evidence",
        (
            ("coverage_ratio", ("evidence", "coverage_ratio")),
            ("accurate_count", ("evidence", "accurate_count")),
            ("citiation_density", ("evidence", "citiation_density")),
        ),
    ),
    (
        "content.segment_level",
        (
            ("insightfulness", ("content", "segment_level", "insightfulness")),
            ("readability", ("content", "segment_level", "readability")),
            ("relevance", ("content", "segment_level", "relevance")),
            ("sufficiency", ("content", "segment_level", "sufficiency")),
        ),
    ),
    (
        "content.report_level",
        (
            ("insightfulness", ("content", "report_level", "insightfulness")),
            ("readability", ("content", "report_level", "readability")),
            ("relevance", ("content", "report_level", "relevance")),
            ("sufficiency", ("content", "report_level", "sufficiency")),
        ),
    ),
)

CONTENT_DIMS = ("insightfulness", "readability", "relevance", "sufficiency")


def resolve_path(path_str: str | None, default: Path) -> Path:
    if not path_str:
        return default
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def method_name(path: Path) -> str:
    suffix = "_benchmark_results.json"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def method_and_judge_name(path: Path) -> tuple[str, str]:
    name = method_name(path)
    if "_" not in name:
        return name, "unknown"
    method, judge_llm = name.rsplit("_", 1)
    return method, judge_llm


def average(values: list[float]) -> float:
    return mean(values) if values else 0.0


def get_nested(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def collect_metric_values(results: list[dict[str, Any]], keys: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for result in results:
        value = to_float(get_nested(result, keys))
        if value is not None:
            values.append(value)
    return values


def collect_content_overall_values(results: list[dict[str, Any]], level: str) -> list[float]:
    values: list[float] = []
    for result in results:
        content = result.get("content")
        if not isinstance(content, dict):
            continue
        level_data = content.get(level)
        if not isinstance(level_data, dict):
            continue

        dim_values = []
        for dim in CONTENT_DIMS:
            value = to_float(level_data.get(dim))
            if value is not None:
                dim_values.append(value)
        if dim_values:
            values.append(average(dim_values))
    return values


def collect_combined_content_values(results: list[dict[str, Any]], dim: str) -> list[float]:
    values: list[float] = []
    for result in results:
        content = result.get("content")
        if not isinstance(content, dict):
            continue
        segment_level = content.get("segment_level")
        report_level = content.get("report_level")
        if not isinstance(segment_level, dict) or not isinstance(report_level, dict):
            continue

        segment_value = to_float(segment_level.get(dim))
        report_value = to_float(report_level.get(dim))
        if segment_value is None or report_value is None:
            continue
        values.append(0.5 * segment_value + 0.5 * report_value)
    return values


def collect_combined_content_overall_values(results: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for result in results:
        content = result.get("content")
        if not isinstance(content, dict):
            continue
        segment_level = content.get("segment_level")
        report_level = content.get("report_level")
        if not isinstance(segment_level, dict) or not isinstance(report_level, dict):
            continue

        dim_values = []
        for dim in CONTENT_DIMS:
            segment_value = to_float(segment_level.get(dim))
            report_value = to_float(report_level.get(dim))
            if segment_value is None or report_value is None:
                continue
            dim_values.append(0.5 * segment_value + 0.5 * report_value)
        if dim_values:
            values.append(average(dim_values))
    return values


def summarize_file(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", []) if isinstance(data, dict) else []
    results = [result for result in results if isinstance(result, dict)]
    method, judge_llm = method_and_judge_name(path)

    lines = [
        f"METHOD: {method}",
        f"JUDGE_LLM: {judge_llm}",
        f"FILE: {path}",
        f"TOTAL: {data.get('total', 0) if isinstance(data, dict) else 0}",
        f"SUCCESSFUL: {data.get('successful', 0) if isinstance(data, dict) else 0}",
        f"RESULTS: {len(results)}",
    ]

    for group_name, metrics in METRIC_GROUPS:
        lines.append(f"【{group_name}】")
        for label, keys in metrics:
            values = collect_metric_values(results, keys)
            if label == "accurate_count":
                lines.append(
                    f"  {label}: avg={average(values):.6f}, sum={sum(values):.0f}, count={len(values)}"
                )
            else:
                lines.append(f"  {label}: avg={average(values):.6f}, count={len(values)}")
        if group_name == "content.segment_level":
            values = collect_content_overall_values(results, "segment_level")
            lines.append(f"  overall: avg={average(values):.6f}, count={len(values)}")
        if group_name == "content.report_level":
            values = collect_content_overall_values(results, "report_level")
            lines.append(f"  overall: avg={average(values):.6f}, count={len(values)}")
        lines.append("")

    lines.append("【content】")
    for dim in CONTENT_DIMS:
        values = collect_combined_content_values(results, dim)
        lines.append(f"  {dim}: avg={average(values):.6f}, count={len(values)}")
    values = collect_combined_content_overall_values(results)
    lines.append(f"  overall: avg={average(values):.6f}, count={len(values)}")
    lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return lines


def build_summary(json_dir: Path, result_files: list[Path]) -> str:
    grouped_files: dict[str, list[Path]] = {}
    for path in result_files:
        _, judge_llm = method_and_judge_name(path)
        grouped_files.setdefault(judge_llm, []).append(path)

    lines = [
        f"JSON dir: {json_dir}",
        f"Files: {len(result_files)}",
        "",
    ]
    for judge_llm in sorted(grouped_files):
        lines.append(f"【JUDGE_LLM: {judge_llm}】")
        lines.append("")
        for path in sorted(grouped_files[judge_llm], key=lambda item: method_and_judge_name(item)[0]):
            lines.extend(summarize_file(path))
            lines.append("")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="汇总 output/*.json benchmark 结果的各维度均值。")
    parser.add_argument(
        "--json-dir",
        default=str(DEFAULT_JSON_DIR),
        help="JSON 目录，默认 output。",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="输出 txt 路径，默认 output/benchmark_json_summary.txt。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    json_dir = resolve_path(args.json_dir, DEFAULT_JSON_DIR)
    output_path = resolve_path(args.output, DEFAULT_OUTPUT_PATH)
    result_files = sorted(json_dir.glob("*_benchmark_results.json"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = build_summary(json_dir, result_files)
    output_path.write_text(output_text, encoding="utf-8")
    print(output_text)
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
