# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "output" / "debug"
DEFAULT_OUTPUT_PATH = DEFAULT_DEBUG_DIR / "content_debug_summary.txt"

LEVELS = ("segment_level", "report_level", "section_level")
DIMS = ("insightfulness", "readability", "relevance", "sufficiency")


def _resolve_path(path_str: str | None, default: Path) -> Path:
    if not path_str:
        return default
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _method_name(path: Path) -> str:
    suffix = "_content_debug_results.json"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def _avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _level_dims(level_data: dict, level: str) -> dict[str, float]:
    if level == "report_level":
        score = level_data.get("score", {}) or {}
        return {
            dim: float((score.get(dim) or {}).get("score", 0.0))
            for dim in DIMS
        }
    average = level_data.get("average", {}) or {}
    return {dim: float(average.get(dim, 0.0)) for dim in DIMS}


def _summarize_file(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", []) or []
    lines = [
        f"METHOD: {_method_name(path)}",
        f"FILE: {path}",
        f"RESULTS: {len(results)}",
    ]

    for level in LEVELS:
        overalls: list[float] = []
        dim_values: dict[str, list[float]] = {dim: [] for dim in DIMS}
        counts: list[int] = []
        failed_counts: list[int] = []

        for result in results:
            level_data = ((result.get("content") or {}).get(level) or {})
            overalls.append(float(level_data.get("overall", 0.0)))
            for dim, value in _level_dims(level_data, level).items():
                dim_values[dim].append(value)

            if level == "segment_level":
                counts.append(int(level_data.get("segment_count", 0)))
                failed_counts.append(int(level_data.get("failed_count", 0)))
            elif level == "section_level":
                counts.append(int(level_data.get("section_count", 0)))

        dim_text = ", ".join(
            f"{dim}={_avg(values):.4f}"
            for dim, values in dim_values.items()
        )
        line = f"  {level}: overall={_avg(overalls):.4f}, {dim_text}"
        if counts:
            line += f", avg_count={_avg([float(v) for v in counts]):.2f}"
        if failed_counts:
            line += f", failed_sum={sum(failed_counts)}"
        lines.append(line)

    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="汇总 output/debug 下 content debug 评估结果均值。"
    )
    parser.add_argument(
        "--debug-dir",
        default=str(DEFAULT_DEBUG_DIR),
        help="debug 目录，默认 output/debug。",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="输出 txt 路径，默认 output/debug/content_debug_summary.txt。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    debug_dir = _resolve_path(args.debug_dir, DEFAULT_DEBUG_DIR)
    output_path = _resolve_path(args.output, DEFAULT_OUTPUT_PATH)
    result_files = sorted(debug_dir.glob("*_content_debug_results.json"))

    lines = [
        f"Debug dir: {debug_dir}",
        f"Files: {len(result_files)}",
        "",
    ]
    for path in result_files:
        lines.extend(_summarize_file(path))
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(output_text, encoding="utf-8")
    print(output_text)
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
