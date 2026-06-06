# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import ast
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

import config
from src.memory.long_term import LongTermMemoryStore
from src.utils.get_entity_info import get_entity_info
from src.utils.task_date import normalize_compact_date, resolve_cur_date


PROJECT_ROOT = Path(__file__).resolve().parent


def read_main_task_desc(main_path: Path) -> str:
    tree = ast.parse(main_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "task_desc":
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
    raise ValueError(f"未能在 {main_path} 中找到字符串形式的 task_desc 赋值。")


def resolve_stock_code(task_desc: str, stock_code: str | None) -> str:
    if stock_code:
        return str(stock_code).strip().zfill(6)

    long_term = LongTermMemoryStore(base_dir=PROJECT_ROOT / "data" / "memory" / "long_term")
    entity = get_entity_info(long_term, task_desc)
    if not entity or not entity.get("code"):
        raise ValueError(f"无法从 task_desc 解析股票实体/代码：{task_desc}")
    return str(entity["code"]).strip().zfill(6)


def resolve_run_date(task_desc: str, date: str | None) -> str:
    return normalize_compact_date(date) if date else resolve_cur_date(task_desc)


def get_report_dirs(llm_name: str, all_report_dirs: bool) -> list[Path]:
    reports_root = PROJECT_ROOT / "output" / "reports"
    if all_report_dirs:
        if not reports_root.exists():
            return []
        return [item for item in reports_root.iterdir() if item.is_dir()]
    return [reports_root / llm_name]


def collect_report_targets(stem: str, report_dirs: list[Path]) -> list[Path]:
    targets: list[Path] = []
    for report_dir in report_dirs:
        targets.extend(
            [
                report_dir / f"{stem}.json",
                report_dir / f"{stem}.md",
                report_dir / f"{stem}.pdf",
                report_dir / f"{stem}_before_refine_outline.json",
                report_dir / f"{stem}_after_refine_outline.json",
                report_dir / stem,
            ]
        )
    return targets


def collect_log_targets(stock_code: str) -> list[Path]:
    patterns = [
        f"log_tracking_{stock_code}_*.txt",
        f"llm_debug_tracking_{stock_code}_*.txt",
        f"verifier_trace_tracking_{stock_code}_*.txt",
    ]
    targets: list[Path] = []
    for pattern in patterns:
        targets.extend(PROJECT_ROOT.glob(pattern))
    return targets


def collect_demonstration_cache_targets(stock_code: str, llm_name: str) -> list[Path]:
    demo_dir = PROJECT_ROOT / "data" / "memory" / "long_term" / "demonstration"
    model_demo_dir = demo_dir / llm_name
    if not demo_dir.exists():
        return []

    patterns = [
        f"{stock_code}_*_outline.json",
        f"{stock_code}_*_outline_only_evidence.json",
    ]
    targets: list[Path] = []
    if model_demo_dir.exists():
        for pattern in patterns:
            targets.extend(model_demo_dir.glob(pattern))

    for demo_markdown in demo_dir.glob(f"{stock_code}_*.md"):
        targets.extend(
            [
                model_demo_dir / f"{demo_markdown.stem}_outline.json",
                model_demo_dir / f"{demo_markdown.stem}_outline_only_evidence.json",
            ]
        )
    return targets


def collect_targets(
    stock_code: str,
    run_date: str,
    llm_name: str,
    all_report_dirs: bool,
    include_logs: bool,
) -> list[Path]:
    stem = f"{stock_code}_{run_date}"
    targets = [PROJECT_ROOT / "data" / "memory" / "short_term" / stem]
    targets.extend(collect_report_targets(stem, get_report_dirs(llm_name, all_report_dirs)))
    if include_logs:
        targets.extend(collect_log_targets(stock_code))
    targets.extend(collect_demonstration_cache_targets(stock_code, llm_name))
    return sorted(set(targets), key=lambda item: str(item).lower())


def remove_target(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def print_targets(targets: list[Path]) -> None:
    existing = [path for path in targets if path.exists()]
    missing = [path for path in targets if not path.exists()]
    print(f"待删除存在项: {len(existing)}")
    for path in existing:
        print(f"  [exists] {path}")
    print(f"不存在项: {len(missing)}")
    for path in missing:
        print(f"  [missing] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清除一次 main.py 运行产生的 short-term、研报、outline 和 PDF 临时文件。"
    )
    parser.add_argument("--task-desc", default="", help="任务描述；默认从 main.py 读取 task_desc。")
    parser.add_argument("--stock-code", default="", help="股票代码；默认从任务描述解析。")
    parser.add_argument("--date", default="", help="运行日期，支持 YYYYMMDD 或 YYYY-MM-DD；默认从任务描述解析。")
    parser.add_argument("--llm-name", default="", help="模型配置 id；默认使用当前 config 的 llm_name。")
    parser.add_argument("--all-report-dirs", action="store_true", help="额外清理所有 output/reports/* 下同名报告；默认只清当前 llm_name。")
    parser.add_argument("--include-logs", action="store_true", help="同时清理该股票的 tracking/log/debug trace 文本日志。")
    parser.add_argument("--yes", action="store_true", help="实际删除；不加时只打印待删除清单。")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    task_desc = args.task_desc.strip() or read_main_task_desc(PROJECT_ROOT / "main.py")
    stock_code = resolve_stock_code(task_desc, args.stock_code.strip() or None)
    run_date = resolve_run_date(task_desc, args.date.strip() or None)
    llm_name = args.llm_name.strip() or config.Config().llm_name
    targets = collect_targets(
        stock_code=stock_code,
        run_date=run_date,
        llm_name=llm_name,
        all_report_dirs=args.all_report_dirs,
        include_logs=args.include_logs,
    )

    print(f"stock_code={stock_code}")
    print(f"date={run_date}")
    print(f"llm_name={llm_name}")
    print_targets(targets)
    if not args.yes:
        print("当前为预览模式；确认无误后加 --yes 执行删除。")
        return

    for target in targets:
        remove_target(target)
    print("清理完成。")


if __name__ == "__main__":
    main()
