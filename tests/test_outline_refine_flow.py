# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import config
from src.memory.long_term import LongTermMemoryStore
from src.pipelines.planning import process_pdf_to_outline
from src.utils.format import _normalize_section_titles
from src.utils.get_entity_info import get_entity_info
from src.utils.instance import (
    formatter,
    llm_instruct,
    llm_outline_refine,
    llm_reasoning,
    outline_refine_formatter,
)
from src.utils.local_file import DEMO_DIR
from src.utils.outline_refine import refine_outline


load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单独调试 outline_refine 流程：仅执行 PDF->Markdown、outline 生成、outline refine。"
    )
    parser.add_argument(
        "stock_code",
        help="股票代码，例如 600699。",
    )
    return parser.parse_args()


def load_benchmark_item(project_root: Path, stock_code: str) -> dict:
    benchmark_path = project_root / "benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"找不到 benchmark.json: {benchmark_path}")
    benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(benchmark_data, list):
        raise ValueError("benchmark.json 应该是一个列表")

    normalized_code = str(stock_code).zfill(6)
    for item in benchmark_data:
        if str(item.get("stock_code", "")).zfill(6) == normalized_code:
            return item
    raise ValueError(f"benchmark.json 中未找到股票代码 {normalized_code} 对应的任务")


def build_task_inputs(project_root: Path, stock_code: str) -> tuple[str, str, Path, dict]:
    normalized_code = str(stock_code).zfill(6)
    benchmark_item = load_benchmark_item(project_root, normalized_code)
    long_term = LongTermMemoryStore(base_dir=project_root / "data" / "memory" / "long_term")
    entity = get_entity_info(long_term, normalized_code)
    if not entity:
        raise ValueError(f"无法获取股票代码 {normalized_code} 的信息")

    stock_name = entity.get("name", "")
    date = benchmark_item.get("date")
    reference = benchmark_item.get("reference", "")
    if not date or not reference:
        raise ValueError(f"benchmark.json 中股票 {normalized_code} 缺少 date 或 reference")

    demo_pdf_path = (DEMO_DIR / reference).resolve()
    if not demo_pdf_path.exists():
        raise FileNotFoundError(f"benchmark 对应的 reference PDF 不存在: {demo_pdf_path}")

    task_desc = f"当前日期是{date}，请帮我调研{stock_name}（股票代码为{normalized_code}）的深度研究报告。"
    return task_desc, date, demo_pdf_path, benchmark_item


def build_default_save_dir(project_root: Path) -> Path:
    return project_root / "data" / "memory" / "long_term" / "demonstration"


def build_default_output_dir(project_root: Path) -> Path:
    cfg = config.Config()
    return project_root / "data" / "output" / "outline_refine_debug" / cfg.outline_refine_name


def print_outline_snapshot(title: str, outline) -> None:
    print(f"\n========== {title} ==========", flush=True)
    print(
        outline.read(
            with_requirements=True,
            with_reference=False,
            with_content=False,
            with_evidence=True,
            read_subsections=True,
        ),
        flush=True,
    )


def save_outline_debug_artifacts(output_dir: Path, demo_pdf_path: Path, initial_outline, refined_outline) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = demo_pdf_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{stem}_{timestamp}"

    initial_json_path = output_dir / f"{base_name}_initial_outline.json"
    refined_json_path = output_dir / f"{base_name}_refined_outline.json"
    initial_txt_path = output_dir / f"{base_name}_initial_outline.txt"
    refined_txt_path = output_dir / f"{base_name}_refined_outline.txt"

    initial_json_path.write_text(initial_outline.to_json(ensure_ascii=False), encoding="utf-8")
    refined_json_path.write_text(refined_outline.to_json(ensure_ascii=False), encoding="utf-8")
    initial_txt_path.write_text(
        initial_outline.read(
            with_requirements=True,
            with_reference=False,
            with_content=False,
            with_evidence=True,
            read_subsections=True,
        ),
        encoding="utf-8",
    )
    refined_txt_path.write_text(
        refined_outline.read(
            with_requirements=True,
            with_reference=False,
            with_content=False,
            with_evidence=True,
            read_subsections=True,
        ),
        encoding="utf-8",
    )

    print("\n========== 调试输出文件 ==========", flush=True)
    print(initial_json_path, flush=True)
    print(refined_json_path, flush=True)
    print(initial_txt_path, flush=True)
    print(refined_txt_path, flush=True)


async def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    task_desc, cur_date, demo_pdf_path, benchmark_item = build_task_inputs(project_root, args.stock_code)
    save_dir = build_default_save_dir(project_root)
    output_dir = build_default_output_dir(project_root)

    print("========== Outline Refine 调试 ==========", flush=True)
    print(f"stock_code: {str(args.stock_code).zfill(6)}", flush=True)
    print(f"cur_date: {cur_date}", flush=True)
    print(f"benchmark_item: {json.dumps(benchmark_item, ensure_ascii=False)}", flush=True)
    print(f"task_desc: {task_desc}", flush=True)
    print(f"demo_pdf_path: {demo_pdf_path}", flush=True)
    print(f"save_dir: {save_dir}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)

    outline = await process_pdf_to_outline(
        demo_pdf_path,
        save_dir,
        llm_reasoning,
        llm_instruct,
        formatter,
    )
    _normalize_section_titles(outline)
    print_outline_snapshot("初始 Outline", outline)

    refined_outline = await refine_outline(
        outline=outline,
        task_desc=task_desc,
        cur_date=cur_date,
        model=llm_outline_refine,
        formatter=outline_refine_formatter,
        model_cfg=config.Config().get_outline_refine_model_cfg(),
        debug_print=True,
    )
    _normalize_section_titles(refined_outline)
    print_outline_snapshot("Refined Outline", refined_outline)

    save_outline_debug_artifacts(output_dir, demo_pdf_path, outline, refined_outline)


if __name__ == "__main__":
    asyncio.run(main())
