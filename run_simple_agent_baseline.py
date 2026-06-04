# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "benchmark.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "reports" / "qwen3.6-27b-baseline"
DEFAULT_MEMORY_ROOT = PROJECT_ROOT / "data" / "memory" / "short_term" / "simple_agent_baseline"
DEFAULT_LLM_NAME = "qwen3.6"

BASELINE_SYS_PROMPT = """你是一个金融研报撰写专家。
你可以使用已注册工具 search_engine 和 fetch_url_page_text 获取网络信息。"""

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

@dataclass
class BenchmarkCase:
    index: int
    stock_code: str
    date: str
    company_name: str
    prompt: str


def sanitize_filename_part(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\r\n\t]+', "_", value.strip())
    value = re.sub(r"\s+", "", value)
    return value or "unknown"


async def run_case(
    case: BenchmarkCase,
    model,
    formatter,
    long_term,
    output_dir: Path,
    memory_root: Path,
    max_iters: int,
    overwrite: bool,
) -> Path:
    from agentscope.agent import ReActAgent
    from agentscope.message import Msg
    from agentscope.tool import Toolkit
    from src.memory.short_term import ShortTermMemoryStore
    from src.tools.material_tools import MaterialTools
    from src.tools.search_tools import SearchTools
    from src.utils.call_with_retry import call_agent_with_retry

    stock_code = sanitize_filename_part(case.stock_code or f"prompt{case.index:02d}")
    date = sanitize_filename_part(case.date or "unknown_date")
    company_name = sanitize_filename_part(case.company_name)
    report_path = output_dir / f"{stock_code}_{date}_{company_name}.md"
    if report_path.exists() and not overwrite:
        print(f"[{case.index:02d}] skip existing: {report_path}", flush=True)
        return report_path

    short_term = ShortTermMemoryStore(
        base_dir=memory_root / f"{case.stock_code}_{case.date}",
        current_date=case.date.replace("-", ""),
    )
    toolkit = Toolkit()
    material_tools = MaterialTools(short_term=short_term, long_term=long_term)
    search_tools = SearchTools(short_term=short_term, long_term=long_term)
    toolkit.register_tool_function(search_tools.search_engine)
    toolkit.register_tool_function(material_tools.fetch_url_page_text)
    agent = ReActAgent(
        name="BaselineAgent",
        sys_prompt=BASELINE_SYS_PROMPT,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        parallel_tool_calls=True,
        max_iters=max_iters,
    )
    msg = Msg(name="user", content=case.prompt, role="user")

    print(f"[{case.index:02d}] start {case.stock_code} {case.date}", flush=True)
    response = await call_agent_with_retry(agent, msg)
    report_text = response.get_text_content().strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", report_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        report_text = match.group(1).strip()
    report_path.write_text(report_text + "\n", encoding="utf-8")
    await agent.memory.clear()
    print(f"[{case.index:02d}] wrote: {report_path}", flush=True)
    return report_path


async def run_baseline(args: argparse.Namespace) -> None:
    if args.start < 1:
        raise ValueError("--start 必须大于等于 1")
    if args.limit < 1:
        raise ValueError("--limit 必须大于等于 1")

    os.environ["LLM_NAME"] = args.llm_name

    import config
    from src.memory.long_term import LongTermMemoryStore
    from src.utils.instance import create_agent_formatter, create_chat_model

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_absolute():
        benchmark_path = PROJECT_ROOT / benchmark_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    memory_root = Path(args.memory_root)
    if not memory_root.is_absolute():
        memory_root = PROJECT_ROOT / memory_root
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_root.mkdir(parents=True, exist_ok=True)

    cfg = config.Config(llm_name=args.llm_name)
    model_cfg = cfg.get_model_cfg()
    model = create_chat_model(reasoning=True, model_cfg=model_cfg)
    formatter = create_agent_formatter(model_cfg=model_cfg)
    long_term = LongTermMemoryStore(base_dir=PROJECT_ROOT / "data" / "memory" / "long_term")

    benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = []
    for index, item in enumerate(benchmark_data, 1):
        stock_code = str(item.get("stock_code", "")).strip()
        date = str(item.get("date", "")).strip()
        company_name = long_term.name_by_code(stock_code) or stock_code
        prompt = (
            f"当前日期是{date}。请浏览网络搜索相关信息，围绕{company_name}（股票代码：{stock_code}）撰写一份深度研究报告。\n"
            "要求：\n"
            "1. 尽量保持专业卖方研报风格；\n"
            "2. 章节结构应完整、层次清晰；\n"
            "3. 结论、数据和判断应尽量与**当前日期**一致；\n"
            "4. 输出为 Markdown 格式，使用清晰的标题层级，如引用外部来源的资料，引用格式固定为：`([可选的来源名称](网页链接))`。"
        )
        cases.append(
            BenchmarkCase(
                index=index,
                stock_code=stock_code,
                date=date,
                company_name=company_name,
                prompt=prompt,
            )
        )
    selected_cases = cases[args.start - 1:args.start - 1 + args.limit]

    print(f"benchmark: {benchmark_path}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"llm_name: {args.llm_name}", flush=True)
    print(f"selected: {len(selected_cases)} / {len(cases)}", flush=True)

    for case in selected_cases:
        try:
            await run_case(
                case=case,
                model=model,
                formatter=formatter,
                long_term=long_term,
                output_dir=output_dir,
                memory_root=memory_root,
                max_iters=args.max_iters,
                overwrite=not args.skip_existing,
            )
        except Exception as exc:
            print(f"[{case.index:02d}] failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            if not args.continue_on_error:
                raise

def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行只注册 search_engine 和 fetch_url_page_text 的简单 agent baseline。"
    )
    parser.add_argument(
        "--benchmark",
        default=str(DEFAULT_BENCHMARK_PATH),
        help="benchmark JSON 文件路径，默认 benchmark.json。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Markdown 报告输出目录，默认 output/reports/qwen3.6-27b。",
    )
    parser.add_argument(
        "--memory-root",
        default=str(DEFAULT_MEMORY_ROOT),
        help="短期 memory 输出目录。",
    )
    parser.add_argument(
        "--llm-name",
        default=os.getenv("LLM_NAME", DEFAULT_LLM_NAME),
        help="config.yaml/config.local.yaml 中的模型 id，默认 qwen3.6 或环境变量 LLM_NAME。",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="从第几个 prompt 开始，默认 1。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="执行 prompt 数量，默认 50。",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=15,
        help="单个 agent 的最大 ReAct 迭代次数，默认 15。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果目标 Markdown 已存在则跳过，默认覆盖写入。",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单条失败后继续后续 prompt，默认失败即停止。",
    )
    args = parser.parse_args()
    asyncio.run(run_baseline(args))


if __name__ == "__main__":
    main()
