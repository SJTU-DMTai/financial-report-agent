# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
import os
import argparse
import traceback
from pathlib import Path

from src.pipelines.workflow_concurrent import run_workflow
from src.memory.long_term import LongTermMemoryStore
from src.utils.get_entity_info import get_entity_info
from src.utils.local_file import DEMO_DIR


async def process_single_task(item, idx, total, long_term_memory, semaphore):
    """
    处理单个任务
    """
    async with semaphore:
        stock_code = item.get("stock_code")
        date = item.get("date")
        reference = item.get("reference", "")

        if not stock_code or not date:
            print(f"任务 {idx}: 缺少必要字段 (stock_code 或 date)，跳过")
            return

        # 通过 get_entity_info 获取股票名称
        entity_info = get_entity_info(long_term_memory, stock_code)

        if not entity_info:
            print(f"任务 {idx}: 无法获取股票代码 {stock_code} 的信息，跳过")
            return

        stock_name = entity_info.get("name", "")

        # 构造 task_desc
        task_desc = f"当前日期是{date}，请帮我调研{stock_name}（股票代码为{stock_code}）的深度研究报告。"

        print(f"{'='*60}")
        print(f"任务 {idx}/{total}")
        print(f"股票代码: {stock_code}")
        print(f"股票名称: {stock_name}")
        print(f"日期: {date}")
        print(f"参考文件: {reference}")
        print(f"任务描述: {task_desc}")
        print(f"{'='*60}\n")

        try:
            # 执行工作流
            await run_workflow(task_desc=task_desc, cur_date=date, demo_pdf_path=DEMO_DIR / reference)
        # print(f"\n✅ 任务 {idx} 完成\n")
        except Exception as e:
            traceback.print_exc()
            print(f"\n❌ 任务 {idx} 执行失败: {str(e)}\n")
            # 可以选择继续执行下一个任务或者中断
            # raise  # 取消注释此行以在出错时中断


async def run_benchmark(batch_size=1):
    """
    读取 benchmark.json 中的任务列表，并行执行每个任务。
    对每个任务：
    1. 设置环境变量 CUR_DATE 为 date 值
    2. 通过 get_entity_info 获取 stock_name
    3. 构造 task_desc 并执行 run_workflow

    Args:
        batch_size: 最大任务级并发数（注意：LLM调用的并发由全局semaphore控制）
    """
    # 读取 benchmark.json
    benchmark_path = Path(__file__).parent / "benchmark.json"

    if not benchmark_path.exists():
        print(f"错误：找不到 benchmark.json 文件：{benchmark_path}")
        return

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    if not isinstance(benchmark_data, list):
        print("错误：benchmark.json 应该是一个列表")
        return

    # 初始化长期记忆存储，用于获取股票名称
    long_term_memory = LongTermMemoryStore(
        base_dir=Path(__file__).parent / "data" / "memory" / "long_term"
    )

    total = len(benchmark_data)
    print(f"共有 {total} 个任务待执行")
    print(f"最大任务级并发数: {batch_size}")
    print(f"注意：LLM调用的并发由全局semaphore控制（N_THREAD环境变量）\n")

    # 创建信号量控制任务级并发数
    semaphore = asyncio.Semaphore(batch_size)

    # 创建所有任务
    tasks = []
    for idx, item in enumerate(benchmark_data, 1):
        task = process_single_task(item, idx, total, long_term_memory, semaphore)
        tasks.append(task)

    # 并行执行所有任务
    await asyncio.gather(*tasks)

    print(f"{'='*60}")
    print("所有任务执行完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 benchmark 测试，支持并行执行")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="最大任务级并发数 (默认: 1)"
    )

    args = parser.parse_args()

    asyncio.run(run_benchmark(batch_size=args.batch_size))
