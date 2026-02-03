# -*- coding: utf-8 -*-
import asyncio

from src.pipelines.workflow_concurrent import run_workflow
# from src.pipelines.workflow import run_workflow
import sys
import io

from utils.get_entity_info import get_entity_info


async def main() -> None:
    get_entity_info()
    task_desc = "当前日期是2025-09-05，请帮我调研比亚迪（股票代码为002594）的深度研究报告。"
    await run_workflow(task_desc=task_desc)
    # print("===== 最终输出=====")
    # print(result)

if __name__ == "__main__":
    asyncio.run(main())
