# -*- coding: utf-8 -*-
import asyncio
from dotenv import load_dotenv
load_dotenv()

from src.pipelines.workflow_concurrent import run_workflow
# from src.pipelines.workflow import run_workflow
import sys
import io


async def main() -> None:
    task_desc = "当前日期是2026-03-22，请帮我调研海兴电力（股票代码为603556）的深度研究报告。"
    await run_workflow(task_desc=task_desc)

if __name__ == "__main__":
    asyncio.run(main())
