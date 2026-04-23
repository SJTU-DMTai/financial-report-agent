# -*- coding: utf-8 -*-
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from src.pipelines.workflow_concurrent import run_workflow
from src.utils.task_date import resolve_cur_date
# from src.pipelines.workflow import run_workflow
import sys
import io


async def main() -> None:
    task_desc = "当前日期是2026-04-21，请帮我调研均胜电子（股票代码为600699）的深度研究报告。"
    await run_workflow(
        task_desc=task_desc,
        cur_date=resolve_cur_date(task_desc),
    )

if __name__ == "__main__":
    asyncio.run(main())
