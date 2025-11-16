import asyncio

from financial_report_agent.pipelines.workflow import run_workflow


async def main() -> None:
    task_desc = "生成当前季度宁德时代（股票代码为300750）的深度研究报告，重点分析盈利能力、估值与风险。"
    result = await run_workflow(task_desc=task_desc)
    print("===== 最终输出=====")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
