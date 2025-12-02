import asyncio

from src.pipelines.workflow import run_workflow


# async def main() -> None:
#     task_desc = "生成2025Q2（即2025年4月到2025年6月） 海康威视（股票代码为002415）的深度研究报告，重点分析盈利结构、估值变化与海外业务风险。"
#     output_filename = "海康威视_2025Q2深度研报"
#     result = await run_workflow(task_desc=task_desc, output_filename=output_filename)
#     print("===== 最终输出=====")
#     print(result)

# async def main() -> None:
#     task_desc = "生成2025Q2（即2025年4月到2025年6月） 贵州茅台（股票代码600519）的深度研究报告，重点分析利润结构、估值变化与渠道改革影响。"
#     output_filename = "贵州茅台_2025Q2深度研报"
#     result = await run_workflow(task_desc=task_desc, output_filename=output_filename)
#     print("===== 最终输出=====")
#     print(result)
          

async def main() -> None:
    task_desc = "生成2024Q4（即2024年10月到2024年12月） 美的集团（股票代码000333）的深度研究报告，重点分析家电主业盈利结构、ToB业务拓展、估值变化与全球化战略进展。"
    output_filename = "美的集团_2024Q4深度研报"
    result = await run_workflow(task_desc=task_desc, output_filename=output_filename)
    print("===== 最终输出=====")
    print(result)

# async def main() -> None:
#     task_desc = "生成2025Q3（即2025年7月到2025年9月） 宁德时代（股票代码为300750）的深度研究报告，重点分析商业模式、现金流质量与创新驱动因素。"
#     output_filename = "宁德时代_2025Q3深度研报"
#     result = await run_workflow(task_desc=task_desc, output_filename=output_filename)
#     print("===== 最终输出=====")
#     print(result)


# async def main() -> None:
#     task_desc = "生成2025Q3（即2025年7月到2025年9月） 迈瑞医疗（股票代码为300760）的深度研究报告，重点分析商业模式、现金流质量与创新驱动因素。"
#     output_filename = "迈瑞医疗2025Q3深度研报"
#     result = await run_workflow(task_desc=task_desc, output_filename=output_filename)
#     print("===== 最终输出=====")
#     print(result)

if __name__ == "__main__":
    asyncio.run(main())
