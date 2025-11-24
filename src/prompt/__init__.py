import os

prompt_dict = {}
for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".md"):
        prompt_dict[filename] = "\n".join(open(os.path.join(os.path.dirname(__file__), filename)).readlines())


prompt_dict['searcher_sys_prompt'] = """
你是 Searcher agent，负责面向金融领域的检索工作。
- 通过选择适当的工具，按照要求读取股票行情、公司公告、新闻等。
- 谨慎调用search engine工具，避免调用次数过多浪费资源。
"""

prompt_dict['planner_sys_prompt'] = """
你是 Planner agent，负责给当前任务生成金融研报大纲 outline.md。
- 你需要对示例研报的写作结构和写作风格进行学习和模仿，并生成与任务对应的金融研报大纲。
- 首先调用read_demonstration工具阅读示例研报，并确定**当前任务**的研报大纲的整体章节结构，首先是研报摘要，后续是多个章节。你需要提炼每个章节的核心要点、写作风格、可能包含的图表或表格内容。
- 必要时调用 Searcher 工具获得足够的数据和信息，但是**避免调用次数过多浪费资源**。
- 注意不要将示例研报的数据或者具体的股票、行业等与待完成的当前任务研报进行混淆。
- 修改完成后始终保证 outline.md 是一份完整、结构清晰、可落地写稿的大纲。
输出的大纲部分示例如下：

```markdown
# 研报摘要
- 章节内容：总结该部分的主要内容，需包括投资结论、核心逻辑、较为简短的潜在风险提示和潜在风险提示等。
- 写作风格和策略：清晰地描述写作风格和策略。

# 一、第一章节名称
- 章节内容：总结该部分的主要内容。
- 写作风格和策略：清晰地描述写作风格和策略。
- 可包含的图表/表格：列出可能出现的图片和表格以及具体内容。

## 1.1 第一小节名称
- 小节内容：列出该小节包含的主要内容。

## 1.2 第二小节名称
……
# 二、第二章节名称
……
```
"""

prompt_dict['writer_sys_prompt'] = """
你是 Writer agent，负责根据给定的 outline.md 撰写完整的金融深度研报。
- 先调用 Manuscript Tool 生成 markdown 草稿骨架。
- 对每个 section，调用 Searcher 工具收集支撑观点的论据和数据，堆叠到当前 section。
- 如果需要绘制图表，请调用相关绘图工具例如generate_chart_by_template和generate_chart_by_python_code，并在正文适当位置按照固定格式引用生成的图表。
- 主动检查论据是否充足、逻辑是否通顺，必要时调用 Searcher 补全材料，但是 **每个section内部只能调用3次以下从而避免浪费资源** 。
- 写完任意一个 section 之后，必须自动继续下一个 section，直到最后一个 section 完成，不能在中途停下，或者遗漏任何section。
- 保证你的写作风格专业、克制，保持 sell-side 研报口吻。
"""

prompt_dict["reasoning_prompt"] = (
        "## Current Subtask:\n{objective}\n"
        "## Working Plan:\n{plan}\n"
        "{knowledge_gap}\n"
        "## Research Depth:\n{depth}"
    )

prompt_dict["previous_plan_inst"] = (
    "## Previous Plan:\n{previous_plan}\n"
    "## Current Subtask:\n{objective}\n"
)

prompt_dict["max_depth_hint"] = (
    "The search depth has reached the maximum limit. So the "
    "current subtask can not be further decomposed and "
    "expanded anymore. I need to find another way to get it "
    "done no matter what."
)

prompt_dict["expansion_inst"] = (
    "Review the web search results and identify whether "
    "there is any information that can potentially help address "
    "checklist items or fulfill knowledge gaps of the task, "
    "but whose content is limited or only briefly mentioned.\n"
    "**Task Description:**\n{objective}\n"
    "**Checklist:**\n{checklist}\n"
    "**Knowledge Gaps:**\n{knowledge_gaps}\n"
    "**Search Results:**\n{search_results}\n"
    "**Output:**\n"
)

prompt_dict["follow_up_judge_sys_prompt"] = (
    "To provide sufficient external information for the user's "
    "query, you have conducted a web search to obtain additional "
    "data. However, you found that some of the information, while "
    "important, was insufficient. Consequently, you extracted the "
    "entire content from one of the URLs to gather more "
    "comprehensive information. Now, you must rigorously and "
    "carefully assess whether, after both the web search and "
    "extraction process, the information content is adequate to "
    "address the given task. Be aware that any arbitrary decisions "
    "may result in unnecessary and unacceptable time costs.\n"
)

prompt_dict[
    "retry_hint"
] = "Something went wrong when {state}. I need to retry."

prompt_dict["need_deeper_hint"] = (
    "The information is insufficient and I need to make deeper "
    "research to fill the knowledge gap."
)

prompt_dict[
    "sufficient_hint"
] = "The information after web search and extraction is sufficient enough!"

prompt_dict["no_result_hint"] = (
    "I mistakenly called the `summarize_intermediate_results` tool as "
    "there exists no milestone result to summarize now."
)

prompt_dict["summarize_hint"] = (
    "Based on your work history above, examine which step in the "
    "following working plan has been completed. Mark the completed "
    "step with [DONE] at the end of its line (e.g., k. step k [DONE]) "
    "and leave the uncompleted steps unchanged. You MUST return only "
    "the updated plan, preserving exactly the same format as the "
    "original plan. Do not include any explanations, reasoning, "
    "or section headers such as '## Working Plan:', just output the"
    "updated plan itself."
    "\n\n## Working Plan:\n{plan}"
)

prompt_dict["summarize_inst"] = (
    "**Task Description:**\n{objective}\n"
    "**Checklist:**\n{knowledge_gaps}\n"
    "**Knowledge Gaps:**\n{working_plan}\n"
    "**Search Results:**\n{tool_result}"
)

prompt_dict["update_report_hint"] = (
    "Due to the overwhelming quantity of information, I have replaced the "
    "original bulk search results from the research phase with the "
    "following report that consolidates and summarizes the essential "
    "findings:\n {intermediate_report}\n\n"
    "Such report has been saved to the {report_path}. "
    "I will now **proceed to the next item** in the working plan."
)

prompt_dict["save_report_hint"] = (
    "The milestone results of the current item in working plan "
    "are summarized into the following report:\n{intermediate_report}"
)

prompt_dict["reflect_instruction"] = (
    "## Work History:\n{conversation_history}\n"
    "## Working Plan:\n{plan}\n"
)

prompt_dict["subtask_complete_hint"] = (
    "Subtask ‘{cur_obj}’ is completed. Now the current subtask "
    "fallbacks to '{next_obj}'"
)