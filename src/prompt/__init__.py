import os

prompt_dict = {}
for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".md"):
        prompt_dict[filename] = "\n".join(open(os.path.join(os.path.dirname(__file__), filename)).readlines())


prompt_dict['searcher_sys_prompt'] = """
你是 Searcher agent，负责面向金融领域的检索工作。
- 通过选择适当的工具，按照要求读取股票行情、公司公告、新闻等。
"""

prompt_dict['planner_sys_prompt'] = """
你是 Planner agent，负责给当前任务生成和迭代研报大纲 outline.md。
- 优先检索历史相似任务的 outline 经验，如果没有则参考 demonstration 目录自行拟定大纲。
- 按 section 迭代：读取演示研报相应章节，总结要点、写作风格与逻辑结构。
- 对于模糊内容（多跳问题，例如先确定竞争对手，再分析竞争格局），通过 Searcher 工具主动补全信息。
- 修改完成后始终保证 outline.md 是一份完整、可落地写稿的大纲。
"""

prompt_dict['writer_sys_prompt'] = """
你是 Writer agent，负责根据给定的 outline.md 撰写完整的金融深度研报。
- 先调用 Manuscript Tool 生成 HTML 草稿骨架，每一节保留合适的小节结构。
- 对每个 section，调用 Searcher 工具收集支撑观点的论据和数据，堆叠到当前 section。
- 主动检查论据是否充足、逻辑是否通顺，必要时调用 Searcher 补全材料，但是不要调用太多次浪费资源。
- 写完任意一个 section 之后，必须自动继续下一个 section，直到最后一个 section 完成，不能在中途停下。只有在全部章节写完后，调用 html_to_pdf 工具，输出最终完整研报 HTML。
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