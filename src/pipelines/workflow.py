from __future__ import annotations

from pathlib import Path

from agentscope.message import Msg

from src.utils.instance import create_chat_model, create_agent_formatter, create_searcher_formatter
from src.memory.short_term import ShortTermMemoryStore
from src.agents.searcher import create_searcher_agent, build_searcher_toolkit
from src.agents.writer import create_writer_agent, build_writer_toolkit
from src.agents.planner import create_planner_agent, build_planner_toolkit
from src.agents.verifier import create_verifier_agent, build_verifier_toolkit

from src.utils.file_converter import md_to_pdf,pdf_to_markdown
from src.utils.parse_verdict import parse_verifier_verdict
import config
async def run_workflow(task_desc: str, output_filename: str) -> str:
    """围绕一个 task description 执行完整的研报生成流程。
    """

    cfg = config.Config()

    # ----- 1. 准备 memory store -----
    short_term = ShortTermMemoryStore(
        base_dir=Path("/financial-report-agent/data/memory/short_term"),
    )

    # 解析demonstration report，第二遍解析同一个report可以注释掉
    # pdf_to_markdown(short_term=short_term)


    # outline_store = OutlineExperienceStore(
    #     base_dir=Path("data/memory/long_term/outlines"),
    # )
    # tool_use_store = ToolUseExperienceStore(
    #     base_path=Path("data/memory/long_term/tool_use"),
    # )

    # ----- 2. 创建底层模型 -----
    model= create_chat_model()

    # ----- 3. 创建 Searcher Agent -----
    searcher_toolkit = build_searcher_toolkit(
        short_term=short_term,
        # tool_use_store=tool_use_store,
    )

    # searcher = create_searcher_agent(model=model, formatter=create_searcher_formatter(), toolkit=searcher_toolkit)
    searcher = create_searcher_agent(model=model, formatter=create_agent_formatter(), toolkit=searcher_toolkit)
    # print("\n=== 打印 JSON Schema (get_json_schemas) ===")
    # schemas = searcher_toolkit.get_json_schemas()
    # print(schemas)

    # ----- 4. 创建 Planner / Writer / Verifier Agent -----
    planner_toolkit = build_planner_toolkit(
        short_term=short_term,
        searcher=searcher,
    )
    planner = create_planner_agent(model=model, formatter=create_agent_formatter(), toolkit=planner_toolkit)

    writer_toolkit = build_writer_toolkit(
        short_term=short_term,
        searcher=searcher,
    )
    writer = create_writer_agent(model=model, formatter=create_agent_formatter(), toolkit=writer_toolkit)

    verifier_toolkit = build_verifier_toolkit(
        short_term=short_term,
    )
    verifier = create_verifier_agent(model=model, formatter=create_agent_formatter(), toolkit=verifier_toolkit)
    


    # ----- 5. 调用 Planner：生成 / 修订 outline.md -----
    planner_input = Msg(
        name="User",
        content=(
            "下面是本次任务描述，请你开始进行大纲撰写：\n\n"
            + task_desc
        ),
        role="user",
    )
    
    outline_msg = await planner(planner_input)
    print(outline_msg.get_text_content())


    # ----- 6. 调用 Writer：基于 outline.md 写 Manuscript 并导出 PDF -----

    # writer_input = Msg(
    #     name="User",
    #     content=(
    #             "下面是本次任务描述，请你基于 outline.md 开始写作：\n\n"
    #             + task_desc
    #     ),
    #     role="user",
    # )
    # final_msg = await writer(writer_input)

    sections = short_term.draft_manuscript_from_outline()
    failed_sections = [] 

    for section_id, title, section_outline in sections:
        print(f"\n====== 开始写作章节 {section_id} ======\n")

        # 先让 Writer 写这一章的初稿
        writer_input = Msg(
            name="User",
            content=(
                "下面是本次任务描述：\n\n"
                f"{task_desc}\n\n"
                f"请你首先阅读 section_id={section_id} 的章节 Markdown 草稿并根据其中内容开始撰写。\n"
            ),
            role="user",
        )


        draft_msg = await writer(writer_input)
        print("[Writer 初稿输出]")
        print(draft_msg.get_text_content())

        max_verify_rounds = cfg.get_max_verify_rounds()
        # 进入 Verifier 审核 loop
        for round_idx in range(1, max_verify_rounds + 1):
            print(f"\n--- Verifier 审核轮次 {round_idx}：章节 {section_id} ---\n")

            # 增补：只有第二轮及之后，提示已修改
            revision_notice = ""
            if round_idx > 1:
                revision_notice = "注意：此章节的文本已经根据你的审核意见完成了修改，请重新审核。\n\n"

            verifier_input = Msg(
                name="User",
                content=(
                    f"下面是任务描述：{task_desc}\n"
                    f"当前章节: section_id={section_id}。\n\n"
                    f"下面是本章节 outline: {section_outline}\n\n"
                    f"{revision_notice}"
                    "请开始调用章节文本读取工具、材料读取工具、字数统计工具进行严格地审核，并给出结构化输出的结论。"
                    "**禁止在未调用工具读取章节文本、或者遗漏其中引用的任何材料的情况下就直接给出结论。**"
                ),
                role="user",
            )

            verify_msg = await verifier(verifier_input)
            verdict_text = verify_msg.get_text_content()
            print("[Verifier 审核结果]")
            print(verdict_text)

            passed, problems, reason = parse_verifier_verdict(verdict_text)

            if passed:
                print(f"[审核通过] 章节 {section_id} 审核通过。进入下一章节。")
                break

            if round_idx == max_verify_rounds:
                print(f"[多次审核未通过] 章节 {section_id} 多次审核未通过，达到最大重写次数，标记为需要人工复核。")

                problems_text = problems if problems else verdict_text

                failed_sections.append(
                    {
                        "section_id": section_id,
                        "title": title,
                        "outline": section_outline,
                        "reason": reason,
                        "problems": problems_text,
                    }
                )
                break

            # 如果没通过，把 Verifier 的结构化结论反馈给 Writer，让其在同一个 section 上重写
            problems_text = problems if problems else verdict_text

            writer_fix_input = Msg(
                name="User",
                content=(
                    "下面是 Verifier 对本章节的详细审核意见，请你在 **同一个 section_id** "
                    f"（section_id={section_id}） 的基础上进行修改，而不是新建章节。\n\n"
                    f"未通过原因：{reason}\n"
                    f"问题如下：{problems_text}\n\n"
                    "请根据这些问题逐条修改本章节内容，并通过工具接口覆盖已有稿件。"
                ),
                role="user",
            )
            draft_msg = await writer(writer_fix_input)
            print("[Writer 根据审核意见修改后的输出]")
            print(draft_msg.get_text_content())
            

    text = md_to_pdf(short_term=short_term,output_filename=output_filename+".pdf")
    print("\n====== 多次审核未通过章节列表 ======")

    if failed_sections:
        for item in failed_sections:
            print(f"section_id={item['section_id']} title={item['title']}\n原因={item['reason']}\n问题={item['problems']}")

    # 返回字符串内容
    return f"{text}\n未通过章节：{failed_sections}"

