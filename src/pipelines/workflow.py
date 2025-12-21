# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path

from agentscope.message import Msg

from memory.working import Section
from src.utils.instance import create_chat_model, create_agent_formatter
from src.memory.short_term import ShortTermMemoryStore
from src.agents.searcher import create_searcher_agent, build_searcher_toolkit
from src.agents.writer import create_writer_agent, build_writer_toolkit
from src.agents.planner import create_planner_agent, build_planner_toolkit
from src.agents.verifier import create_verifier_agent, build_verifier_toolkit

from src.utils.file_converter import md_to_pdf, pdf_to_markdown, section_to_markdown
from src.utils.parse_verdict import parse_verifier_verdict
from src.utils.call_agent_with_retry import call_agent_with_retry
import config
import asyncio

from utils.file_converter import markdown_to_sections
from utils.local_file import STOCK_REPORT_PATHS


async def run_workflow(task_desc: str) -> str:
    """围绕一个 task description 执行完整的研报生成流程。
    """

    cfg = config.Config()

    # ----- 1. 准备 memory store -----

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    short_term_dir = PROJECT_ROOT / "data" / "memory" / "short_term"
    
    short_term = ShortTermMemoryStore(
        base_dir=short_term_dir,
    )

    planner_cfg = cfg.get_planner_cfg()
    use_demo = planner_cfg.get("use_demonstration", False)


    # ----- 2. 创建底层模型 -----
    model= create_chat_model()

    # ----- 3. 创建 Searcher Agent -----
    searcher_toolkit = build_searcher_toolkit(
        short_term=short_term,
        # tool_use_store=tool_use_store,
    )

    searcher = create_searcher_agent(model=model, formatter=create_agent_formatter(), toolkit=searcher_toolkit)
    # print("\n=== 打印 JSON Schema (get_json_schemas) ===")
    # schemas = searcher_toolkit.get_json_schemas()
    # print(schemas)


    # ----- 4. 获取demonstration -----
    searcher_input = Msg(
        name="User",
        content=f"下面是某个任务描述：{task_desc}\n"
                f"首先，你需要识别目标股票代码（如果任务描述中只谈及股票名称，你需要将之转换为股票代码）。"
                f"请只输出纯数字的股票代码，不做其他输出。",
        role="user",
    )
    while True:
        try:
            outline_msg = await call_agent_with_retry(searcher, searcher_input)
            stock_symbol = re.search(r"[0-9]+", outline_msg.get_text_content()).group()
            assert stock_symbol is not None
            print("股票代码：", stock_symbol)
            break
        except AssertionError as e:
            print(e)

    # 解析demonstration report，第二遍解析同一个report可以注释掉
    demo_pdf_path = STOCK_REPORT_PATHS[stock_symbol][-1]
    demo_md_path = short_term_dir / f"demonstration" / (demo_pdf_path.name.split(".")[0] + ".md")
    if not demo_md_path.exists():
        final_text, images = pdf_to_markdown(demo_pdf_path, demo_md_path)
    manuscript = markdown_to_sections(demo_md_path)

    # outline_store = OutlineExperienceStore(
    #     base_dir=Path("data/memory/long_term/outlines"),
    # )
    # tool_use_store = ToolUseExperienceStore(
    #     base_path=Path("data/memory/long_term/tool_use"),
    # )

    # ----- 5. 调用 Planner：生成 / 修订 outline.md -----
    # planner_toolkit = build_planner_toolkit(
    #     short_term=short_term,
    #     searcher=searcher,
    # )
    planner = create_planner_agent(model=model, formatter=create_agent_formatter(), toolkit=None)

    async def dfs_outline(section: Section, parent_id=None):
        if section.subsections is None:
            return
        for subsection in section.subsections:
            section_id = ((parent_id + ".") if parent_id else "") + str(subsection.section_id)
            print(f"\n====== 开始总结章节 {section_id} ======\n")
            await dfs_outline(subsection)
            planner_input = Msg(
                name="User",
                content=subsection.elements[0].example,
                role="user",
            )
            # outline_msg = await planner(planner_input)
            for i in range(10):
                try:
                    outline_msg = await call_agent_with_retry(planner, planner_input)
                    # print(outline_msg.get_text_content())
                    subsection.elements = subsection.parse(outline_msg.get_text_content())
                except AssertionError as e:
                    print(e)
                    planner_input = Msg(
                        name="User",
                        content=str(e),
                        role="user",
                    )
            print(subsection.read(True, True, True, False, False))
    await dfs_outline(manuscript)

    outline = manuscript.read(read_subsections=True)
    (short_term_dir / "outline").write_text(outline)
    print(outline)

    # ----- 6. 调用 Writer：基于 outline.md 写 Manuscript 并导出 PDF -----
    writer_toolkit = build_writer_toolkit(
        short_term=short_term,
        searcher=searcher,
    )
    writer = create_writer_agent(model=model, formatter=create_agent_formatter(), toolkit=writer_toolkit)

    verifier_toolkit = build_verifier_toolkit(
        short_term=short_term,
    )

    async def dfs_report(section: Section, parent_id=None):
        if section.subsections is None:
            return
        for subsection in section.subsections:
            section_id = ((parent_id + ".") if parent_id else "") + str(subsection.section_id)
            print(f"\n====== 开始写作章节 {section_id} ======\n")
            await dfs_report(subsection)
            for element in subsection.elements:
                writer_input = Msg(
                    name="User",
                    content=(
                        f"任务：{task_desc}\n"
                        f"当前需要你撰写要点：{element.summary}\n"
                        f"【写作要求】\n{element.requirements}\n"
                        f"【模版】\n{element.content}\n\n"
                        f"请你开始搜索和撰写。"
                    ),
                    role="user",
                )

                # draft_msg = await writer(writer_input)
                draft_msg = await call_agent_with_retry(writer, writer_input)

                print("[Writer 初稿输出]")
                print(draft_msg.get_text_content())

                max_verify_rounds = cfg.get_max_verify_rounds()
                # 进入 Verifier 审核 loop
                verifier = create_verifier_agent(model=model, formatter=create_agent_formatter(), toolkit=verifier_toolkit)
                for round_idx in range(1, max_verify_rounds + 1):

                    print(f"\n--- Verifier 审核轮次 {round_idx}：章节 {section_id} ---\n")
                    await asyncio.sleep(5)
                    verifier_input = Msg(
                        name="User",
                        content=(
                            f"任务：{task_desc}\n"
                            f"当前正在撰写的要点：{element.summary}\n"
                            f"【写作要求】\n{element.requirements}\n"
                            f"【参考范例】\n{element.example}\n\n"
                            "请调用材料读取工具，不遗漏任何参考材料进行严格地审核，并给出结构化输出的结论。"
                        ),
                        role="user",
                    )

                    # verify_msg = await verifier(verifier_input)
                    verify_msg = await call_agent_with_retry(verifier, verifier_input)
                    verdict_text = verify_msg.get_text_content()
                    print("[Verifier 审核结果]")
                    print(verdict_text)

                    passed, problems, reason = parse_verifier_verdict(verdict_text)

                    if passed:
                        print(f"[审核通过] 章节 {section_id} 审核通过。进入下一章节。")
                        break
                    # 如果没通过，把 Verifier 的结构化结论反馈给 Writer，让其在同一个 section 上重写
                    problems_text = problems if problems else verdict_text

                    writer_fix_input = Msg(
                        name="User",
                        content=(
                            "我给出了一些审核意见。"
                            f"未通过原因：{reason}\n"
                            f"问题如下：{problems_text}\n\n"
                            "请根据这些问题逐条修改本章节内容，返回更正后的新版本。"
                        ),
                        role="user",
                    )
                    # draft_msg = await writer(writer_fix_input)
                    draft_msg = await call_agent_with_retry(writer, writer_fix_input)

                    print("[Writer 根据审核意见修改后的输出]")
                    print(draft_msg.get_text_content())
                element.content = draft_msg.get_text_content()
                element.finished = True
            section_text = "\n".join([e.content for e in subsection.elements])
            draft_msg = await call_agent_with_retry(writer, Msg(
                name="User",
                content=(
                    "以下是所有要点整理后的本章节内容：\n\n"
                    f"{section_text}\n\n"
                    f"参考范例的标题为{subsection.title}\n\n"
                    f"请你根据当前任务撰写的内容起一个新标题。"
                ),
                role="user",
            ))
            element.title = draft_msg.get_text_content()
            print(element.title)

    await dfs_report(manuscript)

    markdown_text = section_to_markdown(manuscript)
    (short_term_dir / "manuscript.md").write_text(markdown_text, encoding="utf-8")
    # md_to_pdf(markdown_text, short_term=short_term)

