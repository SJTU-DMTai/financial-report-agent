# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import os
import time
import json
import pickle
import re
import sys
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path

from agentscope.agent import ReActAgent
from agentscope.message import Msg

from evaluation.eval_content import evaluate_segment
from pipelines.planning import process_pdf_to_outline
from src.memory.working import Section, Segment
from src.prompt import prompt_dict
from src.utils.instance import create_chat_model, create_agent_formatter
from src.memory.short_term import ShortTermMemoryStore
from src.memory.long_term import LongTermMemoryStore
from src.agents.searcher import create_searcher_agent, build_searcher_toolkit
from src.agents.writer import create_writer_agent, build_writer_toolkit
from src.agents.planner import create_planner_agent, build_planner_toolkit
from src.agents.verifier import create_verifier_agent, build_verifier_toolkit

from src.utils.file_converter import md_to_pdf, pdf_to_markdown, section_to_markdown
from src.utils.parse_verdict import parse_verifier_verdict
from src.utils.call_with_retry import call_agent_with_retry
from src.utils.get_entity_info import get_entity_info
from src.utils.file_converter import markdown_to_sections
from src.utils.local_file import STOCK_REPORT_PATHS
import config
import asyncio

from src.utils.call_with_retry import call_chatbot_with_retry
from src.utils.instance import llm_reasoning, llm_instruct, formatter, cfg

CURRENT_RUNNING_TASKS = 0

async def search_evidence(evidence, task_desc, segment_topic, searcher):
    searcher_input = Msg(
        name="user",
        content=(
            f"任务：{task_desc}\n"
            f"当前需要你撰写要点：{segment_topic}\n"
            f"论据所需材料：\n{evidence}\n\n"
            f"请你调用工具搜索，尽量根据多个信息源交叉验证后给出搜索结果。"
        ),
        role="user",
    )
    msg = await call_agent_with_retry(searcher, searcher_input)
    print(f"[Searcher] Finished searching: {evidence[:20]}...")
    return msg.get_text_content()

async def process_single_segment(segment, task_desc, agent_factory, semaphore):
    """并发处理单个 Segment：包含搜索和写作"""
    global CURRENT_RUNNING_TASKS
    async with semaphore:
        CURRENT_RUNNING_TASKS += 1
        print(f"[{time.strftime('%H:%M:%S')}] [并发数: {CURRENT_RUNNING_TASKS}] ✍️ 开始写作: {segment.topic[:15]}...", flush=True)

        searcher, writer = agent_factory()
        for i, evidence in enumerate(segment.evidences):
            segment.evidences[i] = await search_evidence(evidence, task_desc, segment.topic, searcher)
            await searcher.memory.clear()

        try:
            writer_input = Msg(
                name="user",
                content=(
                    f"任务：{task_desc}\n"
                    f"当前步骤需要你撰写要点：\n{segment.topic}\n"
                    f"参考示例、写作要求和相关材料如下：\n\n{str(segment)}\n\n"
                    f"请你开始搜索和撰写。"
                ),
                role="user",
            )

            draft_msg = await call_agent_with_retry(writer, writer_input)
            segment.content = draft_msg.get_text_content()
            print(f"[Writer] Segment finished: {segment.topic}")
            print("[Writer 初稿输出]")
            print(segment.content, flush=True)

            for _ in range(5):
                segment_score, suggestions = await evaluate_segment(create_chat_model(reasoning=False), 
                                                                    create_agent_formatter(), 
                                                                    segment)
                print("修改建议:", suggestions, flush=True)
                if suggestions is None:
                    break
                else:
                    writer_input = Msg(
                        name="user", content=f"经评估：\n{suggestions}\n请你继续修改。", role="user",
                    )
                    draft_msg = await call_agent_with_retry(writer, writer_input)
                    segment.content = draft_msg.get_text_content()
                    print(f"[Writer] Segment finished: {segment.topic}")
                    print(segment.content, flush=True)
            await writer.memory.clear()
            segment.finished = True
        finally:
            CURRENT_RUNNING_TASKS -= 1
            print(f"[{time.strftime('%H:%M:%S')}] [并发数: {CURRENT_RUNNING_TASKS}] ✅ 完成写作: {segment.topic[:15]}.", flush=True)

async def process_section_concurrently(section: Section, parent_id, task_desc, agent_factory,
                                       semaphore, stock_symbol, output_pth, manuscript_root):
    """递归并发处理章节"""

    # 1. 处理子章节 (递归) - 优先启动子任务
    sub_tasks = []
    if section.subsections:
        for subsection in section.subsections:
            section_id = ((parent_id + ".") if parent_id else "") + str(subsection.section_id)
            # 递归调用
            sub_tasks.append(process_section_concurrently(
                subsection, section_id, task_desc, agent_factory, semaphore, stock_symbol,
                output_pth, manuscript_root
            ))

    # 2. 处理当前章节的 Segments (并发)
    seg_tasks = []
    if section.segments:
        print(f"\n====== 启动章节 Segments 并发处理: {parent_id} ======\n")
        for segment in section.segments:
            seg_tasks.append(process_single_segment(
                segment, task_desc, agent_factory, semaphore
            ))

    # 3. 等待所有 Segments 完成
    if seg_tasks:
        await asyncio.gather(*seg_tasks)

        # 4. 生成标题 (Segments 完成后才能做总结)
        # 这里需要一个临时的 writer 来做总结
        global CURRENT_RUNNING_TASKS  # 引入全局变量
        CURRENT_RUNNING_TASKS += 1
        print(
            f"[{time.strftime('%H:%M:%S')}] [并发数: {CURRENT_RUNNING_TASKS}] 🏷️ 生成标题: {section.title[:10]}...", flush=True)

        section_text = "\n".join([s.content for s in section.segments])
        llm_instruct = create_chat_model(reasoning=False)
        formatter = create_agent_formatter()
        def _parse_res(text):
            title = re.search("<title>(.+)</title>", text, re.DOTALL)
            content = re.search("<content>(.+)</content>", text, re.DOTALL)
            assert title is not None and content is not None, "输出格式不对，答案没有被合适的标签包裹住。"
            title = title.group(1).strip().strip("#").strip()
            content = content.group(1).strip()
            return title, content
        title, content = await call_chatbot_with_retry(
            llm_instruct, formatter,
            "你是撰写金融研报的专家。我将提供某一章节初稿，请你删去无意义的部分，输出润色后的内容，不要篡改关键信息。",
            f"金融研报某一章节初稿如下：\n\n{section_text}\n\n"
            f"该章节是参考了小标题为{section.title}的某个范例撰写的，请你根据初稿重新起一个标题，用<title>和</title>包裹住，限十字以内。"
            f"并在初稿基础上稍作润色，更新后的内容用<content>和</content>包裹住。",
            _parse_res, handle_hook_exceptions=(AssertionError, )
        )
        section.title = title
        section.content = content
        print(f"[Final section] {section.title}")
        print(section.content)
        CURRENT_RUNNING_TASKS -= 1

    # 5. 等待子章节递归完成 (如果需要严格的层级顺序保存，可以调整 await 位置)
    if sub_tasks:
        await asyncio.gather(*sub_tasks)

    # 6. 保存中间结果 (可选，防止崩溃全丢)
    # 注意：并发写入文件可能冲突，这里简单处理，实际生产建议用单独的 save 协程或锁
    (output_pth / f"{stock_symbol}_{os.getenv("CUR_DATE", datetime.today().strftime("%Y-%m-%d"))}.json").write_text(manuscript_root.to_json(ensure_ascii=False))


async def run_workflow(task_desc: str):
    """围绕一个 task description 执行完整的研报生成流程。
    """
    # ----- 1. 准备 memory store -----

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    long_term_dir = PROJECT_ROOT / "data" / "memory" / "long_term"
    
    long_term = LongTermMemoryStore(
        base_dir=long_term_dir,
    )


    planner_cfg = cfg.get_planner_cfg()
    use_demo = planner_cfg.get("use_demonstration", False)

    entity = get_entity_info(long_term, task_desc)
    if not entity or not entity.get("code"):
        raise ValueError(f"无法从 task_desc 解析股票实体/代码：{task_desc}")

    stock_symbol = entity["code"]  # 纯数字 6 位代码
    print("股票代码：", stock_symbol)

    filename = f"{stock_symbol}_{os.getenv("CUR_DATE", datetime.today().strftime("%Y-%m-%d"))}"
    short_term_dir = PROJECT_ROOT / "data" / "memory" / "short_term" / filename

    short_term = ShortTermMemoryStore(
        base_dir=short_term_dir,
    )

    # 解析demonstration report，第二遍解析同一个report可以注释掉
    demo_pdf_path = STOCK_REPORT_PATHS[stock_symbol][-1]
    manuscript = await process_pdf_to_outline(demo_pdf_path, long_term_dir / "demonstration",
                                              llm_reasoning, llm_instruct, formatter,)

    verifier_toolkit = build_verifier_toolkit(
        short_term=short_term,
        long_term=long_term,
    )
    verifier = create_verifier_agent(model=llm_reasoning, formatter=formatter, toolkit=verifier_toolkit)

    output_pth = PROJECT_ROOT / "data" / "output" / "reports"

    # 设置并发信号量
    CONCURRENCY_LIMIT = int(os.getenv("N_THREAD", 32))
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    def create_searcher_writer():
        searcher_toolkit = build_searcher_toolkit(
            short_term=short_term,
            long_term=long_term,
        )
        searcher = create_searcher_agent(model=llm_reasoning, formatter=formatter, toolkit=searcher_toolkit)
        writer_toolkit = build_writer_toolkit(
            short_term=short_term,
            long_term=long_term,
            searcher=searcher,
        )
        writer = create_writer_agent(model=llm_reasoning, formatter=formatter, toolkit=writer_toolkit)
        return searcher, writer

    # 启动递归并发处理
    await process_section_concurrently(
        section=manuscript,
        parent_id=None,
        task_desc=task_desc,
        agent_factory=create_searcher_writer,
        semaphore=semaphore,
        stock_symbol=stock_symbol,
        output_pth=output_pth,
        manuscript_root=manuscript # 用于在深层递归中保存完整的 json
    )

    markdown_text = section_to_markdown(manuscript)
    (output_pth / f"{filename}.md").write_text(markdown_text, encoding="utf-8")
    md_to_pdf(markdown_text, short_term=short_term, output_dir=output_pth / f"{filename}.pdf")
