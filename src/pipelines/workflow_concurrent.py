# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import os
import time
import json
import pickle
import re
import sys
import traceback
import warnings
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path

from agentscope.agent import ReActAgent
from agentscope.message import Msg

from src.evaluation.eval_content import evaluate_segment
from src.pipelines.planning import process_pdf_to_outline
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
from src.utils.multi_types_verification import SegmentVerifier
from src.utils.local_file import STOCK_REPORT_PATHS
import config
from src.utils.call_with_retry import call_chatbot_with_retry
from src.utils.instance import llm_reasoning, llm_instruct, llm_judge, formatter

# 用于保护并发写入文件的锁
SAVE_LOCK = asyncio.Lock()

async def search_evidence(query, known_evidence, task_desc, demo_date, segment_topic, searcher, reference=None):
    searcher_input = Msg(
        name="user",
        content=(
            f"任务：{task_desc}\n"
            f"当前需要你撰写要点：{segment_topic}\n"
            f"{known_evidence}\n"
            f"论据还需要的材料：{query}\n\n"
            f"如果该材料并非可以搜索得到的（例如画图要求）、或者不需要搜索的声明（例如数据来源）、或者已收集材料已覆盖的，可以直接返回“{query}”，不做搜索和修改。否则，可以查看\n\n"
            + (f"{demo_date}发布了一份历史研报，以下一段内容可能包含所需材料：{reference}\n"
              f"如果该内容包含所需材料，并且一定不会因时间变化，在当前撰写时间依然成立，则不必调用搜索，摘取出涉及该材料的两三句话作为论据即可，"
              f"并加上 [^cite_id:{demo_date}_reference_report] 作为markdown风格引用。"
              f"如果没有符合时效性论据，\n\n"
            if reference else "") +
            f"请你调用工具搜索，尽量根据多个信息源交叉验证后给出搜索结果。最终给出的答案需要简洁明了。"
        ),
        role="user",
    )
    print(searcher_input.content)
    try:
        msg = await call_agent_with_retry(searcher, searcher_input)
        print(f"[Searcher] Finished searching: {query[:20]}...")
        return msg.get_text_content()
    except Exception as err:
        print(f"[Searcher] Failed searching: {query[:20]}... - Error: {type(err).__name__}: {err}")
        traceback.print_exc()
        raise  # Re-raise the exception to propagate it properly

async def process_single_segment(segment: Segment,
                                 task_desc,
                                 demo_date,
                                 agent_factory,
                                 short_term,
                                 long_term,
                                 multi_source_verification_enabled,
                                 max_verify_rounds):
    """并发处理单个 Segment：包含搜索和写作"""
    print(f"[{time.strftime('%H:%M:%S')}]  ✍️ 开始写作: {segment.topic[:15]}...", flush=True)

    searcher, writer, verifier = agent_factory()
    for i, evidence in enumerate(segment.evidences):
        if segment.reference and '【画图内容要求】' in segment.reference:
            continue
        evidences = [e for e in segment.evidences[:i] if e]
        known_evidence = ("当前已搜索到的论据：\n" + "\n".join(evidences) + "\n") if evidences else ""
        segment.evidences[i] = await search_evidence(evidence, known_evidence, task_desc, demo_date, segment.topic, searcher, reference=segment.reference)
        await searcher.memory.clear()

    try:
        writer_input = Msg(
            name="user",
            content=(
                f"任务：{task_desc}\n"
                f"当前步骤需要你撰写要点：\n{segment.topic}\n"
                f"参考写作模版、写作要求和可能谈及的论据如下：\n\n{str(segment)}\n\n"
                f"请你开始搜索和撰写。"
            ),
            role="user",
        )

        draft_msg = await call_agent_with_retry(writer, writer_input)
        segment.content = draft_msg.get_text_content()
        assert segment.content is not None, str(segment)
        print(f"[Writer] Segment finished: {segment.topic}")
        print("[Writer 初稿输出]")
        print(segment.content, flush=True)

        for _ in range(3):
            await searcher.memory.clear()
            suggestions = await evaluate_segment(llm_judge,
                                                 create_agent_formatter(), segment)
            if suggestions is None:
                break
            else:
                print("修改建议:", suggestions, flush=True)
                writer_input = Msg(
                    name="user", content=f"经评估：\n{suggestions}\n请你继续修改。不要把修改说明放到最终回答中。", role="user",
                )
                draft_msg = await call_agent_with_retry(writer, writer_input)
                segment.content = draft_msg.get_text_content()
                print(f"[Writer] Segment finished: {segment.topic}")
                print(segment.content, flush=True)


        if multi_source_verification_enabled:

            prev_issue_set = set()
            for _ in range(max_verify_rounds):

                verify_issues = await verifier.verify(segment.content)

                # 过滤严重问题
                verify_issues = [
                    iss for iss in verify_issues
                    if iss.severity in ("critical", "major")
                ]
                if not verify_issues:
                    break

                # 收敛检测
                current_set = {
                    (iss.type, iss.description)
                    for iss in verify_issues
                }

                if current_set == prev_issue_set:
                    break
                prev_issue_set = current_set

                # 格式化
                def format_issues(issues):
                    lines = []
                    for i, iss in enumerate(issues[:8], 1):
                        lines.append(
                            f"{i}. [{iss.severity.upper()}] {iss.description}\n"
                            f"   建议: {iss.suggestion}"
                        )
                    return "\n\n".join(lines)

                verify_feedback = format_issues(verify_issues)

                # token 控制
                verify_feedback = verify_feedback[:1500]

                # 生成 rewrite prompt
                writer_input = Msg(
                    name="user",
                    content=(
                        "以下是对你当前段落的事实核验问题，请你直接修正文：\n\n"
                        f"{verify_feedback}\n\n"
                        "要求：\n"
                        "1. 保留可被 cite_id 支持的内容\n"
                        "2. 删除或降级无法验证的断言\n"
                        "3. 必须使用已有 cite_id\n"
                        "4. 禁止新增未验证数字"
                    ),
                    role="user",
                )

                draft_msg = await call_agent_with_retry(writer, writer_input)

                segment.content = draft_msg.get_text_content()

        await writer.memory.clear()
        segment.finished = True
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 完成写作: {segment.topic[:15]}.", flush=True)
    except Exception as e:
        traceback.print_exc()
        raise e

async def process_section_concurrently(section: Section, parent_id, task_desc, demo_date, cur_date,
                                       agent_factory, stock_symbol, output_pth, manuscript_root, short_term, long_term,
                                       multi_source_verification_enabled, max_verify_rounds):
    """递归并发处理章节"""

    # 1. 处理子章节 (递归) - 优先启动子任务
    sub_tasks = []
    if section.subsections:
        for subsection in section.subsections:
            section_id = ((parent_id + ".") if parent_id else "") + str(subsection.section_id)
            # 递归调用
            sub_tasks.append(process_section_concurrently(
                subsection, section_id, task_desc, demo_date, cur_date, agent_factory,
                stock_symbol, output_pth, manuscript_root, short_term, long_term, multi_source_verification_enabled, max_verify_rounds
            ))

    # 2. 处理当前章节的 Segments (并发)
    seg_tasks = []
    if section.segments:
        print(f"\n====== 启动章节 Segments 并发处理: {parent_id} ======\n")
        for segment in section.segments:
            if segment.finished:
                continue
            seg_tasks.append(process_single_segment(
                segment, task_desc, demo_date, agent_factory, short_term, long_term, multi_source_verification_enabled, max_verify_rounds
            ))

    # 3. 等待所有 Segments 完成
    if seg_tasks:
        results = await asyncio.gather(*seg_tasks, return_exceptions=True)

        # 检查是否有失败的任务
        failed_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_count += 1
                warnings.warn(f"❌ Segment 处理失败: {section.segments[i].topic[:30]}... - {type(result).__name__}: {str(result)}")

        if failed_count > 0:
            warnings.warn(f"⚠️  警告: {failed_count}/{len(seg_tasks)} 个 Segment 处理失败")

        # 4. 生成标题 (只有在有成功的 Segments 时才执行)
        successful_segments = [s for s in section.segments if s.finished]
        if successful_segments:
            # 这里需要一个临时的 writer 来做总结
            print(
                f"[{time.strftime('%H:%M:%S')}]  🏷️ 生成标题: {section.title[:10]}...", flush=True)

            section_text = "\n".join([s.content for s in successful_segments if s.content])
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

    # 5. 等待子章节递归完成 (如果需要严格的层级顺序保存，可以调整 await 位置)
    if sub_tasks:
        sub_results = await asyncio.gather(*sub_tasks, return_exceptions=True)

        failed_subsections = 0
        for i, result in enumerate(sub_results):
            if isinstance(result, Exception):
                failed_subsections += 1
                subsection = section.subsections[i]
                warnings.warn(f"❌ 子章节处理失败: {subsection.title[:30]}... - {type(result).__name__}: {str(result)}")

        if failed_subsections > 0:
            warnings.warn(f"⚠️  警告: {failed_subsections}/{len(sub_tasks)} 个子章节处理失败")

    # 6. 保存中间结果 (可选，防止崩溃全丢)
    # 注意：并发写入文件可能冲突，这里简单处理，实际生产建议用单独的 save 协程或锁
    async with SAVE_LOCK:
        (output_pth / f"{stock_symbol}_{cur_date}.json").write_text(manuscript_root.to_json(ensure_ascii=False) ,encoding="utf-8")


async def run_workflow(task_desc: str, cur_date=None, demo_pdf_path=None):
    """围绕一个 task description 执行完整的研报生成流程。
    """
    cur_date = cur_date or os.getenv('CUR_DATE', datetime.now().strftime("%Y%m%d"))
    # ----- 1. 准备 memory store -----

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    long_term_dir = PROJECT_ROOT / "data" / "memory" / "long_term"
    
    long_term = LongTermMemoryStore(
        base_dir=long_term_dir,
    )

    cfg = config.Config()
    multi_source_verification_enabled = cfg.is_multi_source_verification_enabled()
    max_verify_rounds = cfg.get_max_verify_rounds()

    entity = get_entity_info(long_term, task_desc)
    if not entity or not entity.get("code"):
        raise ValueError(f"无法从 task_desc 解析股票实体/代码：{task_desc}")
    stock_symbol = entity["code"] if entity else "unknown"
    print("股票代码：", stock_symbol)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{stock_symbol}_{now_str}.txt"
    log_file = open(log_filename, "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file

    cfg = config.Config()
    filename = f"{stock_symbol}_{cur_date}"
    short_term_dir = PROJECT_ROOT / "data" / "memory" / "short_term" / filename

    short_term = ShortTermMemoryStore(
        base_dir=short_term_dir,
    )

    # 解析demonstration report，第二遍解析同一个report可以注释掉
    if demo_pdf_path is None:
        demo_pdf_path = STOCK_REPORT_PATHS[stock_symbol][-1]
    demo_date = demo_pdf_path.name.split("_")[1]

    output_pth = PROJECT_ROOT / "data" / "output" / "reports" / cfg.llm_name
    output_pth.mkdir(parents=True, exist_ok=True)

    outline = await process_pdf_to_outline(demo_pdf_path, long_term_dir / "demonstration",
                                              llm_reasoning, llm_instruct, formatter,)
    manuscript_path = output_pth / f"{stock_symbol}_{cur_date}.json"
    if manuscript_path.exists():
        manuscript = Section.from_json(manuscript_path.read_text(encoding='utf-8'))
        print("加载已有的 manuscript:", manuscript_path)
    else:
        manuscript = outline
        short_term.save_material(
            cite_id=f"{demo_date}_reference_report",
            content=(long_term_dir / "demonstration" / f"{demo_pdf_path.stem}.md").read_text(encoding='utf-8'),
            description=demo_pdf_path.name,
            source="",
            entity=entity,
            time={"point": str(demo_date)},
        )

    def unfinished(section: Section) -> bool:
        if section.segments:
            for segment in section.segments:
                if not segment.finished:
                    return True
        if section.subsections:
            for subsection in section.subsections:
                if unfinished(subsection):
                    return True
        return False

    if unfinished(manuscript):
        def replace_unfinished_sections(manuscript: Section, outline: Section) -> None:
            """递归替换包含未完成 segment 的 section"""
            # 检查当前 section 是否有未完成的 segment
            has_unfinished = False
            if manuscript.segments:
                for segment in manuscript.segments:
                    if not segment.finished:
                        has_unfinished = True
                        break

            # 如果当前 section 有未完成的 segment，整体替换
            if has_unfinished:
                # 复制 outline 对应 section 的所有属性
                manuscript.title = outline.title
                manuscript.content = outline.content
                manuscript.segments = outline.segments
                manuscript.subsections = outline.subsections
                return

            # 递归处理子 section
            if manuscript.subsections and outline.subsections:
                for i, subsection in enumerate(manuscript.subsections):
                    if i < len(outline.subsections):
                        replace_unfinished_sections(subsection, outline.subsections[i])

        replace_unfinished_sections(manuscript, outline)
        def create_searcher_writer_verifier():
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
            verifier = SegmentVerifier(short_term, long_term)
            return searcher, writer,verifier

        # 启动递归并发处理
        await process_section_concurrently(
            section=manuscript,
            parent_id=None,
            task_desc=task_desc,
            demo_date=demo_date,
            cur_date=cur_date,
            agent_factory=create_searcher_writer_verifier,
            stock_symbol=stock_symbol,
            output_pth=output_pth,
            manuscript_root=manuscript,  # 用于在深层递归中保存完整的 json
            short_term=short_term,
            long_term=long_term,
            multi_source_verification_enabled=multi_source_verification_enabled,
            max_verify_rounds=max_verify_rounds,
        )

    markdown_text = section_to_markdown(manuscript)
    (output_pth / f"{filename}.md").write_text(markdown_text, encoding="utf-8")
    md_to_pdf(markdown_text, short_term=short_term, pdf_path=output_pth / f"{filename}.pdf")
