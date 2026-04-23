# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import os
import time
import json
import pickle
import sys
import traceback
import warnings
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
import pandas as pd

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
from src.utils.file_converter import md_to_pdf, pdf_to_markdown, section_to_markdown
from src.utils.parse_verdict import parse_verifier_verdict
from src.utils.call_with_retry import call_agent_with_retry
from src.utils.get_entity_info import get_entity_info
from src.utils.file_converter import markdown_to_sections
from src.utils.format import (
    _infer_report_title,
    _normalize_report_title,
    _normalize_section_titles,
    _strip_section_number_prefix,
    extract_tagged_text,
    extract_writer_content,
    print_section_reference_warning,
)
from src.utils.multi_types_verification import (
    SegmentVerifier,
    append_verifier_trace,
    set_verifier_trace_path,
    evaluate_claims,
    compute_segment_report,
    format_report_for_writer,
)
from src.utils.local_file import STOCK_REPORT_PATHS
from src.utils.task_date import normalize_compact_date
import config
from src.utils.call_with_retry import call_chatbot_with_retry
from src.utils.instance import llm_reasoning, llm_instruct, llm_judge, formatter
import logging
from typing import List, Optional, Dict, Any
from src.tools.material_tools import MaterialTools

DEFAULT_DISCLOSURE_CATEGORIES = ["年报", "半年报", "一季报", "三季报"]

def _has_preloaded_disclosures(
    short_term: ShortTermMemoryStore,
    symbol: str,
    start_date: str,
    end_date: Optional[str],
    disclosure_categories: List[str],
) -> bool:
    if not short_term or not hasattr(short_term, "_registry"):
        return False

    required_categories = {category for category in disclosure_categories if category}
    if not required_categories:
        return False

    normalized_symbol = (symbol or "").split(".")[0]
    normalized_start = normalize_compact_date(start_date)
    normalized_end = normalize_compact_date(end_date) if end_date else None
    loaded_categories = set()

    for meta in short_term._registry.values():
        if getattr(meta, "source", "") != "AKshare API:CNINFO":
            continue

        entity_code = ((getattr(meta, "entity", {}) or {}).get("code") or "").split(".")[0]
        if entity_code != normalized_symbol:
            continue

        point = normalize_compact_date((getattr(meta, "time", {}) or {}).get("point"))
        if point is not None:
            if point < normalized_start:
                continue
            if normalized_end is not None and point > normalized_end:
                continue

        description = (getattr(meta, "description", "") or "").lower()
        for category in required_categories - loaded_categories:
            if f"category={category}".lower() in description:
                loaded_categories.add(category)

        if loaded_categories == required_categories:
            return True

    return False

async def preload_task_materials(
    tools: MaterialTools, 
    symbol: str, 
    start_date: str, 
    end_date: Optional[str] = None,
    news_keywords: Optional[List[str]] = None,
    disclosure_categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    针对单次任务的前置知识摄取流水线。
    复用 MaterialTools 中的工具，将长文本材料提前存入 ShortTermMemoryStore。
    """
    logging.info(f"[Pre-loader] 🚀 开始为任务预加载知识库: Entity={symbol}, 时间={start_date}至{end_date}")
    print(f"[Pre-loader] 🚀 开始为任务预加载知识库: Entity={symbol}, 时间={start_date}至{end_date}")

    news_keywords = news_keywords or [""]
    disclosure_categories = disclosure_categories or DEFAULT_DISCLOSURE_CATEGORIES
    if _has_preloaded_disclosures(
        tools.short_term,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        disclosure_categories=disclosure_categories,
    ):
        current_registry_size = len(tools.short_term._registry) if tools.short_term else 0
        msg = f"检测到 {symbol} 在目标时间窗内的公告材料已预加载完成，跳过重复预加载。"
        logging.info(f"[Pre-loader] ⏭️ {msg}")
        print(f"[Pre-loader] ⏭️ {msg}")
        return {
            "symbol": symbol,
            "news_batches_loaded": 0,
            "disclosure_categories_loaded": 0,
            "errors": [],
            "total_materials_in_memory": current_registry_size,
            "skipped": True
        }
    print("symbol:", symbol)
    stats = {
        "symbol": symbol,
        "news_batches_loaded": 0,
        "disclosure_categories_loaded": 0,
        "errors": []
    }
    # ==========================================
    # 阶段 2：预加载并解析 PDF 公告
    # ==========================================
    for category in disclosure_categories:
        try:
            logging.info(f"[Pre-loader] 📄 正在拉取并解析公告 (类型: '{category}')")
            print(f"[Pre-loader] 📄 正在拉取并解析公告 (类型: '{category}')")
            # 触发 _fetch_pdf_text 逻辑，
            # 并把每篇公告作为独立的 cite_id 存入 registry
            await tools.fetch_disclosure_material(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                market="沪深京",
                keyword="", 
                category=category
            )
            stats["disclosure_categories_loaded"] += 1
        except Exception as e:
            err_msg = f"公告预加载失败 (类型:{category}): {str(e)}"
            logging.error(f"[Pre-loader] ❌ {err_msg}")
            stats["errors"].append(err_msg)
            
    # ==========================================
    # 阶段 3：缓存对齐与状态确认
    # ==========================================
    registry_size = len(tools.short_term._registry) if tools.short_term else 0
    stats["total_materials_in_memory"] = registry_size
    
    logging.info(f"[Pre-loader] ✅ 预加载完成！当前内存池共有 {registry_size} 条独立材料。")
    print(f"[Pre-loader] ✅ 预加载完成！当前内存池共有 {registry_size} 条独立材料。")
    
    if stats["errors"]:
        logging.warning(f"[Pre-loader] ⚠️ 预加载期间发生了 {len(stats['errors'])} 个错误。")
        
    return stats

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
        segment.content = extract_writer_content(draft_msg.get_text_content())
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
                    name="user",
                    content=(
                        f"经评估：\n{suggestions}\n"
                        "请你继续修改。修改后的正文请使用<content>和</content>包裹。"
                    ),
                    role="user",
                )
                draft_msg = await call_agent_with_retry(writer, writer_input)
                segment.content = extract_writer_content(draft_msg.get_text_content())
                print(f"[Writer] Segment finished: {segment.topic}")
                print(segment.content, flush=True)

        if multi_source_verification_enabled:
            prev_priority_sigs = None
            prev_issue_sig = None

            for round_idx in range(max_verify_rounds):
                current_text = segment.content
                print(f"[Verifier Loop] round={round_idx + 1}/{max_verify_rounds} topic={segment.topic}", flush=True)
                print("[Verifier Checked Text]", flush=True)
                print(current_text, flush=True)

                claims, all_issues = await verifier.verify_with_claims(current_text)
                if not claims:
                    print("[Verifier Loop] No claims extracted, stop")
                    break

                claim_evals = evaluate_claims(claims, all_issues)
                critical_count = sum(ce.signature["critical"] for ce in claim_evals)

                # 动态计算 top_k，避免重复调用 compute_segment_report
                bad_count = sum(1 for ce in claim_evals if ce.claim_score < 3.0)
                top_k = min(5, max(2, bad_count))
                report = compute_segment_report(claim_evals, top_k=top_k)
                print(f"[Verifier Loop] segment_score={report.segment_score}, passed={report.passed}, bad_claims={report.bad_claims_count}", flush=True)

                # 高分通过（必须无 critical issue）
                if report.segment_score >= 80 and critical_count == 0:
                    print(f"[Verifier Loop] High quality (score={report.segment_score}) → stop", flush=True)
                    await append_verifier_trace(
                        topic=segment.topic,
                        round_idx=round_idx + 1,
                        checked_text=current_text,
                        issue_count=0,
                        status="passed_high_score",
                        score=report.segment_score,
                        star_rating=report.star_rating,
                        passed=True,
                        priority_claims_count=0,
                    )
                    break

                # 收敛检测：priority claims 文本不变 + issue 数量不变
                current_priority_sigs = {ce.original_text.strip()[:80] for ce in report.priority_claims}
                current_issue_sig = (
                    sum(len(ce.issues) for ce in report.priority_claims),
                    sum(ce.signature["critical"] for ce in report.priority_claims),
                )
                if prev_priority_sigs is not None and prev_issue_sig is not None:
                    same_priority = current_priority_sigs == prev_priority_sigs
                    same_issues = current_issue_sig == prev_issue_sig
                    if same_priority and same_issues:
                        print(f"[Verifier Loop] Early stop: priority unchanged, issues stuck at {current_issue_sig}", flush=True)
                        break

                prev_priority_sigs = current_priority_sigs
                prev_issue_sig = current_issue_sig

                verify_feedback = format_report_for_writer(report)
                print(f"[Verifier Loop] feedback_chars={len(verify_feedback)}", flush=True)
                print("[Verifier Feedback To Writer]", flush=True)
                print(verify_feedback, flush=True)

                writer_input = Msg(
                    name="user",
                    content=(
                        "以下是你当前段落的**评估报告**。请**优先重写报告中列出的 Priority Claims**，一次性修正它们的所有问题。\n\n"
                        f"{verify_feedback}\n\n"
                        "要求：\n"
                        "1. 对每个 Priority Claim，根据其「问题汇总」和「修改建议」整体重写该句子。\n"
                        "2. 其他未列出的声明可保持不变或微调。\n"
                        "3. 保留可被 cite_id 支持的内容，删除无法验证的断言。\n"
                        "4. 必须使用已有 cite_id，禁止新增未验证数字。\n"
                        "5. 输出的段落正文使用 <content> 和 </content> 包裹，不要包含额外说明。"
                    ),
                    role="user",
                )

                draft_msg = await call_agent_with_retry(writer, writer_input)
                new_content = extract_writer_content(draft_msg.get_text_content())

                if new_content.strip() == current_text.strip():
                    print("[Verifier Loop] Early stop: rewrite produced no change", flush=True)
                    break
                segment.content = new_content

                print("[Writer Rewritten After Verifier]", flush=True)
                print(segment.content, flush=True)

                priority_issue_count = sum(len(ce.issues) for ce in report.priority_claims)
                await append_verifier_trace(
                    topic=segment.topic,
                    round_idx=round_idx + 1,
                    checked_text=current_text,
                    verify_feedback=verify_feedback,
                    rewritten_text=segment.content,
                    issue_count=priority_issue_count,
                    status=f"score={report.segment_score}" if report.segment_score < 75 else "passed",
                    score=report.segment_score,
                    star_rating=report.star_rating,
                    passed=report.passed,
                    priority_claims_count=len(report.priority_claims),
                )
            
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

    tools = MaterialTools(short_term=short_term, long_term=long_term)
    end_date = normalize_compact_date(cur_date)
    start_date = (
        pd.to_datetime(end_date, format="%Y%m%d") - pd.DateOffset(months=6)
    ).strftime("%Y%m%d")
    stats = await preload_task_materials(
        tools=tools,
        symbol=stock_symbol,
        start_date=start_date,
        end_date=end_date,
        disclosure_categories=DEFAULT_DISCLOSURE_CATEGORIES,
    )

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
                title = extract_tagged_text(text, "title")
                content = extract_tagged_text(text, "content")
                assert title is not None and content is not None, "输出格式不对，答案没有被合适的标签包裹住。"
                title = _strip_section_number_prefix(title.strip().strip("#").strip())
                return title, content
            title, content = await call_chatbot_with_retry(
                llm_instruct, formatter,
                "你是撰写金融研报的专家。我将提供某一章节初稿，请你删去无意义的部分，修改不连贯、不流畅的内容，输出润色后的内容，不要篡改关键信息。",
                f"金融研报某一章节初稿如下：\n\n{section_text}\n\n"
                f"该章节是参考了小标题为{section.title}的某个范例撰写的，请你根据初稿重新起一个标题，用<title>和</title>包裹住，限十字以内。"
                "并在初稿基础上稍作润色，更新后的内容用<content>和</content>包裹住。\n\n"
                "额外要求：\n"
                "1. 所有 [^cite_id:xxx] 引用标记都必须原样保留，不允许删除、改写、合并或新增。\n"
                "2. 所有 ![...](chart:chart_xxx) 图表标记都必须原样保留，不允许删除、改写或新增。\n",
                _parse_res, handle_hook_exceptions=(AssertionError, )
            )
            print_section_reference_warning(section.title, section_text, content)
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
    entity = get_entity_info(long_term, task_desc)
    if not entity or not entity.get("code"):
        raise ValueError(f"无法从 task_desc 解析股票实体/代码：{task_desc}")
    stock_symbol = entity["code"] if entity else "unknown"
    print("股票代码：", stock_symbol)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{stock_symbol}_{now_str}.txt"
    verifier_trace_filename = f"verifier_trace_{stock_symbol}_{now_str}.txt"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_filename, "w", encoding="utf-8")
    set_verifier_trace_path(PROJECT_ROOT / verifier_trace_filename)
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        cfg = config.Config()
        multi_source_verification_enabled = cfg.is_multi_source_verification_enabled()
        max_verify_rounds = cfg.get_max_verify_rounds()
        
        filename = f"{stock_symbol}_{cur_date}"
        short_term_dir = PROJECT_ROOT / "data" / "memory" / "short_term" / filename

        short_term = ShortTermMemoryStore(
            base_dir=short_term_dir,
        )
        
        if demo_pdf_path is None:
            demo_pdf_path = STOCK_REPORT_PATHS[stock_symbol][-1]
        demo_date = demo_pdf_path.name.split("_")[1]

        output_pth = PROJECT_ROOT / "data" / "output" / "reports" / cfg.llm_name
        output_pth.mkdir(parents=True, exist_ok=True)

        outline = await process_pdf_to_outline(demo_pdf_path, long_term_dir / "demonstration",
                                                  llm_reasoning, llm_instruct, formatter,)
        _normalize_section_titles(outline)
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
        _normalize_section_titles(manuscript)

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

        _normalize_section_titles(manuscript)
        _normalize_report_title(manuscript, entity, task_desc)
        (output_pth / f"{filename}.json").write_text(manuscript.to_json(ensure_ascii=False), encoding="utf-8")
        markdown_text = section_to_markdown(manuscript)
        (output_pth / f"{filename}.md").write_text(markdown_text, encoding="utf-8")
        md_to_pdf(
            markdown_text,
            short_term=short_term,
            pdf_path=output_pth / f"{filename}.pdf",
            header_title=_infer_report_title(task_desc, entity),
        )
    finally:
        try:
            log_file.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()
            set_verifier_trace_path(None)
