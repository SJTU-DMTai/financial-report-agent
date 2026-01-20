# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import pickle
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Dict

from agentscope.agent import ReActAgent
from agentscope.message import Msg

from src.memory.working import Section, Segment
from src.prompt import prompt_dict
from src.utils.instance import create_chat_model, create_agent_formatter
from src.memory.short_term import ShortTermMemoryStore
from src.agents.searcher import create_searcher_agent, build_searcher_toolkit
from src.agents.writer import create_writer_agent, build_writer_toolkit
from src.agents.planner import create_planner_agent, build_planner_toolkit
# from src.agents.verifier import create_verifier_agent, build_verifier_toolkit
from src.agents.verifier import create_verifier_agent, build_verifier_toolkit, create_all_verifiers, create_final_verifier

from src.utils.file_converter import md_to_pdf, pdf_to_markdown, section_to_markdown
from src.utils.call_agent_with_retry import call_agent_with_retry
import config
import asyncio

from src.utils.file_converter import markdown_to_sections
from src.utils.local_file import STOCK_REPORT_PATHS
from src.evaluation.parse_verifier_verdict import parse_verdict
from src.evaluation.segment_scorer import SegmentScorer, SegmentScore


async def run_workflow(task_desc: str) -> str:
    """围绕一个 task description 执行完整的研报生成流程。
    """

    cfg = config.Config()
    formatter = create_agent_formatter()

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
    model_instruct = create_chat_model(reasoning=False)

    # ----- 初始化评估系统 -----
    print("\n" + "="*60)
    print("初始化评估系统")
    print("="*60)
    
    segment_scorer = SegmentScorer(model, formatter)
    print("评估系统初始化完成")

    # ----- 3. 创建 Searcher Agent -----
    searcher_toolkit = build_searcher_toolkit(
        short_term=short_term,
        # tool_use_store=tool_use_store,
    )

    searcher = create_searcher_agent(model=model, formatter=formatter, toolkit=searcher_toolkit)
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
            print(type(outline_msg))
            print(outline_msg)
            print(outline_msg.get_text_content())

            stock_symbol = re.search(r"[0-9]+", outline_msg.get_text_content()).group()
            assert stock_symbol is not None
            print("股票代码：", stock_symbol)
            await searcher.memory.clear()
            break
        except AssertionError as e:
            print(e)

    # 解析demonstration report，第二遍解析同一个report可以注释掉
    demo_pdf_path = STOCK_REPORT_PATHS[stock_symbol][-1]
    demo_date, demo_name = demo_pdf_path.name.split(".")[0].split("_")[-2:]
    demo_md_path = short_term_dir / f"demonstration" / (demo_pdf_path.name.split(".")[0] + ".md")
    if not demo_md_path.exists():
        final_text, images = pdf_to_markdown(demo_pdf_path, demo_md_path)
    manuscript: Section = markdown_to_sections(demo_md_path)

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

    # planner = create_planner_agent(model=model, formatter=formatter, toolkit=None)

    async def dfs_outline(section: Section, parent_id=None):
        if section.subsections is None:
            return
        for subsection in section.subsections:
            section_id = ((parent_id + ".") if parent_id else "") + str(subsection.section_id)
            print(f"\n====== 开始总结章节 {section_id} ======\n")
            await dfs_outline(subsection)
            if subsection.segments:
                decomposer_input = await formatter.format([
                    Msg("system", prompt_dict["decompose"],"system"),
                    Msg("user", subsection.segments[0].reference.replace("<SEP>", ""), "user", )
                ])
                for i in range(10):
                    try:
                        decomposed_content = await model_instruct(decomposer_input)
                        break
                    except Exception as e:
                        print(e)
                segments = Msg("assistant", decomposed_content.content, "assistant").get_text_content().split("<SEP>")
                subsection.segments = []
                for i, segment in enumerate(segments):
                    planner_input = [
                        Msg("system", prompt_dict["plan_outline"],"system"),
                        Msg(
                            name="user",
                            content=f"当前任务：{task_desc}\n\n为实现当前任务，我找到了某机构在{demo_date}撰写的一份研报，名为{demo_name}。"
                                    f"下文将附上从中摘出的一段参考片段，请你考虑时间差和公司异同，撰写一份用于当前新任务的撰写模版和要求。\n\n"
                                    f"参考片段如下：\n\n{segment}",
                            role="user",
                        )
                    ]
                    # outline_msg = await planner(planner_input)
                    print(segment, flush=True)
                    for i in range(10):
                        try:
                            _input = await formatter.format(planner_input)
                            outline_msg = await model(_input)
                            # print(outline_msg.get_text_content())
                            outline_msg = Msg("assistant", outline_msg.content, "assistant")
                            subsection.segments.append(subsection.parse(outline_msg.get_text_content()))
                            subsection.segments[-1].reference = segment
                            break
                        except AssertionError as e:
                            print(e)
                            planner_input += [
                                outline_msg,
                                Msg("user", str(e), "user")
                            ]
            print(subsection.read(True, True, True, True, False, False))

    outline_json_pth = short_term_dir / "outline.json"
    if not outline_json_pth.exists():
        await dfs_outline(manuscript)
        outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True, fold_other=False)
        print(outline)
        outline_json_pth.write_text(manuscript.to_json(ensure_ascii=False))
    else:
        # outline = outline_md_pth.read_text()
        manuscript = Section.from_json(outline_json_pth.read_text())
        outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True, fold_other=False)
        print(outline)

    # # ----- 6. 创建所有Verifier -----
    # verifiers = create_all_verifiers(model, formatter, short_term)
    # final_verifier = create_final_verifier(model, formatter, short_term)

    # ----- 7. 调用 Writer：基于 outline.md 写 Manuscript 并导出 PDF -----
    writer_toolkit = build_writer_toolkit(
        short_term=short_term,
        searcher=searcher,
    )
    writer = create_writer_agent(model=model, formatter=formatter, toolkit=writer_toolkit)

    # verifier_toolkit = build_verifier_toolkit(
    #     short_term=short_term,
    # )
    # verifier = create_verifier_agent(model=model, formatter=formatter, toolkit=verifier_toolkit)

    output_pth = PROJECT_ROOT / "data" / "output" / "reports"

    # 存储所有segment的评分
    all_segment_scores: List[SegmentScore] = []
    segment_counter = 0
    
    async def dfs_report(section: Section, parent_id=None):
        nonlocal segment_counter
        if section.subsections is None:
            return
        for subsection in section.subsections:
            section_id = ((parent_id + ".") if parent_id else "") + str(subsection.section_id)
            print(f"\n====== 开始写作章节 {section_id} ======\n")
            await dfs_report(subsection)
            for segment_idx, segment in enumerate(subsection.segments):
                await writer.memory.clear()
                if segment.evidences is None:
                    segment.evidences = []
                for i in range(len(segment.evidences)):
                    searcher_input = Msg(
                        name="user",
                        content=(
                            f"任务：{task_desc}\n"
                            f"当前需要你撰写要点：{segment.topic}\n"
                            f"论据所需材料：\n{segment.evidences[i]}\n\n"
                            f"请你调用工具搜索，尽量根据多个信息源交叉验证后给出搜索结果。"
                        ),
                        role="user",
                    )
                    # msg = await call_agent_with_retry(searcher, searcher_input)
                    # msg = msg.get_text_content()
                    # print(f"[Searcher] After searching {segment.evidences[i]}...")
                    # print(msg)
                    # segment.evidences[i] = msg
                    try:
                        msg = await call_agent_with_retry(searcher, searcher_input)
                        # 确保返回的不是 None
                        if msg is not None:
                            msg_text = msg.get_text_content()
                            if msg_text is not None:
                                print(f"[Searcher] After searching {segment.evidences[i]}...")
                                print(msg_text)
                                segment.evidences[i] = msg_text
                            else:
                                print(f"[警告] 搜索 {segment.evidences[i]} 返回空结果")
                                segment.evidences[i] = "(搜索无结果)"
                        else:
                            print(f"[警告] 搜索 {segment.evidences[i]} 返回 None")
                            segment.evidences[i] = "(搜索失败)"
                    except Exception as e:
                        print(f"[错误] 搜索 {segment.evidences[i]} 时出错: {e}")
                        segment.evidences[i] = f"(搜索错误: {str(e)})"
                # 写作
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

                # draft_msg = await writer(writer_input)
                draft_msg = await call_agent_with_retry(writer, writer_input)

                print("[Writer 初稿输出]")
                print(draft_msg.get_text_content())
                segment.content = draft_msg.get_text_content()
                segment.finished = True

                await writer.memory.clear()

                # ----- 9. 评估当前segment -----
                segment_counter += 1
                segment_id = f"{section_id}.{segment_counter}"
                
                print(f"\n{'='*40}")
                print(f"评估segment {segment_id}")
                print(f"{'='*40}")

                try:
                    # 对segment进行评分
                    segment_score = await segment_scorer.score_segment(segment, segment_id)
                    
                    # 记录评分
                    all_segment_scores.append(segment_score)
                    
                    # 打印评分结果
                    print(f"评分结果:")
                    print(f"  全面性: {segment_score.comprehensiveness:.1f}")
                    print(f"  洞察力: {segment_score.insight:.1f}")
                    print(f"  指令遵循: {segment_score.instruction_following:.1f}")
                    print(f"  可读性: {segment_score.readability:.1f}")
                    print(f"  充分性: {segment_score.sufficiency:.1f}")
                    
                    # 保存评分到segment
                    if not hasattr(segment, 'scores'):
                        segment.scores = {}
                    segment.scores = segment_score.to_dict()
                    
                except Exception as e:
                    print(f"评估segment失败: {e}")
                
                # 保存进度
                try:
                    (output_pth / f"{stock_symbol}.json").write_text(
                        manuscript.to_json(ensure_ascii=False)
                    )
                    print(f"[进度保存] Segment 处理完成")
                except Exception as e:
                    print(f"保存进度失败: {e}")

            # 为章节生成标题
            if subsection.segments and any(s.finished for s in subsection.segments):
                section_text = "\n".join([s.content for s in subsection.segments if s.finished])
                if section_text:
                    try:
                        draft_msg = await call_agent_with_retry(writer, Msg(
                            name="user",
                            content=(
                                "以下是所有要点整理后的本章节内容：\n\n"
                                f"{section_text}\n\n"
                                f"参考范例的标题为: {subsection.title}\n\n"
                                f"请你根据当前任务撰写的内容起一个新标题。"
                            ),
                            role="user",
                        ))
                        subsection.title = draft_msg.get_text_content()
                        print(f"[章节标题]: {subsection.title}")
                    except Exception as e:
                        print(f"生成章节标题失败: {e}")

    await dfs_report(manuscript)

     # ----- 计算并输出最终评分结果 -----
    print("\n" + "="*60)
    print("最终评分统计")
    print("="*60)
    
    if all_segment_scores:
        # 计算五个维度的平均分
        avg_comprehensiveness, avg_insight, avg_instruction_following, avg_readability, avg_sufficiency = \
            segment_scorer.calculate_average_scores(all_segment_scores)
        
        print(f"评估完成，共评估 {len(all_segment_scores)} 个segment")
        print(f"\n五个维度的平均分数:")
        print(f"  全面性: {avg_comprehensiveness:.2f}")
        print(f"  洞察力: {avg_insight:.2f}")
        print(f"  指令遵循: {avg_instruction_following:.2f}")
        print(f"  可读性: {avg_readability:.2f}")
        print(f"  充分性: {avg_sufficiency:.2f}")
        
        # 保存详细评分结果
        scoring_results = {
            "task_desc": task_desc,
            "segment_count": len(all_segment_scores),
            "average_scores": {
                "comprehensiveness": avg_comprehensiveness,
                "insight": avg_insight,
                "instruction_following": avg_instruction_following,
                "readability": avg_readability,
                "sufficiency": avg_sufficiency
            },
            "segment_scores": [score.to_dict() for score in all_segment_scores]
        }
        
        # 创建评估目录
        eval_dir = short_term_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = eval_dir / "scoring_results.json"
        results_file.write_text(json.dumps(scoring_results, ensure_ascii=False, indent=2))
        
        # 保存到短期记忆
        short_term.store("content_scoring_results", scoring_results)
        
        print(f"\n详细评分结果已保存到: {results_file}")

    markdown_text = section_to_markdown(manuscript)
    (short_term_dir / "manuscript.md").write_text(markdown_text, encoding="utf-8")
    # md_to_pdf(markdown_text, short_term=short_term)