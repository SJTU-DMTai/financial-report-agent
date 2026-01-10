# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import pickle
import re
import sys
from dataclasses import asdict
from pathlib import Path

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
from src.utils.parse_verdict import parse_verifier_verdict
from src.utils.call_agent_with_retry import call_agent_with_retry
import config
import asyncio

from src.utils.file_converter import markdown_to_sections
from src.utils.local_file import STOCK_REPORT_PATHS
from src.evaluation.parse_verifier_verdict import parse_verdict, VERDICT_PARSERS


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

    # ----- 6. 创建所有Verifier -----
    verifiers = create_all_verifiers(model, formatter, short_term)
    final_verifier = create_final_verifier(model, formatter, short_term)

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

    async def verify_segment_content(verifiers_dict, segment, task_desc, reference_text, materials_text):
        """
        对单个segment执行四个验证环节（numeric → reference → logic → quality），
        遇到第一条不通过立即记录问题并返回，用于writer重写。
        """
        verification_results = {}
        all_problems = []

        # 按顺序执行四个验证
        verifier_order = ["numeric", "reference", "logic", "quality"]

        for verifier_name in verifier_order:
            verifier = verifiers_dict[verifier_name]

            # 构建不同验证器输入
            verifier_content = f"任务：{task_desc}\n\n【写作要点】\n{segment.topic}\n\n"
            if verifier_name in ["numeric", "reference"]:
                verifier_content += f"【可用材料】\n{materials_text}\n\n"
            if verifier_name in ["logic", "quality"]:
                verifier_content += f"【写作要求】\n{segment.requirements}\n\n"
            if verifier_name == "quality":
                verifier_content += f"【参考文本】\n{reference_text}\n\n"
            verifier_content += f"【待审核正文】\n{segment.content}\n\n"

            # 不同验证器提示
            verifier_prompts = {
                "numeric": "请检查正文中所有数字、比例、估值、销量、财务数据、时间区间是否与材料一致。",
                "reference": "请检查正文中所有material_id是否被引用，引用的内容是否与材料本身匹配。",
                "logic": "请检查论点是否由论据支撑，是否存在跳跃、矛盾、自相矛盾的表述，语言是否清晰。",
                "quality": "请对比当前文本与参考文本的写作质量，判断是否达到或超过参考水平。"
            }
            verifier_content += verifier_prompts[verifier_name]

            verifier_input = Msg(name="user", content=verifier_content, role="user")
            print(f"  → 执行 {verifier_name} 验证")

            try:
                response = await call_agent_with_retry(verifier, verifier_input)
                result = parse_verdict(verifier_name, response.get_text_content())
                verification_results[verifier_name] = result

                if not result["passed"]:
                    # 遇到 NO 立即返回结果，让writer修订
                    all_problems.append({
                        "verifier": verifier_name,
                        "passed": False,
                        "problems": result.get("problems", []),
                        "scores": result.get("scores", {}),
                        "raw_result": result
                    })
                    print(f" {verifier_name}验证失败，收集问题并返回")
                    return False, verification_results, all_problems
                else:
                    print(f" {verifier_name}验证通过")

            except Exception as e:
                print(f"  {verifier_name}验证异常: {e}")
                all_problems.append({
                    "verifier": verifier_name,
                    "passed": False,
                    "problems": [f"验证异常: {e}"],
                    "error": str(e)
                })
                return False, verification_results, all_problems

            finally:
                # 清空验证器内存
                await verifier.memory.clear()

        # 全部验证通过
        return True, verification_results, all_problems

    
    def format_problems_for_rewrite(all_problems):
        """格式化问题以便用于重写指令"""
        if not all_problems:
            return ""
        
        formatted = "【验证发现问题】\n\n"
        
        for problem_info in all_problems:
            verifier_name = problem_info["verifier"]
            problems = problem_info.get("problems", [])
            
            # 添加验证器类型说明
            verifier_descriptions = {
                "numeric": "数值一致性检查",
                "reference": "引用正确性检查", 
                "logic": "逻辑一致性检查",
                "quality": "写作质量检查"
            }
            
            description = verifier_descriptions.get(verifier_name, verifier_name)
            formatted += f"{description}发现问题：\n"
            
            if problems:
                if isinstance(problems, list):
                    for i, problem in enumerate(problems[:3]):  # 只取前3个问题
                        if isinstance(problem, dict):
                            # 结构化问题
                            desc = problem.get("description", "")
                            location = problem.get("location", "")
                            suggestion = problem.get("suggestion", "")
                            expected = problem.get("expected", "")
                            actual = problem.get("actual", "")
                            
                            formatted += f"  {i+1}. "
                            if desc:
                                formatted += f"问题: {desc}\n"
                            if expected and actual:
                                formatted += f"     预期: {expected}, 实际: {actual}\n"
                            if location:
                                formatted += f"     位置: {location}\n"
                            if suggestion:
                                formatted += f"     建议: {suggestion}\n"
                        else:
                            # 简单文本问题
                            formatted += f"  {i+1}. {str(problem)}\n"
                else:
                    formatted += f"  {problems}\n"
            else:
                formatted += "  （无具体问题描述）\n"
            
            formatted += "\n"
        
        return formatted
    
    async def dfs_report(section: Section, parent_id=None):
        if section.subsections is None:
            return
        for subsection in section.subsections:
            section_id = ((parent_id + ".") if parent_id else "") + str(subsection.section_id)
            print(f"\n====== 开始写作章节 {section_id} ======\n")
            await dfs_report(subsection)
            for segment in subsection.segments:
                await writer.memory.clear()
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
                    msg = await call_agent_with_retry(searcher, searcher_input)
                    msg = msg.get_text_content()
                    print(f"[Searcher] After searching {segment.evidences[i]}...")
                    print(msg)
                    segment.evidences[i] = msg
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
                # 清空writer内存，准备验证
                await writer.memory.clear()

                max_verify_rounds = cfg.get_max_verify_rounds()
                materials_text = "\n".join(segment.evidences)

                for round_idx in range(1, max_verify_rounds + 1):

                    print(f"\n--- Verifier 审核（Segment 级）轮次 {round_idx} ---\n")

                    # 执行验证，收集所有问题
                    passed, verification_results, all_problems = await verify_segment_content(
                        verifiers, segment, task_desc, segment.reference, materials_text
                    )
                    if passed:
                        segment.finished = True
                        break
                    else:
                        # 构建重写指令
                        if round_idx < max_verify_rounds:
                            rewrite_prompt = "以下是针对【当前这个段落】的审核意见。\n\n"
                            rewrite_prompt += f"【原段落写作要点】\n{segment.topic}\n\n"
                            rewrite_prompt += f"【写作要求】\n{segment.requirements}\n\n"
                            rewrite_prompt += f"【参考范例】\n{segment.reference}\n\n"
                            rewrite_prompt += f"【可用材料】\n{materials_text}\n\n"
                            
                            # 添加格式化的问题详情
                            rewrite_prompt += format_problems_for_rewrite(all_problems)
                            
                            rewrite_prompt += "\n请你根据上述所有问题修改段落内容，确保解决所有验证问题。\n"
                            rewrite_prompt += "请只重写这一段正文，不要提及其他段落，不要总结章节，不要引入新的论点或结论。\n\n"
                            rewrite_prompt += "请直接输出修订后的段落正文。"
                            
                            writer_fix_input = Msg(name="user", content=rewrite_prompt, role="user")
                            draft_msg = await call_agent_with_retry(writer, writer_fix_input)
                            segment.content = draft_msg.get_text_content()
                            print("[Writer 修订输出]")
                            print(segment.content)

                            # 清空writer内存，准备下一轮验证
                            await writer.memory.clear()

                        else:
                            print(f"[达到最大验证轮次 {max_verify_rounds}] Segment 最终未通过验证")
                            segment.finished = False
                            break
                
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

    markdown_text = section_to_markdown(manuscript)
    (short_term_dir / "manuscript.md").write_text(markdown_text, encoding="utf-8")
    # md_to_pdf(markdown_text, short_term=short_term)
