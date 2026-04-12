# 文件名: planning.py (已注释掉 VLM 图片处理部分)

import os
import asyncio
from datetime import datetime

from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from pathlib import Path

from agentscope.model import ChatModelBase

from src.memory.working import Section
from src.prompt import prompt_dict
from src.utils.call_with_retry import call_chatbot_with_retry
from src.utils.file_converter import pdf_to_markdown, markdown_to_sections
# from src.utils.image_analyze import inject_vlm_into_demo_markdown # [修改] 注释掉此导入
from src.utils.instance import create_agent_formatter, create_vlm_model, cfg


async def process_pdf_to_outline(pdf_path: Path, save_dir: Path,
                                 llm_reasoning: ChatModelBase, llm_instruct: ChatModelBase = None,
                                 formatter=None, only_evidence: bool = False, another_stock: bool = False) -> Section:
    """
    处理单个PDF，生成完整的Section对象。
    会检查并使用_outline.json缓存，以避免重复处理。
    """
    # 兼容两种可能的缓存路径
    outline_json_path = save_dir / cfg.llm_name / f'{pdf_path.name.split(".")[0]}_outline.json'
    outline_json_path2 = save_dir / f'{pdf_path.name.split(".")[0]}_outline.json' # 旧路径，为了兼容性保留
    
    if outline_json_path.exists() or outline_json_path2.exists():
        if outline_json_path.exists():
            manuscript = Section.from_json(outline_json_path.read_text(encoding='utf-8')) # [修复] 添加 encoding
        else:
            manuscript = Section.from_json(outline_json_path2.read_text(encoding='utf-8')) # [修复] 添加 encoding
        
        # [修改] 这一段打印代码移到调用者，避免重复打印
        outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True,fold_other=False)
        print(outline)
        return manuscript

    outline_json_path.parent.mkdir(parents=True, exist_ok=True)
    if formatter is None:
        formatter = create_agent_formatter()
    if llm_instruct is None:
        llm_instruct = llm_reasoning
    
    # 从文件名中解析出日期和名称，以便在Prompt中使用
    # 确保文件名格式正确，避免列表越界
    try:
        name_parts = pdf_path.name.split(".")[0].split("_")
        demo_date = name_parts[-2] if len(name_parts) >= 2 else "未知日期"
        demo_name = name_parts[-1] if len(name_parts) >= 1 else pdf_path.stem
    except Exception:
        demo_date = "未知日期"
        demo_name = pdf_path.stem


    print("    - 步骤 1/3: PDF -> Markdown...", flush=True)
    md_path = save_dir / f"{pdf_path.stem}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"output_path: {md_path} (exist: {md_path.exists()})", flush=True)
    if not md_path.exists():
        # [修改] pdf_to_markdown 函数不再返回 images
        # final_text, images = pdf_to_markdown(pdf_path) # 原始代码
        md_text = pdf_to_markdown(pdf_path) # [修改] 新版本
        md_path.write_text(md_text, encoding="utf-8")
        
        # [修改] 注释掉 VLM 图片分析部分
        # await inject_vlm_into_demo_markdown(
        #     demo_md_path=md_path,
        #     images=images,
        #     vlm_model=create_vlm_model(),
        #     image_prompt=prompt_dict["image_analyze"],
        # )

    print("    - 步骤 2/3: Markdown -> 初始章节结构...", flush=True)
    manuscript: Section = markdown_to_sections(md_path)

    print("    - 步骤 3/3: 调用大模型分解并填充内容...", flush=True)

    async def dfs_process_section(section: Section, parent_path_id: str = ""):
        # 构建当前 Section 的路径ID
        current_section_path_id = f"{parent_path_id}_s{section.section_id}" if parent_path_id else f"s{section.section_id}"
        # 使用 asyncio.gather 并行处理子章节，提高效率
        if section.subsections:
            await asyncio.gather(*[dfs_process_section(subsection, current_section_path_id) for subsection in section.subsections])

        # 检查 section.segments 是否存在且有内容，并且第一个 segment 有 reference
        if section.segments and len(section.segments) > 0 and section.segments[0].reference:
            # 调用 decompose Prompt 进行段落分解
            decomposed_response_content = await call_chatbot_with_retry(
                llm_reasoning, formatter,
                prompt_dict["decompose"], section.segments[0].reference.replace("<SEP>", ""),
            )
            # 健壮性检查
            if decomposed_response_content is None:
                print(f"    ! 分解章节 '{section.title}' 失败，返回内容为空。")
                decomposed_segments_text = [section.segments[0].reference] # 退化为使用原始段落
            else:
                decomposed_segments_text = decomposed_response_content.split("<SEP>")
            
            processed_segments = []
            for i, segment_text in enumerate(decomposed_segments_text):
                if not segment_text.strip(): continue # 跳过空段落

                try:
                    # 根据 only_evidence 标志选择不同的Prompt和解析函数
                    prompt_key = "extract_evidence" if only_evidence else "plan_outline"
                    parse_func = section.parse_evidence if only_evidence else section.parse

                    # 构建 Planner Prompt 的内容
                    planner_prompt_content = (
                        f"为了撰写一份新研报，我找到了某机构在过去{demo_date}撰写的一份研报"
                        f"（{'可能是不同公司' if another_stock else '同一公司'}），名为{demo_name}。"
                        f"从中摘出的一段参考片段如下：\n<reference>{segment_text}</reference>\n\n"
                        f"请你考虑时间差和公司异同，抽取用于当前新任务的论据{'' if only_evidence else '、撰写模版、写作要求'}和主题。\n\n"
                        # [新逻辑] 增加一条指令，指导 LLM 在 template 中保留静态知识的具体值
                        f"如果某项论据被标记为静态（static）且其具体值已在参考片段中明确提及，请确保在生成的`<template>`中直接保留其具体值，不要将其替换为占位符。\n\n"
                    )

                    segment = await call_chatbot_with_retry(
                        llm_instruct, formatter,
                        prompt_dict[prompt_key],
                        planner_prompt_content,
                        parse_func, handle_hook_exceptions=(AssertionError,)
                    )
                    
                    if segment: # 确保 segment 对象成功创建
                        # 为 Segment 赋予完整路径ID
                        segment_id_str = f"{current_section_path_id}_p{i+1}"
                        segment.segment_id = segment_id_str
                        
                        # 为 Segment 内的每一个 Evidence 赋予完整路径ID
                        if segment.evidences:
                            for j, evidence in enumerate(segment.evidences):
                                evidence.evidence_id = f"{segment_id_str}_e{j+1}"

                        segment.reference = segment_text
                        processed_segments.append(segment)
                    else:
                        print(f"    ! 处理小段落时，模型返回无效 Segment 对象。")

                except Exception as e:
                    # [新逻辑] 捕获 call_chatbot_with_retry 抛出的最终异常
                    print("\n" + "!"*60)
                    print(f"!!! 警告: 章节 '{section.title}' 的第 {i+1} 个小段落处理彻底失败，将跳过此段落。")
                    print(f"!!! 错误详情: {e}")
                    print("!"*60 + "\n")
                    # 跳过这个 segment，继续下一个循环
                    continue

            section.segments = processed_segments

    await dfs_process_section(manuscript, parent_path_id="") 
    
    # 只有在不需要仅提取论据时才保存完整的outline.json
    if not only_evidence:
        print(f"    - 保存缓存到: {outline_json_path.name}...")
        outline_json_path.write_text(manuscript.to_json(ensure_ascii=False, indent=4), encoding='utf-8') # [修复] 添加 encoding, indent
        print("保存成功！")
    
    return manuscript