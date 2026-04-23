import os
import asyncio
from functools import partial
from datetime import datetime

from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from pathlib import Path

from agentscope.model import ChatModelBase

from src.memory.working import (
    Section,
    _get_outline_cache_paths,
    _load_cached_outline,
    _parse_segment_response,
)
from src.prompt import prompt_dict
from src.utils.call_with_retry import call_chatbot_with_retry
from src.utils.file_converter import pdf_to_markdown, markdown_to_sections
from src.utils.image_analyze import inject_vlm_into_demo_markdown
from src.utils.instance import create_agent_formatter, create_vlm_model


async def process_pdf_to_outline(pdf_path: Path, save_dir: Path,
                                 llm_reasoning: ChatModelBase, llm_instruct: ChatModelBase = None,
                                 formatter=None, only_evidence: bool = False, another_stock: bool = False) -> Section:
    """
    处理单个PDF，生成完整的Section对象。
    会检查并使用_outline.json缓存，以避免重复处理。
    """
    manuscript = _load_cached_outline(pdf_path, save_dir, only_evidence)
    if manuscript is not None:
        outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True,
                                  fold_other=False)
        print(outline)
        return manuscript

    outline_json_path, _ = _get_outline_cache_paths(pdf_path, save_dir, only_evidence)
    outline_json_path.parent.mkdir(parents=True, exist_ok=True)
    if formatter is None:
        formatter = create_agent_formatter()
    if llm_instruct is None:
        llm_instruct = llm_reasoning
    demo_date, demo_name = pdf_path.name.split(".")[0].split("_")[-2:]

    print("    - 步骤 1/3: PDF -> Markdown...", flush=True)
    md_path = save_dir / f"{pdf_path.stem}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"output_path: {md_path} (exist: {md_path.exists()})", flush=True)
    if not md_path.exists():
        md_text, images = pdf_to_markdown(pdf_path)
        md_path.write_text(md_text, encoding="utf-8")
        await inject_vlm_into_demo_markdown(
            demo_md_path=md_path,
            images=images,
            vlm_model=create_vlm_model(),
            image_prompt=prompt_dict["image_analyze"],
        )

    print("    - 步骤 2/3: Markdown -> 初始章节结构...", flush=True)
    manuscript: Section = markdown_to_sections(md_path)

    print("    - 步骤 3/3: 调用大模型分解并填充内容...", flush=True)

    async def dfs_process_section(section: Section):
        if section.subsections:
            await asyncio.gather(*[dfs_process_section(subsection) for subsection in section.subsections])

        if section.segments and len(section.segments) > 0 and section.segments[0].reference:
            decomposed_segments_text = await call_chatbot_with_retry(
                llm_reasoning, formatter,
                prompt_dict["decompose"], section.segments[0].reference.replace("<SEP>", "")
            )
            if decomposed_segments_text is None:
                raise AssertionError("Decomposed segments text is None")
            decomposed_segments_text = decomposed_segments_text.split("<SEP>")
            processed_segments = []
            parse_segment_response = partial(_parse_segment_response, only_evidence=only_evidence)
            for i, segment_text in enumerate(decomposed_segments_text):
                if not segment_text.strip(): continue
                segment_res = await call_chatbot_with_retry(
                    llm_instruct, formatter,
                    prompt_dict["extract_evidence" if only_evidence else "plan_outline"],
                    f"为了撰写一份新研报，我找到了某机构在过去{demo_date}撰写的一份研报"
                    f"（{'可能是不同公司' if another_stock else '同一公司'}），名为{demo_name}。"
                    f"从中摘出的一段参考片段如下：\n<reference>{segment_text}</reference>\n\n"
                    f"请你考虑时间差和公司异同，抽取用于当前新任务的论据{'' if only_evidence else '、撰写模版、写作要求'}和主题。\n"
                    f"{'' if only_evidence else '当某条论据既属于几乎不会随时间变化的静态事实、又能直接从参考片段中确定其具体值或具体情况时，在该条论据末尾标注 (static)。对于被标注为 (static) 的论据，请在 <template> 中直接保留这个具体值，不要替换成占位符。'}\n\n",
                    hook=parse_segment_response,
                    handle_hook_exceptions=(AssertionError,)
                )
                if isinstance(segment_res, str) and "<skip>true</skip>" in segment_res.lower():
                    continue
                segment = segment_res
                segment.reference = segment_text
                processed_segments.append(segment)
            section.segments = processed_segments

    await dfs_process_section(manuscript)
    # outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True,
    #                           fold_other=False)
    # print(outline)
    outline_json_path.write_text(manuscript.to_json(ensure_ascii=False), encoding="utf-8")
    return manuscript
