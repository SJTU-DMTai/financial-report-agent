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
from src.utils.image_analyze import inject_vlm_into_demo_markdown
from src.utils.instance import create_agent_formatter, create_vlm_model

CONCURRENCY_LIMIT = int(os.getenv("N_THREAD", 32))


async def process_pdf_to_outline(pdf_path: Path, save_dir: Path,
                                 llm_reasoning: ChatModelBase, llm_instruct: ChatModelBase = None,
                                 formatter=None, another_stock: bool = False) -> Section:
    """
    处理单个PDF，生成完整的Section对象。
    会检查并使用_outline.json缓存，以避免重复处理。
    """
    outline_json_path = save_dir / (pdf_path.name.split(".")[0] + "_outline.json")
    if outline_json_path.exists():
        # outline = outline_md_pth.read_text()
        manuscript = Section.from_json(outline_json_path.read_text())
        outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True,
                                  fold_other=False)
        print(outline)
        return manuscript

    if formatter is None:
        formatter = create_agent_formatter()
    if llm_instruct is None:
        llm_instruct = llm_reasoning
    demo_date, demo_name = pdf_path.name.split(".")[0].split("_")[-2:]

    print("    - 步骤 1/3: PDF -> Markdown...")
    md_path = save_dir / f"{pdf_path.stem}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"output_path: {md_path}")
    if not md_path.exists():
        md_text, images = pdf_to_markdown(pdf_path)
        md_path.write_text(md_text, encoding="utf-8")
        await inject_vlm_into_demo_markdown(
            demo_md_path=md_path,
            images=images,
            vlm_model=create_vlm_model(),
            image_prompt=prompt_dict["image_analyze"],
        )

    print("    - 步骤 2/3: Markdown -> 初始章节结构...")
    manuscript: Section = markdown_to_sections(md_path)

    print("    - 步骤 3/3: 调用大模型分解并填充内容...")
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def dfs_process_section(section: Section):
        async def process_subsection_with_limit(subsection: Section):
            async with semaphore:
                await dfs_process_section(subsection)

        if section.subsections:
            await asyncio.gather(*[process_subsection_with_limit(subsection) for subsection in section.subsections])

        if section.segments:
            decomposed_segments_text = await call_chatbot_with_retry(
                llm_instruct, formatter,
                prompt_dict["decompose"], section.segments[0].reference.replace("<SEP>", ""),
            )
            decomposed_segments_text = decomposed_segments_text.split("<SEP>")
            processed_segments = []
            for i, segment_text in enumerate(decomposed_segments_text):
                if not segment_text.strip(): continue
                segment = await call_chatbot_with_retry(
                    llm_reasoning, formatter,
                    prompt_dict["plan_outline"],
                    f"为了撰写一份股票研报，我找到了某机构在历史{demo_date}撰写的一份研报"
                    f"（{'可能是不同公司' if another_stock else '同一公司'}），名为{demo_name}。"
                    f"下文将附上从中摘出的一段参考片段，请你考虑时间差和公司异同，撰写一份用于当前新任务的撰写模版和要求。\n\n"
                    f"参考片段如下：\n\n{segment_text}",
                    section.parse, handle_hook_exceptions=(AssertionError,)
                )
                segment.reference = segment_text
                processed_segments.append(segment)
            section.segments = processed_segments

    await dfs_process_section(manuscript)
    outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True,
                              fold_other=False)
    print(outline)
    outline_json_path.write_text(manuscript.to_json(ensure_ascii=False))
    return manuscript