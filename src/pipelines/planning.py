import os
import asyncio
from datetime import datetime
from collections import OrderedDict
from typing import Optional, List

from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from pathlib import Path
from pydantic import BaseModel, Field

from agentscope.model import ChatModelBase

from src.memory.working import Section, Segment
from src.prompt import prompt_dict
from src.utils.call_with_retry import call_chatbot_with_retry
from src.utils.file_converter import pdf_to_markdown, markdown_to_sections
from src.utils.image_analyze import inject_vlm_into_demo_markdown
from src.utils.instance import create_agent_formatter, create_vlm_model, cfg


class SegmentModel(BaseModel):
    """用于 only_evidence=False 时的结构化输出，对应 Segment 的可填充字段"""
    skip: bool = Field(default=False, description="该片段如果与公司无关可以跳过")
    evidences: Optional[List[str]] = Field(default=None, description="论据列表，每条为简短的论据描述")
    template: Optional[str] = Field(default=None, description="写作示例模版")
    topic: str = Field(description="该片段的主题")
    requirements: Optional[str] = Field(default=None, description="写作要求")

class EvidenceEntry(BaseModel):
    description: str = Field(description="论据描述")
    fact: str = Field(description="该论据对应的具体事实或数据")

class EvidenceModel(BaseModel):
    """用于 only_evidence=True 时的结构化输出"""
    skip: bool = Field(default=False, description="该片段如果与公司无关可以跳过")
    evidences: list[EvidenceEntry] = Field(
        description="论据列表，每项为包含抽象描述和具体事实的条目"
    )
    topic: str = Field(description="该片段的主题")


async def process_pdf_to_outline(pdf_path: Path, save_dir: Path,
                                 llm_reasoning: ChatModelBase, llm_instruct: ChatModelBase = None,
                                 formatter=None, only_evidence: bool = False, another_stock: bool = False) -> Section:
    """
    处理单个PDF，生成完整的Section对象。
    会检查并使用_outline.json缓存，以避免重复处理。
    """
    outline_json_path = save_dir / cfg.llm_name / f'{pdf_path.name.split(".")[0]}_outline{"_onlyw_evidence" if only_evidence else ""}.json'
    print(outline_json_path, flush=True)
    outline_json_path2 = save_dir / cfg.llm_name / f'{pdf_path.name.split(".")[0]}_outline{"_onlyw_evidence" if only_evidence else ""}.json'
    if outline_json_path.exists() or outline_json_path2.exists():
        if outline_json_path.exists():
            manuscript = Section.model_validate_json(outline_json_path.read_text(encoding="utf-8"))
        else:
            manuscript = Section.model_validate_json(outline_json_path2.read_text(encoding="utf-8"))
        outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True,
                                  fold_other=False)
        print(outline)
        return manuscript

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
            for i, segment_text in enumerate(decomposed_segments_text):
                if not segment_text.strip(): continue
                _structured_model = EvidenceModel if only_evidence else SegmentModel
                raw = await call_chatbot_with_retry(
                    llm_instruct, formatter,
                    prompt_dict["extract_evidence" if only_evidence else "plan_outline"],
                    f"为了撰写一份新研报，我找到了某机构在过去{demo_date}撰写的一份研报"
                    f"（{'可能是不同公司' if another_stock else '同一公司'}），名为{demo_name}。"
                    f"从中摘出的一段参考片段如下：\n<reference>{segment_text}</reference>\n\n"
                    f"请你考虑时间差和公司异同，抽取用于当前新任务的论据{'' if only_evidence else '、撰写模版、写作要求'}和主题。\n\n",
                    structured_model=_structured_model
                )
                print(raw, flush=True)
                if raw.skip:
                    continue
                if only_evidence:
                    # raw: EvidenceModel
                    segment = Segment(
                        topic=raw.topic,
                        evidences=[(e.description, e.fact) for e in raw.evidences] if raw.evidences else None,
                    )
                else:
                    # raw: SegmentModel
                    segment = Segment(
                        topic=raw.topic,
                        template=raw.template,
                        requirements=raw.requirements,
                        evidences=raw.evidences,
                    )
                if segment:
                    segment.reference = segment_text
                    processed_segments.append(segment)
            section.segments = processed_segments

    await dfs_process_section(manuscript)
    # outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True,
    #                           fold_other=False)
    # print(outline)
    # if not only_evidence:
    outline_json_path.write_text(manuscript.model_dump_json(ensure_ascii=False), encoding="utf-8")
    return manuscript