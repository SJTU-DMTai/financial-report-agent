from __future__ import annotations

from pathlib import Path
import pdfkit
import re
from typing import List, Tuple
from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse

from ..memory.short_term import ShortTermMemoryStore
from .material_tools import *


class ManuscriptTools:

    def __init__(self, short_term: ShortTermMemoryStore):
        self.short_term = short_term
        self.tools = [
            self.draft_manuscript_from_outline,
            self.read_manuscript_section,
            self.replace_manuscript_section,
        ]

    # ---- 内部函数 ----
    def _parse_outline_sections(self, outline: str) -> List[Tuple[str, str, str]]:
        """把 outline.md 划分为若干 section。

        返回: List[(section_id, title, body_markdown)]
        简化策略：
        - 以一级标题 `#` 作为章节分割点
        - section_id 形如 `sec_01_行业分析`，保证字典序 == 章节顺序
        """
        lines = outline.splitlines()
        sections: List[Tuple[str, str, str]] = []

        current_title = None
        current_body_lines: List[str] = []
        index = 0  # 用于编号

        def flush():
            nonlocal current_title, current_body_lines, index
            if current_title is None:
                return
            title = current_title.strip("# ").strip()
            # 简单 slug 化做 section_id
            slug = re.sub(r"\s+", "_", title)
            slug = re.sub(r"[^\w\-一-龥]", "", slug)  # 保留中文和常见字符
            index += 1
            prefix = f"{index:02d}"
            section_id = f"sec_{prefix}_{slug}"
            body = "\n".join(current_body_lines).strip()
            sections.append((section_id, title, body))
            current_title = None
            current_body_lines = []

        for line in lines:
            if line.startswith("# "):  # 一级标题
                flush()
                current_title = line
            else:
                if current_title is None:
                    # 出现在第一个 # 之前的内容可以直接忽略或归入引言
                    continue
                current_body_lines.append(line)

        flush()
        return sections


    # ---- Manuscript Tool ----
    def draft_manuscript_from_outline(
        self,
    ) -> ToolResponse:
        """根据现有的 outline.md 生成按章节拆分的多个 markdown 草稿骨架。
        调用此工具时，将根据大纲内容，自动创建对应章节的初始 markdown 草稿，并返回生成的章节 ID 列表。

        """
        outline = self.short_term.load_outline()
        if not outline.strip():
            return ToolResponse(
                content=[TextBlock(type="text", text="[draft_manuscript_from_outline] outline 为空")],
                metadata={"sections": []},
            )

        sections = self._parse_outline_sections(outline)

        section_ids = []
        for section_id, title, body_md in sections:
            body_markdown = (
                f"# {title}\n\n"
                "（请根据大纲要点在此撰写正文，可调用 Searcher 工具补充材料，调用generate chart工具绘图。）\n\n"
                f"{body_md}\n\n"
                )

            self.short_term.save_manuscript_section(section_id, body_markdown)
            section_ids.append(section_id)

        text = "[draft_manuscript_from_outline] 已生成以下章节草稿:\n" + "\n".join(section_ids)
        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"sections": section_ids},
        )


    def read_manuscript_section(
        self,
        section_id: str,
    ) -> ToolResponse:
        """读取指定章节的 Markdown 草稿。
        可调用此工具获取某个章节点的完整 Markdown 内容，用于阅读。如果该章节不存在，将返回说明信息。

        Args:
            section_id (str):
                要读取的章节唯一标识符。
        """
        markdown = self.short_term.load_manuscript_section(section_id)
        if not markdown:
            markdown = f"[read_manuscript_section] section {section_id} 不存在或为空。"

        return ToolResponse(
            content=[TextBlock(type="text", text=markdown)],
            metadata={"section_id": section_id},
        )


    def replace_manuscript_section(
        self,
        section_id: str,
        new_markdown: str,
    ) -> ToolResponse:
        """
        以新的 Markdown 内容替换指定章节。调用此工具可直接覆盖某个章节的草稿内容。
        调用时需提供章节 ID 以及更新后的完整 Markdown 字符串。

            Args:
                section_id (str):
                    要更新的章节唯一标识符。
                new_markdown (str):
                    用于替换原章节的完整 Markdown 内容。
            """
        self.short_term.save_manuscript_section(section_id, new_markdown)

        return ToolResponse(
            content=[TextBlock(type="text", text=f"[replace_manuscript_section] 已更新 {section_id}")],
            metadata={"section_id": section_id},
        )
    

    