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

  
    
    # ---- Manuscript Tool ----


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
    

    