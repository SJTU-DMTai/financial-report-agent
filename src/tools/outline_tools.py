# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse

from ..memory.working import Section
from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import LongTermMemoryStore


class OutlineTools:

    def __init__(self, manuscript: Section,long_term:LongTermMemoryStore,short_term:ShortTermMemoryStore):
        self.manuscript = manuscript
        self.long_term = long_term
        self.short_term = short_term

    # ---- Outline Tool: Read / Replace / Save to long-term ----
    def read_outline(self) -> ToolResponse:
        """读取当前任务的大纲 outline.md。"""
        content = self.short_term.load_outline()
        return ToolResponse(
            content=[TextBlock(type="text", text=content or "[empty outline]")],
        )

    
    def replace_outline(self, outline_title: str, outline_markdown: str) -> ToolResponse:
        """
        调用此工具可写入新的研报大纲。

        Args:
            outline_title (str):
                研报的自然语言标题。
            outline_markdown (str):
                研报大纲正文的完整 markdown 内容（不包含标题）。
        """
        outline_title = outline_title.replace("#", "").replace("\n", " ").strip()
        new_outline = f"{outline_title}\n{outline_markdown}"
        self.short_term.save_outline(new_outline)
        return ToolResponse(
            content=[TextBlock(type="text", text="[replace_outline] 已更新 outline.md")],
        )
    
    def read_demonstration(self) -> ToolResponse:        
        """读取示例研报的完整内容。
        可调用此工具获取示例研报的完整 markdown 内容，用于阅读和分析。
        """
        demonstration = self.short_term.load_demonstration()
        if not demonstration:
            demonstration = f"[read_demonstration] demonstration 不存在或为空。"
        return ToolResponse(
            content=[TextBlock(type="text", text=demonstration)],
        )
    

    # def save_outline_to_experience(
    #     self,
    #     task_id: str,
    #     short_term: ShortTermMemoryStore,
    #     outline_store: OutlineExperienceStore,
    #     ) -> ToolResponse:
    #     """把当前 outline 保存到长期记忆中。"""
    #     content = short_term.load_outline()
    #     outline_store.save_outline(
    #         task_id=task_id,
    #         outline_content=content,
    #         meta={"task_id": task_id},
    #     )
    #     return ToolResponse(
    #         content=[TextBlock(type="text", text="[save_outline_to_experience] 已保存")],
    #     )


    # def retrieve_outline_experience(
    #     self,
    #     keyword: str,
    #     outline_store: OutlineExperienceStore,
    # ) -> ToolResponse:
    #     """根据关键词粗略检索历史 outline 经验。

    #     简化实现：这里只返回「有哪些历史任务文件名」，真实场景可以做向量检索。
    #     """
    #     paths = outline_store.list_all()
    #     text = (
    #         "[retrieve_outline_experience] 可用 outline:\n"
    #         + "\n".join(p.name for p in paths if keyword.lower() in p.stem.lower())
    #     )
    #     return ToolResponse(
    #         content=[TextBlock(type="text", text=text)],
    #         metadata={"candidates": [str(p) for p in paths]},
    #     )

