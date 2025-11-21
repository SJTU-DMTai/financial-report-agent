from __future__ import annotations

from typing import Callable

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse

from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import OutlineExperienceStore

class PlannerTools:

    def __init__(self, short_term: ShortTermMemoryStore):
        self.short_term = short_term


    # ---- Outline Tool: Read / Replace / Save to long-term ----
    def read_outline(self) -> ToolResponse:
        """读取当前任务的大纲 outline.md。"""
        content = self.short_term.load_outline()
        return ToolResponse(
            content=[TextBlock(type="text", text=content or "[empty outline]")],
        )


    def replace_outline(self, new_outline: str) -> ToolResponse:
        """
        调用此工具可写入新的研报大纲。

            Args:
                new_outline (str):
                    写入研报大纲的完整 markdown 内容。
        """
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


    # ---- 高级工具：调用 Searcher agent ----
    def searcher_tool(self, searcher: ReActAgent) -> Callable[[str], ToolResponse]:
        """把 Searcher agent 封装成 Planner 可见的工具函数。"""

        async def search_with_searcher(query: str) -> ToolResponse:
            """使用指定的 Searcher 工具 基于 query 执行一次检索并返回总结结果。

            Args:
                query (str): 检索需求的自然语言描述。

            """
            msg = Msg(
                name="Planner",
                content=query,
                role="user",
            )
            res = await searcher(msg)
            return ToolResponse(
                content=res.content,
                metadata={"from_agent": searcher.name},
            )

        return search_with_searcher


# ---- Toolkit Builder ----
def build_planner_toolkit(
    short_term: ShortTermMemoryStore,
    searcher: ReActAgent,
) -> Toolkit:
    """创建 Planner 专用 Toolkit。"""
    toolkit = Toolkit()
    planner_tools = PlannerTools(short_term=short_term)
    # toolkit.register_tool_function(planner_tools.read_outline)
    toolkit.register_tool_function(planner_tools.read_demonstration)
    toolkit.register_tool_function(planner_tools.replace_outline)
    toolkit.register_tool_function(planner_tools.searcher_tool(searcher))
    return toolkit
