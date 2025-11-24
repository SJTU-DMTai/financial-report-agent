from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import List, Dict, Any
from ddgs import DDGS
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from agentscope.agent import ReActAgent
from agentscope.message import Msg
from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import ToolUseExperienceStore
from .material_tools import *

#searcher agent使用
async def search_engine(query: str, max_results: int = 10) -> ToolResponse:
    """进行 Web 搜索并返回摘要信息。

    - 调用 DuckDuckGo 的搜索引擎接口，根据给定关键词返回若干条搜索结果的摘要，适合获取大致信息或者是新闻等。
    - 如果需要完整、可核查的原文内容，或者是结构化数据请调用其他工具。
    - 查询语句中的空格被视为逻辑“或”（OR）操作，因此每个关键词会分别参与匹配，请避免加入过于宽泛或无关的词语，导致无法搜索到需要的结果。
    Args:
        query (str):
            搜索内容。
        max_results (int):
            返回的最大结果数量。
    """
    try:
        ddgs = DDGS()
        # 1) 调用 DuckDuckGo 搜索接口
        raw_results = ddgs.text(
            query=query,
            backend="auto",
            region="cn-zh",
            max_results=max_results,
        )

        # 2) 标准化结果结构：title / description
        normalized: List[Dict[str, Any]] = []
        for r in raw_results:
            normalized.append(
                {
                    "title": r.get("title", "无标题"),
                    "description": r.get("body", "无摘要"),
                }
            )

        # 3) 将结果格式化为纯文本，供 TextBlock 使用
        if not normalized:
            text = f"[search_engine] 对查询「{query}」未找到结果。"
        else:
            lines: List[str] = [f"[search_engine] 搜索：{query}", ""]
            for i, item in enumerate(normalized, start=1):
                title = item.get("title", "无标题")
                desc = item.get("description", "无摘要")
                lines.append(f"{i}. {title}")
                lines.append(f"   摘要: {desc}")
                lines.append("")  # 空行分隔
            text = "\n".join(lines)

    except Exception as e:
        text = f"[search_engine] 搜索出错：{e}"

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=text,
            ),
        ],
    )


# planner和writer调用
def searcher_tool(searcher: ReActAgent) -> Callable[[str], ToolResponse]:
        """把 Searcher agent 封装成 agent 可见的工具函数。"""

        async def search_with_searcher(query: str) -> ToolResponse:
            """使用指定的 Searcher 工具 基于 query 执行一次检索并返回总结结果。

            Args:
                query (str): 检索需求的自然语言描述。

            """
            msg = Msg(
                name="user",
                content=query,
                role="user",
            )
            res = await searcher(msg)
            return ToolResponse(
                content=res.content,
                metadata={"from_agent": searcher.name},
            )

        return search_with_searcher
