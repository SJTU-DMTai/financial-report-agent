from __future__ import annotations

from typing import Callable

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse

from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import OutlineExperienceStore


# ---- Outline Tool: Read / Replace / Save to long-term ----
def read_outline(short_term: ShortTermMemoryStore) -> ToolResponse:
    """读取当前任务的 outline.md。"""
    content = short_term.load_outline()
    return ToolResponse(
        content=[TextBlock(type="text", text=content or "[empty outline]")],
    )


def replace_outline(new_outline: str, short_term: ShortTermMemoryStore) -> ToolResponse:
    """用新的大纲内容替换 outline.md。"""
    short_term.save_outline(new_outline)
    return ToolResponse(
        content=[TextBlock(type="text", text="[replace_outline] 已更新 outline.md")],
    )


def save_outline_to_experience(
    task_id: str,
    short_term: ShortTermMemoryStore,
    outline_store: OutlineExperienceStore,
) -> ToolResponse:
    """把当前 outline 保存到长期记忆中。"""
    content = short_term.load_outline()
    outline_store.save_outline(
        task_id=task_id,
        outline_content=content,
        meta={"task_id": task_id},
    )
    return ToolResponse(
        content=[TextBlock(type="text", text="[save_outline_to_experience] 已保存")],
    )


def retrieve_outline_experience(
    keyword: str,
    outline_store: OutlineExperienceStore,
) -> ToolResponse:
    """根据关键词粗略检索历史 outline 经验。

    简化实现：这里只返回「有哪些历史任务文件名」，真实场景可以做向量检索。
    """
    paths = outline_store.list_all()
    text = (
        "[retrieve_outline_experience] 可用 outline:\n"
        + "\n".join(p.name for p in paths if keyword.lower() in p.stem.lower())
    )
    return ToolResponse(
        content=[TextBlock(type="text", text=text)],
        metadata={"candidates": [str(p) for p in paths]},
    )


# ---- 高级工具：调用 Searcher agent ----
def searcher_tool(searcher: ReActAgent) -> Callable[[str], ToolResponse]:
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
        # 直接把 Searcher 的 Msg.content 作为工具返回内容，
        # 这样 Planner 在 ReAct 流程中就能看到原始检索总结
        return ToolResponse(
            content=res.content,
            metadata={"from_agent": searcher.name},
        )

    # closure 的 __name__ 会是 'search_with_searcher'，作为工具名即可。
    return search_with_searcher


# ---- Toolkit Builder ----
def build_outline_toolkit(
    short_term: ShortTermMemoryStore,
    outline_store: OutlineExperienceStore,
    searcher: ReActAgent,
) -> Toolkit:
    """创建 Planner 专用 Toolkit。"""
    toolkit = Toolkit()

    toolkit.register_tool_function(
        read_outline,
        preset_kwargs={"short_term": short_term},
    )
    toolkit.register_tool_function(
        replace_outline,
        preset_kwargs={"short_term": short_term},
    )
    toolkit.register_tool_function(
        save_outline_to_experience,
        preset_kwargs={
            "short_term": short_term,
            "outline_store": outline_store,
        },
    )
    toolkit.register_tool_function(
        retrieve_outline_experience,
        preset_kwargs={"outline_store": outline_store},
    )

    # Searcher 高级工具
    toolkit.register_tool_function(searcher_tool(searcher))

    return toolkit
