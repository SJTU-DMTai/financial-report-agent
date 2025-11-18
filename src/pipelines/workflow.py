from __future__ import annotations

from pathlib import Path

from agentscope.message import Msg

from src.utils.instance import create_chat_model
from src.memory.short_term import ShortTermMemoryStore
from src.tools.searcher_tools import build_searcher_toolkit
from src.tools.writer_tools import build_writer_toolkit
from src.agents.searcher import create_searcher_agent
from src.agents.writer import create_writer_agent


async def run_workflow(task_desc: str) -> str:
    """围绕一个 task description 执行完整的研报生成流程。

    返回值：Writer 最终输出消息中的文本内容（其中包含 PDF 路径）。
    """

    # ----- 1. 准备 memory store -----
    short_term = ShortTermMemoryStore(
        base_dir=Path("data/memory/short_term"),
    )

    # 由于 Tool-Use Experience 和 Outline Experience 还没有实现
    # outline_store = OutlineExperienceStore(
    #     base_dir=Path("data/memory/long_term/outlines"),
    # )
    # tool_use_store = ToolUseExperienceStore(
    #     base_path=Path("data/memory/long_term/tool_use"),
    # )

    # ----- 2. 创建底层模型 -----
    model, formatter = create_chat_model()
    # ----- 3. 创建 Searcher Agent -----
    searcher_toolkit = build_searcher_toolkit(
        short_term=short_term,
        # tool_use_store=tool_use_store,
    )

    searcher = create_searcher_agent(model=model, formatter=formatter, toolkit=searcher_toolkit)

    print("\n=== 打印 JSON Schema (get_json_schemas) ===")
    schemas = searcher_toolkit.get_json_schemas()
    print(schemas)

    # ----- 4. 创建 Planner / Writer Agent -----
    # planner_toolkit = build_planner_toolkit(
    #     short_term=short_term,
    #     outline_store=outline_store,
    #     searcher=searcher,
    # )
    # planner = create_planner_agent(model=model, toolkit=planner_toolkit)

    writer_toolkit = build_writer_toolkit(
        short_term=short_term,
        searcher=searcher,
    )
    writer = create_writer_agent(model=model, formatter=formatter, toolkit=writer_toolkit)

    # ----- 5. 调用 Planner：生成 / 修订 outline.md -----
    # planner_input = Msg(
    #     name="User",
    #     content=task_desc,
    #     role="user",
    # )
    #
    # ----- 6. 调用 Writer：基于 outline.md 写 Manuscript 并导出 PDF -----
    # final_msg = await writer(outline_msg)

    writer_input = Msg(
        name="User",
        content=(
                "下面是本次任务描述，请你基于 outline.md 开始写作：\n\n"
                + task_desc
        ),
        role="user",
    )
    final_msg = await writer(writer_input)

    # Msg.get_text_content() 在官方文档中用于从消息里取纯文本
    return final_msg.get_text_content()
