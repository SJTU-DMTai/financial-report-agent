from __future__ import annotations

from pathlib import Path
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import ToolUseExperienceStore
from .material_tools import *


# -------- Local File Reader --------
def local_file_reader(symbol: str, limit: int = 3) -> ToolResponse:
    """读取本地相关公司的研报，以 demonstration 形式写入 short-term memory。

    Args:
        symbol (str):
            股票代码或公司名称。
        limit (int):
            最多读取多少篇研报。
    """
    # 真实实现中这里应该去 data/reports/ 下面扫文件并做解析
    text = f"[local_file_reader] 读取 {symbol} 的最近 {limit} 篇本地研报（示例占位）。"
    return ToolResponse(
        content=[TextBlock(type="text", text=text)],
        metadata={"symbol": symbol, "limit": limit},
    )





# -------- Memory Tool：经验读写 --------
# def retrieve_tool_use_experience(
#     tool_name: str,
#     store: ToolUseExperienceStore,
# ) -> ToolResponse:
#     """从长期记忆中召回某个工具的历史使用经验。"""
#     exps = store.load_experiences(tool_name)
#     text = f"[retrieve_tool_use_experience] 共召回 {len(exps)} 条与 {tool_name} 相关的经验。"
#     return ToolResponse(
#         content=[TextBlock(type="text", text=text)],
#         metadata={"experiences": exps},
#     )


# def save_tool_use_experience(
#     tool_name: str,
#     exp: dict[str, Any],
#     store: ToolUseExperienceStore,
# ) -> ToolResponse:
#     """保存一条工具使用经验到长期记忆。"""
#     store.append_experience(tool_name, exp)
#     text = f"[save_tool_use_experience] 已保存 {tool_name} 的一条经验。"
#     return ToolResponse(
#         content=[TextBlock(type="text", text=text)],
#         metadata={"saved": True},
#     )


# -------- Toolkit Builder --------
def build_searcher_toolkit(
    short_term: ShortTermMemoryStore,
    tool_use_store: ToolUseExperienceStore,
) -> Toolkit:
    """创建 Searcher 专用 Toolkit。
    """
    toolkit = Toolkit()
    tools = MaterialTools(short_term=short_term)

    # 普通函数直接注册
    # toolkit.register_tool_function(local_file_reader)
    #
    # # 需要额外依赖的工具，通过 preset_kwargs 传入 store
    # toolkit.register_tool_function(
    #     retrieve_tool_use_experience,
    #     preset_kwargs={"store": tool_use_store},
    # )
    # toolkit.register_tool_function(
    #     save_tool_use_experience,
    #     preset_kwargs={"store": tool_use_store},
    # )

    # -------- Material Tools --------

    # ========================================
    # 股价数据 Price
    # ========================================

    toolkit.register_tool_function(
        tools.fetch_realtime_price_material
    )
    
    toolkit.register_tool_function(
        tools.fetch_history_price_material
    )
    
    # ========================================
    # 金融新闻 News
    # ========================================
    
    toolkit.register_tool_function(
        tools.fetch_stock_news_material
    )
    
    toolkit.register_tool_function(
        tools.fetch_disclosure_material
    )
    
    # ========================================
    # 财务报表 Financial Statements
    # ========================================
    

    toolkit.register_tool_function(
        tools.fetch_balance_sheet_material
    )

    toolkit.register_tool_function(
        tools.fetch_profit_table_material
    )
    
    toolkit.register_tool_function(
        tools.fetch_cashflow_table_material
    )
    
    # ========================================
    # 股东信息 Shareholders
    # ========================================
    
    toolkit.register_tool_function(
        tools.fetch_top10_float_shareholders_material
    )
    
    toolkit.register_tool_function(
        tools.fetch_top10_shareholders_material
    )
    
    toolkit.register_tool_function(
        tools.fetch_main_shareholders_material
    )
    
    toolkit.register_tool_function(
        tools.fetch_shareholder_count_detail_material
    )
    
    toolkit.register_tool_function(
        tools.fetch_shareholder_change_material
    )
    
    # ========================================
    # 业务范围 Business Scope
    # ========================================
    
    toolkit.register_tool_function(
        tools.fetch_business_description_material
    )
    
    toolkit.register_tool_function(
        tools.fetch_business_composition_material
    )

    return toolkit
