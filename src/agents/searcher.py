from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit
from agentscope.model import DashScopeChatModel
from agentscope.model import OpenAIChatModel
from ..prompt import prompt_dict

from ..tools.material_tools import *

from ..tools.search_tools import *
from ..memory.short_term import ShortTermMemoryStore

def create_searcher_agent(
    model,
    formatter,
    toolkit: Toolkit,
) -> ReActAgent:
    """Searcher 使用 ReActAgent 实现。
    """
    return ReActAgent(
        name="Searcher",
        sys_prompt=prompt_dict['searcher_sys_prompt'],
        model=model,
        memory=InMemoryMemory(),
        formatter=formatter,
        toolkit=toolkit,
        parallel_tool_calls=True,
    )

def build_searcher_toolkit(
        short_term: ShortTermMemoryStore,
        # tool_use_store: ToolUseExperienceStore,
) -> Toolkit:
    """创建 Searcher 专用 Toolkit。
    """
    toolkit = Toolkit()
    material_tools = MaterialTools(short_term=short_term)

    toolkit.register_tool_function(search_engine)

    # -------- Material Tools --------

    # ========================================
    # 股价数据 Price
    # ========================================

    # toolkit.register_tool_function(
    #     material_tools.fetch_realtime_price_material
    # )

    toolkit.register_tool_function(
        material_tools.fetch_history_price_material
    )

    # ========================================
    # 金融新闻 News
    # ========================================

    # toolkit.register_tool_function(
    #     material_tools.fetch_stock_news_material
    # )

    toolkit.register_tool_function(
        material_tools.fetch_disclosure_material
    )

    # ========================================
    # 财务报表 Financial Statements
    # ========================================

    toolkit.register_tool_function(
        material_tools.fetch_balance_sheet_material
    )

    toolkit.register_tool_function(
        material_tools.fetch_profit_table_material
    )
    toolkit.register_tool_function(
        material_tools.fetch_cashflow_table_material
    )
    # ========================================
    # 股东信息 Shareholders
    # ========================================

    toolkit.register_tool_function(
        material_tools.fetch_top10_float_shareholders_material
    )

    toolkit.register_tool_function(
        material_tools.fetch_top10_shareholders_material
    )

    toolkit.register_tool_function(
        material_tools.fetch_main_shareholders_material
    )

    toolkit.register_tool_function(
        material_tools.fetch_shareholder_count_detail_material
    )

    toolkit.register_tool_function(
        material_tools.fetch_shareholder_change_material
    )

    # ========================================
    # 业务范围 Business Scope
    # ========================================

    toolkit.register_tool_function(
        material_tools.fetch_business_description_material
    )

    toolkit.register_tool_function(
        material_tools.fetch_business_composition_material
    )

    toolkit.register_tool_function(
        material_tools.read_material
    )

    return toolkit
