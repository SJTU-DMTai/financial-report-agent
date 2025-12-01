from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter
from agentscope.tool import Toolkit
from agentscope.model import DashScopeChatModel

from pathlib import Path
from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit
from ..tools.material_tools import *
from ..tools.graphic_tools import *
from ..tools.manuscipt_tools import *
from ..tools.outline_tools import *
from ..tools.search_tools import *
from ..tools.calculate_tools import *
from ..memory.short_term import ShortTermMemoryStore
from ..prompt import prompt_dict


def create_writer_agent(
    model,
    formatter,
    toolkit: Toolkit,
) -> ReActAgent:
    return ReActAgent(
        name="Writer",
        sys_prompt=prompt_dict['writer_sys_prompt'],
        model=model,
        memory=InMemoryMemory(),
        formatter=formatter,
        toolkit=toolkit,
        parallel_tool_calls=True,
        max_iters=50
    )

# ---- Toolkit Builder ----
def build_writer_toolkit(
    short_term: ShortTermMemoryStore,
    searcher: ReActAgent,
) -> Toolkit:
    toolkit = Toolkit()

    manuscript_tools = ManuscriptTools(short_term=short_term)

    toolkit.register_tool_function(manuscript_tools.read_manuscript_section)

    toolkit.register_tool_function(manuscript_tools.replace_manuscript_section)

    search_tools = SearchTools(short_term=short_term)
    toolkit.register_tool_function(search_tools.searcher_tool(searcher))

    chart_tools = GraphicTools(short_term=short_term)
    toolkit.register_tool_function(chart_tools.generate_chart_by_template)
    toolkit.register_tool_function(chart_tools.generate_chart_by_python_code)

    material_tools = MaterialTools(short_term=short_term)
    toolkit.register_tool_function(
        material_tools.read_material
    )
    calculate_tools = CalculateTools(short_term=short_term)
    toolkit.register_tool_function(calculate_tools.calculate_cashflow_metric)
    toolkit.register_tool_function(calculate_tools.calculate_financial_ratio)
    toolkit.register_tool_function(calculate_tools.calculate_forecast_metric)
    toolkit.register_tool_function(calculate_tools.calculate_math_metric)
    toolkit.register_tool_function(calculate_tools.calculate_timeseries_transform)
    toolkit.register_tool_function(calculate_tools.calculate_valuation_metric)
    toolkit.register_tool_function(calculate_tools.calculate_or_analysis_by_python_code)

    return toolkit

