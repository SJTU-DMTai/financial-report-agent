from __future__ import annotations

import os
from agentscope.model import DashScopeChatModel
from agentscope.formatter import (
    DashScopeChatFormatter,
    DashScopeMultiAgentFormatter,
)


def create_chat_model(stream: bool = False) -> DashScopeChatModel:
    """统一创建一个聊天模型实例。
    """
    return DashScopeChatModel(
        model_name=os.environ.get("DASHSCOPE_MODEL", "qwen3-max"),
        api_key=os.environ["DASHSCOPE_API_KEY"],
        stream=stream,
    )


def single_agent_formatter() -> DashScopeChatFormatter:
    """单 Agent 对话用 formatter。"""
    return DashScopeChatFormatter()


def multi_agent_formatter() -> DashScopeMultiAgentFormatter:
    return DashScopeMultiAgentFormatter()
