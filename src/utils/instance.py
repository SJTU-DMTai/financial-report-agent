from __future__ import annotations

import os
from agentscope.model import DashScopeChatModel
from agentscope.formatter import (
    DashScopeChatFormatter,
    DashScopeMultiAgentFormatter,
    OpenAIChatFormatter
)

from agentscope.model import (
    DashScopeChatModel,
    OpenAIChatModel
)

import config

cfg = config.Config()


def create_chat_model() -> DashScopeChatModel:
    """统一创建一个聊天模型实例。
    """
    m = cfg.get_model_cfg()

    provider = m["provider"]
    model_name = m["model_name"]

    base_url = os.environ.get(m.get("base_url_env", ""), None)
    stream = m["stream"]

    if provider == "openrouter":
        return OpenAIChatModel(
            model_name=model_name,
            api_key=os.environ["API_KEY"],
            # base_url=base_url,
            stream=stream,
            client_args={"base_url": base_url}
        ), OpenAIChatFormatter()

    elif provider == "dashscope":
        return DashScopeChatModel(
            model_name=model_name,
            api_key=os.environ["API_KEY"],
            stream=stream,
        ), DashScopeChatFormatter()

    else:
        raise ValueError(f"未知 provider: {provider}")

    # return DashScopeChatModel(
    #     # model_name=os.environ.get("DASHSCOPE_MODEL", "qwen3-max"),
    #     model_name=model_name,
    #     api_key=os.environ["DASHSCOPE_API_KEY"],
    #     stream=stream,
    # )


def single_agent_formatter() -> DashScopeChatFormatter:
    """单 Agent 对话用 formatter。"""
    return DashScopeChatFormatter()


def multi_agent_formatter() -> DashScopeMultiAgentFormatter:
    return DashScopeMultiAgentFormatter()
