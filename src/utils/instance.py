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
from src.utils.token_counter import RoughTokenCounter

import config

cfg = config.Config()


def create_chat_model():
    """统一创建一个聊天模型实例。
    """
    m = cfg.get_model_cfg()

    provider = m["provider"]
    model_name = m["model_name"]

    base_url = m.get("base_url_env", "")
    stream = m["stream"]

    if provider == "openrouter":
        return OpenAIChatModel(
            model_name=model_name,
            api_key=os.environ["API_KEY"],
            # base_url=base_url,
            stream=stream,
            client_args={"base_url": base_url}
        )

    elif provider == "dashscope":
        return DashScopeChatModel(
            model_name=model_name,
            api_key=os.environ["API_KEY"],
            stream=stream,
        )

    else:
        raise ValueError(f"未知 provider: {provider}")

    # return DashScopeChatModel(
    #     # model_name=os.environ.get("DASHSCOPE_MODEL", "qwen3-max"),
    #     model_name=model_name,
    #     api_key=os.environ["DASHSCOPE_API_KEY"],
    #     stream=stream,
    # )

def create_agent_formatter():
    m = cfg.get_model_cfg()
    provider = m["provider"]

    if provider == "openrouter":
        return OpenAIChatFormatter()

    elif provider == "dashscope":
        return DashScopeChatFormatter()
    else:
        raise ValueError(f"未知 provider: {provider}")
    
def create_searcher_formatter():
    m = cfg.get_model_cfg()
    token_counter = RoughTokenCounter(chars_per_token=4)
    provider = m["provider"]

    if provider == "openrouter":
        return OpenAIChatFormatter(token_counter=token_counter,max_tokens=400_000)

    elif provider == "dashscope":
        return DashScopeChatFormatter(token_counter=token_counter,max_tokens=400_000)
    else:
        raise ValueError(f"未知 provider: {provider}")

# def single_agent_formatter() -> DashScopeChatFormatter:
#     """单 Agent 对话用 formatter。"""
#     return DashScopeChatFormatter()


# def multi_agent_formatter() -> DashScopeMultiAgentFormatter:
#     return DashScopeMultiAgentFormatter()
