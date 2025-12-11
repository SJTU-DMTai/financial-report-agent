# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from agentscope.model import DashScopeChatModel
from agentscope.token import HuggingFaceTokenCounter
from agentscope.formatter import (
    DashScopeChatFormatter,
    DashScopeMultiAgentFormatter,
    OpenAIChatFormatter
)

from agentscope.model import (
    DashScopeChatModel,
    OpenAIChatModel
)

from src.utils.patched_openaIchatformatter import PatchedOpenAIChatFormatter
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
            api_key=os.environ.get("API_KEY"),
            stream=stream,
            client_args={"base_url": base_url}
        )

    elif provider == "dashscope":
        return DashScopeChatModel(
            model_name=model_name,
            api_key=os.environ.get("API_KEY"),
            stream=stream,
        )
    elif provider == "modelscope":
        return OpenAIChatModel(
            model_name=model_name,
            api_key=os.environ.get("API_KEY"),
            stream=stream,
            client_args={"base_url": base_url},
            # generate_kwargs={"extra_body":{"enable_thinking": False}}
        )

    else:
        raise ValueError(f"未知 provider: {provider}")

def create_agent_formatter():
    m = cfg.get_model_cfg()
    provider = m["provider"]
    token_counter = HuggingFaceTokenCounter(
        "Qwen/Qwen2.5-7B-Instruct",
        use_mirror=True,
        use_fast=True,
        trust_remote_code=True,
    )

    # if provider == "openrouter":
    #     return OpenAIChatFormatter(token_counter=token_counter, max_tokens=250_000)
    # elif provider == "modelscope":
    #     return PatchedOpenAIChatFormatter(token_counter=token_counter, max_tokens=120_000)
    # elif provider == "dashscope":
    #     return DashScopeChatFormatter(token_counter=token_counter, max_tokens=250_000)
    # else:
    #     raise ValueError(f"未知 provider: {provider}")

    if provider == "openrouter":
        return OpenAIChatFormatter()
    elif provider == "modelscope":
        return PatchedOpenAIChatFormatter()
    elif provider == "dashscope":
        return DashScopeChatFormatter()
    else:
        raise ValueError(f"未知 provider: {provider}")   

# def single_agent_formatter() -> DashScopeChatFormatter:
#     """单 Agent 对话用 formatter。"""
#     return DashScopeChatFormatter()


# def multi_agent_formatter() -> DashScopeMultiAgentFormatter:
#     return DashScopeMultiAgentFormatter()
