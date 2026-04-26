# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from agentscope.model import DashScopeChatModel
from agentscope.token import HuggingFaceTokenCounter
from agentscope.formatter import (
    DashScopeChatFormatter,
    DashScopeMultiAgentFormatter,
    DeepSeekChatFormatter,
    OpenAIChatFormatter
)

from agentscope.model import (
    DashScopeChatModel,
    OpenAIChatModel
)
from openai import OpenAI

from src.utils.format import PatchedOpenAIChatFormatter
import config

cfg = config.Config()


def create_chat_model(reasoning=True, model_cfg=None):
    """统一创建一个聊天模型实例。
    """
    m = model_cfg or cfg.get_model_cfg()

    provider = m["provider"]
    model_name = m["model_name"]

    base_url = m.get("base_url_env", "")
    stream = m["stream"]

    temperature = m.get("temperature", 0)
    top_k = m.get("top_k", 20)
    reasoning_only = m.get("reasoning_only", False)
    non_reasoning_model_name = m.get("non_reasoning_model_name")

    if provider == "openrouter":
        extra_body = {"top_k": top_k}
        if reasoning or reasoning_only:
            extra_body["reasoning"] = {"enabled": True}
        resolved_model_name = model_name
        if not reasoning and not reasoning_only:
            resolved_model_name = non_reasoning_model_name or model_name.replace("-thinking", "")
            extra_body["reasoning"] = {"enabled": False}
        return OpenAIChatModel(
            model_name=resolved_model_name,
            api_key=os.environ.get("API_KEY"),
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"extra_body": extra_body,
                             "temperature": temperature},
        )

    elif provider == "deepseek":
        thinking_enabled = reasoning or reasoning_only
        generate_kwargs = {
            "extra_body": {
                "thinking": {
                    "type": "enabled" if thinking_enabled else "disabled",
                },
            },
        }
        if thinking_enabled:
            generate_kwargs["reasoning_effort"] = m.get("reasoning_effort", "high")
        else:
            generate_kwargs["temperature"] = temperature
        return OpenAIChatModel(
            model_name=model_name,
            api_key=os.environ.get("API_KEY"),
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs=generate_kwargs,
        )

    elif provider == "dashscope":
        return DashScopeChatModel(
            model_name=model_name,
            api_key=os.environ.get("API_KEY"),
            stream=stream,
        )

    elif provider in ["xiaomi"]:
        return OpenAIChatModel(
            model_name=model_name,
            api_key=os.environ.get("API_KEY"),
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"extra_body":{"enable_thinking": reasoning}}
        )

    elif provider in ["ark"]:
        return OpenAIChatModel(
            model_name=model_name,
            api_key=os.environ.get("API_KEY"),
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={'extra_body': {"thinking":{"type": 'enabled' if reasoning else 'disabled'}},
                             "temperature": temperature,
                             'max_tokens': 16384},
        )

    elif provider in ["kalm"]:

        client = OpenAI(api_key=os.environ.get("API_KEY"), base_url=base_url)
        headers = {"ADAMS-BUSINESS": "3939",
                   "Adams-Platform-User": os.environ.get("USER_NAME"),
                   "Adams-User-Token": os.environ.get("USER_TOKEN"),
                   "ADAMS-PREDICT-LIMIT-S": "300",  # 配置前端超时为300秒
                   }
        model_name = client.models.list(extra_headers=headers).data[0].id  # 获取服务对应的模型名
        return OpenAIChatModel(
            model_name=model_name,
            api_key=os.environ.get("API_KEY"),
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={'extra_body': {"thinking":{"type": 'enabled' if reasoning else 'disabled'}},
                             "extra_headers": headers,
                             "temperature": temperature,
                             'max_tokens': 16384},
        )

    else:
        return OpenAIChatModel(
            model_name=model_name,
            api_key=os.environ.get("API_KEY"),
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"extra_body": {"reasoning": {"enabled": reasoning, "exclude": False}},
                             "temperature": temperature},
        )

def create_agent_formatter(model_cfg=None):
    m = model_cfg or cfg.get_model_cfg()
    provider = m["provider"]
    # token_counter = HuggingFaceTokenCounter(
    #     "Qwen/Qwen2.5-7B-Instruct",
    #     use_mirror=True,
    #     use_fast=True,
    #     trust_remote_code=True,
    # )

    # if provider == "openrouter":
    #     return OpenAIChatFormatter(token_counter=token_counter, max_tokens=250_000)
    # elif provider == "modelscope":
    #     return PatchedOpenAIChatFormatter(token_counter=token_counter, max_tokens=120_000)
    # elif provider == "dashscope":
    #     return DashScopeChatFormatter(token_counter=token_counter, max_tokens=250_000)
    # else:
    #     raise ValueError(f"未知 provider: {provider}")

    if provider == "deepseek":
        return DeepSeekChatFormatter()
    elif provider in ("openrouter", "xiaomi"):
        return OpenAIChatFormatter()
    elif provider == "modelscope":
        return OpenAIChatFormatter()
    elif provider == "ark":
        return PatchedOpenAIChatFormatter()
    elif provider == "dashscope":
        return DashScopeChatFormatter()
    else:
        raise ValueError(f"未知 provider: {provider}")   


def create_vlm_model():
    """
    创建支持多模态(图片)的模型实例：
    - 直接使用 model(messages=[...], temperature=0.0) 的调用方式
    - 允许在配置里单独指定 vision_model_name；否则复用 model_name
    """
    m = cfg.get_vlm_cfg()

    provider = m["provider"]
    model_name = m["model_name"]

    base_url = m.get("base_url_env", "")
    stream = m.get("stream", False)
    temperature = m.get("vision_temperature", 0.1)

    if provider == "openrouter":
        return OpenAIChatModel(
            model_name=model_name,
            api_key = os.environ.get("VLM_API_KEY") or os.environ.get("API_KEY"),
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"temperature": temperature},
        )
    elif provider in ["ark"]:
        return OpenAIChatModel(
            model_name=model_name,
            api_key="7e3cde63-e447-4a63-9445-69c8942fdfa9",
            stream=stream,
            client_kwargs={"base_url": base_url},
        )
    elif provider == "xiaomi":
        return OpenAIChatModel(
            model_name=model_name,
            api_key = os.environ.get("VLM_API_KEY") or os.environ.get("API_KEY"),
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"temperature": temperature},
        )

    elif provider in ["kalm"]:

        client = OpenAI(api_key=os.environ.get("API_KEY"), base_url=base_url)
        headers = {"ADAMS-BUSINESS": "deep-research",
                   "Adams-Platform-User": os.environ.get("USER_NAME"),
                   "Adams-User-Token": os.environ.get("USER_TOKEN"),
                   "ADAMS-PREDICT-LIMIT-S": "300",  # 配置前端超时为300秒
                   }
        model_name = client.models.list(extra_headers=headers).data[0].id  # 获取服务对应的模型名
        return OpenAIChatModel(
            model_name=model_name,
            api_key=os.environ.get("API_KEY"),
            stream=stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"extra_headers": headers,
                             "temperature": temperature,
                             'max_tokens': 16384},
        )

    else:
        raise ValueError(f"未知 provider: {provider}")


llm_reasoning = create_chat_model()
llm_instruct = create_chat_model(reasoning=False)
llm_outline_refine = create_chat_model(model_cfg=cfg.get_outline_refine_model_cfg())
llm_judge = create_chat_model(model_cfg=config.Config(llm_name=os.getenv("JUDGE_NAME", None)).get_model_cfg())
formatter = create_agent_formatter()
outline_refine_formatter = create_agent_formatter(model_cfg=cfg.get_outline_refine_model_cfg())
