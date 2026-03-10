# -*- coding: utf-8 -*-
import asyncio
import os
import random
import time
import traceback
import warnings
from copy import deepcopy
from typing import Iterable, Type, Callable, Optional

from agentscope.agent import ReActAgent
from agentscope.formatter import FormatterBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase
from openai import RateLimitError
from pydantic import BaseModel

from .global_semaphore import get_global_semaphore
from .instance import cfg

endpoints = {"ep-20260212213128-4kwzl", "ep-20260205192925-9m7nq", "ep-20250318101605-5c67d"}

class EnvMsg(Exception):
    def __init__(self, *args):
        super().__init__(*args)

async def call_chatbot_with_retry(
    model: ChatModelBase, formatter: FormatterBase,
    sys_prompt: str, user_prompt: str,
    hook: Callable | None = None, max_retries=5,
    handle_hook_exceptions: Iterable[Type[BaseException]] = EnvMsg,
    structured_model: Type[BaseModel] | None = None,
    semaphore: Optional[asyncio.Semaphore] = None
):
    """
    调用 ChatModel 进行评估。
    """
    # 如果没有提供semaphore，使用全局semaphore
    if semaphore is None:
        semaphore = get_global_semaphore()

    assert user_prompt is not None
    messages = [
        Msg("system", sys_prompt, "system"),
        Msg("user", user_prompt, "user"),
    ]

    res = ""
    async with semaphore:
        exceed_tpm_models = set()
        for _ in range(max_retries):
            try:
                _messages = await formatter.format(messages)
                response = await model(_messages, structured_model=structured_model)
                if structured_model is not None:
                    res = response.metadata
                else:
                    res = Msg(role='assistant', content=response.content, name='assistant').get_text_content()
            except RateLimitError as e:
                if cfg.get_model_cfg()['provider'] == 'ark' and os.getenv("LLM_NAME") == 'deepseek-v3.2' and isinstance(e, RateLimitError):
                    print(e)
                    exceed_tpm_models.add(model.model_name)
                    if len(exceed_tpm_models) >= len(endpoints):
                        await asyncio.sleep(60)
                        exceed_tpm_models = set()
                    model.model_name = list(endpoints - exceed_tpm_models)[0]
                    print("切换为", model.model_name, flush=True)
            except Exception as e:
                if "服务限流，请稍后重试":
                    time.sleep(60)
                warnings.warn(f"[调用 ChatModel 失败] 第 {_} 次尝试异常：{type(e).__name__}: {e}，")
                continue
            if res:
                if hook is not None:
                    try:
                        return hook(res)
                    except handle_hook_exceptions as e:
                        warnings.warn(f"user: {user_prompt}\n[调用 ChatModel 失败] 第 {_} 次尝试异常：{type(e).__name__}: {e}，")
                        if structured_model is not None:
                            messages.append(Msg("assistant", str(res), "assistant"))
                        else:
                            messages.append(Msg("assistant", res, "assistant"))
                        messages.append(Msg("user", f"异常：{type(e).__name__}: {e}", "user"))
                    except Exception as e:
                        warnings.warn(f"[调用 ChatModel 失败] prompt:{user_prompt} res:{response} 第 {_} 次尝试异常：{type(e).__name__}: {e}，")
                else:
                    return res
        # traceback.print_exc()
        warnings.warn(f"失败 user: {user_prompt}\nres: {res}")
        raise Exception("调用 ChatModel 多次失败，放弃重试。")


async def call_agent_with_retry(
    agent: ReActAgent,
    msg: Msg,
    max_retries: int = 5,
    base_delay: float = 60.0,
    backoff_factor: float = 2.0,
    non_retry_exceptions: Iterable[Type[BaseException]] = (SystemExit, ),
    semaphore: Optional[asyncio.Semaphore] = None
):
    """
    对 agentscope 的 agent 调用做统一重试。
    - agent: planner / writer / verifier 等 agent 实例
    - msg:   agentscope.message.Msg 实例
    - max_retries: 最大重试次数
    - base_delay: 初始等待时间（秒）
    - backoff_factor: 指数退避倍数
    - non_retry_exceptions: 不参与重试、直接抛出的异常类型集合
    - semaphore: 可选的信号量，用于控制并发（如果为None，使用全局semaphore）
    """
    # 如果没有提供semaphore，使用全局semaphore
    if semaphore is None:
        semaphore = get_global_semaphore()

    last_exc: BaseException | None = None
    async with semaphore:
        exceed_tpm_models = set()
        for attempt in range(1, max_retries + 1):
            try:
                return await agent(msg)
            # except non_retry_exceptions as e:
            #     # 这些异常直接抛出去，不做重试
            #     raise e
            except RateLimitError as e:
                if cfg.get_model_cfg()['provider'] == 'ark' and os.getenv("LLM_NAME") == 'deepseek-v3.2' and isinstance(e, RateLimitError):
                    print(e)
                    exceed_tpm_models.add(agent.model.model_name)
                    if len(exceed_tpm_models) >= len(endpoints):
                        await asyncio.sleep(base_delay)
                        exceed_tpm_models = set()
                    agent.model.model_name = list(endpoints - exceed_tpm_models)[0]
                    print("切换为", agent.model.model_name, flush=True)
                    for pos_id in range(len(agent.memory.content)):
                        if msg.id == agent.memory.content[pos_id][0].id:
                            agent.memory.content = agent.memory.content[:pos_id]
                            break
            except Exception as e:
                print(agent.memory.content, flush=True)
                await agent.memory.clear()
                last_exc = e
                if attempt == max_retries:
                    warnings.warn(f"[重试失败] 第 {attempt} 次仍然报错，放弃重试。异常：{type(e).__name__}: {e}")
                    raise last_exc

                sleep_time = base_delay * (backoff_factor ** (attempt - 1))
                traceback.print_exc()
                warnings.warn(
                    f"[调用 agent 失败] 第 {attempt} 次尝试异常：{type(e).__name__}: {e}，"
                    f"{sleep_time:.1f} 秒后重试..."
                )
                await asyncio.sleep(sleep_time)
