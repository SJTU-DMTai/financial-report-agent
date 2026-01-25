# -*- coding: utf-8 -*-
import asyncio
from typing import Iterable, Type, Callable
import traceback

from agentscope.formatter import FormatterBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase

class EnvMsg(Exception):
    def __init__(self, *args):
        super().__init__(*args)

async def call_chatbot_with_retry(
        model: ChatModelBase, formatter: FormatterBase,
        sys_prompt: str, user_prompt: str,
        hook: Callable | None = None, max_retries=5,
        handle_hook_exceptions: Iterable[Type[BaseException]] = EnvMsg,
):
    """
    调用 ChatModel 进行评估。
    """
    messages = [
        Msg("system", sys_prompt, "system"),
        Msg("user", user_prompt, "user"),
    ]
    res = None
    for _ in range(max_retries):
        try:
            _messages = await formatter.format(messages)
            response = await model(_messages)
            res = Msg(role='assistant', content=response, name='assistant').get_text_content()
        except Exception as e:
            print(f"[调用 ChatModel 失败] 第 {_} 次尝试异常：{type(e).__name__}: {e}，")
        if hook is not None:
            try:
                return hook(res)
            except handle_hook_exceptions as e:
                messages.append(Msg("assistant", res, "assistant"))
                messages.append(Msg("user", f"异常：{type(e).__name__}: {e}", "user"))
            except Exception as e:
                print(f"[调用 ChatModel 失败] 第 {_} 次尝试异常：{type(e).__name__}: {e}，")
        else:
            return res
    raise

async def call_agent_with_retry(
    agent,
    msg,
    max_retries: int = 5,
    base_delay: float = 60.0,
    backoff_factor: float = 2.0,
    non_retry_exceptions: Iterable[Type[BaseException]] = (KeyboardInterrupt, SystemExit),
):
    """
    对 agentscope 的 agent 调用做统一重试。
    - agent: planner / writer / verifier 等 agent 实例
    - msg:   agentscope.message.Msg 实例
    - max_retries: 最大重试次数
    - base_delay: 初始等待时间（秒）
    - backoff_factor: 指数退避倍数
    - non_retry_exceptions: 不参与重试、直接抛出的异常类型集合
    """
    last_exc: BaseException | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await agent(msg)
        except non_retry_exceptions:
            # 这些异常直接抛出去，不做重试
            raise
        except Exception as e:
            print(agent.memory.content, flush=True)
            await agent.memory.clear()
            last_exc = e
            if attempt == max_retries:
                print(f"[重试失败] 第 {attempt} 次仍然报错，放弃重试。异常：{type(e).__name__}: {e}")
                raise

            sleep_time = base_delay * (backoff_factor ** (attempt - 1))
            print(
                f"[调用 agent 失败] 第 {attempt} 次尝试异常：{type(e).__name__}: {e}，"
                f"{sleep_time:.1f} 秒后重试..."
            )
            await asyncio.sleep(sleep_time)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("未知错误")
