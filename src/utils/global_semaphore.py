# -*- coding: utf-8 -*-
"""
全局信号量管理模块
用于控制整个应用的并发LLM调用数量
"""
import asyncio
import os
from typing import Optional

# 全局信号量实例
_global_semaphore: Optional[asyncio.Semaphore] = None


def get_global_semaphore() -> asyncio.Semaphore:
    """
    获取全局信号量实例。
    如果尚未初始化，则使用环境变量 N_THREAD 的值进行初始化（默认32）。

    Returns:
        asyncio.Semaphore: 全局信号量实例
    """
    global _global_semaphore
    if _global_semaphore is None:
        concurrency_limit = int(os.getenv("N_THREAD", 32))
        _global_semaphore = asyncio.Semaphore(concurrency_limit)
        print(f"[全局信号量] 初始化完成，并发限制: {concurrency_limit}")
    return _global_semaphore


def set_global_semaphore(semaphore: asyncio.Semaphore) -> None:
    """
    设置全局信号量实例（用于测试或自定义配置）。

    Args:
        semaphore: 要设置的信号量实例
    """
    global _global_semaphore
    _global_semaphore = semaphore


def reset_global_semaphore() -> None:
    """
    重置全局信号量（主要用于测试）。
    """
    global _global_semaphore
    _global_semaphore = None
