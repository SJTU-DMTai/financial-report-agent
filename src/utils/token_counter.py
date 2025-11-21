from __future__ import annotations

from agentscope.token import TokenCounterBase
from typing import Any, List
from abc import ABC, abstractmethod

class RoughTokenCounter(TokenCounterBase):
    """极简、轻量级 token 计数器.

    - 实现 TokenCounterBase 接口: async count(messages: list[dict]) -> int
    """

    def __init__(self, chars_per_token: int = 4, min_tokens: int = 1) -> None:
        self.chars_per_token = chars_per_token
        self.min_tokens = min_tokens

    def _count_sync(self, messages: list[dict]) -> int:
        total_chars = 0

        for m in messages:
            for v in m.values():
                if isinstance(v, str):
                    total_chars += len(v)

        tokens = total_chars // self.chars_per_token
        return max(self.min_tokens, tokens)

    async def count(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> int:
        return self._count_sync(messages)


