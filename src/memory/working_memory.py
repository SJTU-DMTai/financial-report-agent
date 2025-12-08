from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from typing import Union, Iterable

class SlidingWindowMemory(InMemoryMemory):
    """带最大长度限制的memory，超出后丢掉最早的消息。"""

    def __init__(self, max_messages: int = 30) -> None:
        super().__init__()
        self.max_messages = max_messages

    async def add(
        self,
        memories: Union[list[Msg], Msg, None],
        allow_duplicates: bool = False,
    ) -> None:
        await super().add(memories, allow_duplicates=allow_duplicates)

        # 再做截断：只保留最近 max_messages 条
        overflow = len(self.content) - self.max_messages
        if overflow > 0:
            # 丢掉最早的 overflow 条
            self.content = self.content[overflow:]
