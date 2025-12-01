import asyncio

async def call_agent_with_retry(agent, msg, max_retries=3, delay=5):
    """
    尝试调用 agent，如果发生异常（如网络中断），则进行重试。
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            # 尝试调用 Agent
            return await agent(msg)
        except Exception as e:
            last_exception = e
            print(f"\n[网络波动警告] 调用 {agent.name} 失败 (第 {attempt + 1}/{max_retries} 次尝试)。")
            print(f"错误信息: {e}")
            if attempt < max_retries - 1:
                print(f"将在 {delay} 秒后重试...")
                await asyncio.sleep(delay)
            else:
                print("[错误] 重试次数已用尽，放弃本次操作。")
    
    # 如果重试都失败了，抛出最后一次的异常
    raise last_exception