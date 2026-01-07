# -*- coding: utf-8 -*-

import re
from typing import Optional, Dict, List, Iterable, Tuple
from src.memory.long_term import LongTermMemoryStore
import jieba

# 常见 A 股代码表达：600519 / sh600519 / 600519.SH / SZ000001 等
_CODE_PATTERNS = [
    re.compile(r"^(?:sh|sz)?(\d{6})$", re.IGNORECASE),
    re.compile(r"^(?:sh|sz)?(\d{6})\.(?:sh|sz)$", re.IGNORECASE),
    re.compile(r"^(?:sh|sz)(\d{6})$", re.IGNORECASE),
]

# 用于把标点等分隔为“空格”，减少分词干扰（可按需扩充）
_SEP_RE = re.compile(r"[，,。.;；:：/\\|()（）\[\]{}<>《》“”\"'!?！？\t\r\n]+")


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # 将常见分隔符替换为单空格，避免黏连
    text = _SEP_RE.sub(" ", text)
    # 多空格收敛
    text = re.sub(r"\s+", " ", text)
    return text


def _iter_tokens(text: str) -> List[str]:
    """
    先按空格切分，再对每个 chunk 分词。
    """
    norm = _normalize_text(text)
    if not norm:
        return []

    raw_chunks = [c for c in norm.split(" ") if c]
    tokens: List[str] = []

    for chunk in raw_chunks:
        tokens.append(chunk)
        for w in jieba.cut(chunk, HMM=True):
            w = w.strip()
            if w:
                tokens.append(w)

    return tokens
def _merge_adjacent_tokens(tokens_seq: List[str], max_window: int = 3) -> List[str]:
    """
    将 tokens_seq 中相邻 token 进行拼接，生成候选：
      - window=2: t[i] + t[i+1]
      - window=3: t[i] + t[i+1] + t[i+2]
    返回去重（保序）的合并候选列表。
    """
    if not tokens_seq:
        return []

    merged: List[str] = []
    n = len(tokens_seq)

    # 仅合并相邻 token，不插入空格
    for w in range(2, max_window + 1):
        for i in range(0, n - w + 1):
            parts = tokens_seq[i : i + w]
            s = "".join(p.strip() for p in parts if p and p.strip())
            if not s:
                continue
            # 可选：做长度控制，避免爆炸式候选
            if len(s) < 2:
                continue
            merged.append(s)

    # 去重（保序）
    seen = set()
    dedup: List[str] = []
    for t in merged:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup


def _extract_code(token: str) -> Optional[str]:
    t = (token or "").strip()
    if not t:
        return None
    # 兼容 600519.SH / sh600519 / SZ000001 等
    for pat in _CODE_PATTERNS:
        m = pat.match(t)
        if m:
            return m.group(1)
    # 兜底：从字符串中抽取 6 位数字（例如 “贵州茅台600519” 这种黏连）
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", t)
    if m:
        return m.group(1)
    return None



def get_entity_info(long_term:LongTermMemoryStore, text: str) -> Optional[Dict[str, str]]:
    """
    从任意字符串（如搜索 query）中解析 A 股实体（code, name）。

    解析策略：
    1) 生成候选 tokens：先按空格切分，再对每段做 jieba 分词。
    2) 优先识别股票代码：从 token 中抽取 6 位代码，调用 long_term.name_by_code(code)；
       若能查到 name，则直接返回。
    3) 若未命中代码，再识别股票名称：对 tokens 做“精确匹配”查表（fuzzy=False）；
       若命中则返回（若多个命中，取第一个）。
    4) 若仍未命中：基于 tokens 序列，尝试合并相邻 tokens
    返回：
      {"code": "600519", "name": "贵州茅台"} 或 None
    """
    tokens = _iter_tokens(text)
    if not tokens:
        return None

    # 1) 代码优先
    for tok in tokens:
        code = _extract_code(tok)
        if not code:
            continue
        name = long_term.name_by_code(code)
        if name:
            return {"code": str(code).zfill(6), "name": str(name)}

    # 2) 名称匹配
    name_tokens = sorted(
        (t for t in tokens if len(t) >= 2),  # 过滤过短 token
        key=len,
        reverse=True,
    )
    for tok in name_tokens:
        hits = long_term.codes_by_name(tok, fuzzy=False)
        if hits:
            # 若多个命中，取第一个；也可以改为返回候选列表
            return {"code": hits[0]["code"], "name": hits[0]["name"]}

    tokens_seq = _iter_tokens(text)
    merged_tokens = _merge_adjacent_tokens(tokens_seq, max_window=3)
    if not merged_tokens:
        return None

    merged_name_tokens = sorted(
        (t for t in merged_tokens if len(t) >= 2),
        key=len,
        reverse=True,
    )
    for tok in merged_name_tokens:
        hits = long_term.codes_by_name(tok, fuzzy=False)
        if hits:
            return {"code": hits[0]["code"], "name": hits[0]["name"]}

    return None
