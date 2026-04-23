# -*- coding: utf-8 -*-
import re
from urllib.parse import urlparse

CITE_RE = re.compile(r"\[\^cite_id:([A-Za-z0-9_\-]+)(?:\|[^\]]*)?\]")

def material_source_key(
    short_term,  # ShortTermMemoryStore
    cite_id: str,
) -> str:
    """
    给定 cite_id，返回“来源归一化 key”，用于判定两个 material 是否来自同一来源。

    规则：
      - search_engine：按 domain 聚合（同域名视为同来源）
      - 计算工具：每个 cite_id 视为独立来源
      - AKshare API：按api的数据来源
      - 其它：按 meta.source 归一化（source 为空则回退到 filename）

    返回示例：
      - "web:zhihu.com"
      - "calc:calculate_financial_ratio_result_1767927143"
      - "akshare:eastmoney"
      - "src:xxx"
      - "file:xxx.csv"
    """
    meta = short_term.get_material_meta(cite_id)
    if meta is None:
        raise ValueError(f"Material cite_id='{cite_id}' 不存在于 registry。")

    src_raw = (meta.source or "").strip()
    src_lower = src_raw.lower()

    # 1) AKshare：api的数据来源，如eastmoney，同花顺等
    if src_lower.startswith("akshare"):
        provider = ""
        if ":" in src_raw:
            provider = src_raw.split(":", 1)[1].strip().lower()
        provider = provider or "unknown"
        return f"akshare:{provider}"

    # 2) 计算工具：每个结果独立算一个来源
    if cite_id.startswith("calculate_"):
        return f"calc:{cite_id}"

    # 3) Search engine：按 domain 聚合
    if cite_id.startswith("search_engine") or ("search engine" in src_lower):
        domain = None
        if not src_raw:
            domain = None
        _DOMAIN_IN_SOURCE_RE = re.compile(
            r"(?:来源|source)\s*[：:]\s*([a-z0-9][a-z0-9.-]*\.[a-z]{2,})",
            flags=re.IGNORECASE,
        )
        _URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", flags=re.IGNORECASE)
        m = _DOMAIN_IN_SOURCE_RE.search(src_raw)
        if m:
            domain=m.group(1)
        um = _URL_RE.search(src_raw)
        if um:
            try:
                domain=urlparse(um.group(0)).netloc
            except Exception:
                domain = None

    
        if not domain:
            # 从 material 原文解析 link/url
            obj = short_term.load_material(cite_id)
            url = obj[0].get("link","")
            if url:
                domain = urlparse(url).netloc

        domain = (domain or "").strip().lower()
        if domain and domain.startswith("www."):
            domain = domain[4:]

        return f"web:{domain or 'unknown'}"

    # 4) 其它：按 source 归一化；若 source 为空，退化为 filename
    if src_raw:
        norm_src = re.sub(r"\s+", " ", src_raw).lower()
        return f"src:{norm_src}"
    return f"file:{(meta.filename or '').strip().lower()}"
