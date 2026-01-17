# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import math
import re
from ..utils.get_entity_info import get_entity_info
from ..memory.short_term import ShortTermMemoryStore, MaterialMeta
from ..memory.long_term import LongTermMemoryStore

# ========= 时间解析 =========
_RE_YEAR = re.compile(r"\b(20\d{2})\b")
_RE_DATE1 = re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_RE_DATE2 = re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b")  # yyyymmdd
_RE_Q = re.compile(r"\b(20\d{2})\s*[qQ]\s*([1-4])\b")
_RE_Q_ZH = re.compile(r"(20\d{2})\s*年?\s*第?\s*([一二三四1-4])\s*季(?:度)?")

_Q_ZH_MAP = {"一": "1", "二": "2", "三": "3", "四": "4"}

def _extract_query_time_signals(q: str) -> Dict[str, Any]:
    q = (q or "")
    years = set(m.group(1) for m in _RE_YEAR.finditer(q))

    dates = set()
    for m in _RE_DATE1.finditer(q):
        y, mm, dd = m.group(1), int(m.group(2)), int(m.group(3))
        dates.add(f"{y}-{mm:02d}-{dd:02d}")
    for m in _RE_DATE2.finditer(q):
        y, mm, dd = m.group(1), int(m.group(2)), int(m.group(3))
        # 粗过滤：防止把 6 位股票代码误判为日期
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            dates.add(f"{y}-{mm:02d}-{dd:02d}")

    quarters = set()
    for m in _RE_Q.finditer(q):
        quarters.add((m.group(1), m.group(2)))
    for m in _RE_Q_ZH.finditer(q):
        y = m.group(1)
        qq = _Q_ZH_MAP.get(m.group(2), m.group(2))
        if qq in {"1", "2", "3", "4"}:
            quarters.add((y, qq))

    return {"years": years, "dates": dates, "quarters": quarters}


def _meta_time_values(meta: MaterialMeta) -> List[str]:
    """
    把 meta.time 里可能出现的日期字符串抽出来，统一用于匹配。
    time 允许：{}, {"point":...}, {"start":...,"end":...}
    """
    out = []
    t = meta.time or {}
    for k in ("point", "start", "end"):
        v = t.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


# ========= tokenization for BM25 =========

_RE_CJK_SEQ = re.compile(r"[\u4e00-\u9fff]+")
_RE_ALNUM_SEQ = re.compile(r"[a-zA-Z]+|\d+(?:\.\d+)?")

def _tokenize_for_bm25(text: str) -> List[str]:
    """
    面向中文短文本：使用 CJK bigram + 英文/数字序列 token。
    """
    text = (text or "").strip().lower()
    if not text:
        return []

    toks: List[str] = []

    # 英文/数字序列
    toks.extend(_RE_ALNUM_SEQ.findall(text))

    # 中文 bigram
    for seq in _RE_CJK_SEQ.findall(text):
        if len(seq) == 1:
            toks.append(seq)
        else:
            toks.extend(seq[i:i+2] for i in range(len(seq) - 1))

    return toks


# ========= 规则召回 =========

_DEFAULT_KEYWORDS = [
    # 银行/金融常见指标
    "不良贷款率", "不良率", "拨备覆盖率", "拨备", "净息差", "nim", "资本充足率", "核心一级资本充足率",
    "资产质量", "贷款", "存款", "对公", "零售", "手续费及佣金", "非息收入", "利息收入", "利息支出",
    "净利润", "营收", "营业收入", "成本收入比", "roe", "roa", "拨贷比", "逾期", "核销",
    # 三大报表/常见表名
    "资产负债表", "利润表", "现金流量表", "现金流", "财务报表", "按报告期", "按年度", "按季度",
    # 常见任务词
    "估值", "分红", "股息率", "风险", "监管", "业绩", "展望", "预测",
]


def _rule_score(
    query: str,
    meta: MaterialMeta,
    entity_terms: Dict[str, Any],
    time_sig: Dict[str, Any],
    keywords: List[str],
) -> float:

    desc = meta.description or ""
    if not desc:
        return 0.0

    score = 0.0

    # 1) entity
    code = entity_terms.get("code") or ""
    name = entity_terms.get("name") or ""

    if code:
        if (meta.entity or {}).get("code") == code:
            score += 1.2
        if code in desc:
            score += 0.6

    if name:
        if (meta.entity or {}).get("name") == name:
            score += 0.6
        if name in desc:
            score += 0.3

    # 2) 时间加权
    years_q = time_sig.get("years") or set()
    dates_q = time_sig.get("dates") or set()
    quarters_q = time_sig.get("quarters") or set()

    meta_times = _meta_time_values(meta)
    meta_time_text = " ".join(meta_times)

    # 日期精确命中优先
    for d in dates_q:
        if d in meta_time_text or d in desc:
            score += 0.7
            break

    # 年份命中
    if years_q:
        # meta.time 或 description 命中任一年
        if any(y in meta_time_text for y in years_q) or any(y in desc for y in years_q):
            score += 0.25

    # 季度命中（允许字面出现 2025q3）
    for (y, q) in quarters_q:
        if (f"{y}q{q}" in desc) or (f"{y} q{q}" in desc) or (y in desc and q in desc):
            score += 0.25
            break

    # 3) 关键词命中（指标/任务词）
    hit = 0
    active = [kw for kw in keywords if kw and kw in query]
    if active:
        hit = sum(1 for kw in active if kw in desc)
        score += min(0.12 * hit, 0.6)

    return score

def _bm25_scores(
    query_tokens: List[str],
    docs_tokens: List[List[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """
    标准 Okapi BM25，对输入 docs_tokens（候选集）计算 BM25 分数。
    """
    N = len(docs_tokens)
    if N == 0 or not query_tokens:
        return [0.0] * N

    # 文档长度与 avgdl
    doc_lens = [len(toks) for toks in docs_tokens]
    avgdl = (sum(doc_lens) / N) if N > 0 else 0.0
    avgdl = max(avgdl, 1.0)

    # df 统计
    df: Dict[str, int] = {}
    for toks in docs_tokens:
        seen = set(toks)
        for t in seen:
            df[t] = df.get(t, 0) + 1

    # idf（加 1 避免负值/极端）
    idf: Dict[str, float] = {}
    for t, dfi in df.items():
        idf[t] = math.log((N - dfi + 0.5) / (dfi + 0.5) + 1.0)

    # query tf
    qtf: Dict[str, int] = {}
    for t in query_tokens:
        qtf[t] = qtf.get(t, 0) + 1

    scores = [0.0] * N
    for i, toks in enumerate(docs_tokens):
        # doc tf
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1

        dl = max(doc_lens[i], 1)
        denom_base = k1 * (1 - b + b * (dl / avgdl))

        s = 0.0
        for t, qt in qtf.items():
            if t not in tf:
                continue
            # 可选：query term 频率权重（这里简单乘 qt）
            f = tf[t]
            term = idf.get(t, 0.0) * ((f * (k1 + 1)) / (f + denom_base)) * qt
            s += term

        scores[i] = s
    return scores


# ========= 主入口：retrieve =========

def retrieve_in_memory(
    short_term: Optional[ShortTermMemoryStore],
    long_term: Optional[LongTermMemoryStore],
    query: str,
    top_k: int = 5,
    pre_k: int = 50,
    min_rule_score: float = 0.05,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    mix_rule_weight: float = 0.35,  # 最终分 = bm25 + mix_rule_weight * rule
) -> List[Dict[str, Any]]:
    """
    三层策略：
      层0：结构化过滤/加权（entity / time）
      层1：规则召回（关键词/代码/简称命中），截断到 pre_k
      层2：对候选集使用 BM25 重排

    返回：按 final_score 降序的候选 material 列表（含 ref_id、分数、desc 等）
    """
    if short_term is None:
        return []

    query = (query or "").strip()
    if not query:
        return []

    # 取 registry（只取有 description 的）
    registry = getattr(short_term, "_registry", None) or {}
    metas_all: List[MaterialMeta] = [m for m in registry.values() if (m.description or "").strip()]
    if not metas_all:
        return []

    # 层0：entity 过滤（同 code 优先，过滤为空则回退全量）
    entity_terms = get_entity_info(long_term=long_term, text=query) or {}
    code = entity_terms.get("code") or ""
    name = entity_terms.get("name") or ""

    metas = metas_all
    if code:
        same_code = [m for m in metas_all if (m.entity or {}).get("code") == code]
        if same_code:
            metas = same_code
    elif name:
        same_name = [m for m in metas_all if (m.entity or {}).get("name") == name]
        if same_name:
            metas = same_name

    time_sig = _extract_query_time_signals(query)
    kw = _DEFAULT_KEYWORDS

    # 层1：规则召回打分并截断
    rule_scored: List[Tuple[float, MaterialMeta]] = []
    for m in metas:
        s = _rule_score(query, m, entity_terms, time_sig, kw)
        if s >= min_rule_score:
            rule_scored.append((s, m))

    # 若规则召回完全为空：保底拿 entity 过滤后的全集（否则会“检索不到任何材料”）
    if not rule_scored:
        rule_scored = [(0.0, m) for m in metas]

    rule_scored.sort(key=lambda x: x[0], reverse=True)
    candidates = rule_scored[: max(pre_k, top_k)]

    # 层2：BM25 仅在候选集内计算
    cand_metas = [m for _, m in candidates]
    docs_tokens = [_tokenize_for_bm25(m.description or "") for m in cand_metas]

    # query tokens：把 query + entity 信息一起纳入（提升稳定性）
    q_extra = []
    if code:
        q_extra.append(code)
    if name:
        q_extra.append(name)

    q_text = query + " " + " ".join(q_extra)
    q_tokens = _tokenize_for_bm25(q_text)

    bm25_scores = _bm25_scores(q_tokens, docs_tokens, k1=bm25_k1, b=bm25_b)

    # 最终融合：BM25 主导，规则分做轻量先验
    final: List[Tuple[float, float, float, MaterialMeta]] = []
    for (rule_s, m), bm25_s in zip(candidates, bm25_scores):
        final_s = float(bm25_s) + mix_rule_weight * float(rule_s)
        final.append((final_s, float(bm25_s), float(rule_s), m))

    final.sort(key=lambda x: x[0], reverse=True)

    # ===================== DEBUG PRINT (before top_k truncation) =====================

    print("\n" + "=" * 90)
    print("[retrieve_in_memory][DEBUG] before top_k truncation")
    print(f"query={query!r} | entity_terms={entity_terms} | total_candidates={len(final)}")
    for idx, (final_s, bm25_s, rule_s, m) in enumerate(final, 1):
        desc_preview = (m.description[:320] + "…") if (m.description and len(m.description) > 320) else (m.description or "")
        row = {
            "rank": idx,
            "ref_id": m.ref_id,
            "final_score": round(final_s, 6),
            "bm25_score": round(bm25_s, 6),
            "rule_score": round(rule_s, 6),
            "m_type": getattr(m.m_type, "value", str(m.m_type)),
            "filename": m.filename,
            "entity": m.entity or {},
            "time": m.time or {},
            "source": m.source or "",
            "description": desc_preview,
        }
        print(row)
    print("=" * 90 + "\n")

    # ===================== DEBUG PRINT END =====================
    final = final[: max(int(top_k), 0)]

    out: List[Dict[str, Any]] = []
    for final_s, bm25_s, rule_s, m in final:
        out.append({
            "ref_id": m.ref_id,
            "final_score": round(final_s, 6),
            "bm25_score": round(bm25_s, 6),
            "rule_score": round(rule_s, 6),
            "m_type": getattr(m.m_type, "value", str(m.m_type)),
            "filename": m.filename,
            "entity": m.entity or {},
            "time": m.time or {},
            "source": m.source or "",
            "description": (m.description[:320] + "…") if (m.description and len(m.description) > 320) else (m.description or ""),
        })
    return out
