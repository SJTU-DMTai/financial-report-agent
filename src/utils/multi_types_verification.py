"""
Architecture:
    Claim Extractor → [FactChecker, NumericChecker, TemporalChecker] → ConsistencyFusion

External API:
    verify_issues = await verify_segment_content(segment, short_term, long_term)
"""

from __future__ import annotations

import asyncio
import json
import re
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from agentscope.message import Msg
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from src.agents.verifier import create_three_verifiers
from src.memory.short_term import ShortTermMemoryStore
from src.memory.long_term import LongTermMemoryStore
from src.tools.material_tools import MaterialTools
from src.utils.cite_id import is_calc_cite_id
from src.utils.instance import create_chat_model, create_agent_formatter
from src.utils.call_with_retry import call_agent_with_retry
from src.prompt import prompt_dict
from pathlib import Path
import config

# ---------------------------------------------------------------------------
# Verifier Trace
# ---------------------------------------------------------------------------
VERIFIER_TRACE_LOCK = asyncio.Lock()
VERIFIER_TRACE_PATH: Optional[Path] = None


def set_verifier_trace_path(path: Optional[Path]) -> None:
    global VERIFIER_TRACE_PATH
    VERIFIER_TRACE_PATH = path
    if VERIFIER_TRACE_PATH is not None:
        VERIFIER_TRACE_PATH.write_text("", encoding="utf-8")


async def append_verifier_trace(
    topic,
    round_idx,
    checked_text,
    verify_feedback=None,
    rewritten_text=None,
    issue_count=None,
    status="issues_found",
):
    if VERIFIER_TRACE_PATH is None:
        return

    sections = [
        "=" * 80,
        f"[Verifier Trace] topic={topic}",
        f"round={round_idx}",
        f"status={status}",
    ]
    if issue_count is not None:
        sections.append(f"issue_count={issue_count}")
    sections.extend(
        [
            "",
            "[Checked Text]",
            checked_text or "",
        ]
    )
    if verify_feedback is not None:
        sections.extend(
            [
                "",
                "[Feedback To Writer]",
                verify_feedback,
            ]
        )
    if rewritten_text is not None:
        sections.extend(
            [
                "",
                "[Writer Rewritten Text]",
                rewritten_text,
            ]
        )
    sections.append("\n")

    async with VERIFIER_TRACE_LOCK:
        with open(VERIFIER_TRACE_PATH, "a", encoding="utf-8") as trace_file:
            trace_file.write("\n".join(sections))


async def append_verifier_trace_log(title: str, message: str, payload: Optional[str] = None):
    if VERIFIER_TRACE_PATH is None:
        return

    sections = [
        "-" * 80,
        f"[{title}]",
        message,
    ]
    if payload is not None:
        sections.extend(
            [
                "",
                payload,
            ]
        )
    sections.append("\n")

    async with VERIFIER_TRACE_LOCK:
        with open(VERIFIER_TRACE_PATH, "a", encoding="utf-8") as trace_file:
            trace_file.write("\n".join(sections))

# ---------------------------------------------------------------------------
# 安全 JSON 解析（统一使用）
# ---------------------------------------------------------------------------
def _extract_balanced_json_at(text: str, start: int, open_ch: str, close_ch: str) -> Optional[str]:
    depth = 1
    in_string = False
    escaped = False
    for j in range(start + 1, len(text)):
        current = text[j]
        if escaped:
            escaped = False
            continue
        if current == "\\":
            escaped = True
            continue
        if current == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if current == open_ch:
            depth += 1
        elif current == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    return None


def _iter_balanced_json_candidates(text: str):
    pairs = {"{": "}", "[": "]"}
    for index, ch in enumerate(text):
        close_ch = pairs.get(ch)
        if close_ch is None:
            continue
        extracted = _extract_balanced_json_at(text, index, ch, close_ch)
        if extracted:
            yield extracted


def _safe_parse_json(text: str) -> Any:
    """Parse a JSON object/array from raw LLM text without truncating arrays."""
    text = str(text or "").strip()

    for fence_match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        fenced_text = fence_match.group(1).strip()
        try:
            return json.loads(fenced_text)
        except json.JSONDecodeError:
            continue

    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for extracted in _iter_balanced_json_candidates(text):
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            continue

    raise ValueError("Failed to parse JSON from response")

def fix_encoding(s: str) -> str:
    try:
        return s.encode('latin-1').decode('utf-8')
    except Exception:
        return s


def _extract_text_response(response_msg: Any) -> str:
    if response_msg is None:
        raise ValueError("Agent returned None response")
    if not hasattr(response_msg, "get_text_content"):
        raise TypeError(f"Agent returned unsupported response type: {type(response_msg).__name__}")

    text = response_msg.get_text_content()
    if text is None or not str(text).strip():
        raise ValueError("Agent returned empty text content")
    return str(text)


def _strip_chart_references_for_claim_extract(text: str) -> str:
    if not text:
        return ""

    cleaned = re.sub(
        r'!\[[^\]]*]\(chart:[a-zA-Z0-9_\-]+\)',
        "",
        text,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalize_claim_extract_segments(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        segments = data.get("segments", [])
    elif isinstance(data, list):
        segments = data
    else:
        return []

    if not isinstance(segments, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for segment in segments:
        if isinstance(segment, dict):
            normalized.append(segment)
    return normalized


# ---------------------------------------------------------------------------
# §2  Claim Extractor - 完全由 LLM 驱动的原子声明提取器
# ---------------------------------------------------------------------------
class ClaimType(str, Enum):
    FACTUAL = "factual"
    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    FACTUAL_NUMERIC = "factual_numeric"
    NUMERIC_TEMPORAL = "numeric_temporal"
    COMPOSITE = "composite"


@dataclass
class Claim:
    claim_id: str
    claim_type: ClaimType
    original_text: str
    normalized_text: str
    slots: Dict[str, Any] = field(default_factory=dict)
    source_span: Tuple[int, int] = (0, 0)
    cite_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """用于批量验证的序列化方法"""
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type.value,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "slots": self.slots,
            "cite_ids": self.cite_ids,
        }


FINANCIAL_METRIC_KEYWORDS = ("收入", "营收", "营业收入", "利润", "净利润", "归母净利", "扣非", "毛利", "毛利率", "净利率", "费用率", "ROE", "现金流", "资产负债率")
GROWTH_COMPARISON_KEYWORDS = ("同比", "环比", "增长", "提升", "下降", "下滑", "改善", "恶化", "高于", "低于", "领先", "落后", "超过", "不及")
BUSINESS_EVENT_KEYWORDS = ("订单", "定点", "客户", "量产", "投产", "产能", "扩产", "并表", "收购", "出售", "剥离", "重组", "战略合作", "新项目", "市占率", "市场份额")
FORECAST_VALUATION_KEYWORDS = ("预测", "预计", "有望", "目标价", "估值", "PE", "PB", "PEG", "EPS", "Forward", "TTM", "市盈率", "市净率", "上行空间")
ASSERTIVE_KEYWORDS = ("首次", "唯一", "第一", "最大", "最小", "领先", "龙头", "核心", "显著", "大幅", "突破", "创历史")
CAUSAL_CONCLUSION_KEYWORDS = ("主要由于", "主要系", "受益于", "驱动", "源于", "带动", "归因于", "得益于", "导致")
RISK_UNITS = ("亿元", "万元", "%", "pct", "倍", "元", "万台", "万辆")

def keyword_score(text: str, keywords: Tuple[str, ...], weight: int, cap: int) -> int:
    hits = sum(1 for keyword in keywords if keyword in text)
    return min(hits * weight, cap)


def score_claim_risk(claim: Claim) -> int:
    text = f"{claim.original_text} {claim.normalized_text}"
    number_count = len(re.findall(r"\d+(?:\.\d+)?", text))
    financial_score = keyword_score(text, FINANCIAL_METRIC_KEYWORDS, 10, 35)
    growth_score = keyword_score(text, GROWTH_COMPARISON_KEYWORDS, 8, 28)
    business_score = keyword_score(text, BUSINESS_EVENT_KEYWORDS, 9, 32)
    forecast_score = keyword_score(text, FORECAST_VALUATION_KEYWORDS, 11, 38)
    unit_score = keyword_score(text, RISK_UNITS, 4, 16)
    score = 0

    if number_count:
        score += 28 + min(number_count * 5, 25)

    score += financial_score
    score += growth_score
    score += business_score
    score += forecast_score
    score += keyword_score(text, ASSERTIVE_KEYWORDS, 7, 21)
    score += keyword_score(text, CAUSAL_CONCLUSION_KEYWORDS, 6, 18)
    score += unit_score

    if re.search(r"(20\d{2}|Q[1-4]|[一二三四]季度|H[12]|上半年|下半年|前三季度)", text):
        score += 10
    if claim.slots.get("numeric"):
        score += 8
    if claim.slots.get("temporal") and (number_count or business_score):
        score += 5
    if claim.cite_ids:
        score += 4
    if not claim.cite_ids and number_count and (
        financial_score or growth_score or business_score or forecast_score or unit_score
    ):
        score += 8

    if not number_count and len(text.strip()) < 15:
        score -= 12

    return max(score, 0)


def select_high_risk_claims(claims: List[Claim], max_claims: int) -> List[Claim]:
    if max_claims <= 0 or len(claims) <= max_claims:
        return list(claims)

    ranked = sorted(
        enumerate(claims),
        key=lambda item: (-score_claim_risk(item[1]), item[0]),
    )
    selected = sorted(ranked[:max_claims], key=lambda item: item[0])
    return [claim for _, claim in selected]


class ClaimExtractor:
    """
    从文本片段中提取原子声明。
    """
    CITE_RE = re.compile(r"\[\^cite_id:([A-Za-z0-9_\-]+)(?:\|[^\]]*)?\]")

    def __init__(self, agent):
        self.agent = agent

    # ---------- 主提取方法 ----------
    async def extract(self, text: str) -> List[Claim]:
        if not text.strip():
            return []

        if hasattr(self.agent, "memory") and self.agent.memory is not None:
            await self.agent.memory.clear()

        msg = Msg(
            name="extractor",
            role="user",
            content=text
        )

        last_exc: Exception | None = None
        raw = ""

        try:
            response_msg = await call_agent_with_retry(self.agent, msg)
            raw = _extract_text_response(response_msg)
        except Exception as e:
            last_exc = e
            await append_verifier_trace_log(
                "ClaimExtractor",
                f"[ClaimExtractor] invalid text response after call_agent_with_retry: "
                f"{type(e).__name__}: {e}"
            )
            if hasattr(self.agent, "memory") and self.agent.memory is not None:
                await self.agent.memory.clear()

        if last_exc is not None and not raw:
            raise last_exc

        # print("\n====== RAW LLM OUTPUT ======")
        # print(type(raw))
        # print("====== END ======\n")

        # 安全解析 JSON
        try:
            data = _safe_parse_json(raw)
        except Exception as e:
            await append_verifier_trace_log(
                "ClaimExtractor",
                f"[ClaimExtractor] Failed to parse JSON: {e}",
                payload=f"Raw text:\n{raw}",
            )
            return []

        segments = _normalize_claim_extract_segments(data)
        if not segments:
            await append_verifier_trace_log(
                "ClaimExtractor",
                f"[ClaimExtractor] No valid segments returned from LLM; parsed_type={type(data).__name__}",
                payload=f"Raw text:\n{raw}",
            )
            return []

        claims: List[Claim] = []
        for seg in segments:
            seg_cite_ids = seg.get("cite_ids", [])
            segment_claims = seg.get("claims", [])
            if not isinstance(segment_claims, list):
                continue
            for item in segment_claims:
                if not isinstance(item, dict):
                    continue
                claim_type_str = item.get("claim_type", "factual")
                try:
                    claim_type = ClaimType(claim_type_str)
                except ValueError:
                    claim_type = ClaimType.FACTUAL

                claim = Claim(
                    claim_id=f"c{len(claims)}",
                    claim_type=claim_type,
                    original_text=item.get("original_text", ""),
                    normalized_text=item.get("normalized_text", ""),
                    slots=item.get("slots", {}),
                    source_span=(0, 0),  # 可选：后面再做
                    cite_ids=item.get("cite_ids", seg_cite_ids)
                )
                claims.append(claim)

        return self._post_process(claims)

    # ---------- 后处理：去重 + ID 重编号 ----------
    def _post_process(self, claims: List[Claim]) -> List[Claim]:
        filtered = []
        seen = set()
        for c in claims:
            key = (
                c.claim_type,
                json.dumps(c.slots, sort_keys=True, ensure_ascii=False),
                c.original_text[:50]
            )
            if key in seen:
                continue
            seen.add(key)

            if len(c.original_text.strip()) < 2:
                continue

            c.claim_id = f"c{len(filtered)}"
            filtered.append(c)

        return filtered

# ---------------------------------------------------------------------------
# §3  Data Structure
# ---------------------------------------------------------------------------

@dataclass
class EvidenceSpan:
    cite_id: str
    text: str
    score: Optional[float] = None   # 相关性/置信度

@dataclass
class ClaimIssue:
    type: str
    description: str
    severity: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    suggestion: str = ""

@dataclass
class ClaimResult:
    claim_id: str
    issues: List[ClaimIssue]


SEGMENT_CITE_RE = re.compile(r"\[\^\s*(?:(?:cite_id)\s*[:=]\s*)?([^\]\|\s]+)(?:\|[^\]]*)?\]", re.IGNORECASE)
MAX_MATERIAL_PREVIEW_CHARS = 1600
MAX_CALC_CODE_CHARS = 4000
GLOBAL_MATERIAL_DAG_CACHE: Dict[str, List[ClaimIssue]] = {}
GLOBAL_MATERIAL_DAG_CACHE_LOCKS: Dict[str, asyncio.Lock] = {}
GLOBAL_MATERIAL_DAG_CACHE_LOCKS_LOCK = asyncio.Lock()


async def get_material_dag_cache_lock(cite_id: str) -> asyncio.Lock:
    async with GLOBAL_MATERIAL_DAG_CACHE_LOCKS_LOCK:
        lock = GLOBAL_MATERIAL_DAG_CACHE_LOCKS.get(cite_id)
        if lock is None:
            lock = asyncio.Lock()
            GLOBAL_MATERIAL_DAG_CACHE_LOCKS[cite_id] = lock
        return lock


def extract_segment_cite_ids(text: str) -> List[str]:
    cite_ids: List[str] = []
    for match in SEGMENT_CITE_RE.finditer(text or ""):
        cite_id = match.group(1).strip()
        if cite_id and cite_id not in cite_ids:
            cite_ids.append(cite_id)
    return cite_ids


def extract_calc_root_cite_ids(text: str) -> List[str]:
    return [cite_id for cite_id in extract_segment_cite_ids(text) if is_calc_cite_id(cite_id)]


def truncate_for_verifier(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[内容过长，已截断]"


def first_calc_record(content: Any) -> Dict[str, Any] | None:
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return dict(content[0])
    if isinstance(content, dict):
        return dict(content)
    return None


def calc_content_payload(content: Any) -> Dict[str, Any] | None:
    record = first_calc_record(content)
    if record is None:
        return None
    payload = {
        "description": record.get("description", ""),
        "tool": record.get("tool", ""),
        "sub_type": record.get("sub_type", ""),
        "parameters": record.get("parameters", {}),
        "param_sources": record.get("param_sources", {}),
        "result_type": record.get("result_type", ""),
        "result": record.get("result"),
        "code": truncate_for_verifier(record.get("code", ""), MAX_CALC_CODE_CHARS),
    }
    return payload


def material_meta_payload(short_term: ShortTermMemoryStore, cite_id: str) -> Dict[str, Any] | None:
    meta = short_term.get_material_meta(cite_id)
    if meta is None:
        return None
    return {
        "description": meta.description,
        "source": meta.source,
        "filename": meta.filename,
        "m_type": getattr(meta.m_type, "value", str(meta.m_type)),
        "entity": meta.entity,
        "time": meta.time,
        "upstream_cite_ids": list(getattr(meta, "upstream_cite_ids", []) or []),
    }


def material_node_payload(short_term: ShortTermMemoryStore, cite_id: str) -> Dict[str, Any] | None:
    meta_payload = material_meta_payload(short_term, cite_id)
    if meta_payload is None:
        return None

    node = {
        "cite_id": cite_id,
        "is_calculation": is_calc_cite_id(cite_id),
        "meta": meta_payload,
    }
    content = short_term.load_material(cite_id)
    if is_calc_cite_id(cite_id):
        node["content"] = calc_content_payload(content)
    else:
        node["preview"] = short_term.load_material_preview(
            cite_id,
            max_chars=MAX_MATERIAL_PREVIEW_CHARS,
        )
    return node


def collect_material_graph(
    short_term: ShortTermMemoryStore,
    root_cite_ids: List[str],
) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []
    missing: List[str] = []
    cycles: List[List[str]] = []
    visited: set[str] = set()
    stack: List[Tuple[str, List[str]]] = [(cite_id, []) for cite_id in root_cite_ids]

    while stack:
        cite_id, path = stack.pop()
        if cite_id in path:
            cycles.append(path + [cite_id])
            continue
        if cite_id in visited:
            continue
        node = material_node_payload(short_term, cite_id)
        if node is None:
            if cite_id not in missing:
                missing.append(cite_id)
            continue
        visited.add(cite_id)
        nodes.append(node)
        upstream_ids = node["meta"].get("upstream_cite_ids", []) or []
        next_path = path + [cite_id]
        for upstream_cite_id in upstream_ids:
            upstream = str(upstream_cite_id).strip()
            if not upstream:
                continue
            edges.append({"from": cite_id, "to": upstream})
            stack.append((upstream, next_path))

    return {
        "nodes": nodes,
        "edges": edges,
        "missing": missing,
        "cycles": cycles,
    }


def calc_node_missing_param_sources(node: Dict[str, Any]) -> bool:
    content = node.get("content")
    if not isinstance(content, dict):
        return True
    param_sources = content.get("param_sources")
    return not isinstance(param_sources, dict) or not param_sources


def calc_node_parameters_without_sources(node: Dict[str, Any]) -> List[str]:
    content = node.get("content")
    if not isinstance(content, dict):
        return []
    parameters = content.get("parameters")
    param_sources = content.get("param_sources")
    if not isinstance(parameters, dict) or not isinstance(param_sources, dict):
        return []
    return [param_name for param_name in parameters if param_name not in param_sources]


def material_issue(
    issue_type: str,
    severity: str,
    description: str,
    suggestion: str,
    evidence: List[Dict[str, Any]] | None = None,
) -> ClaimIssue:
    return ClaimIssue(
        type=issue_type,
        description=f"[material_dag] {description}",
        severity=severity,
        evidence=evidence or [],
        suggestion=suggestion,
    )


def local_material_graph_issues(graph: Dict[str, Any]) -> List[ClaimIssue]:
    issues: List[ClaimIssue] = []
    for cite_id in graph.get("missing", []) or []:
        issues.append(material_issue(
            "upstream_missing",
            "critical",
            f"material 不存在或无法读取：{cite_id}",
            f"删除或改写依赖 material {cite_id} 的计算表述；如果必须保留该计算，需要重新生成有效 material 后再引用。",
            evidence=[{"cite_id": cite_id, "text": "material missing"}],
        ))
    for cycle in graph.get("cycles", []) or []:
        cycle_text = " -> ".join(cycle)
        issues.append(material_issue(
            "material_dag_cycle",
            "critical",
            f"material DAG 存在循环依赖：{cycle_text}",
            "删除或改写涉及该循环依赖的计算表述；如果必须保留该计算，需要重新生成无循环依赖的计算 material。",
            evidence=[{"text": cycle_text}],
        ))
    for node in graph.get("nodes", []) or []:
        if not node.get("is_calculation"):
            continue
        cite_id = str(node.get("cite_id") or "")
        if calc_node_missing_param_sources(node):
            issues.append(material_issue(
                "provenance_incomplete",
                "major",
                f"计算 material {cite_id} 缺少 param_sources，无法沿参数溯源核验。",
                "删除或改写正文中对该计算结果的引用；如果必须保留该计算，需要重新生成带 param_sources 的计算 material。",
                evidence=[{"cite_id": cite_id, "text": "missing param_sources"}],
            ))
            continue
        missing_params = calc_node_parameters_without_sources(node)
        for param_name in missing_params:
            issues.append(material_issue(
                "provenance_incomplete",
                "major",
                f"计算 material {cite_id} 的参数 {param_name} 缺少来源。",
                f"删除或改写正文中对该计算结果的引用；如果必须保留该计算，需要为参数 {param_name} 补充 param_sources 后重新计算。",
                evidence=[{"cite_id": cite_id, "text": f"parameter without source: {param_name}"}],
            ))
    return issues


def normalize_material_verifier_severity(value: Any) -> str:
    severity = str(value or "").lower()
    if severity in {"critical", "high"}:
        return "critical"
    if severity in {"major", "medium"}:
        return "major"
    if severity in {"minor", "low"}:
        return "minor"
    return "major"


def material_verifier_issues_from_payload(payload: Any) -> List[ClaimIssue]:
    if isinstance(payload, dict):
        raw_issues = payload.get("issues", [])
    elif isinstance(payload, list):
        raw_issues = payload
    else:
        raw_issues = []

    issues: List[ClaimIssue] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            continue
        calc_cite_id = str(item.get("calc_cite_id") or item.get("root_cite_id") or "").strip()
        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        description = str(item.get("description") or "").strip()
        if calc_cite_id and calc_cite_id not in description:
            description = f"{calc_cite_id}: {description}"
        issues.append(material_issue(
            str(item.get("type") or "material_dag_issue"),
            normalize_material_verifier_severity(item.get("severity")),
            description or "计算材料溯源存在问题。",
            str(item.get("suggestion") or "").strip(),
            evidence=evidence,
        ))
    return issues


# ---------------------------------------------------------------------------
# §4  Verifier Router + Issue Fusion
# ---------------------------------------------------------------------------

class VerifierRouter:
    def __init__(self, verifiers: Dict[str, any]):
        """verifiers: {"fact": agent, "numeric": agent, "temporal": agent}"""
        self.verifiers = verifiers

    async def verify_claims(
        self,
        claims: List[Claim],
        context: Optional[Dict[str, str]] = None,
    ) -> List[ClaimResult]:
        """批量验证所有 claims，每个类型只调用一次 agent。"""
        # 按类型分组
        claims_by_type: Dict[ClaimType, List[Claim]] = {}
        for claim in claims:
            claims_by_type.setdefault(claim.claim_type, []).append(claim)

        # 需要验证的类型映射（与 agent 类型对应）
        type_to_verifier = {
            ClaimType.FACTUAL: "fact",
            ClaimType.NUMERIC: "numeric",
            ClaimType.TEMPORAL: "temporal",
            ClaimType.FACTUAL_NUMERIC: ["fact", "numeric"],  # 需要两个 agent
            ClaimType.NUMERIC_TEMPORAL: ["numeric", "temporal"],
            ClaimType.COMPOSITE: ["fact", "numeric", "temporal"],
        }

        # 收集所有任务（每个 verifier 类型对应一个批量任务）
        tasks = []
        verifier_to_claims: Dict[str, List[Claim]] = {}

        for ct, clist in claims_by_type.items():
            needed = type_to_verifier.get(ct, [])
            if isinstance(needed, str):
                needed = [needed]
            for vname in needed:
                verifier_to_claims.setdefault(vname, []).extend(clist)

        # 去重（一个 claim 可能被多个 verifier 处理，如 factual_numeric）
        # 但我们仍要按 verifier 分别批量调用
        for vname, vclaims in verifier_to_claims.items():
            tasks.append(self._run_batch(vname, vclaims, context or {}))

        # 并行执行所有批量任务
        results = await asyncio.gather(*tasks)

        # 合并结果：将 issue 分配回各个 claim
        claim_issues: Dict[str, List[ClaimIssue]] = {c.claim_id: [] for c in claims}
        for result in results:  # result 是 {claim_id: [issues]}
            for cid, issues in result.items():
                claim_issues[cid].extend(issues)

        # 构建 ClaimResult 列表
        claim_results = []
        for c in claims:
            # 去重合并（同一 claim 可能有来自多个 verifier 的相同类型 issue）
            merged = self._merge_issues(claim_issues[c.claim_id])
            claim_results.append(ClaimResult(claim_id=c.claim_id, issues=merged))

        return claim_results

    #-----------------------------------------------------------------------
    # 批量执行单个 verifier
    # -----------------------------------------------------------------------
    async def _run_batch(
        self,
        verifier_name: str,
        claims: List[Claim],
        context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[ClaimIssue]]:
        """一次性处理多个 claims，返回 {claim_id: [issues]}"""
        agent = self.verifiers[verifier_name]

        # 每次调用前清空 agent 的记忆，防止不同批次之间上下文污染。
        if hasattr(agent, "memory") and agent.memory is not None:
            await agent.memory.clear()

        payload = self._batch_payload_for_verifier(verifier_name, claims, context)

        try:
            msg = Msg(
                role="user",
                content=json.dumps(payload, ensure_ascii=False),
                name="Verifier"
            )

            last_exc: Exception | None = None
            text = ""
            try:
                response_msg = await call_agent_with_retry(agent, msg)
                text = _extract_text_response(response_msg)
            except Exception as e:
                last_exc = e

                await append_verifier_trace_log(
                    f"{verifier_name.upper()} verifier",
                    f"[{verifier_name} verifier] invalid text response after "
                    f"call_agent_with_retry: {type(e).__name__}: {e}"
                )
                if hasattr(agent, "memory") and agent.memory is not None:
                    await agent.memory.clear()

            if last_exc is not None and not text:
                raise last_exc

            # print(f"\n====== [{verifier_name.upper()} BATCH RAW OUTPUT] ======")
            # if text is None:
            #     print("TEXT: <None>")
            # else:
            #     print("TEXT:", text[:500] + "..." if len(text) > 500 else text)
            # print("====================================\n")

            # 安全解析
            data = _safe_parse_json(text)

            # 期望 data 是一个 dict，包含 issues 列表，每个 issue 有 claim_id
            if isinstance(data, list):
                issues_list = data
            elif isinstance(data, dict):
                issues_list = data.get("issues", [])
            else:
                issues_list = []

            # 按 claim_id 分组
            result: Dict[str, List[ClaimIssue]] = {c.claim_id: [] for c in claims}
            skipped_items = []
            for item in issues_list:
                if not isinstance(item, dict):
                    skipped_items.append(item)
                    continue
                cid = item.get("claim_id")
                if not cid or cid not in result:
                    skipped_items.append(item)
                    continue

                result[cid].append(ClaimIssue(
                    type=item.get("type", "unknown"),
                    description=f"[{verifier_name}] {item.get('description', '')}",
                    severity=self._normalize_severity(item.get("severity")),
                    suggestion=self._normalize_suggestion(item.get("suggestion")),
                    evidence=self._normalize_evidence(item.get("evidence")),
                ))
            if skipped_items:
                await append_verifier_trace_log(
                    f"{verifier_name.upper()} verifier",
                    f"[{verifier_name} verifier] dropped {len(skipped_items)} "
                    f"issue items without valid claim_id",
                    payload=json.dumps(skipped_items, ensure_ascii=False, indent=2),
                )
            return result

        except Exception as e:
            await append_verifier_trace_log(
                f"{verifier_name.upper()} BATCH ERROR",
                f"[{verifier_name.upper()} BATCH ERROR] {type(e).__name__}: {e}",
                payload=traceback.format_exc(),
            )
            return {c.claim_id: [] for c in claims}

    # -----------------------------------------------------------------------
    # Issue 合并（同一 claim 内去重）
    # -----------------------------------------------------------------------
    def _merge_issues(self, issues: List[ClaimIssue]) -> List[ClaimIssue]:
        merged = {}
        for iss in issues:
            key = (iss.type, iss.description)
            if key not in merged:
                merged[key] = iss
            else:
                # 升级 severity
                if self._severity_priority(iss.severity) > self._severity_priority(merged[key].severity):
                    merged[key].severity = iss.severity
                # 补充 suggestion
                if not merged[key].suggestion and iss.suggestion:
                    merged[key].suggestion = iss.suggestion
        return list(merged.values())

    # -----------------------------------------------------------------------
    # 工具函数
    # -----------------------------------------------------------------------
    def _claim_payload_for_verifier(self, claim: Claim, verifier_name: str) -> Dict[str, Any]:
        payload = {
            "claim_id": claim.claim_id,
            "original_text": claim.original_text,
            "normalized_text": claim.normalized_text,
            "cite_ids": claim.cite_ids,
        }
        if verifier_name == "fact":
            payload["factual"] = claim.slots.get("factual", {})
        elif verifier_name == "numeric":
            payload["numeric"] = claim.slots.get("numeric", [])
        elif verifier_name == "temporal":
            payload["temporal"] = claim.slots.get("temporal", [])
        return payload

    def _batch_payload_for_verifier(
        self,
        verifier_name: str,
        claims: List[Claim],
        context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "claims": [self._claim_payload_for_verifier(c, verifier_name) for c in claims]
        }
        if context:
            cleaned_context = {
                key: value
                for key, value in context.items()
                if value
            }
            if cleaned_context:
                payload["context"] = cleaned_context
        return payload

    def _normalize_suggestion(self, suggestion: Any) -> str:
        if suggestion is None:
            return ""
        if isinstance(suggestion, str):
            return suggestion.strip()
        if isinstance(suggestion, (dict, list)):
            return json.dumps(suggestion, ensure_ascii=False)
        return str(suggestion).strip()

    def _normalize_evidence(self, evidence: Any) -> List[Dict[str, Any]]:
        if evidence is None:
            return []
        if isinstance(evidence, dict):
            evidence_items = [evidence]
        elif isinstance(evidence, list):
            evidence_items = evidence
        else:
            evidence_items = [{"text": str(evidence)}]

        normalized = []
        for item in evidence_items:
            if isinstance(item, EvidenceSpan):
                normalized.append({
                    "cite_id": item.cite_id,
                    "text": item.text,
                    **({"score": item.score} if item.score is not None else {}),
                })
                continue
            if isinstance(item, dict):
                normalized_item = dict(item)
                if "cite_id" in normalized_item:
                    normalized_item["cite_id"] = str(normalized_item["cite_id"])
                if "text" in normalized_item:
                    normalized_item["text"] = str(normalized_item["text"])
                normalized.append(normalized_item)
                continue
            normalized.append({"text": str(item)})
        return normalized

    def _normalize_severity(self, s: str) -> str:
        if not s:
            return "minor"
        s = str(s).lower()
        if s in ["critical", "high"]:
            return "critical"
        if s in ["major", "medium"]:
            return "major"
        if s in ["minor", "low"]:
            return "minor"
        if s in ["info"]:
            return "info"
        return "minor"

    def _severity_priority(self, s: str) -> int:
        mapping = {"critical": 3, "major": 2, "minor": 1, "info": 0}
        return mapping.get(s, 1)



# ---------------------------------------------------------------------------
# §5  SegmentVerifier — 总控制器
# ---------------------------------------------------------------------------

class MaterialDAGVerifier:
    def __init__(
        self,
        short_term: ShortTermMemoryStore,
        long_term: LongTermMemoryStore,
        model,
        formatter,
    ) -> None:
        toolkit = Toolkit()
        material_tools = MaterialTools(short_term=short_term, long_term=long_term)
        toolkit.register_tool_function(material_tools.read_material)
        self.short_term = short_term
        self.agent = ReActAgent(
            name="MaterialDAGVerifier",
            sys_prompt=prompt_dict["material_dag_verifier_prompt"],
            model=model,
            memory=InMemoryMemory(),
            formatter=formatter,
            toolkit=toolkit,
            parallel_tool_calls=False,
            max_iters=10,
        )

    async def verify(
        self,
        segment: str,
        company_name: Optional[str] = None,
        report_date: Optional[str] = None,
        root_calc_cite_ids: Optional[List[str]] = None,
    ) -> List[ClaimIssue]:
        root_ids = list(root_calc_cite_ids or extract_calc_root_cite_ids(segment))
        if not root_ids:
            return []

        graph = collect_material_graph(self.short_term, root_ids)
        local_issues = local_material_graph_issues(graph)
        payload = {
            "context": {
                "company_name": company_name or "",
                "report_date": report_date or "",
            },
            "segment_text": segment,
            "root_calc_cite_ids": root_ids,
            "material_graph": graph,
        }

        await append_verifier_trace_log(
            "MaterialDAGVerifier",
            "material DAG collected for calculation citations",
            payload=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        )
        if hasattr(self.agent, "memory") and self.agent.memory is not None:
            await self.agent.memory.clear()

        try:
            msg = Msg(
                role="user",
                content=json.dumps(payload, ensure_ascii=False, default=str),
                name="MaterialDAGVerifier",
            )
            response_msg = await call_agent_with_retry(self.agent, msg)
            response_text = _extract_text_response(response_msg)
            parsed = _safe_parse_json(response_text)
            llm_issues = material_verifier_issues_from_payload(parsed)
            return local_issues + llm_issues
        except Exception as exc:
            await append_verifier_trace_log(
                "MaterialDAGVerifier",
                f"material DAG verification failed: {type(exc).__name__}: {exc}",
                payload=traceback.format_exc(),
            )
            return local_issues


class SegmentVerifier:
    def __init__(self, short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore):
        self.model = create_chat_model(reasoning=False)
        self.formatter = create_agent_formatter()
        self.max_claims_per_segment = config.Config().get_max_claims_per_segment()

        # ---------- 创建 ClaimExtractor  ----------
        self.extractor_agent = ReActAgent(
            name="ClaimExtractor",
            sys_prompt=prompt_dict["claim_extract_sys_prompt"],
            model=self.model,
            memory=InMemoryMemory(),
            formatter=self.formatter,
            toolkit=Toolkit(),
            max_iters=1,
        )
        self.extractor = ClaimExtractor(agent=self.extractor_agent)

        # ---------- 创建三个 Verifier Agents ----------
        self.verifiers = create_three_verifiers(
            model=self.model,
            formatter=self.formatter,
            short_term=short_term,
            long_term=long_term
        )

        self.router = VerifierRouter(self.verifiers)
        self.material_dag_verifier = MaterialDAGVerifier(
            short_term=short_term,
            long_term=long_term,
            model=self.model,
            formatter=self.formatter,
        )
        self.material_dag_cache = GLOBAL_MATERIAL_DAG_CACHE

    async def verify_material_dag(
        self,
        segment: str,
        company_name: Optional[str] = None,
        report_date: Optional[str] = None,
    ) -> List[ClaimIssue]:
        root_calc_cite_ids = extract_calc_root_cite_ids(segment)
        if not root_calc_cite_ids:
            return []

        cached_issues: List[ClaimIssue] = []
        for cite_id in root_calc_cite_ids:
            if cite_id in self.material_dag_cache:
                cached_issues.extend(self.material_dag_cache[cite_id])
                continue
            async with await get_material_dag_cache_lock(cite_id):
                if cite_id not in self.material_dag_cache:
                    self.material_dag_cache[cite_id] = await self.material_dag_verifier.verify(
                        segment,
                        company_name=company_name,
                        report_date=report_date,
                        root_calc_cite_ids=[cite_id],
                    )
                cached_issues.extend(self.material_dag_cache[cite_id])

        return cached_issues

    async def verify(
        self,
        segment: str,
        company_name: Optional[str] = None,
        report_date: Optional[str] = None,
    ) -> List[ClaimIssue]:
        """主入口：segment → claims → batch verification → issues"""
        material_issues = await self.verify_material_dag(
            segment,
            company_name=company_name,
            report_date=report_date,
        )
        segment_for_extract = _strip_chart_references_for_claim_extract(segment)
        if company_name or report_date:
            context_lines = ["# 任务上下文"]
            if company_name:
                context_lines.append(f"公司名称：{company_name}")
            if report_date:
                context_lines.append(f"研报日期：{report_date}")
            context_lines.append("上下文信息可用于解析正文中的公司简称、公司/本公司等指代，以及相对时间。")
            segment_for_extract = "\n".join(context_lines) + "\n\n# 待抽取正文\n" + segment_for_extract
        claims = await self.extractor.extract(segment_for_extract)
        if not claims:
            return material_issues
        selected_claims = select_high_risk_claims(claims, self.max_claims_per_segment)
        await append_verifier_trace_log(
            "SegmentVerifier",
            (
                "claims selected for verification: "
                f"extracted={len(claims)}, selected={len(selected_claims)}, "
                f"max_claims_per_segment={self.max_claims_per_segment}"
            ),
        )
        await append_verifier_trace_log(
            "SegmentVerifier",
            "======== claims抽取已完成 ========",
            payload=json.dumps(
                [
                    {
                        **claim.to_dict(),
                        "risk_score": score_claim_risk(claim),
                    }
                    for claim in selected_claims
                ],
                ensure_ascii=False,
                indent=2,
            ),
        )

        claim_results = await self.router.verify_claims(
            selected_claims,
            context={
                "company_name": company_name or "",
                "report_date": report_date or "",
            },
        )

        # 展平所有 issues
        flat = []
        flat.extend(material_issues)
        for cr in claim_results:
            flat.extend(cr.issues)
        return flat
