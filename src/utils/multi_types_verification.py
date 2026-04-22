"""
Triple-Check Verifier — 论文级三路交叉验证系统
=================================================
Architecture:
    Claim Extractor → [FactVerifier, NumericVerifier, TemporalVerifier] → Issue Fusion

External API:
    verify_issues = await verify(segment, short_term, long_term)
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

from src.agents.verifier import create_three_verifiers
from src.memory.short_term import ShortTermMemoryStore, MaterialType
from src.memory.long_term import LongTermMemoryStore
from src.utils.instance import create_chat_model, create_agent_formatter
from src.utils.call_with_retry import call_agent_with_retry, call_chatbot_with_retry
from src.prompt import prompt_dict
from src.utils.multi_source_verification import material_source_key
from pathlib import Path

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
    score: Optional[int] = None,           
    star_rating: Optional[int] = None,     
    passed: Optional[bool] = None,         
    priority_claims_count: Optional[int] = None,  
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
    if score is not None:
        sections.append(f"segment_score={score}")
    if star_rating is not None:
        sections.append(f"star_rating={star_rating}")
    if passed is not None:
        sections.append(f"passed={passed}")
    if priority_claims_count is not None:
        sections.append(f"priority_claims_count={priority_claims_count}")
    
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
def _safe_parse_json(text: str) -> Dict[str, Any]:
    """移除 markdown 代码块并提取 JSON 对象/数组，然后解析。"""
    # 移除可能的 markdown 代码块
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # 提取第一个 JSON 对象或数组
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if not match:
        raise ValueError("No JSON structure found in response")
    json_str = match.group(1)

    return json.loads(json_str)

class ClaimExtractionError(Exception):
    """Claim 提取失败的异常，携带原始响应文本。"""
    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


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


# claim_type → verifier name 列表的全局映射（避免多处硬编码）
CLAIM_TYPE_TO_VERIFIERS = {
    ClaimType.FACTUAL: ["fact"],
    ClaimType.NUMERIC: ["numeric"],
    ClaimType.TEMPORAL: ["temporal"],
    ClaimType.FACTUAL_NUMERIC: ["fact", "numeric"],
    ClaimType.NUMERIC_TEMPORAL: ["numeric", "temporal"],
    ClaimType.COMPOSITE: ["fact", "numeric", "temporal"],
}


@dataclass
class Claim:
    claim_id: str
    claim_type: ClaimType
    original_text: str
    normalized_text: str
    slots: Dict[str, Any] = field(default_factory=dict)
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

class ClaimExtractor:
    """
    从文本片段中提取原子声明。
    直接使用 model + formatter 进行纯文本生成，避免 ReActAgent 的开销。
    """
    CITE_RE = re.compile(r"\[\^cite_id:([A-Za-z0-9_\-]+)(?:\|[^\]]*)?\]")

    def __init__(self, model, formatter, sys_prompt: str):
        """
        :param model: ChatModel 实例
        :param formatter: Formatter 实例
        :param sys_prompt: 系统提示词
        """
        self.model = model
        self.formatter = formatter
        self.sys_prompt = sys_prompt

    # ---------- 主提取方法 ----------
    async def extract(self, text: str) -> List[Claim]:
        if not text.strip():
            return []

        last_error = None
        raw = ""

        for attempt in range(2):
            try:
                raw = await call_chatbot_with_retry(
                    self.model, self.formatter,
                    self.sys_prompt, text,
                    max_retries=3,
                )
            except Exception as e:
                await append_verifier_trace_log(
                    "ClaimExtractor",
                    f"[ClaimExtractor] LLM call failed: {type(e).__name__}: {e}"
                )
                raise ClaimExtractionError(f"LLM call failed: {e}", raw="")

            try:
                data = _safe_parse_json(raw)
                break
            except Exception as e:
                last_error = e
                await append_verifier_trace_log(
                    "ClaimExtractor",
                    f"[ClaimExtractor] JSON parse failed (attempt {attempt + 1}): {e}",
                    payload=f"Raw text:\n{raw[:2000]}",
                )
                if attempt == 0:
                    text = f"{text}\n\n注意：上次输出无法解析为合法 JSON。请严格输出合法 JSON，不要包含任何 markdown 代码块或解释文字。"
                else:
                    raise ClaimExtractionError(f"JSON parse failed after retry: {last_error}", raw=raw)
        else:
            raise ClaimExtractionError(f"JSON parse failed after retry: {last_error}", raw=raw)

        segments = data.get("segments", [])
        if not segments:
            await append_verifier_trace_log(
                "ClaimExtractor",
                "[ClaimExtractor] No segments returned from LLM",
                payload=f"Raw text:\n{raw[:2000]}",
            )
            return []

        claims: List[Claim] = []
        for seg in segments:
            seg_cite_ids = seg.get("cite_ids", [])
            for item in seg.get("claims", []):
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
                c.original_text[:60]
            )
            if key in seen:
                continue
            seen.add(key)

            if len(c.original_text.strip()) < 2:
                continue

            # 交叉校验：提取了 numeric slot 但 claim_type 为 factual → 升级为 factual_numeric
            if c.claim_type == ClaimType.FACTUAL and c.slots.get("numeric"):
                c.claim_type = ClaimType.FACTUAL_NUMERIC

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
    source: str = ""           # 证据来源链路：llm / numeric / fact_store

@dataclass
class ClaimIssue:
    claim_id: str
    type: str
    description: str
    severity: str
    confidence: float = 1.0
    source: str = ""           # fact / numeric / temporal
    evidence: List[EvidenceSpan] = field(default_factory=list)
    suggestion: str = ""


@dataclass
class ClaimEvaluation:
    """单个声明的评估结果"""
    claim_id: str
    original_text: str
    issues: List[ClaimIssue]

    @property
    def claim_score(self) -> float:
        """连续化评分：5分制，无问题=5.0"""
        return compute_claim_score(self.issues)

    @property
    def signature(self) -> Dict[str, int]:
        return {
            "critical": sum(1 for i in self.issues if i.severity == "critical"),
            "major": sum(1 for i in self.issues if i.severity == "major"),
            "minor": sum(1 for i in self.issues if i.severity == "minor"),
        }

    @property
    def combined_suggestion(self) -> str:
        critical_issues = [i for i in self.issues if i.severity == "critical"]
        major_issues = [i for i in self.issues if i.severity == "major"]
        if not self.issues:
            return ""
        if critical_issues:
            return critical_issues[0].suggestion
        if major_issues:
            return major_issues[0].suggestion
        if self.issues[0].suggestion:
            return self.issues[0].suggestion
        return ""

@dataclass
class SegmentEvaluationReport:
    """段落级评估报告"""
    segment_score: int        # 0-100
    star_rating: int          # 1-5
    passed: bool
    total_claims: int
    bad_claims_count: int
    priority_claims: List[ClaimEvaluation]

# ---------------------------------------------------------------------------
# §4  Verifier Router + Issue Fusion
# ---------------------------------------------------------------------------


class VerifierRouter:
    def __init__(self, verifiers: Dict[str, Any], short_term: Optional[Any] = None, max_concurrent: int = 3):
        """
        :param verifiers: {"fact": agent, "numeric": agent, "temporal": agent}
        :param short_term: ShortTermMemoryStore，用于 evidence 来源独立性分析
        :param max_concurrent: 最大并行验证请求数（限流）
        """
        self.verifiers = verifiers
        self.short_term = short_term
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def verify_claims(self, claims: List[Claim]) -> Dict[str, List[ClaimIssue]]:
        """批量验证所有 claims，每个类型只调用一次 agent，并受并发限流控制"""
        # 按类型分组
        claims_by_type: Dict[ClaimType, List[Claim]] = {}
        for claim in claims:
            claims_by_type.setdefault(claim.claim_type, []).append(claim)

        # 构建 verifier -> claims 映射（可能有重复）
        verifier_to_claims: Dict[str, List[Claim]] = {}
        for ct, clist in claims_by_type.items():
            needed = CLAIM_TYPE_TO_VERIFIERS.get(ct, [])
            for vname in needed:
                verifier_to_claims.setdefault(vname, []).extend(clist)

        # 去重（按 claim_id）
        for vname in list(verifier_to_claims.keys()):
            claims_list = verifier_to_claims[vname]
            seen = set()
            unique = []
            for c in claims_list:
                if c.claim_id not in seen:
                    seen.add(c.claim_id)
                    unique.append(c)
            verifier_to_claims[vname] = unique

        # 创建带信号量控制的协程任务
        async def limited_run(vname: str, vclaims: List[Claim]):
            async with self._semaphore:
                return await self._run_batch(vname, vclaims)

        tasks = [limited_run(vname, vclaims) for vname, vclaims in verifier_to_claims.items()]
        results = await asyncio.gather(*tasks)

        # 合并结果
        claim_issues: Dict[str, List[ClaimIssue]] = {c.claim_id: [] for c in claims}
        for result in results:
            for cid, issues in result.items():
                claim_issues[cid].extend(issues)

        # 按 claim_id 合并 issues 并赋予系统 confidence
        result: Dict[str, List[ClaimIssue]] = {}
        for c in claims:
            merged = self._merge_issues(claim_issues[c.claim_id])
            for issue in merged:
                self._assign_system_confidence(issue, c.cite_ids)
            result[c.claim_id] = merged
        return result

    # ---------- _run_batch 拆分为三步：调用 → 解析 → 校验 ----------

    async def _call_agent(self, verifier_name: str, payload: dict) -> str:
        """调用 agent 获取原始文本响应。"""
        agent = self.verifiers[verifier_name]
        msg = Msg(role="user", content=json.dumps(payload, ensure_ascii=False), name="Verifier")
        response_msg = await call_agent_with_retry(agent, msg)
        return _extract_text_response(response_msg)

    async def _parse_response(self, verifier_name: str, raw_text: str, payload: dict) -> Tuple[Optional[Any], str]:
        """解析 JSON 响应，失败时重试一次 agent 调用。
        返回: (解析后的数据, 最后使用的 raw_text)。解析彻底失败返回 (None, raw_text)。
        """
        data = None
        parse_error = None
        for attempt in range(2):
            try:
                data = _safe_parse_json(raw_text)
                return data, raw_text
            except (json.JSONDecodeError, ValueError) as e:
                parse_error = e
                await append_verifier_trace_log(
                    f"{verifier_name.upper()} PARSE ERROR",
                    f"JSON parse failed (attempt {attempt + 1}): {e}",
                    payload=raw_text[:2000]
                )
                if attempt == 0:
                    try:
                        retry_payload = json.dumps(payload, ensure_ascii=False) + "\n\n注意：上次输出无法解析为合法 JSON。请严格输出合法 JSON 数组，不要包含任何 markdown 代码块或解释文字。"
                        msg = Msg(role="user", content=retry_payload, name="Verifier")
                        agent = self.verifiers[verifier_name]
                        response_msg = await call_agent_with_retry(agent, msg)
                        raw_text = _extract_text_response(response_msg)
                    except Exception as retry_e:
                        await append_verifier_trace_log(
                            f"{verifier_name.upper()} RETRY ERROR",
                            f"Retry call failed: {retry_e}",
                        )
                        break
                else:
                    break

        if data is None:
            await append_verifier_trace_log(
                f"{verifier_name.upper()} PARSE FATAL",
                f"JSON parse failed after retry: {parse_error}",
                payload=raw_text[:2000]
            )
        return data, raw_text

    async def _validate_schema(self, verifier_name: str, items: List[Any], claims: List[Claim]) -> Dict[str, List[ClaimIssue]]:
        """对解析后的 issue items 做类型安全处理和 schema 校验。"""
        result: Dict[str, List[ClaimIssue]] = {c.claim_id: [] for c in claims}
        for item in items:
            if not isinstance(item, dict):
                await append_verifier_trace_log(
                    f"{verifier_name.upper()} SCHEMA ERROR",
                    "Issue item is not a dict",
                    payload=str(item)[:200]
                )
                continue

            claim_id = item.get("claim_id")
            if not claim_id or not isinstance(claim_id, str) or claim_id not in result:
                await append_verifier_trace_log(
                    f"{verifier_name.upper()} SCHEMA ERROR",
                    f"Missing or invalid claim_id: {claim_id}",
                    payload=json.dumps(item, ensure_ascii=False)[:200]
                )
                continue

            issue_type = item.get("type")
            if not isinstance(issue_type, str):
                issue_type = "unknown"

            description = item.get("description", "")
            if not isinstance(description, str):
                description = str(description)

            severity = item.get("severity")
            if severity and not isinstance(severity, str):
                severity = "minor"

            evidence_spans = []
            ev_list = item.get("evidence", [])
            if isinstance(ev_list, list):
                for ev in ev_list:
                    if isinstance(ev, dict):
                        evidence_spans.append(EvidenceSpan(
                            cite_id=ev.get("cite_id", ""),
                            text=ev.get("text", ""),
                            source="agent",
                        ))
                    elif isinstance(ev, EvidenceSpan):
                        evidence_spans.append(ev)

            suggestion_raw = item.get("suggestion", "")
            if isinstance(suggestion_raw, dict):
                suggestion_str = suggestion_raw.get("content", str(suggestion_raw))
            elif isinstance(suggestion_raw, str):
                suggestion_str = suggestion_raw
            else:
                suggestion_str = ""

            issue = ClaimIssue(
                claim_id=claim_id,
                type=issue_type,
                description=f"[{verifier_name}] {description}",
                severity=self._normalize_severity(severity),
                source=verifier_name,
                evidence=evidence_spans,
                suggestion=suggestion_str
            )
            result[claim_id].append(issue)

        return result

    async def _run_batch(self, verifier_name: str, claims: List[Claim]) -> Dict[str, List[ClaimIssue]]:
        """一次性处理多个 claims：调用 → 解析 → 校验。"""
        payload = {"claims": [c.to_dict() for c in claims]}

        try:
            raw_text = await self._call_agent(verifier_name, payload)
            data, final_text = await self._parse_response(verifier_name, raw_text, payload)

            if data is None:
                return {
                    c.claim_id: [ClaimIssue(
                        claim_id=c.claim_id,
                        type="parse_error",
                        description=f"[{verifier_name}] JSON parse failed",
                        severity="critical",
                        evidence=[EvidenceSpan(cite_id="", text=final_text[:2000], source="parser")],
                        suggestion=f"Verifier {verifier_name} 返回无法解析的响应，请检查输出格式",
                    )]
                    for c in claims
                }

            if isinstance(data, list):
                issues_list = data
            elif isinstance(data, dict):
                issues_list = data.get("issues", [])
            else:
                issues_list = []

            return await self._validate_schema(verifier_name, issues_list, claims)

        except Exception as e:
            await append_verifier_trace_log(
                f"{verifier_name.upper()} BATCH ERROR",
                f"{type(e).__name__}: {e}",
                payload=traceback.format_exc(),
            )
            return {
                c.claim_id: [ClaimIssue(
                    claim_id=c.claim_id,
                    type="parse_error",
                    description=f"[{verifier_name}] Batch execution failed: {type(e).__name__}: {e}",
                    severity="critical",
                    evidence=[EvidenceSpan(cite_id="", text=traceback.format_exc()[:1000], source="system")],
                    suggestion=f"Verifier {verifier_name} 执行异常，请检查输入或重试",
                )]
                for c in claims
            }

    # ---------- 系统生成 confidence：基于 evidence 独立来源数量 ----------
    def _assign_system_confidence(self, issue: ClaimIssue, fallback_cite_ids: List[str] = None) -> None:
        if issue.type in ("parse_error",):
            issue.confidence = 0.20
            return

        cite_ids = [ev.cite_id for ev in issue.evidence if ev.cite_id]
        if not cite_ids and fallback_cite_ids:
            cite_ids = fallback_cite_ids

        unique_cites = set(cite_ids)
        n_cites = len(unique_cites)

        if self.short_term and n_cites > 0:
            source_keys = set()
            for cid in unique_cites:
                try:
                    source_keys.add(material_source_key(self.short_term, cid))
                except Exception:
                    source_keys.add(f"cite:{cid}")
            n_sources = len(source_keys)
        else:
            n_sources = n_cites

        if n_sources >= 2:
            issue.confidence = min(1.0, 0.80 + n_sources * 0.05)
        elif n_sources == 1:
            issue.confidence = 0.60
        else:
            issue.confidence = 0.30

    # ---------- 合并：用 (type, claim_id, source) 三元组去重 ----------
    def _merge_issues(self, issues: List[ClaimIssue]) -> List[ClaimIssue]:
        merged: Dict[Tuple[str, str, str], ClaimIssue] = {}
        for iss in issues:
            key = (iss.type, iss.claim_id, iss.source)
            if key not in merged:
                merged[key] = iss
                continue
            existing = merged[key]
            # 保留更高 severity
            if self._severity_priority(iss.severity) > self._severity_priority(existing.severity):
                existing.severity = iss.severity
            # 保留更高 confidence
            if iss.confidence > existing.confidence:
                existing.confidence = iss.confidence
            # 合并 description（不同则拼接）
            if iss.description and iss.description != existing.description:
                existing.description = f"{existing.description}; {iss.description}"
            # 合并 evidence（去重）
            seen_ev = {(e.cite_id, e.text) for e in existing.evidence}
            for ev in iss.evidence:
                if (ev.cite_id, ev.text) not in seen_ev:
                    existing.evidence.append(ev)
                    seen_ev.add((ev.cite_id, ev.text))
            # 补充 suggestion
            if not existing.suggestion and iss.suggestion:
                existing.suggestion = iss.suggestion
        return list(merged.values())

    # ---------- 辅助方法（保持不变）----------
    def _normalize_severity(self, s: str) -> str:
        if not s:
            return "minor"
        s = s.lower()
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

class SegmentVerifier:
    def __init__(self, short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore):
        self.model = create_chat_model()
        self.formatter = create_agent_formatter()

        # ---------- 创建 ClaimExtractor（轻量调用，无 ReActAgent 开销）----------
        self.extractor = ClaimExtractor(
            model=self.model,
            formatter=self.formatter,
            sys_prompt=prompt_dict["claim_extract_sys_prompt"],
        )

        # ---------- 创建三个 Verifier Agents ----------
        self.verifiers = create_three_verifiers(
            model=self.model,
            formatter=self.formatter,
            short_term=short_term,
            long_term=long_term
        )

        self.router = VerifierRouter(self.verifiers, short_term=short_term)

    async def verify(self, segment: str) -> List[ClaimIssue]:
        """原有接口：只返回 issues（兼容旧代码）"""
        try:
            claims = await self.extractor.extract(segment)
        except ClaimExtractionError as e:
            await append_verifier_trace_log(
                "SegmentVerifier",
                f"Claim extraction failed: {e}",
                payload=e.raw[:2000],
            )
            return [ClaimIssue(
                claim_id="extract_failed",
                type="parse_error",
                description=f"Claim extraction failed: {e}",
                severity="critical",
                evidence=[EvidenceSpan(cite_id="", text=e.raw[:2000], source="parser")],
                suggestion="请检查输入文本格式或重试",
            )]

        if not claims:
            return []
        await append_verifier_trace_log(
            "SegmentVerifier",
            "======== claims抽取已完成 ========",
            payload=json.dumps(
                [claim.to_dict() for claim in claims],
                ensure_ascii=False,
                indent=2,
            ),
        )

        claim_issues = await self.router.verify_claims(claims)
        issues = []
        for iss_list in claim_issues.values():
            issues.extend(iss_list)
        return issues

    async def verify_with_claims(self, segment: str) -> Tuple[List[Claim], List[ClaimIssue]]:
        """新接口：返回提取的 claims 和对应的 issues，用于评估报告"""
        try:
            claims = await self.extractor.extract(segment)
        except ClaimExtractionError as e:
            await append_verifier_trace_log(
                "SegmentVerifier",
                f"Claim extraction failed: {e}",
                payload=e.raw[:2000],
            )
            issue = ClaimIssue(
                claim_id="extract_failed",
                type="parse_error",
                description=f"Claim extraction failed: {e}",
                severity="critical",
                evidence=[EvidenceSpan(cite_id="", text=e.raw[:2000], source="parser")],
                suggestion="请检查输入文本格式或重试",
            )
            return [], [issue]

        if not claims:
            return [], []

        await append_verifier_trace_log(
            "SegmentVerifier",
            "======== claims抽取已完成 ========",
            payload=json.dumps(
                [claim.to_dict() for claim in claims],
                ensure_ascii=False,
                indent=2,
            ),
        )
        claim_issues = await self.router.verify_claims(claims)
        issues = []
        for iss_list in claim_issues.values():
            issues.extend(iss_list)
        return claims, issues


# ---------------------------------------------------------------------------
# Claim-Centric Evaluation Functions
# ---------------------------------------------------------------------------
def compute_claim_score(issues: List[ClaimIssue]) -> float:
    """连续化评分：5分制，无问题=5.0"""
    if not issues:
        return 5.0
    W_CRITICAL = 0.5
    W_MAJOR = 0.25
    W_MINOR = 0.1
    penalty = 0.0
    for i in issues:
        w = W_CRITICAL if i.severity == "critical" else W_MAJOR if i.severity == "major" else W_MINOR if i.severity == "minor" else 0.0
        multiplier = 1.0 + max(0.0, min(1.0, i.confidence))
        penalty += w * multiplier
    return round(max(0.0, 5.0 * (1.0 - penalty)), 2)


def evaluate_claims(claims: List[Claim], all_issues: List[ClaimIssue]) -> List[ClaimEvaluation]:
    issues_by_claim: Dict[str, List[ClaimIssue]] = {c.claim_id: [] for c in claims}
    for iss in all_issues:
        if iss.claim_id in issues_by_claim:
            issues_by_claim[iss.claim_id].append(iss)
        else:
            issues_by_claim[iss.claim_id] = [iss]

    return [
        ClaimEvaluation(
            claim_id=claim.claim_id,
            original_text=claim.original_text,
            issues=issues_by_claim.get(claim.claim_id, []),
        )
        for claim in claims
    ]

def compute_segment_report(claim_evaluations: List[ClaimEvaluation], top_k: int = 3) -> SegmentEvaluationReport:
    # 按 original_text 去重：同一文本保留问题最多（score 最低）的代表
    text_to_best: Dict[str, ClaimEvaluation] = {}
    for ce in claim_evaluations:
        existing = text_to_best.get(ce.original_text)
        if existing is None or ce.claim_score < existing.claim_score:
            text_to_best[ce.original_text] = ce
    claim_evaluations = list(text_to_best.values())

    total_claims = len(claim_evaluations)
    if total_claims == 0:
        return SegmentEvaluationReport(
            segment_score=100,
            star_rating=5,
            passed=True,
            total_claims=0,
            bad_claims_count=0,
            priority_claims=[]
        )

    # 基础平均分 (claim_score 范围 0.0~5.0)
    avg_score = sum(ce.claim_score for ce in claim_evaluations) / total_claims
    segment_score = int(avg_score * 20)

    # 连续化惩罚：critical 数量越多分数衰减越明显
    critical_count = sum(1 for ce in claim_evaluations if ce.signature["critical"] > 0)
    if critical_count > 0:
        factor = max(0.35, 1.0 - 0.18 * critical_count)
        segment_score = int(segment_score * factor)

    # 星数映射（基于 0~100 的 segment_score）
    if segment_score >= 90:
        star_rating = 5
    elif segment_score >= 75:
        star_rating = 4
    elif segment_score >= 60:
        star_rating = 3
    elif segment_score >= 40:
        star_rating = 2
    else:
        star_rating = 1

    # 通过条件：分数≥60 且没有 critical 声明
    passed = (segment_score >= 60) and (critical_count == 0)

    # 筛选有问题的声明（score < 5.0 表示存在问题）
    problematic_claims = [ce for ce in claim_evaluations if ce.claim_score < 5.0]
    # bad_claims：严重问题（score < 3.0）
    bad_claims = [ce for ce in claim_evaluations if ce.claim_score < 3.0]
    bad_claims_count = len(bad_claims)

    # 保留全部 critical，非 critical 按 top_k 截断
    critical_claims = [ce for ce in problematic_claims if ce.signature["critical"] > 0]
    non_critical = sorted(
        [ce for ce in problematic_claims if ce.signature["critical"] == 0],
        key=lambda ce: (-ce.signature["major"], ce.claim_score, -len(ce.issues))
    )
    remaining = max(0, top_k - len(critical_claims))
    priority = critical_claims + non_critical[:remaining]
    
    return SegmentEvaluationReport(
        segment_score=segment_score,
        star_rating=star_rating,
        passed=passed,
        total_claims=total_claims,
        bad_claims_count=len(bad_claims),
        priority_claims=priority,
    )

def format_report_for_writer(report: SegmentEvaluationReport) -> str:
    """将评估报告格式化为 writer 易读的 Markdown 文本，包含 claim_id"""
    lines = []
    lines.append("## 段落评估报告\n")
    lines.append(f"**段落评分**：{report.segment_score}/100 →({report.star_rating}/5)")
    status_icon = "通过" if report.passed else "不通过（需修改）"
    lines.append(f"**判定**：{status_icon}\n")
    lines.append(f"**概览**：共 {report.total_claims} 个声明，其中 {report.bad_claims_count} 个存在问题。\n")
    
    if not report.priority_claims:
        lines.append("所有声明质量良好，无需修改。")
        return "\n".join(lines)
    
    lines.append(f"**优先修改以下 {len(report.priority_claims)} 个声明**：\n")
    
    for idx, ce in enumerate(report.priority_claims, 1):
        # 显示 claim_id，方便 writer 定位
        score_display = f"{ce.claim_score:.1f}/5.0"
        lines.append(f"### 声明 {idx} (ID: `{ce.claim_id}`, 得分: {score_display}, 严重问题: critical={ce.signature['critical']}, major={ce.signature['major']})")
        lines.append(f"> {ce.original_text}\n")
        lines.append("**问题汇总**：")
        for iss in ce.issues:
            lines.append(f"- **[{iss.severity.upper()}]** {iss.description}")
        lines.append(f"\n**修改建议**：{ce.combined_suggestion}\n")
        lines.append("---\n")
    
    lines.append("**要求**：请优先重写上述声明，一次性修正其所有问题。其他声明可保持不变或微调。")
    return "\n".join(lines)

# # ---------------------------------------------------------------------------
# # §6  Test / Demo
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     segment = (
#         """公司已逐步将产业链一体化优势转化为产品力优势。通过实现新能源汽车产业链的全覆盖，有效降低了原材料价格剧烈变化的风险，同时有效降低成本，提高生产效率，并因此拥有了一定的定价主动权 [^cite_id:2024-06-24_reference_report]。

# 从产品策略来看，公司已将成本端与企业运营端优势转化为产品力优势：2024年2月，王朝/海洋系列在短短两周内密集投放五波荣耀版车型，包括秦PLUS、驱逐舰05、海豚、汉、唐以及宋PLUS、宋Pro，覆盖从7.98万元到24.98万元的小型、紧凑型以及中大型车市场，较冠军版起售价最高降低了6万元，实现了"加量还降价" [^cite_id:2024-06-24_reference_report]。其中，秦PLUS荣耀版以"日系省油、德系驾驶、美系智能"的赞誉上市，首周便取得了23,590辆的新车订单 [^cite_id:2024-06-24_reference_report]。

# 2024年5月，公司开启了基于全新混动平台DM5.0的全新车型序列的逐步上市，率先上市的比亚迪秦L与海豹06定位紧凑型级别，但车身尺寸和轴距已经是中型轿车水平 [^cite_id:2024-06-24_reference_report]。该车型序列叠加同级领先的油耗、智能化与电气化水平，在"油电同价"的基础上进一步升级为"电比油低"，充分体现了公司对于10万至20万元细分市场的竞争力与志在必得的决心 [^cite_id:2024-06-24_reference_report]。上市不到2周的时间已经累计获得8万台订单，充分体现了消费者对该系列车型的充分认可 [^cite_id:2024-06-24_reference_report]。

# ![新车型与竞品核心参数对比](chart:chart_1774964357334)

# 数据显示，2026年5月上市的秦L DM-i与海豹06 DM-i在核心参数上显著优于同级别传统燃油竞品：WLTC综合油耗分别为1.11L/100km和1.36L/100km，远低于轩逸2024款经典（5.94L/100km）和朗逸2024款（5.92L/100km）[^cite_id:2024-06-24_reference_report]。车身尺寸方面，秦L与海豹06的轴距达到2790mm，已超过轩逸（2700mm）和朗逸（2688mm）的中型轿车水平 [^cite_id:2024-06-24_reference_report]。智能化配置上，秦L与海豹06标配倒车影像、定速巡航、车载智能系统及完整的手机App远程控制功能，而轩逸和朗逸在同价位车型中上述配置多数缺失 [^cite_id:2024-06-24_reference_report]。
# """
#     )

#     PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#     long_term_dir = PROJECT_ROOT / "data" / "memory" / "long_term"
    
#     long_term = LongTermMemoryStore(
#         base_dir=long_term_dir,
#     )

#     short_term_dir = PROJECT_ROOT / "data" / "memory" / "short_term" / "002594_20260411"

#     short_term = ShortTermMemoryStore(
#         base_dir=short_term_dir,
#     )

#     # ---- 注册缺失的材料 ----
#     cite_id = "2024-06-24_reference_report"
#     material_path = short_term_dir / "material" / f"{cite_id}.txt"

#     # 检查元数据中是否已有
#     meta = short_term.get_material_meta(cite_id)
#     if not meta:
#         if material_path.exists():
#             with open(material_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#             short_term.save_material(
#                 cite_id=cite_id,
#                 content=content,
#                 description="比亚迪首次覆盖报告（2024-06-24）",
#                 source="测试材料",
#             )
#             print(f"材料 {cite_id} 已注册")
#         else:
#             print(f"警告：材料文件 {material_path} 不存在，请检查路径")
#     else:
#         print(f"材料 {cite_id} 已在元数据中")

#     trace_path = PROJECT_ROOT / "verifier_trace.log"
#     set_verifier_trace_path(trace_path)
    
    
#     verifier = SegmentVerifier(short_term=short_term, long_term=long_term)

#     async def test():
#         print("\n================ STEP 1: Claim Extraction ================\n")

#         claims, issues = await verifier.verify_with_claims(segment)

#         print(f"Total Claims: {len(claims)}")
#         for c in claims:
#             print(f"[{c.claim_id}] ({c.claim_type}) {c.original_text}")

#         print("\n================ STEP 2: Issues ================\n")

#         for iss in issues:
#             print(f"[{iss.severity}] {iss.type} → {iss.description}")

#         print("\n================ STEP 3: Claim Evaluation (Rubrics) ================\n")

#         claim_evals = evaluate_claims(claims, issues)

#         for ce in claim_evals:
#             print(f"\n--- Claim {ce.claim_id} ---")
#             print(f"Text: {ce.original_text}")
#             print(f"Score: {ce.claim_score}/5")
#             print(f"Signature: {ce.signature}")
#             print(f"Issues: {len(ce.issues)}")
#             print(f"Suggestion: {ce.combined_suggestion}")

#         print("\n================ STEP 4: Segment Report ================\n")

#         report = compute_segment_report(claim_evals)

#         print(f"Segment Score: {report.segment_score}")
#         print(f"Stars: {report.star_rating}")
#         print(f"Passed: {report.passed}")
#         print(f"Bad Claims: {report.bad_claims_count}")

#         print("\n================ STEP 5: Writer Feedback ================\n")

#         feedback = format_report_for_writer(report)
#         print(feedback)

#     asyncio.run(test())
