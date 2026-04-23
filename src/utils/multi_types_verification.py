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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentscope.message import Msg
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from src.agents.verifier import create_three_verifiers
from src.memory.short_term import ShortTermMemoryStore, MaterialType
from src.memory.long_term import LongTermMemoryStore
from src.utils.instance import create_chat_model, create_agent_formatter
from src.utils.call_with_retry import call_agent_with_retry, call_chatbot_with_retry
from src.prompt import prompt_dict
from src.utils.multi_source_verification import material_source_key

# ---------------------------------------------------------------------------
# Verifier Trace (replaced synchronous I/O with async-safe executor)
# ---------------------------------------------------------------------------
VERIFIER_TRACE_LOCK = asyncio.Lock()
VERIFIER_TRACE_PATH: Optional[Path] = None


def set_verifier_trace_path(path: Optional[Path]) -> None:
    global VERIFIER_TRACE_PATH
    VERIFIER_TRACE_PATH = path
    if VERIFIER_TRACE_PATH is not None:
        # 清空文件
        VERIFIER_TRACE_PATH.write_text("", encoding="utf-8")


async def _write_text_async(path: Path, text: str) -> None:
    """使用线程池执行同步 I/O，避免阻塞事件循环"""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: path.open("a", encoding="utf-8").write(text))


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
        await _write_text_async(VERIFIER_TRACE_PATH, "\n".join(sections))


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
        await _write_text_async(VERIFIER_TRACE_PATH, "\n".join(sections))


# ---------------------------------------------------------------------------
# 安全 JSON 解析（非贪婪，更鲁棒）
# ---------------------------------------------------------------------------
def _safe_parse_json(text: str) -> Dict[str, Any]:
    # 移除可能的 markdown 代码块
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # 优先尝试直接解析整个文本（处理纯净 JSON 的情况）
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 括号计数提取最外层 JSON 对象或数组
    def _extract_balanced(s: str, open_ch: str, close_ch: str):
        for i, ch in enumerate(s):
            if ch == open_ch:
                depth = 1
                for j in range(i + 1, len(s)):
                    if s[j] == open_ch:
                        depth += 1
                    elif s[j] == close_ch:
                        depth -= 1
                        if depth == 0:
                            return s[i:j + 1]
        return None

    # 优先找对象 {...}，再找数组 [...]
    for opener, closer in (('{', '}'), ('[', ']')):
        extracted = _extract_balanced(text, opener, closer)
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                continue

    raise ValueError("Failed to parse JSON from response")


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


def route_verifiers(claim: Dict[str, Any]) -> List[str]:
    """
    Slot-level routing：根据 claim 实际包含的 slot 决定需要哪些验证器。
    claim 可以是 Claim 对象或字典（含 slots）。
    """
    slots = claim.get("slots", {}) if isinstance(claim, dict) else claim.slots
    v = []
    if slots.get("factual"):
        v.append("fact")
    if slots.get("numeric"):
        v.append("numeric")
    if slots.get("temporal"):
        v.append("temporal")
    # 若 slot 为空，回退到按 claim_type 路由
    if not v:
        ct = claim.get("claim_type", "") if isinstance(claim, dict) else claim.claim_type
        if ct in ("factual",):
            v = ["fact"]
        elif ct in ("numeric",):
            v = ["numeric"]
        elif ct in ("temporal",):
            v = ["temporal"]
        elif ct in ("factual_numeric",):
            v = ["fact", "numeric"]
        elif ct in ("numeric_temporal",):
            v = ["numeric", "temporal"]
        elif ct in ("composite",):
            v = ["fact", "numeric", "temporal"]
    return v


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
    使用 ChatAgent（无工具）进行纯文本生成，避免 ReAct 的开销。
    """
    CITE_RE = re.compile(r"\[\^cite_id:([A-Za-z0-9_\-]+)(?:\|[^\]]*)?\]")

    def __init__(self, agent):
        self.agent = agent

    async def extract(self, text: str) -> List[Claim]:
        if not text.strip():
            return []

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

        try:
            data = _safe_parse_json(raw)
        except Exception as e:
            await append_verifier_trace_log(
                "ClaimExtractor",
                f"[ClaimExtractor] Failed to parse JSON: {e}",
                payload=f"Raw text:\n{raw}",
            )
            return []
        
        # ---- 加这行调试 ----
        await append_verifier_trace_log(
            "ClaimExtractor-Debug",
            f"Parsed data type: {type(data).__name__}",
            payload=json.dumps(data, ensure_ascii=False, indent=2) if not isinstance(data, str) else data,
        )
        # -------------------

        # 兼容顶层数组和对象两种格式
        if isinstance(data, list):
            segments = data
        elif isinstance(data, dict):
            segments = data.get("segments", [])
        else:
            segments = []

        if not segments:
            await append_verifier_trace_log(
                "ClaimExtractor",
                "[ClaimExtractor] No segments returned from LLM",
                payload=f"Raw text:\n{raw}",
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
    source: str = ""           # 证据来源链路：agent / numeric / fact_store / invalid_cite
    valid: bool = True         # cite_id 在短期记忆中是否可查证


@dataclass
class ClaimIssue:
    claim_id: str
    type: str
    description: str
    severity: str
    confidence: float = 1.0
    source: str = ""           # 多来源用“/”分隔，如 "fact/numeric"
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
        """返回优先级最高的修改建议，critical → major → 任意"""
        critical_issues = [i for i in self.issues if i.severity == "critical"]
        major_issues = [i for i in self.issues if i.severity == "major"]
        if not self.issues:
            return ""
        if critical_issues:
            return critical_issues[0].suggestion
        if major_issues:
            return major_issues[0].suggestion
        # 合并所有建议
        suggestions = [i.suggestion for i in self.issues if i.suggestion]
        if suggestions:
            return "；".join(suggestions)
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
    def __init__(self, verifiers: Dict[str, Any], short_term: Optional[Any] = None):
        """verifiers: {"fact": agent, "numeric": agent, "temporal": agent}"""
        self.verifiers = verifiers
        self.short_term = short_term

    async def verify_claims(self, claims: List[Claim]) -> Dict[str, List[ClaimIssue]]:
        """批量验证所有 claims，每个 verifier 处理其负责的子集。"""
        # 按类型分组，再根据每个声明的 slots 决定需要哪些 verifier
        claims_by_type: Dict[ClaimType, List[Claim]] = {}
        for claim in claims:
            claims_by_type.setdefault(claim.claim_type, []).append(claim)

        # 根据 slots 动态构建每个 verifier 负责的列表
        verifier_to_claims: Dict[str, List[Claim]] = {}
        for ct, clist in claims_by_type.items():
            for claim in clist:
                needed = route_verifiers(claim.to_dict())
                for vname in needed:
                    verifier_to_claims.setdefault(vname, []).append(claim)

        # 去重（同一个 claim 可能被多个验证器处理）
        for vname in list(verifier_to_claims.keys()):
            claims_list = verifier_to_claims[vname]
            seen = set()
            unique = []
            for c in claims_list:
                if c.claim_id not in seen:
                    seen.add(c.claim_id)
                    unique.append(c)
            verifier_to_claims[vname] = unique

        tasks = []
        for vname, vclaims in verifier_to_claims.items():
            tasks.append(self._run_batch(vname, vclaims))

        results = await asyncio.gather(*tasks)

        # 合并结果
        claim_issues: Dict[str, List[ClaimIssue]] = {c.claim_id: [] for c in claims}
        for result in results:
            for cid, issues in result.items():
                claim_issues[cid].extend(issues)

        # 融合并赋予系统 confidence
        merged_result: Dict[str, List[ClaimIssue]] = {}
        for c in claims:
            merged = self._merge_issues(claim_issues[c.claim_id])
            for issue in merged:
                self._validate_and_assign_confidence(issue, c.cite_ids)
            merged_result[c.claim_id] = merged
        return merged_result

    async def _run_batch(self, verifier_name: str, claims: List[Claim]) -> Dict[str, List[ClaimIssue]]:
        """一次性处理多个 claims，返回 {claim_id: [issues]}"""
        agent = self.verifiers[verifier_name]

        # 每次调用前清空 agent 的记忆，防止上下文污染
        if hasattr(agent, "memory") and agent.memory is not None:
            await agent.memory.clear()

        payload = {"claims": [c.to_dict() for c in claims]}

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

            data = _safe_parse_json(text)

            if isinstance(data, list):
                issues_list = data
            elif isinstance(data, dict):
                issues_list = data.get("issues", [])
            else:
                issues_list = []

            # 建立 claim_id 到 Claim 的映射，用于模糊匹配
            claim_map: Dict[str, Claim] = {c.claim_id: c for c in claims}
            result: Dict[str, List[ClaimIssue]] = {c.claim_id: [] for c in claims}
            unmatched = []

            for item in issues_list:
                cid = item.get("claim_id")
                # 如果 LLM 输出的 claim_id 无法直接匹配，尝试用 original_text 模糊查找
                if not cid or cid not in claim_map:
                    original = item.get("original_text", "")
                    matched_id = None
                    for c in claims:
                        if c.original_text.strip() == original.strip():
                            matched_id = c.claim_id
                            break
                    if matched_id:
                        cid = matched_id
                    else:
                        unmatched.append(item)
                        continue

                evidence_spans = []
                for ev in item.get("evidence", []):
                    if isinstance(ev, dict):
                        evidence_spans.append(EvidenceSpan(
                            cite_id=ev.get("cite_id", ""),
                            text=ev.get("text", ""),
                            source="agent",
                            valid=True  # 初始设为有效，后续由系统校验
                        ))

                suggestion_raw = item.get("suggestion", "")
                if isinstance(suggestion_raw, dict):
                    suggestion_str = suggestion_raw.get("content", str(suggestion_raw))
                elif isinstance(suggestion_raw, str):
                    suggestion_str = suggestion_raw
                else:
                    suggestion_str = ""

                result[cid].append(ClaimIssue(
                    claim_id=cid,
                    type=item.get("type", "unknown"),
                    description=f"[{verifier_name}] {item.get('description', '')}",
                    severity=self._normalize_severity(item.get("severity")),
                    confidence=1.0,  # 将在后续统一校准
                    source=verifier_name,
                    evidence=evidence_spans,
                    suggestion=suggestion_str
                ))

            if unmatched:
                await append_verifier_trace_log(
                    f"{verifier_name.upper()} verifier",
                    f"[{verifier_name}] {len(unmatched)} issue(s) could not be mapped to any claim",
                    payload=json.dumps(unmatched, ensure_ascii=False, indent=2),
                )
            return result

        except Exception as e:
            await append_verifier_trace_log(
                f"{verifier_name.upper()} BATCH ERROR",
                f"[{verifier_name.upper()} BATCH ERROR] {type(e).__name__}: {e}",
                payload=traceback.format_exc(),
            )
            return {c.claim_id: [] for c in claims}

    def _merge_issues(self, issues: List[ClaimIssue]) -> List[ClaimIssue]:
        """融合多个验证器对同一声明的重复发现，保留所有来源信息"""
        merged: Dict[Tuple[str, str], ClaimIssue] = {}
        for iss in issues:
            key = (iss.type, iss.description)
            if key not in merged:
                merged[key] = ClaimIssue(
                    claim_id=iss.claim_id,
                    type=iss.type,
                    description=iss.description,
                    severity=iss.severity,
                    confidence=iss.confidence,
                    source=iss.source,
                    evidence=list(iss.evidence),
                    suggestion=iss.suggestion,
                )
            else:
                existing = merged[key]
                # 升级 severity
                if self._severity_priority(iss.severity) > self._severity_priority(existing.severity):
                    existing.severity = iss.severity
                # 合并来源字符串
                if iss.source not in existing.source:
                    existing.source += f"/{iss.source}"
                # 补充 evidence（去重 cite_id）
                existing_cite_ids = {e.cite_id for e in existing.evidence}
                for ev in iss.evidence:
                    if ev.cite_id not in existing_cite_ids:
                        existing.evidence.append(ev)
                        existing_cite_ids.add(ev.cite_id)
                # 合并建议：优先保留最严重级别的建议，否则拼接
                if not existing.suggestion:
                    existing.suggestion = iss.suggestion
                elif (
                    self._severity_priority(iss.severity) >= self._severity_priority(existing.severity)
                    and iss.suggestion
                ):
                    existing.suggestion = iss.suggestion  # 用更严重问题的建议覆盖
                # 置信度取多个来源的平均值（初步，后面会被系统校准覆盖）
                existing.confidence = 1.0  # 将在后续统一计算
        return list(merged.values())

    def _validate_and_assign_confidence(self, issue: ClaimIssue, fallback_cite_ids: List[str]) -> None:
        """
        基于 evidence 中 cite_id 在短期记忆中的真实存在性，重新计算置信度。
        标记无效证据并降权。
        """
        if issue.type in ("parse_error",):
            issue.confidence = 0.20
            return

        # 收集所有 cite_id，包括 fallback
        cite_ids = [ev.cite_id for ev in issue.evidence if ev.cite_id]
        if not cite_ids and fallback_cite_ids:
            cite_ids = fallback_cite_ids

        valid_cite_ids = []
        for ev in issue.evidence:
            if not ev.cite_id:
                continue
            if self._is_cite_valid(ev.cite_id):
                valid_cite_ids.append(ev.cite_id)
            else:
                ev.valid = False
                ev.source = "invalid_cite"

        # 也可考虑 fallback_cite_ids 中的有效部分
        unique_valid_cites = set()
        for cid in valid_cite_ids:
            unique_valid_cites.add(cid)
        if fallback_cite_ids:
            for cid in fallback_cite_ids:
                if self._is_cite_valid(cid):
                    unique_valid_cites.add(cid)

        # 统计独立来源
        source_keys = set()
        for cid in unique_valid_cites:
            try:
                source_keys.add(material_source_key(self.short_term, cid))
            except Exception:
                source_keys.add(f"cite:{cid}")
        n_sources = len(source_keys)

        if n_sources >= 2:
            issue.confidence = min(1.0, 0.80 + n_sources * 0.05)
        elif n_sources == 1:
            issue.confidence = 0.60
        else:
            issue.confidence = 0.30

        # 如果原始 evidence 都被标记为无效，追加提示到描述
        if issue.evidence and all(not ev.valid for ev in issue.evidence):
            issue.description += " (警告：所有引用证据无法在材料库中查证)"
            issue.confidence = max(0.10, issue.confidence - 0.3)

    def _is_cite_valid(self, cite_id: str) -> bool:
        """验证 cite_id 是否在短期记忆中真实存在"""
        if self.short_term is None:
            return False
        try:
            # 若 material_source_key 成功返回则存在
            material_source_key(self.short_term, cite_id)
            return True
        except Exception:
            return False

    @staticmethod
    def _normalize_severity(s: str) -> str:
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

    @staticmethod
    def _severity_priority(s: str) -> int:
        mapping = {"critical": 3, "major": 2, "minor": 1, "info": 0}
        return mapping.get(s, 1)


# ---------------------------------------------------------------------------
# §5  SegmentVerifier — 总控制器
# ---------------------------------------------------------------------------
class SegmentVerifier:
    def __init__(self, short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore):
        self.model = create_chat_model()
        self.formatter = create_agent_formatter()

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
            segment_for_extract = _strip_chart_references_for_claim_extract(segment)
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
    W_CRITICAL = 0.8
    W_MAJOR = 0.5
    W_MINOR = 0.1
    penalty = 0.0
    for i in issues:
        w = W_CRITICAL if i.severity == "critical" else W_MAJOR if i.severity == "major" else W_MINOR if i.severity == "minor" else 0.0
        multiplier = 0.5 + 0.5 * max(0.0, min(1.0, i.confidence))
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
    """
    生成段落级评估报告。
    - 统计基于原始所有声明（不去重）。
    - 优先修改列表按 original_text 去重展示，避免重复修改同一句子。
    """
    # 原始总数统计
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

    avg_score = sum(ce.claim_score for ce in claim_evaluations) / total_claims
    segment_score = int(avg_score * 20)

    # 分析所有原始声明
    critical_count = sum(1 for ce in claim_evaluations if ce.signature["critical"] > 0)
    if critical_count > 0:
        factor = max(0.35, 1.0 - 0.18 * critical_count)
        segment_score = int(segment_score * factor)

    # 星数映射
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

    passed = (segment_score >= 60) and (critical_count == 0)

    # 有问题的声明（score < 5.0）
    problematic_claims = [ce for ce in claim_evaluations if ce.claim_score < 5.0]
    bad_claims = [ce for ce in claim_evaluations if ce.claim_score < 3.0]
    bad_claims_count = len(bad_claims)

    # 去重：按 original_text 保留问题最严重的代表，用于生成 writer 的优先修改列表
    text_to_best: Dict[str, ClaimEvaluation] = {}
    for ce in problematic_claims:
        existing = text_to_best.get(ce.original_text)
        if existing is None or ce.claim_score < existing.claim_score:
            text_to_best[ce.original_text] = ce
    deduped_problematic = list(text_to_best.values())

    critical_deduped = [ce for ce in deduped_problematic if ce.signature["critical"] > 0]
    non_critical = sorted(
        [ce for ce in deduped_problematic if ce.signature["critical"] == 0],
        key=lambda ce: (-ce.signature["major"], ce.claim_score, -len(ce.issues))
    )
    remaining = max(0, top_k - len(critical_deduped))
    priority_claims = critical_deduped + non_critical[:remaining]

    return SegmentEvaluationReport(
        segment_score=segment_score,
        star_rating=star_rating,
        passed=passed,
        total_claims=total_claims,
        bad_claims_count=bad_claims_count,
        priority_claims=priority_claims,
    )


def format_report_for_writer(report: SegmentEvaluationReport) -> str:
    """将评估报告格式化为 writer 易读的 Markdown 文本，包含 claim_id"""
    lines = []
    lines.append("## 段落评估报告\n")
    lines.append(f"**段落评分**：{report.segment_score}/100 →({report.star_rating}/5)")
    status_icon = "通过" if report.passed else "不通过（需修改）"
    lines.append(f"**判定**：{status_icon}\n")
    lines.append(f"**概览**：共 {report.total_claims} 个声明，其中 {report.bad_claims_count} 个存在严重问题。\n")

    if not report.priority_claims:
        lines.append("所有声明质量良好，无需修改。")
        return "\n".join(lines)

    lines.append(f"**优先修改以下 {len(report.priority_claims)} 个声明**：\n")

    for idx, ce in enumerate(report.priority_claims, 1):
        score_display = f"{ce.claim_score:.1f}/5.0"
        lines.append(
            f"### 声明 {idx} (ID: `{ce.claim_id}`, 得分: {score_display}, "
            f"严重问题: critical={ce.signature['critical']}, major={ce.signature['major']})"
        )
        lines.append(f"> {ce.original_text}\n")
        lines.append("**问题汇总**：")
        for iss in ce.issues:
            lines.append(f"- **[{iss.severity.upper()}]** {iss.description}")
        lines.append(f"\n**修改建议**：{ce.combined_suggestion}\n")
        lines.append("---\n")

    lines.append("**要求**：请优先重写上述声明，一次性修正其所有问题。其他声明可保持不变或微调。")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# §6  Test / Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    segment = (
        """公司已逐步将产业链一体化优势转化为产品力优势。通过实现新能源汽车产业链的全覆盖，有效降低了原材料价格剧烈变化的风险，同时有效降低成本，提高生产效率，并因此拥有了一定的定价主动权 [^cite_id:2024-06-24_reference_report]。

从产品策略来看，公司已将成本端与企业运营端优势转化为产品力优势：2024年2月，王朝/海洋系列在短短两周内密集投放五波荣耀版车型，包括秦PLUS、驱逐舰05、海豚、汉、唐以及宋PLUS、宋Pro，覆盖从7.98万元到24.98万元的小型、紧凑型以及中大型车市场，较冠军版起售价最高降低了6万元，实现了"加量还降价" [^cite_id:2024-06-24_reference_report]。其中，秦PLUS荣耀版以"日系省油、德系驾驶、美系智能"的赞誉上市，首周便取得了23,590辆的新车订单 [^cite_id:2024-06-24_reference_report]。

2024年5月，公司开启了基于全新混动平台DM5.0的全新车型序列的逐步上市，率先上市的比亚迪秦L与海豹06定位紧凑型级别，但车身尺寸和轴距已经是中型轿车水平 [^cite_id:2024-06-24_reference_report]。该车型序列叠加同级领先的油耗、智能化与电气化水平，在"油电同价"的基础上进一步升级为"电比油低"，充分体现了公司对于10万至20万元细分市场的竞争力与志在必得的决心 [^cite_id:2024-06-24_reference_report]。上市不到2周的时间已经累计获得8万台订单，充分体现了消费者对该系列车型的充分认可 [^cite_id:2024-06-24_reference_report]。

![新车型与竞品核心参数对比](chart:chart_1774964357334)

数据显示，2026年5月上市的秦L DM-i与海豹06 DM-i在核心参数上显著优于同级别传统燃油竞品：WLTC综合油耗分别为1.11L/100km和1.36L/100km，远低于轩逸2024款经典（5.94L/100km）和朗逸2024款（5.92L/100km）[^cite_id:2024-06-24_reference_report]。车身尺寸方面，秦L与海豹06的轴距达到2790mm，已超过轩逸（2700mm）和朗逸（2688mm）的中型轿车水平 [^cite_id:2024-06-24_reference_report]。智能化配置上，秦L与海豹06标配倒车影像、定速巡航、车载智能系统及完整的手机App远程控制功能，而轩逸和朗逸在同价位车型中上述配置多数缺失 [^cite_id:2024-06-24_reference_report]。
"""
    )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    long_term_dir = PROJECT_ROOT / "data" / "memory" / "long_term"
    
    long_term = LongTermMemoryStore(
        base_dir=long_term_dir,
    )

    short_term_dir = PROJECT_ROOT / "data" / "memory" / "short_term" / "002594_20260411"

    short_term = ShortTermMemoryStore(
        base_dir=short_term_dir,
    )

    # ---- 注册缺失的材料 ----
    cite_id = "2024-06-24_reference_report"
    material_path = short_term_dir / "material" / f"{cite_id}.txt"

    # 检查元数据中是否已有
    meta = short_term.get_material_meta(cite_id)
    if not meta:
        if material_path.exists():
            with open(material_path, "r", encoding="utf-8") as f:
                content = f.read()
            short_term.save_material(
                cite_id=cite_id,
                content=content,
                description="比亚迪首次覆盖报告（2024-06-24）",
                source="测试材料",
            )
            print(f"材料 {cite_id} 已注册")
        else:
            print(f"警告：材料文件 {material_path} 不存在，请检查路径")
    else:
        print(f"材料 {cite_id} 已在元数据中")

    trace_path = PROJECT_ROOT / "verifier_trace.log"
    set_verifier_trace_path(trace_path)
    
    
    verifier = SegmentVerifier(short_term=short_term, long_term=long_term)

    async def test():
        print("\n================ STEP 1: Claim Extraction ================\n")

        claims, issues = await verifier.verify_with_claims(segment)

        print(f"Total Claims: {len(claims)}")
        for c in claims:
            print(f"[{c.claim_id}] ({c.claim_type}) {c.original_text}")

        print("\n================ STEP 2: Issues ================\n")

        for iss in issues:
            print(f"[{iss.severity}] {iss.type} → {iss.description}")
        
         # --- 追加：issues 日志 ---
        await append_verifier_trace_log(
              "TEST-Issues",
              f"Found {len(issues)} issues",
              payload=json.dumps(
                  [
                      {
                          "claim_id": i.claim_id,
                          "type": i.type,
                          "severity": i.severity,
                          "description": i.description,
                          "confidence": i.confidence,
                          "source": i.source,
                          "suggestion": i.suggestion,
                      }
                      for i in issues
                  ],
                  ensure_ascii=False, indent=2, default=str
              ),
          )

        print("\n================ STEP 3: Claim Evaluation (Rubrics) ================\n")

        claim_evals = evaluate_claims(claims, issues)

        for ce in claim_evals:
            print(f"\n--- Claim {ce.claim_id} ---")
            print(f"Text: {ce.original_text}")
            print(f"Score: {ce.claim_score}/5")
            print(f"Signature: {ce.signature}")
            print(f"Issues: {len(ce.issues)}")
            print(f"Suggestion: {ce.combined_suggestion}")

        print("\n================ STEP 4: Segment Report ================\n")

        report = compute_segment_report(claim_evals)

        print(f"Segment Score: {report.segment_score}")
        print(f"Stars: {report.star_rating}")
        print(f"Passed: {report.passed}")
        print(f"Bad Claims: {report.bad_claims_count}")

        print("\n================ STEP 5: Writer Feedback ================\n")

        feedback = format_report_for_writer(report)
        print(feedback)

        await append_verifier_trace(
              topic="test-segment",
              round_idx=1,
              checked_text=segment[:500],
              verify_feedback=feedback,
              issue_count=report.bad_claims_count,
              status="passed" if report.passed else "issues_found",
              score=report.segment_score,
              star_rating=report.star_rating,
              passed=report.passed,
              priority_claims_count=len(report.priority_claims),
          )
    asyncio.run(test())