"""
Triple-Check Verifier — 论文级三路交叉验证系统
=================================================
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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
from urllib.parse import urlparse

from agentscope.message import Msg
from agentscope.agent import ReActAgent  
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from src.agents.verifier import create_three_verifiers, build_verifier_toolkit
from src.memory.short_term import ShortTermMemoryStore, MaterialType
from src.memory.long_term import LongTermMemoryStore
from src.utils.instance import create_chat_model, create_agent_formatter
from src.utils.call_with_retry import call_agent_with_retry
from src.prompt import prompt_dict
from pathlib import Path
from src.memory.short_term import MaterialType

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

class ClaimExtractor:
    """
    从文本片段中提取原子声明。
    使用 ChatAgent（无工具）进行纯文本生成，避免 ReAct 的开销。
    """
    CITE_RE = re.compile(r"\[\^cite_id:([A-Za-z0-9_\-]+)(?:\|[^\]]*)?\]")

    def __init__(self, agent):
        """
        :param agent: AgentScope ChatAgent 实例（已配置好系统提示词）
        """
        self.agent = agent

    # ---------- 主提取方法 ----------
    async def extract(self, text: str) -> List[Claim]:
        if not text.strip():
            return []

        # 构建用户消息
        msg = Msg(
            name="extractor",
            role="user",
            content=text
        )

        # 使用 call_agent_with_retry 调用 agent，并额外校验文本响应
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

        segments = data.get("segments", [])
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
                c.original_text[:60]
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
    claim_id: str
    type: str
    description: str
    severity: str
    evidence: List[EvidenceSpan] = field(default_factory=list)
    suggestion: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClaimResult:
    claim_id: str
    issues: List[ClaimIssue]

@dataclass
class ClaimEvaluation:
    """单个声明的评估结果"""
    claim_id: str
    original_text: str
    claim_score: int          # 1-5, 1最差, 5完美
    issues: List[ClaimIssue]
    combined_suggestion: str
    signature: Dict[str, int] = field(default_factory=dict)  # {"critical": n, "major": n, "minor": n}

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
import traceback


class VerifierRouter:
    def __init__(self, verifiers: Dict[str, Any], max_concurrent: int = 3):
        """
        :param verifiers: {"fact": agent, "numeric": agent, "temporal": agent}
        :param max_concurrent: 最大并行验证请求数（限流）
        """
        self.verifiers = verifiers
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def verify_claims(self, claims: List[Claim]) -> List[ClaimResult]:
        """批量验证所有 claims，每个类型只调用一次 agent，并受并发限流控制"""
        # 按类型分组
        claims_by_type: Dict[ClaimType, List[Claim]] = {}
        for claim in claims:
            claims_by_type.setdefault(claim.claim_type, []).append(claim)

        # 需要验证的类型映射（与 agent 类型对应）
        type_to_verifier = {
            ClaimType.FACTUAL: "fact",
            ClaimType.NUMERIC: "numeric",
            ClaimType.TEMPORAL: "temporal",
            ClaimType.FACTUAL_NUMERIC: ["fact", "numeric"],
            ClaimType.NUMERIC_TEMPORAL: ["numeric", "temporal"],
            ClaimType.COMPOSITE: ["fact", "numeric", "temporal"],
        }

        # 构建 verifier -> claims 映射（可能有重复）
        verifier_to_claims: Dict[str, List[Claim]] = {}
        for ct, clist in claims_by_type.items():
            needed = type_to_verifier.get(ct, [])
            if isinstance(needed, str):
                needed = [needed]
            for vname in needed:
                verifier_to_claims.setdefault(vname, []).extend(clist)

        # ---------- 去重（按 claim_id）----------
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
        for result in results:   # result 是 {claim_id: [issues]}
            for cid, issues in result.items():
                claim_issues[cid].extend(issues)

        # 构建 ClaimResult 列表，并合并 issues
        claim_results = []
        for c in claims:
            merged = self._merge_issues(claim_issues[c.claim_id])
            claim_results.append(ClaimResult(claim_id=c.claim_id, issues=merged))
        return claim_results

    async def _run_batch(self, verifier_name: str, claims: List[Claim]) -> Dict[str, List[ClaimIssue]]:
        """一次性处理多个 claims，返回 {claim_id: [issues]}，并保证类型安全"""
        agent = self.verifiers[verifier_name]
        payload = {"claims": [c.to_dict() for c in claims]}

        try:
            msg = Msg(role="user", content=json.dumps(payload, ensure_ascii=False), name="Verifier")
            response_msg = await call_agent_with_retry(agent, msg)
            text = _extract_text_response(response_msg)

            try:
                data = _safe_parse_json(text)
            except (json.JSONDecodeError, ValueError) as e:
                await append_verifier_trace_log(
                    f"{verifier_name.upper()} PARSE ERROR",
                    f"JSON parse failed: {e}",
                    payload=text[:500]
                )
                return {c.claim_id: [] for c in claims}  # 解析失败，返回空 issues

            if isinstance(data, list):
                issues_list = data
            elif isinstance(data, dict):
                issues_list = data.get("issues", [])
            else:
                issues_list = []

            result: Dict[str, List[ClaimIssue]] = {c.claim_id: [] for c in claims}
            for item in issues_list:
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

                # 类型安全处理
                issue_type = item.get("type")
                if not isinstance(issue_type, str):
                    issue_type = "unknown"

                description = item.get("description", "")
                if not isinstance(description, str):
                    description = str(description)

                severity = item.get("severity")
                if severity and not isinstance(severity, str):
                    severity = "minor"

                # 转换 evidence
                evidence_spans = []
                ev_list = item.get("evidence", [])
                if isinstance(ev_list, list):
                    for ev in ev_list:
                        if isinstance(ev, dict):
                            evidence_spans.append(EvidenceSpan(
                                cite_id=ev.get("cite_id", ""),
                                text=ev.get("text", ""),
                                score=ev.get("score")
                            ))
                        elif isinstance(ev, EvidenceSpan):
                            evidence_spans.append(ev)

                # 统一 suggestion 为字典
                suggestion_raw = item.get("suggestion", {})
                if isinstance(suggestion_raw, str):
                    suggestion_dict = {"content": suggestion_raw}
                elif isinstance(suggestion_raw, dict):
                    suggestion_dict = suggestion_raw
                else:
                    suggestion_dict = {}

                issue = ClaimIssue(
                    claim_id=claim_id,
                    type=issue_type,
                    description=f"[{verifier_name}] {description}",
                    severity=self._normalize_severity(severity),
                    evidence=evidence_spans,
                    suggestion=suggestion_dict
                )
                result[claim_id].append(issue)

            return result

        except Exception as e:
            await append_verifier_trace_log(
                f"{verifier_name.upper()} BATCH ERROR",
                f"{type(e).__name__}: {e}",
                payload=traceback.format_exc(),
            )
            return {c.claim_id: [] for c in claims}

    # ---------- 合并 key 使用标准化 description ----------
    def _merge_issues(self, issues: List[ClaimIssue]) -> List[ClaimIssue]:
        merged = {}
        for iss in issues:
            # 标准化 description：去除首尾空格，转小写
            norm_desc = iss.description.strip().lower()
            key = (iss.type, norm_desc)
            if key not in merged:
                merged[key] = iss
            else:
                # 保留更高 severity 的 issue
                if self._severity_priority(iss.severity) > self._severity_priority(merged[key].severity):
                    merged[key].severity = iss.severity
                # 如果已有 issue 没有 suggestion，用新的补充
                if not merged[key].suggestion and iss.suggestion:
                    merged[key].suggestion = iss.suggestion
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

    async def verify(self, segment: str) -> List[ClaimIssue]:
        """主入口：segment → claims → batch verification → issues"""
        segment_for_extract = _strip_chart_references_for_claim_extract(segment) # 清洗chart id标记，防止图表引用标记被抽取成claim
        claims = await self.extractor.extract(segment_for_extract)
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
        
        claim_results = await self.router.verify_claims(claims)
        issues = []
        for cr in claim_results:
            issues.extend(cr.issues)
        return issues

    async def verify_with_claims(self, segment: str) -> Tuple[List[Claim], List[ClaimIssue]]:
        """新接口：返回提取的 claims 和对应的 issues，用于评估报告"""
        claims = await self.extractor.extract(segment)
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
        claim_results = await self.router.verify_claims(claims)
        issues = []
        for cr in claim_results:
            issues.extend(cr.issues)
        return claims, issues


# ---------------------------------------------------------------------------
# Claim-Centric Evaluation Functions
# ---------------------------------------------------------------------------
def evaluate_claims(claims: List[Claim], all_issues: List[ClaimIssue]) -> List[ClaimEvaluation]:
    issues_by_claim = {c.claim_id: [] for c in claims}
    for iss in all_issues:
        if iss.claim_id in issues_by_claim:
            issues_by_claim[iss.claim_id].append(iss)
        else:
            issues_by_claim[iss.claim_id] = [iss]

    def compute_signature(issues: List[ClaimIssue]) -> Dict[str, int]:
        return {
            "critical": sum(1 for i in issues if i.severity == "critical"),
            "major": sum(1 for i in issues if i.severity == "major"),
            "minor": sum(1 for i in issues if i.severity == "minor"),
        }

    def compute_claim_score(signature: Dict[str, int]) -> int:
        if signature["critical"] > 0:
            return 1
        if signature["major"] >= 2:
            return 2
        if signature["major"] == 1:
            return 3
        if signature["minor"] > 0:
            return 4
        return 5

    claim_evaluations = []
    for claim in claims:
        issues = issues_by_claim.get(claim.claim_id, [])
        signature = compute_signature(issues)
        claim_score = compute_claim_score(signature)

        # 生成 combined_suggestion
        combined_suggestion = ""
        critical_issues = [i for i in issues if i.severity == "critical"]
        major_issues = [i for i in issues if i.severity == "major"]
        if critical_issues:
            combined_suggestion = critical_issues[0].suggestion
        elif major_issues:
            combined_suggestion = major_issues[0].suggestion
        elif issues and issues[0].suggestion:
            combined_suggestion = issues[0].suggestion
        else:
            combined_suggestion = "请对照材料核实并修正该声明。"

        claim_evaluations.append(ClaimEvaluation(
            claim_id=claim.claim_id,
            original_text=claim.original_text,
            claim_score=claim_score,
            issues=issues,
            combined_suggestion=combined_suggestion,
            signature=signature,
        ))
    return claim_evaluations

def compute_segment_report(claim_evaluations: List[ClaimEvaluation], top_k: int = 3) -> SegmentEvaluationReport:
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
    
    # 基础平均分
    avg_score = sum(ce.claim_score for ce in claim_evaluations) / total_claims
    segment_score = int(avg_score * 20)
    
    # 惩罚：存在多个 critical 声明时降低分数
    critical_count = sum(1 for ce in claim_evaluations if ce.signature["critical"] > 0)
    if critical_count >= 2:
        segment_score = min(segment_score, 40)
    elif critical_count == 1:
        segment_score = min(segment_score, 60)
    
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
    
    # 通过条件：分数≥60 且没有 critical 声明
    passed = (segment_score >= 60) and (critical_count == 0)
    bad_claims = [ce for ce in claim_evaluations if ce.claim_score <= 3]

    # 筛选有问题的声明（score < 5）
    problematic_claims = [ce for ce in claim_evaluations if ce.claim_score < 5]
    bad_claims_count = len(problematic_claims)

     # 排序（按 signature 和 score）
    sorted_claims = sorted(
        problematic_claims,  # 只排序有问题的
        key=lambda ce: (
            -ce.signature["critical"],
            -ce.signature["major"],
            ce.claim_score,
            -len(ce.issues)
        )
    )
    priority = sorted_claims[:top_k]
    
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
        lines.append(f"### 声明 {idx} (ID: `{ce.claim_id}`, 得分: {ce.claim_score}/5, 严重问题: critical={ce.signature['critical']}, major={ce.signature['major']})")
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
