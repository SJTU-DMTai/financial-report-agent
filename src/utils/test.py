"""
Triple-Check Verifier — 论文级三路交叉验证系统
=================================================
Architecture:
    Claim Extractor → [FactChecker, NumericChecker, TemporalChecker] → ConsistencyFusion → Report + RewriteController

External API:
    verify_issues = await verify_segment_content(segment, short_term, long_term)

Author: AI Assistant (Claude Opus 4.6)
Date: 2026-03-23
"""

from __future__ import annotations

import asyncio
import json
import re
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("TripleCheckVerifier")


# ---------------------------------------------------------------------------
# §1  Enumerations & Data Models
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    """声明类型分类 — 决定路由到哪些检查器"""
    FACTUAL = "factual"           # 事实性断言 (人名/事件/属性)
    NUMERIC = "numeric"           # 数值/统计/计算断言
    TEMPORAL = "temporal"         # 时间/日期/时序断言
    FACTUAL_NUMERIC = "factual_numeric"
    FACTUAL_TEMPORAL = "factual_temporal"
    NUMERIC_TEMPORAL = "numeric_temporal"
    COMPOSITE = "composite"       # 三类均涉及


class Verdict(str, Enum):
    """验证判决"""
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"
    UNVERIFIABLE = "unverifiable"


class Severity(str, Enum):
    """错误严重程度"""
    CRITICAL = "critical"     # 核心事实错误，必须修正
    MAJOR = "major"           # 重要偏差，强烈建议修正
    MINOR = "minor"           # 轻微问题，可选修正
    INFO = "info"             # 信息提示，无需修正


class RewriteAction(str, Enum):
    """重写动作类型"""
    NO_ACTION = "no_action"
    REPLACE_VALUE = "replace_value"
    REPHRASE = "rephrase"
    ADD_QUALIFIER = "add_qualifier"
    REMOVE_CLAIM = "remove_claim"
    RESTRUCTURE = "restructure"


@dataclass
class Claim:
    """从文本中提取的单条声明"""
    claim_id: str
    claim_type: ClaimType
    original_text: str                   # 原始文本片段
    normalized_text: str                 # 标准化后的声明
    slots: Dict[str, Any] = field(default_factory=dict)
    # slots 示例:
    #   factual:  {"subject": "...", "predicate": "...", "object": "..."}
    #   numeric:  {"entity": "...", "metric": "...", "value": 123, "unit": "万"}
    #   temporal: {"event": "...", "time_expr": "2024-01", "relation": "before/after/during"}
    source_span: Tuple[int, int] = (0, 0)  # 在 segment 中的字符位置

    def involves_fact(self) -> bool:
        return self.claim_type in (
            ClaimType.FACTUAL, ClaimType.FACTUAL_NUMERIC,
            ClaimType.FACTUAL_TEMPORAL, ClaimType.COMPOSITE,
        )

    def involves_numeric(self) -> bool:
        return self.claim_type in (
            ClaimType.NUMERIC, ClaimType.FACTUAL_NUMERIC,
            ClaimType.NUMERIC_TEMPORAL, ClaimType.COMPOSITE,
        )

    def involves_temporal(self) -> bool:
        return self.claim_type in (
            ClaimType.TEMPORAL, ClaimType.FACTUAL_TEMPORAL,
            ClaimType.NUMERIC_TEMPORAL, ClaimType.COMPOSITE,
        )


@dataclass
class Evidence:
    """单条证据"""
    source: str               # 来源标识 (e.g., "short_term_memory", "long_term_kb", "calculation")
    content: str              # 证据文本
    relevance_score: float    # 相关性 [0, 1]
    reliability: float = 0.8  # 来源可信度 [0, 1]


@dataclass
class CheckResult:
    """单个检查器对单条 Claim 的检查结果"""
    checker_name: str
    claim_id: str
    verdict: Verdict
    confidence: float          # [0, 1]
    evidences: List[Evidence] = field(default_factory=list)
    reasoning: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    # detail 可存放检查器特有信息
    #   FactChecker:    {"matched_facts": [...], "contradiction_points": [...]}
    #   NumericChecker: {"expected_value": ..., "actual_value": ..., "tolerance": ...}
    #   TemporalChecker:{"timeline": [...], "conflict_pairs": [...]}


@dataclass
class FusedResult:
    """融合后的最终验证结果（单条 Claim）"""
    claim: Claim
    final_verdict: Verdict
    final_confidence: float
    severity: Severity
    check_results: List[CheckResult] = field(default_factory=list)
    conflict_resolution_note: str = ""
    suggested_correction: str = ""
    rewrite_action: RewriteAction = RewriteAction.NO_ACTION


@dataclass
class VerificationReport:
    """整个 Segment 的验证报告"""
    segment_text: str
    overall_score: float                       # [0, 100]
    total_claims: int
    issues: List[FusedResult] = field(default_factory=list)  # 仅含有问题的
    all_results: List[FusedResult] = field(default_factory=list)
    summary: str = ""
    rewrite_needed: bool = False
    rewrite_instructions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_text": self.segment_text,
            "overall_score": self.overall_score,
            "total_claims": self.total_claims,
            "issues_count": len(self.issues),
            "issues": [
                {
                    "claim_id": r.claim.claim_id,
                    "original_text": r.claim.original_text,
                    "verdict": r.final_verdict.value,
                    "confidence": round(r.final_confidence, 3),
                    "severity": r.severity.value,
                    "correction": r.suggested_correction,
                    "rewrite_action": r.rewrite_action.value,
                    "details": [
                        {
                            "checker": cr.checker_name,
                            "verdict": cr.verdict.value,
                            "confidence": round(cr.confidence, 3),
                            "reasoning": cr.reasoning,
                        }
                        for cr in r.check_results
                    ],
                    "conflict_note": r.conflict_resolution_note,
                }
                for r in self.issues
            ],
            "summary": self.summary,
            "rewrite_needed": self.rewrite_needed,
            "rewrite_instructions": self.rewrite_instructions,
        }


# ---------------------------------------------------------------------------
# §2  LLM Abstraction Layer
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """LLM 调用抽象层 — 可替换为 OpenAI / Anthropic / 本地模型"""

    @abstractmethod
    async def generate(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        ...


class MockLLMProvider(LLMProvider):
    """
    基于规则的 Mock LLM — 用于无外部 API 时的本地运行与测试。
    在生产环境中替换为真实 LLM 调用。
    """

    async def generate(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        # 简单启发式：解析 prompt 中的 JSON 指令并返回合理结构
        await asyncio.sleep(0.01)  # 模拟异步 I/O

        if "extract_claims" in system.lower() or "extract claims" in prompt.lower():
            return self._mock_claim_extraction(prompt)
        if "fact_check" in system.lower() or "fact check" in prompt.lower():
            return self._mock_fact_check(prompt)
        if "numeric_check" in system.lower() or "numeric check" in prompt.lower():
            return self._mock_numeric_check(prompt)
        if "temporal_check" in system.lower() or "temporal check" in prompt.lower():
            return self._mock_temporal_check(prompt)
        if "suggest_correction" in system.lower() or "correction" in prompt.lower():
            return self._mock_correction(prompt)

        return '{"result": "no_op"}'

    def _mock_claim_extraction(self, prompt: str) -> str:
        """模拟声明提取 — 基于简单文本分析"""
        claims = []
        sentences = re.split(r'[。！？\.\!\?]', prompt)
        claim_idx = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 5:
                continue

            claim_type = "factual"
            slots: Dict[str, Any] = {}

            # 检测数值
            num_patterns = re.findall(
                r'(\d+[\d,\.]*)\s*(万|亿|千|百|%|美元|元|人|吨|km|公里|米|年)?', sent
            )
            has_numeric = len(num_patterns) > 0

            # 检测时间
            time_patterns = re.findall(
                r'(\d{4})\s*年|(\d{1,2})\s*月|(\d{1,2})\s*日|'
                r'(公元前?\d+)|'
                r'(前|后|之前|之后|期间|同时|早于|晚于)',
                sent
            )
            has_temporal = len(time_patterns) > 0

            if has_numeric and has_temporal:
                claim_type = "numeric_temporal"
            elif has_numeric:
                claim_type = "numeric"
                if num_patterns:
                    slots["value"] = num_patterns[0][0].replace(",", "")
                    slots["unit"] = num_patterns[0][1] if num_patterns[0][1] else ""
            elif has_temporal:
                claim_type = "temporal"
                for tp in time_patterns:
                    if tp[0]:
                        slots["year"] = tp[0]

            # 判定是否也含事实成分（大部分自然语句都有）
            if claim_type == "numeric":
                claim_type = "factual_numeric"
            elif claim_type == "temporal":
                claim_type = "factual_temporal"
            elif claim_type == "numeric_temporal":
                claim_type = "composite"
            else:
                claim_type = "factual"

            claims.append({
                "claim_id": f"c{claim_idx}",
                "claim_type": claim_type,
                "original_text": sent,
                "normalized_text": sent,
                "slots": slots,
            })
            claim_idx += 1

        return json.dumps({"claims": claims}, ensure_ascii=False)

    def _mock_fact_check(self, prompt: str) -> str:
        return json.dumps({
            "verdict": "supported",
            "confidence": 0.75,
            "reasoning": "Based on available context, the claim appears consistent with known information.",
            "evidences": []
        })

    def _mock_numeric_check(self, prompt: str) -> str:
        return json.dumps({
            "verdict": "supported",
            "confidence": 0.80,
            "reasoning": "Numeric values are within expected ranges based on context.",
            "detail": {}
        })

    def _mock_temporal_check(self, prompt: str) -> str:
        return json.dumps({
            "verdict": "supported",
            "confidence": 0.78,
            "reasoning": "Temporal ordering is consistent with known timeline.",
            "detail": {}
        })

    def _mock_correction(self, prompt: str) -> str:
        return json.dumps({
            "suggested_correction": "",
            "rewrite_action": "no_action"
        })


# ---------------------------------------------------------------------------
# §3  Memory / Knowledge Context
# ---------------------------------------------------------------------------

@dataclass
class MemoryContext:
    """
    统一的记忆上下文接口。
    short_term: 近期对话/写作上下文 (dict or object with .get()/.search())
    long_term:  长期知识库 (dict or object with .get()/.search())
    """
    short_term: Any = None
    long_term: Any = None

    def search_short_term(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """从短期记忆检索相关信息"""
        if self.short_term is None:
            return []
        if isinstance(self.short_term, dict):
            results = []
            for key, value in self.short_term.items():
                text = str(value)
                # 简单关键词匹配
                score = self._simple_relevance(query, text)
                if score > 0.1:
                    results.append({"key": key, "content": text, "score": score})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        if hasattr(self.short_term, "search"):
            return self.short_term.search(query, top_k=top_k)
        if isinstance(self.short_term, list):
            results = []
            for i, item in enumerate(self.short_term):
                text = str(item)
                score = self._simple_relevance(query, text)
                if score > 0.1:
                    results.append({"key": f"item_{i}", "content": text, "score": score})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        return []

    def search_long_term(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """从长期知识库检索相关信息"""
        if self.long_term is None:
            return []
        if isinstance(self.long_term, dict):
            results = []
            for key, value in self.long_term.items():
                text = str(value)
                score = self._simple_relevance(query, text)
                if score > 0.1:
                    results.append({"key": key, "content": text, "score": score})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        if hasattr(self.long_term, "search"):
            return self.long_term.search(query, top_k=top_k)
        if isinstance(self.long_term, list):
            results = []
            for i, item in enumerate(self.long_term):
                text = str(item)
                score = self._simple_relevance(query, text)
                if score > 0.1:
                    results.append({"key": f"item_{i}", "content": text, "score": score})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        return []

    @staticmethod
    def _simple_relevance(query: str, text: str) -> float:
        """
        基于字符级 n-gram 的简易相关性打分。
        生产环境应替换为 embedding cosine similarity。
        """
        if not query or not text:
            return 0.0
        query_lower = query.lower()
        text_lower = text.lower()

        # 词级 Jaccard
        q_tokens = set(re.findall(r'\w+', query_lower))
        t_tokens = set(re.findall(r'\w+', text_lower))
        if not q_tokens:
            return 0.0
        intersection = q_tokens & t_tokens
        union = q_tokens | t_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        # 子串匹配加成
        substr_bonus = 0.0
        for token in q_tokens:
            if token in text_lower and len(token) > 1:
                substr_bonus += 0.1
        substr_bonus = min(substr_bonus, 0.4)

        return min(jaccard + substr_bonus, 1.0)


# ---------------------------------------------------------------------------
# §4  Claim Extractor
# ---------------------------------------------------------------------------

class ClaimExtractor:
    """
    从文本段落中提取结构化声明。
    使用 LLM 进行语义分析 + 规则后处理。
    """

    SYSTEM_PROMPT = """You are a precise claim extractor. Extract all verifiable claims from the given text.
For each claim, identify:
1. claim_type: one of [factual, numeric, temporal, factual_numeric, factual_temporal, numeric_temporal, composite]
2. original_text: the exact text span
3. normalized_text: a clear, unambiguous restatement
4. slots: structured key-value pairs relevant to the claim type

Output JSON: {"claims": [{"claim_id": "c0", "claim_type": "...", "original_text": "...", "normalized_text": "...", "slots": {...}}, ...]}
Only output valid JSON. Do not include markdown formatting."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
    
    def _safe_json_load(self, raw: str) -> Dict:
        try:
            return json.loads(raw)
        except:
            matches = re.findall(r'\{.*?\}', raw, re.DOTALL)
            for m in matches:
                try:
                    return json.loads(m)
                except:
                    continue
        return {"claims": []}
    
    def _find_span(self, segment: str, text: str, start_pos=0):
        idx = segment.find(text, start_pos)
        if idx == -1:
            return (0, 0), start_pos
        return (idx, idx + len(text)), idx + len(text)
    
    def _split_atomic(self, claim: Claim) -> List[Claim]:
        text = claim.original_text
        results = []

        # 时间
        for y in re.findall(r'(\d{4})年', text):
            results.append(Claim(
                claim_id="",
                claim_type=ClaimType.TEMPORAL,
                original_text=f"{y}年",
                normalized_text=f"year={y}",
                slots={"year": int(y)}
            ))

        # 数值
        for m in re.finditer(r'(\d+[\d,\.]*)\s*(亿|万|%)', text):
            value = float(m.group(1).replace(",", ""))
            unit = m.group(2)

            normalized = value * (1e8 if unit == "亿" else 1e4 if unit == "万" else 1)

            results.append(Claim(
                claim_id="",
                claim_type=ClaimType.NUMERIC,
                original_text=m.group(0),
                normalized_text=m.group(0),
                slots={
                    "value": value,
                    "unit": unit,
                    "normalized": normalized
                }
            ))

        return results if results else [claim]
    
    async def extract(self, segment: str) -> List[Claim]:
        raw = await self.llm.generate(segment, system=self.SYSTEM_PROMPT)

        data = self._safe_json_load(raw)

        claims = []
        cursor = 0

        for item in data.get("claims", []):
            original = item.get("original_text", "")

            span, cursor = self._find_span(segment, original, cursor)

            base_claim = Claim(
                claim_id="",
                claim_type=ClaimType(item.get("claim_type", "factual")),
                original_text=original,
                normalized_text=item.get("normalized_text", original),
                slots=item.get("slots", {}),
                source_span=span
            )

            # atomic 拆分
            atomic = self._split_atomic(base_claim)
            claims.extend(atomic)

        claims = self._post_process_v2(claims)

        return claims
    
    def _post_process_v2(self, claims: List[Claim]) -> List[Claim]:
        filtered = []
        seen = set()

        for c in claims:
            key = (
                c.claim_type,
                json.dumps(c.slots, sort_keys=True)
            )

            if key in seen:
                continue
            seen.add(key)

            if len(c.original_text.strip()) < 2:
                continue

            c.claim_id = f"c{len(filtered)}"
            filtered.append(c)

        return filtered

#     async def extract(self, segment: str) -> List[Claim]:
#         """提取 Segment 中的所有可验证声明"""
#         prompt = f"""extract_claims

# Text to analyze:
# \"\"\"
# {segment}
# \"\"\"

# Extract all verifiable claims. Output JSON only."""

#         raw = await self.llm.generate(prompt, system=self.SYSTEM_PROMPT)
#         claims = self._parse_response(raw, segment)
#         claims = self._post_process(claims, segment)

#         logger.info(f"Extracted {len(claims)} claims from segment ({len(segment)} chars)")
#         return claims

    def _parse_response(self, raw: str, segment: str) -> List[Claim]:
        """解析 LLM 返回的 JSON"""
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)

            claims = []
            for item in data.get("claims", []):
                claim_type_str = item.get("claim_type", "factual")
                try:
                    claim_type = ClaimType(claim_type_str)
                except ValueError:
                    claim_type = ClaimType.FACTUAL

                original = item.get("original_text", "")
                # 计算 source_span
                start = segment.find(original)
                span = (start, start + len(original)) if start >= 0 else (0, 0)

                claims.append(Claim(
                    claim_id=item.get("claim_id", f"c{len(claims)}"),
                    claim_type=claim_type,
                    original_text=original,
                    normalized_text=item.get("normalized_text", original),
                    slots=item.get("slots", {}),
                    source_span=span,
                ))
            return claims

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse claim extraction response: {e}")
            return self._fallback_extract(segment)

    def _fallback_extract(self, segment: str) -> List[Claim]:
        """当 LLM 解析失败时的规则降级提取"""
        claims = []
        sentences = re.split(r'(?<=[。！？\.!\?])', segment)

        for idx, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 4:
                continue

            claim_type = self._classify_sentence(sent)
            slots = self._extract_slots(sent, claim_type)

            claims.append(Claim(
                claim_id=f"c{idx}",
                claim_type=claim_type,
                original_text=sent,
                normalized_text=sent,
                slots=slots,
                source_span=(segment.find(sent), segment.find(sent) + len(sent)),
            ))

        return claims

    def _classify_sentence(self, sent: str) -> ClaimType:
        """规则分类"""
        has_num = bool(re.search(r'\d+[\d,\.]*\s*(万|亿|千|百|%|美元|元|人|吨|km|公里)?', sent))
        has_time = bool(re.search(
            r'\d{4}\s*年|\d{1,2}\s*月|\d{1,2}\s*日|公元|世纪|前|后|期间', sent
        ))

        if has_num and has_time:
            return ClaimType.COMPOSITE
        elif has_num:
            return ClaimType.FACTUAL_NUMERIC
        elif has_time:
            return ClaimType.FACTUAL_TEMPORAL
        else:
            return ClaimType.FACTUAL

    def _extract_slots(self, sent: str, claim_type: ClaimType) -> Dict[str, Any]:
        """规则提取 slots"""
        slots: Dict[str, Any] = {}

        nums = re.findall(r'(\d+[\d,\.]*)\s*(万|亿|千|百|%|美元|元|人|吨|km|公里|米)?', sent)
        if nums:
            slots["values"] = [{"number": n[0].replace(",", ""), "unit": n[1]} for n in nums]

        times = re.findall(r'(\d{4})\s*年', sent)
        if times:
            slots["years"] = times

        months = re.findall(r'(\d{1,2})\s*月', sent)
        if months:
            slots["months"] = months

        return slots

    # def _post_process(self, claims: List[Claim], segment: str) -> List[Claim]:
    #     """后处理：去重、过滤过短声明、重新编号"""
    #     seen_texts = set()
    #     filtered = []
    #     for c in claims:
    #         key = c.normalized_text.strip()
    #         if key in seen_texts or len(key) < 3:
    #             continue
    #         seen_texts.add(key)
    #         c.claim_id = f"c{len(filtered)}"
    #         filtered.append(c)
    #     return filtered


# ---------------------------------------------------------------------------
# §5  Base Checker & Three Independent Checkers
# ---------------------------------------------------------------------------

class BaseChecker(ABC):
    """检查器基类"""

    def __init__(self, llm: LLMProvider, memory: MemoryContext):
        self.llm = llm
        self.memory = memory

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def check(self, claim: Claim) -> CheckResult:
        ...

    def _build_evidence_from_memory(
        self, query: str, source_label: str
    ) -> List[Evidence]:
        """从记忆上下文构建证据列表"""
        evidences = []

        for item in self.memory.search_short_term(query, top_k=3):
            evidences.append(Evidence(
                source=f"short_term:{item.get('key', '?')}",
                content=str(item.get("content", "")),
                relevance_score=float(item.get("score", 0.5)),
                reliability=0.85,
            ))

        for item in self.memory.search_long_term(query, top_k=3):
            evidences.append(Evidence(
                source=f"long_term:{item.get('key', '?')}",
                content=str(item.get("content", "")),
                relevance_score=float(item.get("score", 0.5)),
                reliability=0.90,
            ))

        return evidences


class FactChecker(BaseChecker):
    """
    事实验证检查器
    ─────────────
    多源检索 + LLM 推理验证事实性声明。
    验证方法：
      1. 从 short_term / long_term 检索相关事实
      2. 交叉比对证据
      3. LLM 推理最终判定
    """

    SYSTEM_PROMPT = """You are a rigorous fact checker. Given a claim and supporting evidence, determine:
1. verdict: one of [supported, refuted, uncertain, unverifiable]
2. confidence: float between 0 and 1
3. reasoning: detailed explanation

Fact check the following. Output JSON: {"verdict": "...", "confidence": 0.0, "reasoning": "...", "evidences": []}"""

    @property
    def name(self) -> str:
        return "FactChecker"

    async def check(self, claim: Claim) -> CheckResult:
        query = claim.normalized_text
        evidences = self._build_evidence_from_memory(query, "fact")

        evidence_text = "\n".join(
            f"[{e.source}] (rel={e.relevance_score:.2f}) {e.content}"
            for e in evidences
        ) or "No direct evidence found in memory."

        prompt = f"""Fact check the following claim.

Claim: {claim.normalized_text}
Original text: {claim.original_text}
Slots: {json.dumps(claim.slots, ensure_ascii=False)}

Available evidence:
{evidence_text}

Provide your assessment as JSON: {{"verdict": "...", "confidence": 0.0, "reasoning": "..."}}"""

        raw = await self.llm.generate(prompt, system=self.SYSTEM_PROMPT)

        try:
            json_match = re.search(r'\{[\s\S]*?\}', raw)
            data = json.loads(json_match.group()) if json_match else json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            data = {"verdict": "uncertain", "confidence": 0.5, "reasoning": "Parse failure"}

        verdict_str = data.get("verdict", "uncertain")
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.UNCERTAIN

        confidence = float(data.get("confidence", 0.5))

        # 根据证据质量调整置信度
        if evidences:
            avg_relevance = sum(e.relevance_score for e in evidences) / len(evidences)
            avg_reliability = sum(e.reliability for e in evidences) / len(evidences)
            evidence_factor = (avg_relevance * 0.4 + avg_reliability * 0.6)
            confidence = confidence * 0.6 + evidence_factor * 0.4
        else:
            # 无证据时降低置信度
            confidence = min(confidence, 0.4)

        return CheckResult(
            checker_name=self.name,
            claim_id=claim.claim_id,
            verdict=verdict,
            confidence=round(confidence, 4),
            evidences=evidences,
            reasoning=data.get("reasoning", ""),
            detail={"matched_facts": [], "contradiction_points": []},
        )


class NumericChecker(BaseChecker):
    """
    数值验证检查器
    ─────────────
    验证方法：
      1. 提取声明中的数值
      2. 从上下文获取参考值
      3. 数值计算验证（算术校验、量级校验、比例校验）
      4. 交叉比对多源数值
    """

    SYSTEM_PROMPT = """You are a numeric verification specialist. Given a claim with numeric values and context, verify:
1. Are the numbers factually correct?
2. Are calculations (if any) accurate?
3. Are the magnitudes reasonable?

Output JSON: {"verdict": "...", "confidence": 0.0, "reasoning": "...", "detail": {"expected_value": null, "actual_value": null, "tolerance": null}}"""

    @property
    def name(self) -> str:
        return "NumericChecker"

    async def check(self, claim: Claim) -> CheckResult:
        # 1. 提取数值
        numbers = self._extract_numbers(claim)

        if not numbers:
            return CheckResult(
                checker_name=self.name,
                claim_id=claim.claim_id,
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.9,
                reasoning="No numeric values found in claim.",
            )

        # 2. 上下文检索
        query = claim.normalized_text
        evidences = self._build_evidence_from_memory(query, "numeric")

        # 3. 本地数值校验
        local_issues = self._local_numeric_checks(numbers, claim)

        # 4. LLM 辅助验证
        evidence_text = "\n".join(
            f"[{e.source}] {e.content}" for e in evidences
        ) or "No reference data found."

        prompt = f"""Numeric check the following claim.

Claim: {claim.normalized_text}
Extracted numbers: {json.dumps(numbers, ensure_ascii=False)}
Local check issues: {json.dumps(local_issues, ensure_ascii=False)}

Reference data:
{evidence_text}

Output JSON: {{"verdict": "...", "confidence": 0.0, "reasoning": "...", "detail": {{}}}}"""

        raw = await self.llm.generate(prompt, system=self.SYSTEM_PROMPT)

        try:
            json_match = re.search(r'\{[\s\S]*?\}', raw)
            data = json.loads(json_match.group()) if json_match else json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            data = {"verdict": "uncertain", "confidence": 0.5, "reasoning": "Parse failure"}

        verdict_str = data.get("verdict", "uncertain")
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.UNCERTAIN

        confidence = float(data.get("confidence", 0.5))

        # 如果本地检查发现问题，加权影响
        if local_issues:
            severity_weight = sum(
                1.0 if iss.get("severity") == "critical" else 0.5
                for iss in local_issues
            )
            if severity_weight > 0 and verdict == Verdict.SUPPORTED:
                verdict = Verdict.UNCERTAIN
                confidence *= 0.6

        return CheckResult(
            checker_name=self.name,
            claim_id=claim.claim_id,
            verdict=verdict,
            confidence=round(confidence, 4),
            evidences=evidences,
            reasoning=data.get("reasoning", ""),
            detail={
                "extracted_numbers": numbers,
                "local_issues": local_issues,
                **(data.get("detail", {})),
            },
        )

    def _extract_numbers(self, claim: Claim) -> List[Dict[str, Any]]:
        """从声明中提取所有数值"""
        text = claim.original_text
        results = []

        # 带单位的数值
        patterns = [
            (r'(\d+[\d,]*\.?\d*)\s*(万亿|万|亿|千|百|%|美元|元|人|吨|km|公里|米|年|月|日|倍)',
             lambda m: {"raw": m.group(0), "value": float(m.group(1).replace(",", "")),
                        "unit": m.group(2)}),
            # 分数
            (r'(\d+)\s*/\s*(\d+)',
             lambda m: {"raw": m.group(0), "value": float(m.group(1)) / max(float(m.group(2)), 1),
                        "unit": "fraction"}),
            # 百分比
            (r'(\d+\.?\d*)\s*%',
             lambda m: {"raw": m.group(0), "value": float(m.group(1)), "unit": "%"}),
        ]

        seen_spans = set()
        for pattern, extractor in patterns:
            for m in re.finditer(pattern, text):
                span = (m.start(), m.end())
                if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                    continue
                seen_spans.add(span)
                try:
                    results.append(extractor(m))
                except (ValueError, ZeroDivisionError):
                    pass

        # 从 slots 补充
        if "value" in claim.slots:
            try:
                val = float(str(claim.slots["value"]).replace(",", ""))
                unit = claim.slots.get("unit", "")
                if not any(abs(r["value"] - val) < 0.001 for r in results):
                    results.append({"raw": f"{val}{unit}", "value": val, "unit": unit})
            except (ValueError, TypeError):
                pass

        if "values" in claim.slots:
            for v_item in claim.slots["values"]:
                try:
                    val = float(str(v_item.get("number", "0")).replace(",", ""))
                    unit = v_item.get("unit", "")
                    if not any(abs(r["value"] - val) < 0.001 for r in results):
                        results.append({"raw": f"{val}{unit}", "value": val, "unit": unit})
                except (ValueError, TypeError):
                    pass

        return results

    def _local_numeric_checks(
        self, numbers: List[Dict[str, Any]], claim: Claim
    ) -> List[Dict[str, str]]:
        """本地数值合理性检查"""
        issues = []

        for num_info in numbers:
            val = num_info["value"]
            unit = num_info.get("unit", "")

            # 量级异常检测
            if unit == "%" and (val < 0 or val > 100):
                # 允许超过 100% 的情况（增长率等），但标记
                if val > 1000 or val < -100:
                    issues.append({
                        "type": "magnitude_anomaly",
                        "severity": "major",
                        "detail": f"Percentage value {val}% seems extreme",
                        "number": num_info,
                    })

            if unit == "万亿" and val > 1000:
                issues.append({
                    "type": "magnitude_anomaly",
                    "severity": "minor",
                    "detail": f"Value {val}万亿 is unusually large",
                    "number": num_info,
                })

            if unit == "年" and val > 0:
                current_year = 2026
                if val > current_year + 50 or (val > 100 and val < 1000):
                    issues.append({
                        "type": "year_anomaly",
                        "severity": "major",
                        "detail": f"Year value {val} seems incorrect",
                        "number": num_info,
                    })

        # 内部一致性检查：如果有多个数值，检查是否存在矛盾
        if len(numbers) >= 2:
            percentage_nums = [n for n in numbers if n.get("unit") == "%"]
            if len(percentage_nums) >= 2:
                total = sum(n["value"] for n in percentage_nums)
                if total > 100.5:
                    issues.append({
                        "type": "sum_inconsistency",
                        "severity": "major",
                        "detail": f"Percentages sum to {total}% (>100%)",
                        "numbers": percentage_nums,
                    })

        return issues


class TemporalChecker(BaseChecker):
    """
    时间逻辑验证检查器
    ─────────────────
    验证方法：
      1. 提取时间表达式并规范化
      2. 构建事件时间线
      3. 检查时序一致性（因果关系、先后顺序）
      4. 与已知事件日期交叉验证
    """

    SYSTEM_PROMPT = """You are a temporal logic verification specialist. Given a claim with temporal expressions, verify:
1. Are dates/times factually correct?
2. Is the temporal ordering logically consistent?
3. Are there any anachronisms?

Output JSON: {"verdict": "...", "confidence": 0.0, "reasoning": "...", "detail": {"timeline": [], "conflict_pairs": []}}"""

    @property
    def name(self) -> str:
        return "TemporalChecker"

    async def check(self, claim: Claim) -> CheckResult:
        # 1. 提取时间表达式
        time_exprs = self._extract_temporal_expressions(claim)

        if not time_exprs:
            return CheckResult(
                checker_name=self.name,
                claim_id=claim.claim_id,
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.9,
                reasoning="No temporal expressions found in claim.",
            )

        # 2. 上下文检索
        query = claim.normalized_text
        evidences = self._build_evidence_from_memory(query, "temporal")

        # 3. 本地时序校验
        local_issues = self._local_temporal_checks(time_exprs, claim)

        # 4. LLM 辅助验证
        evidence_text = "\n".join(
            f"[{e.source}] {e.content}" for e in evidences
        ) or "No temporal reference data found."

        prompt = f"""Temporal check the following claim.

Claim: {claim.normalized_text}
Extracted temporal expressions: {json.dumps(time_exprs, ensure_ascii=False)}
Local check issues: {json.dumps(local_issues, ensure_ascii=False)}

Reference data:
{evidence_text}

Output JSON: {{"verdict": "...", "confidence": 0.0, "reasoning": "...", "detail": {{}}}}"""

        raw = await self.llm.generate(prompt, system=self.SYSTEM_PROMPT)

        try:
            json_match = re.search(r'\{[\s\S]*?\}', raw)
            data = json.loads(json_match.group()) if json_match else json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            data = {"verdict": "uncertain", "confidence": 0.5, "reasoning": "Parse failure"}

        verdict_str = data.get("verdict", "uncertain")
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.UNCERTAIN

        confidence = float(data.get("confidence", 0.5))

        # 本地问题影响
        if local_issues:
            severity_weight = sum(
                1.0 if iss.get("severity") == "critical" else 0.5
                for iss in local_issues
            )
            if severity_weight > 0:
                if verdict == Verdict.SUPPORTED:
                    verdict = Verdict.UNCERTAIN
                confidence = max(confidence * 0.5, 0.2)

        return CheckResult(
            checker_name=self.name,
            claim_id=claim.claim_id,
            verdict=verdict,
            confidence=round(confidence, 4),
            evidences=evidences,
            reasoning=data.get("reasoning", ""),
            detail={
                "time_expressions": time_exprs,
                "local_issues": local_issues,
                **(data.get("detail", {})),
            },
        )

    def _extract_temporal_expressions(self, claim: Claim) -> List[Dict[str, Any]]:
        """提取并规范化时间表达式"""
        text = claim.original_text
        exprs = []

        # 完整日期 YYYY年MM月DD日
        for m in re.finditer(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', text):
            exprs.append({
                "raw": m.group(0),
                "type": "date",
                "year": int(m.group(1)),
                "month": int(m.group(2)),
                "day": int(m.group(3)),
                "normalized": f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
            })

        # YYYY年MM月
        for m in re.finditer(r'(\d{4})\s*年\s*(\d{1,2})\s*月(?!\s*\d)', text):
            if not any(m.start() >= e.get("_start", -1) and m.end() <= e.get("_end", -1)
                       for e in exprs):
                exprs.append({
                    "raw": m.group(0),
                    "type": "year_month",
                    "year": int(m.group(1)),
                    "month": int(m.group(2)),
                    "normalized": f"{m.group(1)}-{int(m.group(2)):02d}",
                })

        # YYYY年
        for m in re.finditer(r'(\d{4})\s*年', text):
            year = int(m.group(1))
            already_covered = any(
                e.get("year") == year and e["type"] in ("date", "year_month")
                for e in exprs
            )
            if not already_covered:
                exprs.append({
                    "raw": m.group(0),
                    "type": "year",
                    "year": year,
                    "normalized": str(year),
                })

        # 相对时间
        for m in re.finditer(r'(之前|之后|以前|以后|前|后|期间|同时|早于|晚于)', text):
            exprs.append({
                "raw": m.group(0),
                "type": "relative",
                "relation": m.group(1),
            })

        # 世纪
        for m in re.finditer(r'(\d{1,2})\s*世纪', text):
            exprs.append({
                "raw": m.group(0),
                "type": "century",
                "century": int(m.group(1)),
                "year_range": ((int(m.group(1)) - 1) * 100 + 1, int(m.group(1)) * 100),
            })

        # 从 slots 补充
        if "years" in claim.slots:
            for y in claim.slots["years"]:
                year = int(y)
                if not any(e.get("year") == year for e in exprs):
                    exprs.append({
                        "raw": f"{year}年",
                        "type": "year",
                        "year": year,
                        "normalized": str(year),
                    })

        return exprs

    def _local_temporal_checks(
        self, time_exprs: List[Dict[str, Any]], claim: Claim
    ) -> List[Dict[str, str]]:
        """本地时序合理性检查"""
        issues = []
        current_year = 2026

        for expr in time_exprs:
            # 未来日期检查（如果是叙述过去事件）
            if expr.get("type") in ("date", "year_month", "year"):
                year = expr.get("year", 0)
                if year > current_year:
                    issues.append({
                        "type": "future_date",
                        "severity": "major",
                        "detail": f"Date {expr.get('normalized', expr['raw'])} is in the future",
                        "expression": expr,
                    })
                if year < 0:
                    issues.append({
                        "type": "invalid_date",
                        "severity": "critical",
                        "detail": f"Negative year: {year}",
                        "expression": expr,
                    })

            # 日期有效性
            if expr.get("type") == "date":
                try:
                    datetime(expr["year"], expr["month"], expr["day"])
                except ValueError:
                    issues.append({
                        "type": "invalid_date",
                        "severity": "critical",
                        "detail": f"Invalid date: {expr['normalized']}",
                        "expression": expr,
                    })

            if expr.get("type") == "year_month":
                if not (1 <= expr.get("month", 0) <= 12):
                    issues.append({
                        "type": "invalid_month",
                        "severity": "critical",
                        "detail": f"Invalid month: {expr.get('month')}",
                        "expression": expr,
                    })

        # 时序一致性：年份应该是递增的（在叙述中）
        years = [e.get("year") for e in time_exprs if e.get("year")]
        if len(years) >= 2:
            # 检查是否存在不合理的时间跳跃
            for i in range(len(years) - 1):
                gap = abs(years[i + 1] - years[i])
                if gap > 500:
                    issues.append({
                        "type": "large_time_gap",
                        "severity": "minor",
                        "detail": f"Large time gap: {years[i]} → {years[i+1]} ({gap} years)",
                    })

        return issues


# ---------------------------------------------------------------------------
# §6  Consistency Fusion Engine
# ---------------------------------------------------------------------------

class ConsistencyFusion:
    """
    三路结果融合引擎
    ─────────────
    融合策略：
      1. 加权投票 (权重 = checker_confidence × checker_reliability_prior)
      2. 冲突消解规则（硬性规则优先于软判断）
      3. 最终置信度 = 加权平均 × 一致性系数
    """

    # 各检查器的先验可靠性权重
    CHECKER_WEIGHTS = {
        "FactChecker": 0.40,
        "NumericChecker": 0.35,
        "TemporalChecker": 0.25,
    }

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def fuse(
        self, claim: Claim, check_results: List[CheckResult]
    ) -> FusedResult:
        """融合多个检查器的结果"""

        if not check_results:
            return FusedResult(
                claim=claim,
                final_verdict=Verdict.UNVERIFIABLE,
                final_confidence=0.0,
                severity=Severity.INFO,
            )

        # 过滤掉 UNVERIFIABLE 的结果（检查器认为不在其范围内）
        verifiable_results = [r for r in check_results if r.verdict != Verdict.UNVERIFIABLE]

        if not verifiable_results:
            return FusedResult(
                claim=claim,
                final_verdict=Verdict.UNVERIFIABLE,
                final_confidence=0.8,
                severity=Severity.INFO,
                check_results=check_results,
            )

        # 1. 加权投票
        verdict_scores: Dict[Verdict, float] = {v: 0.0 for v in Verdict}
        total_weight = 0.0

        for result in verifiable_results:
            weight = self.CHECKER_WEIGHTS.get(result.checker_name, 0.3)
            effective_weight = weight * result.confidence
            verdict_scores[result.verdict] += effective_weight
            total_weight += effective_weight

        # 2. 冲突检测
        verdicts_set = set(r.verdict for r in verifiable_results)
        has_conflict = (Verdict.SUPPORTED in verdicts_set and Verdict.REFUTED in verdicts_set)

        conflict_note = ""

        # 3. 冲突消解
        if has_conflict:
            conflict_note = self._resolve_conflict(verifiable_results)

            # 硬规则：如果 NumericChecker 说 REFUTED 且置信度 > 0.7，
            # 即使 FactChecker 说 SUPPORTED，也倾向 REFUTED
            numeric_results = [r for r in verifiable_results if r.checker_name == "NumericChecker"]
            if numeric_results and numeric_results[0].verdict == Verdict.REFUTED:
                if numeric_results[0].confidence > 0.7:
                    # 数值错误是硬错误，加倍权重
                    verdict_scores[Verdict.REFUTED] *= 1.5
                    conflict_note += " [NumericChecker override: numeric errors are hard errors]"

            # 硬规则：TemporalChecker 发现无效日期
            temporal_results = [r for r in verifiable_results if r.checker_name == "TemporalChecker"]
            if temporal_results:
                tr = temporal_results[0]
                local_issues = tr.detail.get("local_issues", [])
                critical_temporal = any(
                    iss.get("severity") == "critical" for iss in local_issues
                )
                if critical_temporal and tr.verdict == Verdict.REFUTED:
                    verdict_scores[Verdict.REFUTED] *= 1.5
                    conflict_note += " [TemporalChecker override: critical temporal error]"

        # 4. 确定最终判决
        if total_weight > 0:
            normalized_scores = {
                v: s / total_weight for v, s in verdict_scores.items()
            }
        else:
            normalized_scores = verdict_scores

        final_verdict = max(normalized_scores, key=normalized_scores.get)

        # 5. 计算最终置信度
        # 一致性系数：如果所有检查器一致，系数为 1；有冲突时降低
        if has_conflict:
            consistency_coeff = 0.65
        elif len(verdicts_set) == 1:
            consistency_coeff = 1.0
        else:
            consistency_coeff = 0.85

        raw_confidence = normalized_scores[final_verdict]
        final_confidence = raw_confidence * consistency_coeff

        # Clamp
        final_confidence = max(0.0, min(1.0, final_confidence))

        # 6. 确定严重程度
        severity = self._determine_severity(final_verdict, final_confidence, check_results)

        # 7. 如果 refuted 或 uncertain，生成修正建议
        suggested_correction = ""
        rewrite_action = RewriteAction.NO_ACTION

        if final_verdict in (Verdict.REFUTED, Verdict.UNCERTAIN):
            suggested_correction, rewrite_action = await self._generate_correction(
                claim, check_results, final_verdict
            )

        return FusedResult(
            claim=claim,
            final_verdict=final_verdict,
            final_confidence=round(final_confidence, 4),
            severity=severity,
            check_results=check_results,
            conflict_resolution_note=conflict_note,
            suggested_correction=suggested_correction,
            rewrite_action=rewrite_action,
        )

    def _resolve_conflict(self, results: List[CheckResult]) -> str:
        """生成冲突消解说明"""
        supported = [r for r in results if r.verdict == Verdict.SUPPORTED]
        refuted = [r for r in results if r.verdict == Verdict.REFUTED]

        note_parts = ["CONFLICT DETECTED:"]
        for r in supported:
            note_parts.append(f"  {r.checker_name} → SUPPORTED (conf={r.confidence:.2f}): {r.reasoning[:100]}")
        for r in refuted:
            note_parts.append(f"  {r.checker_name} → REFUTED (conf={r.confidence:.2f}): {r.reasoning[:100]}")

        return " | ".join(note_parts)

    def _determine_severity(
        self, verdict: Verdict, confidence: float, results: List[CheckResult]
    ) -> Severity:
        """确定错误严重程度"""
        if verdict == Verdict.SUPPORTED:
            return Severity.INFO

        if verdict == Verdict.UNVERIFIABLE:
            return Severity.INFO

        if verdict == Verdict.REFUTED:
            if confidence > 0.7:
                return Severity.CRITICAL
            elif confidence > 0.4:
                return Severity.MAJOR
            else:
                return Severity.MINOR

        # UNCERTAIN
        if confidence > 0.6:
            return Severity.MAJOR
        elif confidence > 0.3:
            return Severity.MINOR
        else:
            return Severity.INFO

    async def _generate_correction(
        self, claim: Claim, results: List[CheckResult], verdict: Verdict
    ) -> Tuple[str, RewriteAction]:
        """生成修正建议"""
        # 收集所有证据中的修正信息
        all_reasoning = "\n".join(
            f"[{r.checker_name}] {r.reasoning}" for r in results if r.reasoning
        )

        prompt = f"""Based on the verification results, suggest a correction for this claim.

Claim: {claim.original_text}
Verdict: {verdict.value}

Checker findings:
{all_reasoning}

Output JSON: {{"suggested_correction": "...", "rewrite_action": "replace_value|rephrase|add_qualifier|remove_claim|restructure|no_action"}}"""

        raw = await self.llm.generate(
            prompt,
            system="You suggest corrections for factual errors. Output JSON only. suggest_correction"
        )

        try:
            json_match = re.search(r'\{[\s\S]*?\}', raw)
            data = json.loads(json_match.group()) if json_match else json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            data = {}

        correction = data.get("suggested_correction", "")
        action_str = data.get("rewrite_action", "no_action")
        try:
            action = RewriteAction(action_str)
        except ValueError:
            action = RewriteAction.REPHRASE if verdict == Verdict.REFUTED else RewriteAction.ADD_QUALIFIER

        return correction, action

# # ---------------------------------------------------------------------------
# # §5  Base Checker & Three Independent Checkers
# # ---------------------------------------------------------------------------

# class BaseChecker(ABC):
#     """检查器基类"""

#     def __init__(self, llm: LLMProvider, memory: MemoryContext):
#         self.llm = llm
#         self.memory = memory

#     @property
#     @abstractmethod
#     def name(self) -> str:
#         ...

#     @abstractmethod
#     async def check(self, claim: Claim) -> CheckResult:
#         ...

#     def _build_evidence_from_memory(
#         self, query: str, source_label: str
#     ) -> List[Evidence]:
#         """从记忆上下文构建证据列表"""
#         evidences = []

#         for item in self.memory.search_short_term(query, top_k=3):
#             evidences.append(Evidence(
#                 source=f"short_term:{item.get('key', '?')}",
#                 content=str(item.get("content", "")),
#                 relevance_score=float(item.get("score", 0.5)),
#                 reliability=0.85,
#             ))

#         for item in self.memory.search_long_term(query, top_k=3):
#             evidences.append(Evidence(
#                 source=f"long_term:{item.get('key', '?')}",
#                 content=str(item.get("content", "")),
#                 relevance_score=float(item.get("score", 0.5)),
#                 reliability=0.90,
#             ))

#         return evidences


# class FactChecker(BaseChecker):
#     """
#     事实验证检查器
#     ─────────────
#     多源检索 + LLM 推理验证事实性声明。
#     验证方法：
#       1. 从 short_term / long_term 检索相关事实
#       2. 交叉比对证据
#       3. LLM 推理最终判定
#     """

#     SYSTEM_PROMPT = """You are a rigorous fact checker. Given a claim and supporting evidence, determine:
# 1. verdict: one of [supported, refuted, uncertain, unverifiable]
# 2. confidence: float between 0 and 1
# 3. reasoning: detailed explanation

# Fact check the following. Output JSON: {"verdict": "...", "confidence": 0.0, "reasoning": "...", "evidences": []}"""

#     @property
#     def name(self) -> str:
#         return "FactChecker"

#     async def check(self, claim: Claim) -> CheckResult:
#         query = claim.normalized_text
#         evidences = self._build_evidence_from_memory(query, "fact")

#         evidence_text = "\n".join(
#             f"[{e.source}] (rel={e.relevance_score:.2f}) {e.content}"
#             for e in evidences
#         ) or "No direct evidence found in memory."

#         prompt = f"""Fact check the following claim.

# Claim: {claim.normalized_text}
# Original text: {claim.original_text}
# Slots: {json.dumps(claim.slots, ensure_ascii=False)}

# Available evidence:
# {evidence_text}

# Provide your assessment as JSON: {{"verdict": "...", "confidence": 0.0, "reasoning": "..."}}"""

#         raw = await self.llm.generate(prompt, system=self.SYSTEM_PROMPT)

#         try:
#             json_match = re.search(r'\{[\s\S]*?\}', raw)
#             data = json.loads(json_match.group()) if json_match else json.loads(raw)
#         except (json.JSONDecodeError, AttributeError):
#             data = {"verdict": "uncertain", "confidence": 0.5, "reasoning": "Parse failure"}

#         verdict_str = data.get("verdict", "uncertain")
#         try:
#             verdict = Verdict(verdict_str)
#         except ValueError:
#             verdict = Verdict.UNCERTAIN

#         confidence = float(data.get("confidence", 0.5))

#         # 根据证据质量调整置信度
#         if evidences:
#             avg_relevance = sum(e.relevance_score for e in evidences) / len(evidences)
#             avg_reliability = sum(e.reliability for e in evidences) / len(evidences)
#             evidence_factor = (avg_relevance * 0.4 + avg_reliability * 0.6)
#             confidence = confidence * 0.6 + evidence_factor * 0.4
#         else:
#             # 无证据时降低置信度
#             confidence = min(confidence, 0.4)

#         return CheckResult(
#             checker_name=self.name,
#             claim_id=claim.claim_id,
#             verdict=verdict,
#             confidence=round(confidence, 4),
#             evidences=evidences,
#             reasoning=data.get("reasoning", ""),
#             detail={"matched_facts": [], "contradiction_points": []},
#         )


# class NumericChecker(BaseChecker):
#     """
#     数值验证检查器
#     ─────────────
#     验证方法：
#       1. 提取声明中的数值
#       2. 从上下文获取参考值
#       3. 数值计算验证（算术校验、量级校验、比例校验）
#       4. 交叉比对多源数值
#     """

#     SYSTEM_PROMPT = """You are a numeric verification specialist. Given a claim with numeric values and context, verify:
# 1. Are the numbers factually correct?
# 2. Are calculations (if any) accurate?
# 3. Are the magnitudes reasonable?

# Output JSON: {"verdict": "...", "confidence": 0.0, "reasoning": "...", "detail": {"expected_value": null, "actual_value": null, "tolerance": null}}"""

#     @property
#     def name(self) -> str:
#         return "NumericChecker"

#     async def check(self, claim: Claim) -> CheckResult:
#         # 1. 提取数值
#         numbers = self._extract_numbers(claim)

#         if not numbers:
#             return CheckResult(
#                 checker_name=self.name,
#                 claim_id=claim.claim_id,
#                 verdict=Verdict.UNVERIFIABLE,
#                 confidence=0.9,
#                 reasoning="No numeric values found in claim.",
#             )

#         # 2. 上下文检索
#         query = claim.normalized_text
#         evidences = self._build_evidence_from_memory(query, "numeric")

#         # 3. 本地数值校验
#         local_issues = self._local_numeric_checks(numbers, claim)

#         # 4. LLM 辅助验证
#         evidence_text = "\n".join(
#             f"[{e.source}] {e.content}" for e in evidences
#         ) or "No reference data found."

#         prompt = f"""Numeric check the following claim.

# Claim: {claim.normalized_text}
# Extracted numbers: {json.dumps(numbers, ensure_ascii=False)}
# Local check issues: {json.dumps(local_issues, ensure_ascii=False)}

# Reference data:
# {evidence_text}

# Output JSON: {{"verdict": "...", "confidence": 0.0, "reasoning": "...", "detail": {{}}}}"""

#         raw = await self.llm.generate(prompt, system=self.SYSTEM_PROMPT)

#         try:
#             json_match = re.search(r'\{[\s\S]*?\}', raw)
#             data = json.loads(json_match.group()) if json_match else json.loads(raw)
#         except (json.JSONDecodeError, AttributeError):
#             data = {"verdict": "uncertain", "confidence": 0.5, "reasoning": "Parse failure"}

#         verdict_str = data.get("verdict", "uncertain")
#         try:
#             verdict = Verdict(verdict_str)
#         except ValueError:
#             verdict = Verdict.UNCERTAIN

#         confidence = float(data.get("confidence", 0.5))

#         # 如果本地检查发现问题，加权影响
#         if local_issues:
#             severity_weight = sum(
#                 1.0 if iss.get("severity") == "critical" else 0.5
#                 for iss in local_issues
#             )
#             if severity_weight > 0 and verdict == Verdict.SUPPORTED:
#                 verdict = Verdict.UNCERTAIN
#                 confidence *= 0.6

#         return CheckResult(
#             checker_name=self.name,
#             claim_id=claim.claim_id,
#             verdict=verdict,
#             confidence=round(confidence, 4),
#             evidences=evidences,
#             reasoning=data.get("reasoning", ""),
#             detail={
#                 "extracted_numbers": numbers,
#                 "local_issues": local_issues,
#                 **(data.get("detail", {})),
#             },
#         )

#     def _extract_numbers(self, claim: Claim) -> List[Dict[str, Any]]:
#         """从声明中提取所有数值"""
#         text = claim.original_text
#         results = []

#         # 带单位的数值
#         patterns = [
#             (r'(\d+[\d,]*\.?\d*)\s*(万亿|万|亿|千|百|%|美元|元|人|吨|km|公里|米|年|月|日|倍)',
#              lambda m: {"raw": m.group(0), "value": float(m.group(1).replace(",", "")),
#                         "unit": m.group(2)}),
#             # 分数
#             (r'(\d+)\s*/\s*(\d+)',
#              lambda m: {"raw": m.group(0), "value": float(m.group(1)) / max(float(m.group(2)), 1),
#                         "unit": "fraction"}),
#             # 百分比
#             (r'(\d+\.?\d*)\s*%',
#              lambda m: {"raw": m.group(0), "value": float(m.group(1)), "unit": "%"}),
#         ]

#         seen_spans = set()
#         for pattern, extractor in patterns:
#             for m in re.finditer(pattern, text):
#                 span = (m.start(), m.end())
#                 if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
#                     continue
#                 seen_spans.add(span)
#                 try:
#                     results.append(extractor(m))
#                 except (ValueError, ZeroDivisionError):
#                     pass

#         # 从 slots 补充
#         if "value" in claim.slots:
#             try:
#                 val = float(str(claim.slots["value"]).replace(",", ""))
#                 unit = claim.slots.get("unit", "")
#                 if not any(abs(r["value"] - val) < 0.001 for r in results):
#                     results.append({"raw": f"{val}{unit}", "value": val, "unit": unit})
#             except (ValueError, TypeError):
#                 pass

#         if "values" in claim.slots:
#             for v_item in claim.slots["values"]:
#                 try:
#                     val = float(str(v_item.get("number", "0")).replace(",", ""))
#                     unit = v_item.get("unit", "")
#                     if not any(abs(r["value"] - val) < 0.001 for r in results):
#                         results.append({"raw": f"{val}{unit}", "value": val, "unit": unit})
#                 except (ValueError, TypeError):
#                     pass

#         return results

#     def _local_numeric_checks(
#         self, numbers: List[Dict[str, Any]], claim: Claim
#     ) -> List[Dict[str, str]]:
#         """本地数值合理性检查"""
#         issues = []

#         for num_info in numbers:
#             val = num_info["value"]
#             unit = num_info.get("unit", "")

#             # 量级异常检测
#             if unit == "%" and (val < 0 or val > 100):
#                 # 允许超过 100% 的情况（增长率等），但标记
#                 if val > 1000 or val < -100:
#                     issues.append({
#                         "type": "magnitude_anomaly",
#                         "severity": "major",
#                         "detail": f"Percentage value {val}% seems extreme",
#                         "number": num_info,
#                     })

#             if unit == "万亿" and val > 1000:
#                 issues.append({
#                     "type": "magnitude_anomaly",
#                     "severity": "minor",
#                     "detail": f"Value {val}万亿 is unusually large",
#                     "number": num_info,
#                 })

#             if unit == "年" and val > 0:
#                 current_year = 2026
#                 if val > current_year + 50 or (val > 100 and val < 1000):
#                     issues.append({
#                         "type": "year_anomaly",
#                         "severity": "major",
#                         "detail": f"Year value {val} seems incorrect",
#                         "number": num_info,
#                     })

#         # 内部一致性检查：如果有多个数值，检查是否存在矛盾
#         if len(numbers) >= 2:
#             percentage_nums = [n for n in numbers if n.get("unit") == "%"]
#             if len(percentage_nums) >= 2:
#                 total = sum(n["value"] for n in percentage_nums)
#                 if total > 100.5:
#                     issues.append({
#                         "type": "sum_inconsistency",
#                         "severity": "major",
#                         "detail": f"Percentages sum to {total}% (>100%)",
#                         "numbers": percentage_nums,
#                     })

#         return issues


# class TemporalChecker(BaseChecker):
#     """
#     时间逻辑验证检查器
#     ─────────────────
#     验证方法：
#       1. 提取时间表达式并规范化
#       2. 构建事件时间线
#       3. 检查时序一致性（因果关系、先后顺序）
#       4. 与已知事件日期交叉验证
#     """

#     SYSTEM_PROMPT = """You are a temporal logic verification specialist. Given a claim with temporal expressions, verify:
# 1. Are dates/times factually correct?
# 2. Is the temporal ordering logically consistent?
# 3. Are there any anachronisms?

# Output JSON: {"verdict": "...", "confidence": 0.0, "reasoning": "...", "detail": {"timeline": [], "conflict_pairs": []}}"""

#     @property
#     def name(self) -> str:
#         return "TemporalChecker"

#     async def check(self, claim: Claim) -> CheckResult:
#         # 1. 提取时间表达式
#         time_exprs = self._extract_temporal_expressions(claim)

#         if not time_exprs:
#             return CheckResult(
#                 checker_name=self.name,
#                 claim_id=claim.claim_id,
#                 verdict=Verdict.UNVERIFIABLE,
#                 confidence=0.9,
#                 reasoning="No temporal expressions found in claim.",
#             )

#         # 2. 上下文检索
#         query = claim.normalized_text
#         evidences = self._build_evidence_from_memory(query, "temporal")

#         # 3. 本地时序校验
#         local_issues = self._local_temporal_checks(time_exprs, claim)

#         # 4. LLM 辅助验证
#         evidence_text = "\n".join(
#             f"[{e.source}] {e.content}" for e in evidences
#         ) or "No temporal reference data found."

#         prompt = f"""Temporal check the following claim.

# Claim: {claim.normalized_text}
# Extracted temporal expressions: {json.dumps(time_exprs, ensure_ascii=False)}
# Local check issues: {json.dumps(local_issues, ensure_ascii=False)}

# Reference data:
# {evidence_text}

# Output JSON: {{"verdict": "...", "confidence": 0.0, "reasoning": "...", "detail": {{}}}}"""

#         raw = await self.llm.generate(prompt, system=self.SYSTEM_PROMPT)

#         try:
#             json_match = re.search(r'\{[\s\S]*?\}', raw)
#             data = json.loads(json_match.group()) if json_match else json.loads(raw)
#         except (json.JSONDecodeError, AttributeError):
#             data = {"verdict": "uncertain", "confidence": 0.5, "reasoning": "Parse failure"}

#         verdict_str = data.get("verdict", "uncertain")
#         try:
#             verdict = Verdict(verdict_str)
#         except ValueError:
#             verdict = Verdict.UNCERTAIN

#         confidence = float(data.get("confidence", 0.5))

#         # 本地问题影响
#         if local_issues:
#             severity_weight = sum(
#                 1.0 if iss.get("severity") == "critical" else 0.5
#                 for iss in local_issues
#             )
#             if severity_weight > 0:
#                 if verdict == Verdict.SUPPORTED:
#                     verdict = Verdict.UNCERTAIN
#                 confidence = max(confidence * 0.5, 0.2)

#         return CheckResult(
#             checker_name=self.name,
#             claim_id=claim.claim_id,
#             verdict=verdict,
#             confidence=round(confidence, 4),
#             evidences=evidences,
#             reasoning=data.get("reasoning", ""),
#             detail={
#                 "time_expressions": time_exprs,
#                 "local_issues": local_issues,
#                 **(data.get("detail", {})),
#             },
#         )

#     def _extract_temporal_expressions(self, claim: Claim) -> List[Dict[str, Any]]:
#         """提取并规范化时间表达式"""
#         text = claim.original_text
#         exprs = []

#         # 完整日期 YYYY年MM月DD日
#         for m in re.finditer(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', text):
#             exprs.append({
#                 "raw": m.group(0),
#                 "type": "date",
#                 "year": int(m.group(1)),
#                 "month": int(m.group(2)),
#                 "day": int(m.group(3)),
#                 "normalized": f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
#             })

#         # YYYY年MM月
#         for m in re.finditer(r'(\d{4})\s*年\s*(\d{1,2})\s*月(?!\s*\d)', text):
#             if not any(m.start() >= e.get("_start", -1) and m.end() <= e.get("_end", -1)
#                        for e in exprs):
#                 exprs.append({
#                     "raw": m.group(0),
#                     "type": "year_month",
#                     "year": int(m.group(1)),
#                     "month": int(m.group(2)),
#                     "normalized": f"{m.group(1)}-{int(m.group(2)):02d}",
#                 })

#         # YYYY年
#         for m in re.finditer(r'(\d{4})\s*年', text):
#             year = int(m.group(1))
#             already_covered = any(
#                 e.get("year") == year and e["type"] in ("date", "year_month")
#                 for e in exprs
#             )
#             if not already_covered:
#                 exprs.append({
#                     "raw": m.group(0),
#                     "type": "year",
#                     "year": year,
#                     "normalized": str(year),
#                 })

#         # 相对时间
#         for m in re.finditer(r'(之前|之后|以前|以后|前|后|期间|同时|早于|晚于)', text):
#             exprs.append({
#                 "raw": m.group(0),
#                 "type": "relative",
#                 "relation": m.group(1),
#             })

#         # 金融场景：同比/环比（优先提供给 LLM 用于时序语义判定）
#         # 这些不是“date/year/relative(之前/之后)”的简单形式，因此单独注入为相对时间锚点。
#         if re.search(r'(同比|YoY)', text):
#             exprs.append({
#                 "raw": "同比",
#                 "type": "relative",
#                 "relation": "yoy",
#                 "reference": "last_year",
#             })
#         if re.search(r'(环比|QoQ)', text):
#             exprs.append({
#                 "raw": "环比",
#                 "type": "relative",
#                 "relation": "qoq",
#                 "reference": "last_period",
#             })

#         # 世纪
#         for m in re.finditer(r'(\d{1,2})\s*世纪', text):
#             exprs.append({
#                 "raw": m.group(0),
#                 "type": "century",
#                 "century": int(m.group(1)),
#                 "year_range": ((int(m.group(1)) - 1) * 100 + 1, int(m.group(1)) * 100),
#             })

#         # 从 slots 补充（slots 统一为顶层 factual/numeric/temporal）
#         temporal = claim.slots.get("temporal", {}) if isinstance(claim.slots, dict) else {}
#         if isinstance(temporal, dict):
#             time_expr = temporal.get("time_expr")
#             if isinstance(time_expr, str) and time_expr:
#                 # YYYY-Qn
#                 m_q = re.match(r'^(\d{4})-Q([1-4])$', time_expr)
#                 if m_q:
#                     y, q = int(m_q.group(1)), int(m_q.group(2))
#                     exprs.append({
#                         "raw": time_expr,
#                         "type": "quarter",
#                         "year": y,
#                         "quarter": q,
#                         "normalized": f"{y}-Q{q}",
#                     })
#                     # 继续走下一个插入逻辑会重复判断；这里直接跳过其它时间粒度
#                     # （后续分支需要额外条件防止重复插入）

#                 # YYYY-Hn
#                 m_h = re.match(r'^(\d{4})-H([1-2])$', time_expr)
#                 if m_h:
#                     y, h = int(m_h.group(1)), int(m_h.group(2))
#                     exprs.append({
#                         "raw": time_expr,
#                         "type": "half_year",
#                         "year": y,
#                         "half": h,
#                         "normalized": f"{y}-H{h}",
#                     })
                    

#                 # rolling_3y / rolling_5y
#                 if time_expr.startswith("rolling_"):
#                     exprs.append({
#                         "raw": time_expr,
#                         "type": "rolling",
#                         "normalized": time_expr,
#                     })
                    

#                 # YYYY-MM-DD
#                 m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', time_expr)
#                 if m:
#                     y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
#                     exprs.append({
#                         "raw": time_expr,
#                         "type": "date",
#                         "year": y,
#                         "month": mo,
#                         "day": d,
#                         "normalized": f"{y}-{mo:02d}-{d:02d}",
#                     })
#                 # YYYY-MM
#                 m2 = re.match(r'^(\d{4})-(\d{2})$', time_expr)
#                 if not m and m2:
#                     y, mo = int(m2.group(1)), int(m2.group(2))
#                     exprs.append({
#                         "raw": time_expr,
#                         "type": "year_month",
#                         "year": y,
#                         "month": mo,
#                         "normalized": f"{y}-{mo:02d}",
#                     })
#                 # YYYY
#                 m3 = re.match(r'^(\d{4})$', time_expr)
#                 if not m and not m2 and m3:
#                     y = int(m3.group(1))
#                     if not any(e.get("year") == y for e in exprs):
#                         exprs.append({
#                             "raw": f"{y}年",
#                             "type": "year",
#                             "year": y,
#                             "normalized": str(y),
#                         })

#         return exprs

#     def _local_temporal_checks(
#         self, time_exprs: List[Dict[str, Any]], claim: Claim
#     ) -> List[Dict[str, str]]:
#         """本地时序合理性检查"""
#         issues = []
#         current_year = 2026

#         for expr in time_exprs:
#             # 未来日期检查（如果是叙述过去事件）
#             if expr.get("type") in ("date", "year_month", "year", "quarter", "half_year"):
#                 year = expr.get("year", 0)
#                 if year > current_year:
#                     issues.append({
#                         "type": "future_date",
#                         "severity": "major",
#                         "detail": f"Date {expr.get('normalized', expr['raw'])} is in the future",
#                         "expression": expr,
#                     })
#                 if year < 0:
#                     issues.append({
#                         "type": "invalid_date",
#                         "severity": "critical",
#                         "detail": f"Negative year: {year}",
#                         "expression": expr,
#                     })

#             # 日期有效性
#             if expr.get("type") == "date":
#                 try:
#                     datetime(expr["year"], expr["month"], expr["day"])
#                 except ValueError:
#                     issues.append({
#                         "type": "invalid_date",
#                         "severity": "critical",
#                         "detail": f"Invalid date: {expr['normalized']}",
#                         "expression": expr,
#                     })

#             if expr.get("type") == "year_month":
#                 if not (1 <= expr.get("month", 0) <= 12):
#                     issues.append({
#                         "type": "invalid_month",
#                         "severity": "critical",
#                         "detail": f"Invalid month: {expr.get('month')}",
#                         "expression": expr,
#                     })

#         # 时序一致性：年份应该是递增的（在叙述中）
#         years = [e.get("year") for e in time_exprs if e.get("year")]
#         if len(years) >= 2:
#             # 检查是否存在不合理的时间跳跃
#             for i in range(len(years) - 1):
#                 gap = abs(years[i + 1] - years[i])
#                 if gap > 500:
#                     issues.append({
#                         "type": "large_time_gap",
#                         "severity": "minor",
#                         "detail": f"Large time gap: {years[i]} → {years[i+1]} ({gap} years)",
#                     })

#         return issues


# # ---------------------------------------------------------------------------
# # §4  Claim Extractor
# # ---------------------------------------------------------------------------

# import re
# import json
# import logging
# from typing import Dict, Any, List, Optional, Tuple, Union
# from dataclasses import dataclass, field
# from enum import Enum
# from abc import ABC, abstractmethod

# # ---------- 基础类型定义 ----------
# class ClaimType(str, Enum):
#     FACTUAL = "factual"
#     NUMERIC = "numeric"
#     TEMPORAL = "temporal"
#     FACTUAL_NUMERIC = "factual_numeric"
#     FACTUAL_TEMPORAL = "factual_temporal"
#     NUMERIC_TEMPORAL = "numeric_temporal"
#     COMPOSITE = "composite"

# @dataclass
# class Claim:
#     claim_id: str
#     claim_type: ClaimType
#     original_text: str
#     normalized_text: str
#     slots: Dict[str, Any] = field(default_factory=dict)
#     source_span: Tuple[int, int] = (0, 0)
#     is_atomic: bool = False
#     parent_id: Optional[str] = None
#     cite_ids: List[str] = field(default_factory=list)

# class LLMProvider(ABC):
#     @abstractmethod
#     async def generate(self, prompt: str, system: Optional[str] = None) -> str:
#         pass

# # ---------- 日志配置 ----------
# logger = logging.getLogger(__name__)

# # ---------- 改进后的 ClaimExtractor ----------
# class ClaimExtractor:
#     """
#     从文本段落中提取结构化声明。
#     slots 最终格式：{"factual": {...}, "numeric": List[...], "temporal": List[...]}
#     原子拆分默认关闭，可通过 enable_atomic 开关控制。
#     """
#     CITE_RE = re.compile(r"\[\^cite_id:([A-Za-z0-9_\-]+)(?:\|[^\]]*)?\]")
    
#     SYSTEM_PROMPT = """You are a precise claim extractor. Extract all verifiable claims from the given text.
# For each claim, identify:
# 1. claim_type: one of [factual, numeric, temporal, factual_numeric, factual_temporal, numeric_temporal, composite]
# 2. original_text: the exact text span
# 3. normalized_text: a clear, unambiguous restatement
# 4. slots: structured key-value pairs relevant to the claim type
# 5. cite_ids: a list of reference IDs supporting this claim (from inline citations like [^cite_id:...]).

# slots schema:
# - factual: {"subject": "...", "predicate": "...", "object": "..."}
# - numeric: list of objects with the following fields:
#     * entity: the subject the number refers to (e.g., "公司A")
#     * metric: the measured quantity (e.g., "收入", "利润", "同比增长", "环比增长")
#     * value: numeric value (as number)
#     * unit: unit string (e.g., "万", "%", "元")
#     * period (optional): time period the number belongs to (e.g., "2025-Q1")
#     * comparison (optional): for growth metrics, provide object with:
#         - type: "yoy" or "qoq"
#         - base_period: the comparison period (e.g., "2024-Q1")
# - temporal: list of {"event": "...", "time_expr": "...", "relation": "before/after/during"}

# For composite or mixed claim types, use the same list structures.

# Output JSON: {"claims": [{"claim_id": "c0", "claim_type": "...", "original_text": "...", "normalized_text": "...", "slots": {...}, "cite_ids": [...]}, ...]}
# Only output valid JSON. Do not include markdown formatting
# """

#     def __init__(self, llm: LLMProvider, enable_atomic: bool = False):
#         self.llm = llm
#         self.enable_atomic = enable_atomic
#         # 扩展的数值单位列表（用于回退提取）
#         raw_units = [
#         '万辆', '亿辆', '万份', '亿份', '万吨', '亿吨', '万千瓦时', '亿千瓦时',
#         '万亿元', '亿元', '万元', '元', '美元', '人', '吨', 'km', '公里', '米', '倍',
#         '个百分点', '百分点', '%', '千', '百', '次', '小时', '天', '月', '年'
#     ]
#         self._numeric_units = sorted(raw_units, key=lambda x: -len(x))

#     # ---------- 改进的 JSON 解析 ----------
#     def _safe_json_load(self, raw: str) -> Dict[str, Any]:
#         """更健壮的 JSON 解析，处理 markdown 代码块和常见格式错误"""
#         # 移除可能的 markdown 代码块标记（支持 ```json ... ``` 或 ``` ... ```）
#         raw = re.sub(r'^```(?:json)?\s*\n?', '', raw, flags=re.MULTILINE)
#         raw = re.sub(r'\n?```\s*$', '', raw, flags=re.MULTILINE)
#         raw = raw.strip()

#         if not raw:
#             logger.warning("Empty response from LLM")
#             return {"claims": [], "__parse_failed": True}

#         # 尝试直接解析
#         try:
#             data = json.loads(raw)
#         except json.JSONDecodeError:
#             # 尝试提取第一个 JSON 对象或数组
#             match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw)
#             if match:
#                 try:
#                     data = json.loads(match.group(1))
#                 except json.JSONDecodeError as e:
#                     logger.warning(f"Failed to parse JSON from LLM response: {e}")
#                     return {"claims": [], "__parse_failed": True}
#             else:
#                 logger.warning("No JSON structure found in LLM response")
#                 return {"claims": [], "__parse_failed": True}

#         # 统一为 {"claims": [...]} 格式
#         if isinstance(data, list):
#             data = {"claims": data}
#         elif not isinstance(data, dict):
#             return {"claims": [], "__parse_failed": True}
#         return data

#     def _find_span(self, segment: str, text: str, start_pos: int = 0) -> Tuple[Tuple[int, int], int]:
#         """在 segment 中查找 text 的起止位置"""
#         if not text:
#             return (0, 0), start_pos
#         # 从 start_pos 开始查找
#         idx = segment.find(text, start_pos)
#         if idx != -1:
#             return (idx, idx + len(text)), idx + len(text)
#         # 如果完全匹配失败，尝试模糊匹配（去除空格）
#         stripped = re.sub(r'\s+', '', text)
#         seg_stripped = re.sub(r'\s+', '', segment[start_pos:])
#         pos = seg_stripped.find(stripped)
#         if pos != -1:
#             # 尝试映射回原始位置（粗略估计）
#             orig_idx = start_pos + pos
#             return (orig_idx, orig_idx + len(text)), orig_idx + len(text)
#         return (0, 0), start_pos

#     # ---------- 改进的实体推断 ----------
#     def _infer_entity(self, text: str) -> str:
#         """更通用的实体推断，匹配公司、人名、组织等常见模式"""
#         # 先尝试匹配公司模式（扩展）
#         patterns = [
#             r"([\u4e00-\u9fffA-Za-z0-9]{2,20}(?:公司|集团|股份|有限|责任|中心|研究所|大学))",
#             r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # 英文人名/组织
#             r"([\u4e00-\u9fff]{2,10}(?:先生|女士|教授|博士))",  # 中文称谓
#             r"([\u4e00-\u9fff]{2,10}(?:省|市|县|区|镇|乡))",  # 地名
#         ]
#         for pat in patterns:
#             m = re.search(pat, text)
#             if m:
#                 return m.group(1)
#         return ""

#     def _infer_metric(self, text_segment: str, full_text: str) -> str:
#         """
#         改进的指标推断，考虑上下文窗口，并避免误判。
#         """
#         if not text_segment:
#             return ""
#         # 局部窗口
#         window = text_segment
#         # 优先匹配特定指标
#         metric_keywords = [
#             ("销量", "销量"), ("收入", "收入"), ("利润", "利润"), ("净利润", "净利润"),
#             ("毛利率", "毛利率"), ("同比增长", "同比增长"), ("环比增长", "环比增长"),
#             ("增长", "增长"), ("下降", "下降"), ("减少", "减少"), ("增加", "增加")
#         ]
#         for kw, metric in metric_keywords:
#             if kw in window:
#                 return metric
#         # 如果文本包含同比/环比且当前数值是百分比，推断为增长率
#         if ("同比" in full_text or "环比" in full_text) and "%" in window:
#             return "同比增长" if "同比" in full_text else "环比增长"
#         return ""

#     # ---------- 数值槽位处理 ----------
#     def _coerce_numeric_slots(self, src: Any, original_text: str) -> List[Dict[str, Any]]:
#         """
#         将 LLM 返回的原始数值槽位转换为标准格式。
#         优先使用 LLM 提供的 entity, metric, comparison 字段，
#         仅当缺失时才用回退逻辑推断 entity 和 metric。
#         """
#         result = []

#         # 1. 提取 LLM 返回的数值项（保持原始结构）
#         if isinstance(src, list):
#             for item in src:
#                 if isinstance(item, dict) and "value" in item:
#                     numeric = {
#                         "entity": str(item.get("entity", "")),
#                         "metric": str(item.get("metric", "")),
#                         "value": item.get("value"),
#                         "unit": str(item.get("unit", "")),
#                         "period": str(item.get("period", "")),
#                         "comparison": item.get("comparison"),  # 直接保留 LLM 提供的 comparison
#                     }
#                     result.append(numeric)
#         elif isinstance(src, dict) and "value" in src:
#             result.append({
#                 "entity": str(src.get("entity", "")),
#                 "metric": str(src.get("metric", "")),
#                 "value": src.get("value"),
#                 "unit": str(src.get("unit", "")),
#                 "period": str(src.get("period", "")),
#                 "comparison": src.get("comparison"),
#             })
        
#          # 如果LLM未提供entity或metric，使用原有推断作为回退
#         if not result:
#             unit_pattern = r'(-?\d+[\d,\.]*)\s*(' + '|'.join(re.escape(u) for u in self._numeric_units) + r')'
#             matches = list(re.finditer(unit_pattern, original_text))
#             if not matches:
#                 # 如果没有匹配到单位，则提取所有数字（可能没有单位）
#                 num_pattern = r'(-?\d+[\d,\.]*)'
#                 matches = list(re.finditer(num_pattern, original_text))

#             for m in matches:
#                 value_str = m.group(1).replace(",", "")
#                 try:
#                     value = float(value_str)
#                     # 如果 value 是整数且没有小数点，保留为 int
#                     if value.is_integer():
#                         value = int(value)
#                 except (ValueError, TypeError):
#                     continue

#                 # 单位：如果有则取，否则为空
#                 unit = m.group(2) if len(m.groups()) > 1 else ""
#                 # 局部上下文
#                 start = max(0, m.start() - 20)
#                 end = min(len(original_text), m.end() + 20)
#                 local_text = original_text[start:end]
#                 metric = self._infer_metric(local_text, original_text)
#                 # 如果没有 metric，尝试从全局推断（比如“同比增长”）
#                 if not metric and unit == "%" and ("同比" in original_text or "环比" in original_text):
#                     metric = "同比增长" if "同比" in original_text else "环比增长"
#                 # 如果仍然没有 metric 且单位非空，尝试从单位推断（如“元”->“收入”）
#                 if not metric and unit:
#                     if unit in ["元", "美元", "万元", "亿元"]:
#                         metric = "金额"
#                     elif unit in ["%", "个百分点"]:
#                         metric = "百分比"
#                     elif unit in ["吨", "公斤", "斤"]:
#                         metric = "重量"
#                     elif unit in ["公里", "米", "km"]:
#                         metric = "长度"
#                 result.append({
#                     "entity": self._infer_entity(original_text),
#                     "metric": metric,
#                     "value": value,
#                     "unit": unit,
#                 })
#         else:
#             # 对每个数值补充period和comparison（仍由代码推断）
#             period = self._extract_time_period(original_text)
#             for item in result:
#                 if not item["entity"]:
#                     item["entity"] = self._infer_entity(original_text)  # 回退实体推断
#                 if not item["metric"]:
#                     item["metric"] = self._infer_metric(original_text, original_text)  # 回退指标推断
#                 item["period"] = period
#                 # if item.get("unit") == "%" and ("同比" in original_text or "环比" in original_text):
#                 #     comp_type = "yoy" if "同比" in original_text else "qoq"
#                 #     base_period = self._infer_base_period(period, comp_type)
#                 #     item["comparison"] = {
#                 #         "type": comp_type,
#                 #         "base_period": base_period
#                 #     }
#         return result


#     # ---------- 时间提取与规范化 ----------
#     def _extract_time_period(self, text: str) -> str:
#         """从文本中提取时间区间（改进版，优先匹配完整时间）"""
#         # 1. 完整日期
#         dm = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', text)
#         if dm:
#             y, mo, d = dm.group(1), dm.group(2), dm.group(3)
#             return f"{int(y)}-{int(mo):02d}-{int(d):02d}"

#         # 2. 季度（先处理中文）
#         cn_q = re.search(r'(\d{4})\s*年\s*第?([一二三四])季度', text)
#         if cn_q:
#             y = int(cn_q.group(1))
#             q_map = {'一':1, '二':2, '三':3, '四':4}
#             q = q_map.get(cn_q.group(2))
#             if q:
#                 return f"{y}-Q{q}"
#         # 英文季度
#         q_match = re.search(r'[Qq]([1-4])', text)
#         y_match = re.search(r'(\d{4})', text)
#         if q_match and y_match:
#             return f"{int(y_match.group(1))}-Q{int(q_match.group(1))}"

#         # 3. 半年
#         cn_h = re.search(r'(\d{4})\s*年\s*(上|下)半年', text)
#         if cn_h:
#             y = int(cn_h.group(1))
#             half = 'H1' if cn_h.group(2) == '上' else 'H2'
#             return f"{y}-{half}"
#         h_match = re.search(r'[Hh]([12])', text)
#         if h_match and y_match:
#             return f"{int(y_match.group(1))}-H{h_match.group(1)}"

#         # 4. 年月
#         ym = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月', text)
#         if ym:
#             y, mo = ym.group(1), ym.group(2)
#             return f"{int(y)}-{int(mo):02d}"

#         # 5. 年份
#         y_only = re.search(r'(\d{4})\s*年', text)
#         if y_only:
#             return str(int(y_only.group(1)))

#         # 6. 回退：纯数字年份
#         fallback = re.search(r'\b(\d{4})\b', text)
#         if fallback:
#             return fallback.group(1)

#         return ""

#     def _infer_base_period(self, period: str, comp_type: str) -> str:
#         """根据比较类型推断基期（改进，增加异常处理）"""
#         if not period:
#             return ""
#         try:
#             if comp_type == "yoy":
#                 if "-Q" in period:
#                     year, q = period.split("-Q")
#                     return f"{int(year)-1}-Q{q}"
#                 elif "-H" in period:
#                     year, h = period.split("-H")
#                     return f"{int(year)-1}-H{h}"
#                 elif "-" in period and len(period.split("-")) == 2:
#                     year, month = period.split("-")
#                     return f"{int(year)-1}-{month}"
#                 else:
#                     return str(int(period) - 1)
#             elif comp_type == "qoq":
#                 if "-Q" in period:
#                     year, q = period.split("-Q")
#                     q = int(q)
#                     if q == 1:
#                         return f"{int(year)-1}-Q4"
#                     else:
#                         return f"{year}-Q{q-1}"
#         except (ValueError, TypeError):
#             logger.warning(f"Invalid period format: {period} for comparison {comp_type}")
#         return ""

#     def _coerce_temporal_slots(self, src: Any, original_text: str) -> List[Dict[str, Any]]:
#         """将原始时间槽位转换为标准格式列表，并增强回退提取"""
#         result = []
#         if isinstance(src, dict) and "time_expr" in src:
#             result.append({
#                 "event": str(src.get("event", "")),
#                 "time_expr": src.get("time_expr", ""),
#                 "relation": str(src.get("relation", "during")),
#             })
#         elif isinstance(src, list):
#             for item in src:
#                 if isinstance(item, dict):
#                     result.append({
#                         "event": str(item.get("event", "")),
#                         "time_expr": item.get("time_expr", ""),
#                         "relation": str(item.get("relation", "during")),
#                     })

#         # 回退：从原文提取时间
#         if not result:
#             time_expr = self._extract_time_period(original_text)
#             if time_expr:
#                 result.append({"event": "", "time_expr": time_expr, "relation": "during"})

#         # 规范化时间表达式（如年份扩展为季度）
#         for t in result:
#             te = t["time_expr"]
#             if te and re.match(r'^\d{4}$', te):
#                 year = int(te)
#                 q_match = re.search(r'[Qq]([1-4])', original_text)
#                 if q_match:
#                     q_num = int(q_match.group(1))
#                     t["time_expr"] = f"{year}-Q{q_num}"
#                 elif re.search(r'上半年', original_text):
#                     t["time_expr"] = f"{year}-H1"
#                 elif re.search(r'下半年', original_text):
#                     t["time_expr"] = f"{year}-H2"
#             # 确保 relation 默认为 during
#             t["relation"] = "during"
#         return result

#     # ---------- 声明规范化 ----------
#     def _normalize_claim_slots(self, claim: Claim) -> Claim:
#         """规范化槽位，确保格式统一，并重建 normalized_text"""
#         raw = claim.slots if isinstance(claim.slots, dict) else {}
#         new_slots = {"factual": {}, "numeric": [], "temporal": []}

#         if claim.is_atomic:
#             # 保留原始 numeric 和 temporal 列表（已包含正确的 period, comparison 等）
#             numeric_list = raw.get("numeric", [])
#             temporal_list = raw.get("temporal", [])
#             # 确保每个数值项有 entity 和 metric（若缺失，可回退，但原子声明应由父声明提供）
#             # 这里不做覆盖，避免丢失 period 和 comparison
#             new_slots["numeric"] = numeric_list
#             new_slots["temporal"] = temporal_list
#             # 原子声明不保留 factual
#             new_slots["factual"] = {}
        
#         else:
#             # factual
#             factual_src = raw.get("factual") if isinstance(raw, dict) else None
#             if isinstance(factual_src, dict):
#                 subject = str(factual_src.get("subject", ""))
#                 predicate = str(factual_src.get("predicate", ""))
#                 obj = str(factual_src.get("object", ""))
#             else:
#                 subject = predicate = obj = ""
#             if not subject:
#                 subj, pred, o = self._simple_svo(claim.original_text)
#                 subject, predicate, obj = subj, pred, o
#             if not subject:
#                 subject = (claim.original_text or "").strip()[:20].strip() or "unknown"
#             new_slots["factual"] = {"subject": subject, "predicate": predicate, "object": obj}

#             # numeric
#             numeric_src = raw.get("numeric") if isinstance(raw, dict) else raw
#             numeric_list = self._coerce_numeric_slots(numeric_src, claim.original_text)
#             if numeric_list:
#                 new_slots["numeric"] = numeric_list

#             # temporal
#             temporal_src = raw.get("temporal") if isinstance(raw, dict) else raw
#             temporal_list = self._coerce_temporal_slots(temporal_src, claim.original_text)
#             if temporal_list:
#                 new_slots["temporal"] = temporal_list
        
#         claim.slots = new_slots

#         # 重建 normalized_text
#         lines = []

#         # 1. 数值槽位（可能多个）
#         for num in new_slots.get("numeric", []):
#             num_parts = []
#             if num.get("entity"):
#                 num_parts.append(f"entity={num['entity']}")
#             if num.get("metric"):
#                 num_parts.append(f"metric={num['metric']}")
#             if "value" in num:
#                 val = num["value"]
#                 # 保留原样，如果是整数则不带小数点
#                 val_str = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
#                 num_parts.append(f"value={val_str}")
#             if num.get("unit"):
#                 num_parts.append(f"unit={num['unit']}")
#             if num.get("period"):
#                 num_parts.append(f"period={num['period']}")

#             if num_parts:
#                 lines.append(f"NUM[{', '.join(num_parts)}]")

#             # 如果该数值带有 comparison 信息，单独输出 CMP 行
#             cmp = num.get("comparison")
#             if cmp and isinstance(cmp, dict):
#                 cmp_parts = []
#                 if cmp.get("type"):
#                     cmp_parts.append(f"type={cmp['type']}")
#                 if cmp.get("base_period"):
#                     cmp_parts.append(f"base_period={cmp['base_period']}")
#                 if cmp_parts:
#                     lines.append(f"CMP[{', '.join(cmp_parts)}]")

#         # 2. 时间槽位（可能多个）
#         for tmp in new_slots.get("temporal", []):
#             tmp_parts = []
#             if tmp.get("time_expr"):
#                 tmp_parts.append(f"time_expr={tmp['time_expr']}")
#             if tmp.get("event"):
#                 tmp_parts.append(f"event={tmp['event']}")
#             if tmp.get("relation"):
#                 tmp_parts.append(f"relation={tmp['relation']}")
#             if tmp_parts:
#                 lines.append(f"TMP[{', '.join(tmp_parts)}]")

#         # 3. 事实槽位
#         factual = new_slots.get("factual", {})
#         fact_parts = []
#         if factual.get("subject"):
#             fact_parts.append(f"subject={factual['subject']}")
#         if factual.get("predicate"):
#             fact_parts.append(f"predicate={factual['predicate']}")
#         if factual.get("object"):
#             fact_parts.append(f"object={factual['object']}")
#         if fact_parts:
#             lines.append(f"FAC[{', '.join(fact_parts)}]")

#         # 组装最终字符串
#         claim.normalized_text = "\n".join(lines) if lines else claim.original_text
#         return claim

#     def _simple_svo(self, text: str) -> Tuple[str, str, str]:
#         """简单的 SVO 提取（用于回退）"""
#         text = (text or "").strip()
#         m = re.search(r"(.+?)在.+?(增长|达到|为|下降)(.+)", text)
#         if m:
#             return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
#         m2 = re.search(r"(.+?)(是|为)(.+)", text)
#         if m2:
#             return m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()
#         return "", "", ""

#     # ---------- 原子拆分 ----------
#     # def _should_split_atomic(self, claim: Claim) -> bool:
#     #     """判断是否应该拆分为原子声明"""
#     #     text = claim.original_text
#     #     # 统计数值出现次数（带单位数值或百分比）
#     #     num_count = len(re.findall(r'\d+[\d,\.]*\s*(万|亿|%)', text))
#     #     # 统计年份出现次数
#     #     time_count = len(re.findall(r'\d{4}年', text))
#     #     # 典型财务句子：单一指标（1-2个数值、1个时间点）不拆分
#     #     if num_count <= 2 and time_count <= 1:
#     #         return False
#     #     return True
#     def _should_split_atomic(self, claim: Claim) -> bool:
#         """判断是否应该拆分为原子声明"""
#         text = claim.original_text
        
#         # 规则1: 包含多种信息的复合类型强制拆分
#         if claim.claim_type in [ClaimType.COMPOSITE, ClaimType.NUMERIC_TEMPORAL]:
#             # 检查是否真的包含多种信息
#             has_numeric = bool(re.search(r'\d+[\d,\.]*\s*(?:万|亿|%|元)', text))
#             has_temporal = bool(re.search(r'\d{4}年|\d{4}年\s*第?[一二三四]季度|\d{4}年\s*(?:上|下)半年', text))
            
#             # 如果同时包含数值和时间，需要拆分
#             if has_numeric and has_temporal:
#                 return True
        
#         # 规则2: COMPOSITE 且包含多个数值或时间点
#         if claim.claim_type == ClaimType.COMPOSITE:
#             num_count = len(re.findall(r'\d+[\d,\.]*\s*(?:万|亿|%)', text))
#             time_count = len(re.findall(r'\d{4}年', text))
            
#             # 多个数值或时间点时拆分
#             if num_count > 1 or time_count > 1:
#                 return True
    
#         return False
    
#     def _overlap(self, start: int, end: int, seen_spans: set) -> bool:
#         """判断区间 (start, end) 是否与 seen_spans 中的任何区间重叠"""
#         for (s0, e0) in seen_spans:
#             if s0 < end and e0 > start:
#                 return True
#         return False

#     def _split_atomic(self, claim: Claim) -> List[Claim]:
#         text = claim.original_text
#         base_offset = claim.source_span[0] if claim.source_span else 0
#         results = []
#         seen_spans = set()  # 记录已被原子占据的 span（包括时间和数值）

#         parent_numerics = claim.slots.get("numeric", []) if claim.slots else []
#         parent_temporals = claim.slots.get("temporal", []) if claim.slots else []
#         num_idx = 0
#         temp_idx = 0

#         # ---------- 1. 时间原子提取（主动屏蔽子匹配）----------
#         # 定义时间模式（按粒度从细到粗，便于后续优先级排序）
#         time_patterns = [
#             # 粒度优先级：数字越小越细
#             (0, r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', 
#             lambda m: f"{int(m.group(1))}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
#             (1, r'(\d{4})\s*年\s*第?([一二三四])季度', 
#             lambda m: f"{int(m.group(1))}-Q{ {'一':1,'二':2,'三':3,'四':4}.get(m.group(2), '') }"),
#             (2, r'(\d{4})\s*[Qq]([1-4])', 
#             lambda m: f"{int(m.group(1))}-Q{int(m.group(2))}"),
#             (3, r'(\d{4})\s*年\s*(上|下)半年', 
#             lambda m: f"{int(m.group(1))}-{'H1' if m.group(2)=='上' else 'H2'}"),
#             (4, r'(\d{4})\s*[Hh]([12])', 
#             lambda m: f"{int(m.group(1))}-H{int(m.group(2))}"),
#             (5, r'(\d{4})\s*年\s*(\d{1,2})\s*月', 
#             lambda m: f"{int(m.group(1))}-{int(m.group(2)):02d}"),
#             (6, r'(\d{4})\s*年', 
#             lambda m: str(int(m.group(1)))),
#         ]

#         # 收集所有匹配候选 (start, end, priority, formatter, matched_text)
#         candidates = []
#         for priority, pattern, formatter in time_patterns:
#             for m in re.finditer(pattern, text):
#                 candidates.append((m.start(), m.end(), priority, formatter, m))

#         # 按起始位置排序，然后选择最细粒度且不重叠的
#         candidates.sort(key=lambda x: (x[0], x[2]))  # 按 start 升序，同 start 按优先级细→粗（数字小优先）

#         selected = []
#         last_end = 0
#         for start, end, priority, formatter, m in candidates:
#             if start >= last_end:  # 不重叠
#                 selected.append((start, end, priority, formatter, m))
#                 last_end = end
#             # 否则跳过，因为细粒度的已经优先被选（排序保证了同 start 细优先）
        
#         # print("CANDIDATES:", [(text[s:e], p) for s,e,p,_,_ in candidates])
#         # print("SELECTED:", [text[s:e] for s,e,_,_,_ in selected])

#         # 根据选中的候选生成时间原子
#         for start, end, priority, formatter, m in selected:
#             time_expr = formatter(m)
#             # 从父声明按顺序获取 temporal 信息（保持原逻辑）
#             temp_info = parent_temporals[temp_idx] if temp_idx < len(parent_temporals) else {}
#             temp_idx += 1
#             atomic = Claim(
#                 claim_id="",
#                 claim_type=ClaimType.TEMPORAL,
#                 original_text=text[start:end],
#                 normalized_text=text[start:end],  # 临时，稍后会被覆盖
#                 slots={"temporal": [{
#                     "event": temp_info.get("event", ""),
#                     "time_expr": time_expr,
#                     "relation": temp_info.get("relation", "during")
#                 }]},
#                 source_span=(base_offset + start, base_offset + end),
#                 is_atomic=True,
#                 parent_id=claim.claim_id,
#                 cite_ids=claim.cite_ids, 
#             )
#             results.append(atomic)
#             seen_spans.add((start, end))

#         # ---------- 2. 数值原子提取（跳过已被时间占用的 span）----------
#         unit_pattern = r'(-?\d+[\d,\.]*)\s*(' + '|'.join(re.escape(u) for u in self._numeric_units) + r')'
#         for m in re.finditer(unit_pattern, text):
#             start, end = m.start(), m.end()
#             # 跳过已经被时间原子占用的区域
#             if self._overlap(start, end, seen_spans):
#                 continue
#             # 跳过单独的年份片段（例如“2025年”），因为这些已被时间原子覆盖
#             if re.fullmatch(r'\d{4}\s*年', text[start:end]):
#                 continue
#             seen_spans.add((start, end))

#             value_str = m.group(1).replace(",", "")
#             try:
#                 value = float(value_str)
#                 if value.is_integer():
#                     value = int(value)
#             except (ValueError, TypeError):
#                 continue
#             unit = m.group(2)

#             # 从父声明按顺序获取数值信息
#             num_info = parent_numerics[num_idx] if num_idx < len(parent_numerics) else {}
#             num_idx += 1

#             entity = num_info.get("entity", "")
#             metric = num_info.get("metric", "")
#             period = num_info.get("period", "")
#             comparison = num_info.get("comparison")  # 直接继承，不重新推断

#             # 回退推断（仅当父声明未提供时）
#             if not entity:
#                 entity = self._infer_entity(text)
#             if not metric:
#                 local_text = text[max(0, start-20):min(len(text), end+20)]
#                 metric = self._infer_metric(local_text, text)
#                 if not metric:
#                     # 单位推断
#                     if unit in ["元", "美元", "万元", "亿元"]:
#                         metric = "金额"
#                     elif unit in ["%", "个百分点"]:
#                         metric = "百分比"
#                     elif unit in ["吨", "公斤", "斤"]:
#                         metric = "重量"
#                     elif unit in ["公里", "米", "km"]:
#                         metric = "长度"
#             if not period:
#                 period = self._extract_time_period(text)

#             numeric_slot = {
#                 "entity": entity,
#                 "metric": metric,
#                 "value": value,
#                 "unit": unit,
#                 "period": period,
#             }
#             if comparison is not None:
#                 numeric_slot["comparison"] = comparison

#             atomic = Claim(
#                 claim_id="",
#                 claim_type=ClaimType.NUMERIC,
#                 original_text=text[start:end],
#                 normalized_text=text[start:end],
#                 slots={"numeric": [numeric_slot]},
#                 source_span=(base_offset + start, base_offset + end),
#                 is_atomic=True,
#                 parent_id=claim.claim_id,
#                 cite_ids=claim.cite_ids,
#             )
#             results.append(atomic)

#         return results

#     # ---------- 主提取方法 ----------
#     async def extract(self, segment: str, cite_ids: List[str]) -> List[Claim]:
#         """
#         从文本片段中提取声明（强制绑定 cite_ids）。
#         :param segment: 文本片段（应已去除引用标记）
#         :param cite_ids: 该片段绑定的引用ID列表（必须提供）
#         """
#         if cite_ids is None:
#             raise ValueError("extract() 必须提供 cite_ids，请先调用 split_text_with_citations 获得片段和引用ID")
        
#         # 清理文本中的引用标记（以防万一）
#         seg_to_use = self.CITE_RE.sub('', segment).strip()
#         if not seg_to_use:
#             return []  # 无内容，返回空列表

#         raw = await self.llm.generate(seg_to_use, system=self.SYSTEM_PROMPT)
#         data = self._safe_json_load(raw)
#         items = data.get("claims", [])

#         if data.get("__parse_failed") or not items:
#             # 回退提取
#             fallback = self._fallback_extract(seg_to_use, cite_ids=cite_ids)
#             fallback = [self._normalize_claim_slots(c) for c in fallback]
#             return self._post_process(fallback)

#         claims = []
#         cursor = 0
#         for idx, item in enumerate(items):
#             original = item.get("original_text", "")
#             span, cursor = self._find_span(seg_to_use, original, cursor)

#             claim_type_str = item.get("claim_type", "factual")
#             try:
#                 claim_type = ClaimType(claim_type_str)
#             except ValueError:
#                 claim_type = ClaimType.COMPOSITE

#             base_claim = Claim(
#                 claim_id=f"tmp_{idx}",
#                 claim_type=claim_type,
#                 original_text=original,
#                 normalized_text=item.get("normalized_text", original),
#                 slots=item.get("slots", {}),
#                 source_span=span,
#                 is_atomic=False,
#                 parent_id=None,
#                 cite_ids=cite_ids,   # 唯一来源
#             )
#             base_claim = self._normalize_claim_slots(base_claim)
#             claims.append(base_claim)

#             if self.enable_atomic and self._should_split_atomic(base_claim):
#                 atomic_claims = self._split_atomic(base_claim)
#                 for ac in atomic_claims:
#                     ac = self._normalize_claim_slots(ac)
#                     claims.append(ac)

#         return self._post_process(claims)

#     def _post_process(self, claims: List[Claim]) -> List[Claim]:
#         """去重、分配最终 ID、修复父子关系"""
#         filtered = []
#         seen = set()
#         id_map = {}

#         for c in claims:
#             key = (
#                 c.claim_type,
#                 c.is_atomic,
#                 json.dumps(c.slots, sort_keys=True, ensure_ascii=False),
#                 c.original_text[:30],
#                 c.source_span,
#             )
#             if key in seen:
#                 continue
#             seen.add(key)
#             if len(c.original_text.strip()) < 2:
#                 continue

#             old_id = c.claim_id
#             c.claim_id = f"c{len(filtered)}"
#             id_map[old_id] = c.claim_id
#             filtered.append(c)

#         for c in filtered:
#             if c.is_atomic and c.parent_id:
#                 c.parent_id = id_map.get(c.parent_id)
#                 if not c.parent_id:
#                     c.parent_id = None
        
#         # ========== 新增：时间原子冗余去重 ==========
#         if self.enable_atomic:
#             temporal_atoms = [c for c in filtered if c.claim_type == ClaimType.TEMPORAL and c.is_atomic]
#             if len(temporal_atoms) > 1:
#                 seen_time_expr = set()
#                 new_filtered = []
#                 for c in filtered:
#                     if c.claim_type == ClaimType.TEMPORAL and c.is_atomic:
#                         te = c.slots.get("temporal", [{}])[0].get("time_expr", "")
#                         if te in seen_time_expr:
#                             continue  # 跳过重复的时间原子
#                         seen_time_expr.add(te)
#                     new_filtered.append(c)
#                 filtered = new_filtered

#         return filtered

#     def _fallback_extract(self, segment: str, cite_ids: List[str]) -> List[Claim]:
#         """回退提取：整个片段作为一个复合声明"""
#         if not segment:
#             return []
#         claim = Claim(
#             claim_id="fallback_0",
#             claim_type=ClaimType.COMPOSITE,
#             original_text=segment,
#             normalized_text=segment,
#             slots={"factual": {}, "numeric": [], "temporal": []},
#             source_span=(0, len(segment)),
#             is_atomic=False,
#             cite_ids=cite_ids,
#         )
#         return [claim]

# class MockLLM:
#     async def generate(self, text, system=None):
#         # 假设文本已经用 split_text_with_citations 处理，每个片段传入对应 cite_ids
#         # 返回每个 claim 都带上 cite_ids
#         return json.dumps({
#             "claims": [
#                 {
#                     "claim_id": "c0",
#                     "claim_type": "composite",
#                     "original_text": "比亚迪2025年第一季度销量达到41万辆，同比增长72%，环比增长15%，市场占有率达到30%。",
#                     "normalized_text": "比亚迪2025年Q1销量41万辆，同比增长72%，环比增长15%，市场占有率30%",
#                     "slots": {
#                         "factual": {"subject": "比亚迪2025年第一季度销量", "predicate": "达到", "object": "41万辆"},
#                         "numeric": [
#                             {"entity": "比亚迪", "metric": "销量", "value": 41.0, "unit": "万辆", "period": "2025-Q1"},
#                             {"entity": "比亚迪", "metric": "同比增长", "value": 72.0, "unit": "%", "period": "2025-Q1",
#                              "comparison": {"type": "yoy", "base_period": "2024-Q1"}},
#                             {"entity": "比亚迪", "metric": "环比增长", "value": 15.0, "unit": "%", "period": "2025-Q1",
#                              "comparison": {"type": "qoq", "base_period": "2024-Q4"}},
#                             {"entity": "比亚迪", "metric": "市场占有率", "value": 30.0, "unit": "%", "period": "2025-Q1"}
#                         ],
#                         "temporal": [{"event": "", "time_expr": "2025-Q1", "relation": "during"}]
#                     },
#                     "cite_ids": ["search_engine_1", "akshare_3"]
#                 }
#             ]
#         })
    
# import asyncio

# async def test():
#     extractor = ClaimExtractor(MockLLM(), enable_atomic=True)

#     # # 情况1：无引用标记的文本（整段没有引用）
#     # text1 = "比亚迪2025年第一季度销量达到41万辆，同比增长72%，环比增长15%，市场占有率达到30%。"
#     # segments1 = split_text_with_citations(text1)  # 返回 [(text1, [])]
#     # all_claims1 = []
#     # for seg, cite_ids in segments1:
#     #     if seg:
#     #         # 传入空列表作为 cite_ids
#     #         claims = await extractor.extract(seg, cite_ids=cite_ids)
#     #         all_claims1.extend(claims)

#     # print("=== 无引用情况 ===")
#     # for c in all_claims1:
#     #     print("id:", c.claim_id)
#     #     print("type:", c.claim_type)
#     #     print("text:", c.original_text)
#     #     print("norm:", c.normalized_text)
#     #     print("slots:", json.dumps(c.slots, indent=2, ensure_ascii=False))
#     #     print("atomic:", c.is_atomic)
#     #     print(f"cite_ids: {c.cite_ids}")
#     #     print("---")

#     # 情况2：有引用标记的文本（按标记切分）
#     text2 = "比亚迪2025年第一季度销量达到41万辆[^cite_id:search_engine_1]，同比增长72%[^cite_id:akshare_3]，环比增长15%，市场占有率达到30%。"
#     segments2 = split_text_with_citations(text2)
#     all_claims2 = []
#     for seg, cite_ids in segments2:
#         if seg:
#             # 每个片段传入其绑定的引用ID
#             claims = await extractor.extract(seg, cite_ids=cite_ids)
#             all_claims2.extend(claims)

#     print("=== 有引用情况 ===")
#     for c in all_claims2:
#         print("id:", c.claim_id)
#         print("type:", c.claim_type)
#         print("text:", c.original_text)
#         print("norm:", c.normalized_text)
#         print("slots:", json.dumps(c.slots, indent=2, ensure_ascii=False))
#         print("atomic:", c.is_atomic)
#         print(f"cite_ids: {c.cite_ids}")
#         print("---")


# asyncio.run(test())