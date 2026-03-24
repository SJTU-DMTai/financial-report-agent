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
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from agentscope.message import Msg
from agentscope.tool._coding._python import execute_python_code

from src.agents.verifier import create_verifier_agent, build_verifier_toolkit
from src.memory.short_term import ShortTermMemoryStore, MaterialType
from src.memory.long_term import LongTermMemoryStore
from src.utils.instance import create_chat_model, create_agent_formatter
from src.utils.call_with_retry import call_agent_with_retry
from src.prompt import prompt_dict

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("TripleCheckVerifier")



# ---------------------------------------------------------------------------
# § 1Memory / Knowledge Context
# ---------------------------------------------------------------------------

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
    
# ---------------------------------------------------------------------------
# §2  Claim Extractor - 完全由 LLM 驱动的原子声明提取器
# ---------------------------------------------------------------------------

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# ---------- 基础类型定义 ----------
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

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        pass

class AgentScopeLLM(LLMProvider):
    def __init__(self, model):
        self.model = model

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.model(messages)

        # 优先处理 ChatResponse
        if hasattr(response, "content"):
            content = response.content

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")

            # fallback
            return str(content)

        # OpenAI 兼容
        if hasattr(response, "choices"):
            return response.choices[0].message.content

        if isinstance(response, str):
            return response

        return str(response)

logger = logging.getLogger(__name__)

class ClaimExtractor:
    """
    从文本片段中提取原子声明（单一事实、单一数值、单一时间）。
    所有提取逻辑由 LLM 完成，本类只负责调用 LLM、解析 JSON、计算源位置和去重。
    """
    CITE_RE = re.compile(r"\[\^cite_id:([A-Za-z0-9_\-]+)(?:\|[^\]]*)?\]")

    def __init__(self, llm: LLMProvider, system_prompt: str):
        """
        :param llm: 大模型接口
        :param system_prompt: 系统提示词（应要求输出原子声明）
        """
        self.llm = llm
        self.system_prompt = system_prompt

    # ---------- JSON 解析 ----------
    def _safe_json_load(self, raw: str) -> Dict[str, Any]:
        """健壮的 JSON 解析，处理 markdown 代码块"""
        raw = re.sub(r'^```(?:json)?\s*\n?', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\n?```\s*$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()

        if not raw:
            logger.warning("Empty response from LLM")
            return {"claims": [], "__parse_failed": True}

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from LLM response: {e}")
                    return {"claims": [], "__parse_failed": True}
            else:
                logger.warning("No JSON structure found in LLM response")
                return {"claims": [], "__parse_failed": True}

        if isinstance(data, list):
            data = {"claims": data}
        elif not isinstance(data, dict):
            return {"claims": [], "__parse_failed": True}
        return data

    # ---------- 精确计算原文位置 ----------
    def _find_span(self, segment: str, text: str, start_pos: int = 0) -> Tuple[Tuple[int, int], int]:
        """
        在 segment 中查找 text 的起止位置（精确匹配）。
        返回 ((start, end), next_start)，用于顺序查找。
        """
        if not text:
            return (0, 0), start_pos
        idx = segment.find(text, start_pos)
        if idx != -1:
            return (idx, idx + len(text)), idx + len(text)
        # 若未找到，记录警告并返回默认值
        logger.warning(f"Could not find substring '{text[:50]}...' in segment (start_pos={start_pos})")
        return (0, 0), start_pos

    # ---------- 主提取方法 ----------
    async def extract(self, text: str) -> List[Claim]:
        if not text.strip():
            return []

        # 调 LLM（注意：不再去掉 cite）
        raw = await self.llm.generate(text, system=self.system_prompt)

        print("\n====== RAW LLM OUTPUT ======")
        print(raw)
        print("====== END ======\n")

        data = self._safe_json_load(raw)

        segments = data.get("segments", [])
        if not segments:
            logger.error("No segments returned from LLM")
            return []

        claims: List[Claim] = []

        # 遍历 segments
        for seg_idx, seg in enumerate(segments):
            seg_text = seg.get("text", "")
            seg_cite_ids = seg.get("cite_ids", [])

            for idx, item in enumerate(seg.get("claims", [])):
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
                    cite_ids=item.get("cite_ids", seg_cite_ids)  #  核心
                )

                claims.append(claim)

        return self._post_process(claims)

    # ---------- 后处理：去重 + ID 重编号 ----------
    def _post_process(self, claims: List[Claim]) -> List[Claim]:
        """
        去重（基于 claim_type + slots 序列化 + original_text 前50字符），
        并分配连续的 claim_id。
        """
        filtered = []
        seen = set()

        for c in claims:
            # 构造唯一标识：类型 + slots 序列化 + 原文片段
            key = (
                c.claim_type,
                json.dumps(c.slots, sort_keys=True, ensure_ascii=False),
                c.original_text[:50]
            )
            if key in seen:
                continue
            seen.add(key)

            # 过滤过于简短的声明（避免空值或单个字符）
            if len(c.original_text.strip()) < 2:
                continue

            c.claim_id = f"c{len(filtered)}"
            filtered.append(c)

        return filtered
    
# ---------------------------------------------------------------------------
# §3  Data Structure
# ---------------------------------------------------------------------------

@dataclass
class ClaimIssue:
    type: str
    description: str
    severity: str
    suggestion: str


@dataclass
class ClaimResult:
    claim_id: str
    issues: List[ClaimIssue]

# ---------------------------------------------------------------------------
# §4  Verifier Router + Issue Fusion
# ---------------------------------------------------------------------------

class VerifierRouter:
    def __init__(self, verifiers: Dict[str, any]):
        """
        verifiers:
        {
            "fact": agent,
            "numeric": agent,
            "temporal": agent
        }
        """
        self.verifiers = verifiers

    async def verify_claim(self, claim) -> ClaimResult:
        tasks = []

        # ===== 路由逻辑 =====
        if claim.claim_type in [ClaimType.FACTUAL, ClaimType.FACTUAL_NUMERIC, ClaimType.COMPOSITE]:
            tasks.append(self._run("fact", claim))

        if claim.claim_type in [ClaimType.NUMERIC, ClaimType.FACTUAL_NUMERIC, ClaimType.NUMERIC_TEMPORAL, ClaimType.COMPOSITE]:
            tasks.append(self._run("numeric", claim))

        if claim.claim_type in [ClaimType.TEMPORAL, ClaimType.NUMERIC_TEMPORAL, ClaimType.COMPOSITE]:
            tasks.append(self._run("temporal", claim))

        if not tasks:
            return ClaimResult(claim.claim_id, [])

        results = await asyncio.gather(*tasks)

        merged_issues = self._merge_results(results)

        return ClaimResult(
            claim_id=claim.claim_id,
            issues=merged_issues
        )

    # -----------------------------------------------------------------------
    # 单个 verifier 执行
    # -----------------------------------------------------------------------
    async def _run(self, verifier_name: str, claim):
        agent = self.verifiers[verifier_name]

        payload = {
            "original_text": claim.original_text,
            "normalized_text": claim.normalized_text,
            "slots": claim.slots,
            "cite_ids": claim.cite_ids
        }

        try:
            response = await agent(payload)

            print(f"\n====== [{verifier_name.upper()} RAW OUTPUT] ======")
            print(response)
            print("====================================\n")

            # 统一解析
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response

            #兼容 list / dict
            if isinstance(data, list):
                issues = data
            else:
                issues = data.get("issues", [])

            # 标准化
            normalized_issues = []
            for item in issues:
                desc = item.get("description", "")

                normalized_issues.append({
                    "type": item.get("type", "unknown"),
                    "description": f"[{verifier_name}] {desc}",
                    "severity": self._normalize_severity(item.get("severity")),
                    "suggestion": item.get("suggestion", "")
                })

            return {"issues": normalized_issues}

        except Exception as e:
            return {
                "issues": [
                    {
                        "type": "verifier_runtime_error",
                        "severity": "major",
                        "description": f"[{verifier_name}] verifier failed: {str(e)}",
                        "suggestion": "检查 verifier 输出格式"
                    }
                ]
            }

    # -----------------------------------------------------------------------
    # Issue Fusion（核心）
    # -----------------------------------------------------------------------
    def _merge_results(self, results: List[Dict]) -> List[ClaimIssue]:
        merged = {}

        for r in results:
            for item in r.get("issues", []):
                key = (
                    item.get("type"),
                    item.get("description")
                )

                severity = item.get("severity", "minor")

                if key not in merged:
                    merged[key] = ClaimIssue(
                        type=item.get("type", "unknown"),
                        description=item.get("description", ""),
                        severity=severity,
                        suggestion=item.get("suggestion", "")
                    )
                else:
                    prev = merged[key]

                    # severity 升级规则
                    if self._severity_priority(severity) > self._severity_priority(prev.severity):
                        prev.severity = severity

                    # suggestion 补充
                    if not prev.suggestion and item.get("suggestion"):
                        prev.suggestion = item.get("suggestion")

        final_issues = list(merged.values())

        print("\n====== MERGED ISSUES ======")
        for iss in final_issues:
            print(f"[{iss.severity}] {iss.type} → {iss.description}")
        print("===========================\n")

        return final_issues

    # -----------------------------------------------------------------------
    # 工具函数
    # -----------------------------------------------------------------------

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
        mapping = {
            "critical": 3,
            "major": 2,
            "minor": 1,
            "info": 0
        }
        return mapping.get(s, 1)

class SegmentVerifier:
    def __init__(self, short_term, long_term):
        self.model = create_chat_model()
        self.formatter = create_agent_formatter()
        self.llm = AgentScopeLLM(self.model)

        self.extractor = ClaimExtractor(
            llm=self.llm,
            system_prompt=prompt_dict["claim_extract_sys_prompt"]
        )

        self.verifiers = {}

        for name in ["fact", "numeric", "temporal"]:
            toolkit = build_verifier_toolkit(
                short_term=short_term,
                long_term=long_term
            )

            agent = create_verifier_agent(
                model=self.model,
                formatter=self.formatter,
                toolkit=toolkit,
                verifier_type=name
            )

            self.verifiers[name] = agent

        self.router = VerifierRouter(self.verifiers)

    async def verify(self, segment: str) -> List[ClaimIssue]:
        claims = await self.extractor.extract(segment)
        if not claims:
            return []

        tasks = [self.router.verify_claim(c) for c in claims]
        results = await asyncio.gather(*tasks)

        flat = []
        for r in results:
            flat.extend(r.issues)

        return flat


# async def verify_segment_content(
#     segment: str,
#     short_term: ShortTermMemoryStore,
#     long_term: LongTermMemoryStore,
# ) -> List[ClaimIssue]:
#     """
#     外部唯一入口：

#     segment → claim extraction → multi-verifier → issue fusion

#     return:
#         List[ClaimIssue]
#     """

#     # --------------------------------------------------
#     # 初始化模型 & LLM
#     # --------------------------------------------------
#     model = create_chat_model()
#     formatter = create_agent_formatter()
#     llm = AgentScopeLLM(model)

#     extractor = ClaimExtractor(
#         llm=llm,
#         system_prompt=prompt_dict["claim_extract_sys_prompt"]
#     )

#     # --------------------------------------------------
#     # 构建三个 Verifier Agents
#     # --------------------------------------------------
#     verifiers = {}

#     for name in ["fact", "numeric", "temporal"]:
#         toolkit = build_verifier_toolkit(
#             short_term=short_term,
#             long_term=long_term
#         )

#         agent = create_verifier_agent(
#             model=model,
#             formatter=formatter,
#             toolkit=toolkit,
#             verifier_type=name
#         )

#         verifiers[name] = agent

#     router = VerifierRouter(verifiers)

#     # --------------------------------------------------
#     # Claim Extraction
#     # --------------------------------------------------
#     claims: List[Claim] = await extractor.extract(segment)

#     if not claims:
#         return []

#     # --------------------------------------------------
#     # Claim-level 并行验证
#     # --------------------------------------------------
#     tasks = [router.verify_claim(claim) for claim in claims]

#     results: List[ClaimResult] = await asyncio.gather(*tasks)

#     # flatten
#     flat_issues: List[ClaimIssue] = []

#     for r in results:
#         flat_issues.extend(r.issues)

#         if not r.issues:
#             print("✅ No issues")
#             continue

#         for iss in r.issues:
#             print(f"[{iss.severity.upper()}] {iss.type}")
#             print(f"  描述: {iss.description}")
#             print(f"  严重性: {iss.severity}")
#             print(f"  建议: {iss.suggestion}")
#             print()


#     return flat_issues


# async def test_extract():
#     # 1️⃣ 创建模型
#     model = create_chat_model()
#     llm = AgentScopeLLM(model)

#     # 2️⃣ prompt
#     system_prompt = prompt_dict["claim_extract_sys_prompt"]

#     # 3️⃣ extractor
#     extractor = ClaimExtractor(llm, system_prompt)

#     # 4️⃣ 测试文本（关键：必须带 cite_id）
#     segment = (
#         "特斯拉在2025年第一季度销量为41万辆，同比增长12%[^cite_id:tesla_q1_2025]。"
#     )

#     try:
#         # 5️⃣ 执行提取（新版接口）
#         claims = await extractor.extract(segment)

#         # 👉 防御：过滤无 cite（强烈建议）
#         claims = [c for c in claims if c.cite_ids]

#         # 6️⃣ 打印结果
#         print(f"\n=== 提取结果：{len(claims)} 条 claims ===")

#         for claim in claims:
#             print("\n------------------------------")
#             print(f"ID: {claim.claim_id}")
#             print(f"Type: {claim.claim_type}")
#             print(f"Original: {claim.original_text}")
#             print(f"Normalized: {claim.normalized_text}")
#             print(f"Cite IDs: {claim.cite_ids}")
#             print(f"Slots:")
#             print(json.dumps(claim.slots, ensure_ascii=False, indent=2))

#     except Exception as e:
#         print(f"❌ 提取失败: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     asyncio.run(test_extract())