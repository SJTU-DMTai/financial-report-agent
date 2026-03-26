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
import logging
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("TripleCheckVerifier")

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

        # 使用 call_agent_with_retry 调用 agent
        response_msg = await call_agent_with_retry(self.agent, msg)

        # 获取文本内容（已正确解码）
        raw = response_msg.get_text_content()

        print("\n====== RAW LLM OUTPUT ======")
        print(type(raw))
        print(repr(raw[:200]))
        print("FIXED repr:", repr(fix_encoding(raw[:200])))
        print("====== END ======\n")

        # 安全解析 JSON
        try:
            data = _safe_parse_json(raw)
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}\nRaw text: {raw}")
            return []

        segments = data.get("segments", [])
        if not segments:
            logger.error("No segments returned from LLM")
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
        """verifiers: {"fact": agent, "numeric": agent, "temporal": agent}"""
        self.verifiers = verifiers

    async def verify_claims(self, claims: List[Claim]) -> List[ClaimResult]:
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
            tasks.append(self._run_batch(vname, vclaims))

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
    async def _run_batch(self, verifier_name: str, claims: List[Claim]) -> Dict[str, List[ClaimIssue]]:
        """一次性处理多个 claims，返回 {claim_id: [issues]}"""
        agent = self.verifiers[verifier_name]

        # 构建批量 payload
        payload = {
            "claims": [c.to_dict() for c in claims]
        }

        try:
            msg = Msg(
                role="user",
                content=json.dumps(payload, ensure_ascii=False),
                name="Verifier"
            )

            response_msg = await call_agent_with_retry(agent, msg)
            text = response_msg.get_text_content()

            print(f"\n====== [{verifier_name.upper()} BATCH RAW OUTPUT] ======")
            print("TEXT:", text[:500] + "..." if len(text) > 500 else text)
            print("====================================\n")

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
            for item in issues_list:
                cid = item.get("claim_id")
                if not cid or cid not in result:
                    continue
                result[cid].append(ClaimIssue(
                    type=item.get("type", "unknown"),
                    description=f"[{verifier_name}] {item.get('description', '')}",
                    severity=self._normalize_severity(item.get("severity")),
                    suggestion=item.get("suggestion", "")
                ))
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            # 出错时，为每个 claim 返回一个错误 issue
            result = {}
            for c in claims:
                result[c.claim_id] = [ClaimIssue(
                    type="verifier_runtime_error",
                    severity="major",
                    description=f"[{verifier_name}] verifier batch failed: {str(e)}",
                    suggestion="检查 verifier 输出格式（必须是 JSON）"
                )]
            return result

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
        claims = await self.extractor.extract(segment)
        if not claims:
            return []
        print("======== claims抽取已完成 =======")

        claim_results = await self.router.verify_claims(claims)

        # 展平所有 issues
        flat = []
        for cr in claim_results:
            flat.extend(cr.issues)
        return flat


# # ---------------------------------------------------------------------------
# # §6  Test / Demo
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     segment = (
#         "销量的爆发式增长直接转化为对固定成本的强力摊薄。随着产量从2019年的97.27万辆攀升至2023年的304.52万辆，公司的固定资产折旧、研发费用等固定成本被巨大的销量基数有效稀释。"
#         "数据显示，公司研发费用从2019年的56.29亿元增长至2023年的395.75亿元，增幅达6.03倍[^cite_id:002594_profit_按年度][^cite_id:002594_balance_按年度_1773318841]。"
#         "更关键的是，规模扩张显著增强了公司对上游供应商的议价能力。公司应付账款从2019年的361.68亿元激增至2023年的1,984.83亿元，增长4.49倍[^cite_id:002594_balance_按年度_1773318841][^cite_id:002594_profit_按年度]，"
#         "这直接反映了公司对上游供应商的强势地位，能够有效延长付款周期、压低采购成本。"
#     )

#     short_term = ShortTermMemoryStore(base_dir=Path("D:/code/2026/Agent/financial-report-agent/tests/test_verifier_material"))

    # # 保存测试材料
    # short_term.save_material(
    #     cite_id="002594_profit_按年度",
    #     content="公司研发费用从2019年的56.29亿元增长至2023年的395.75亿元，增幅达6.03倍。",
    #     description="研发费用年度数据",
    #     source="财报"
    # )
    # short_term.save_material(
    #     cite_id="002594_balance_按年度_1773318841",
    #     content="公司应付账款从2019年的361.68亿元激增至2023年的1,984.83亿元，增长4.49倍。",
    #     description="应付账款年度数据",
    #     source="财报"
    # )
    # short_term.save_material(
    #     cite_id="002594_balance_按年度",
    #     content="2019年产量97.27万辆，2023年产量304.52万辆，固定资产折旧、研发费用被摊薄。",
    #     description="产量和固定成本数据",
    #     source="财报"
    # )

    # long_term = LongTermMemoryStore(base_dir=Path("D:/code/2026/Agent/financial-report-agent/tests/long_term"))

    # verifier = SegmentVerifier(short_term=short_term, long_term=long_term)

    # async def test():
    #     issues = await verifier.verify(segment)
    #     print("\n=== 验证结果 ===")
    #     for iss in issues:
    #         print(f"[{iss.severity}] {iss.type} → {iss.description}")
    #         print(f"  建议: {iss.suggestion}")

    # asyncio.run(test())