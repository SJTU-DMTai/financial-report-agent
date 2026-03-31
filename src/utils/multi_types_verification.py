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
from src.memory.short_term import MaterialType

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

        # 获取文本内容
        raw = response_msg.get_text_content()

        # print("\n====== RAW LLM OUTPUT ======")
        # print(type(raw))
        # print("====== END ======\n")

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
class EvidenceSpan:
    cite_id: str
    text: str
    score: Optional[float] = None   # 相关性/置信度

@dataclass
class ClaimIssue:
    type: str
    description: str
    severity: str
    evidence: List[EvidenceSpan] = field(default_factory=list)
    suggestion: Dict[str, Any] = field(default_factory=dict)

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
            for item in issues_list:
                cid = item.get("claim_id")
                if not cid or cid not in result:
                    continue

                result[cid].append(ClaimIssue(
                    type=item.get("type", "unknown"),
                    description=f"[{verifier_name}] {item.get('description', '')}",
                    severity=self._normalize_severity(item.get("severity")),
                    suggestion=item.get("suggestion", {}),
                    evidence=item.get("evidence", [])
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
#         """公司已逐步将产业链一体化优势转化为产品力优势。通过实现新能源汽车产业链的全覆盖，有效降低了原材料价格剧烈变化的风险，同时有效降低成本，提高生产效率，并因此拥有了一定的定价主动权 [^cite_id:2024-06-24_reference_report]。

# 从产品策略来看，公司已将成本端与企业运营端优势转化为产品力优势：2024年2月，王朝/海洋系列在短短两周内密集投放五波荣耀版车型，包括秦PLUS、驱逐舰05、海豚、汉、唐以及宋PLUS、宋Pro，覆盖从7.98万元到24.98万元的小型、紧凑型以及中大型车市场，较冠军版起售价最高降低了6万元，实现了"加量还降价" [^cite_id:2024-06-24_reference_report]。其中，秦PLUS荣耀版以"日系省油、德系驾驶、美系智能"的赞誉上市，首周便取得了23,590辆的新车订单 [^cite_id:2024-06-24_reference_report]。

# 2024年5月，公司开启了基于全新混动平台DM5.0的全新车型序列的逐步上市，率先上市的比亚迪秦L与海豹06定位紧凑型级别，但车身尺寸和轴距已经是中型轿车水平 [^cite_id:2024-06-24_reference_report]。该车型序列叠加同级领先的油耗、智能化与电气化水平，在"油电同价"的基础上进一步升级为"电比油低"，充分体现了公司对于10万至20万元细分市场的竞争力与志在必得的决心 [^cite_id:2024-06-24_reference_report]。上市不到2周的时间已经累计获得8万台订单，充分体现了消费者对该系列车型的充分认可 [^cite_id:2024-06-24_reference_report]。

# ![新车型与竞品核心参数对比](chart:chart_1774964357334)

# 数据显示，2026年5月上市的秦L DM-i与海豹06 DM-i在核心参数上显著优于同级别传统燃油竞品：WLTC综合油耗分别为1.11L/100km和1.36L/100km，远低于轩逸2024款经典（5.94L/100km）和朗逸2024款（5.92L/100km）[^cite_id:2024-06-24_reference_report]。车身尺寸方面，秦L与海豹06的轴距达到2790mm，已超过轩逸（2700mm）和朗逸（2688mm）的中型轿车水平 [^cite_id:2024-06-24_reference_report]。智能化配置上，秦L与海豹06标配倒车影像、定速巡航、车载智能系统及完整的手机App远程控制功能，而轩逸和朗逸在同价位车型中上述配置多数缺失 [^cite_id:2024-06-24_reference_report]。
# """
#     )

#     # short_term = ShortTermMemoryStore(base_dir=Path("D:/code/2026/Agent/financial-report-agent/tests/test_verifier_material"))

#     # long_term = LongTermMemoryStore(base_dir=Path("D:/code/2026/Agent/financial-report-agent/tests/long_term"))
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#     long_term_dir = PROJECT_ROOT / "data" / "memory" / "long_term"
    
#     long_term = LongTermMemoryStore(
#         base_dir=long_term_dir,
#     )

#     short_term_dir = PROJECT_ROOT / "data" / "memory" / "short_term" / "002594_20260331"

#     short_term = ShortTermMemoryStore(
#         base_dir=short_term_dir,
#     )

#     # # ---- 注册缺失的材料 ----
#     # cite_id = "2024-06-24_reference_report"
#     # material_path = short_term_dir / "material" / f"{cite_id}.txt"

#     # # 检查元数据中是否已有
#     # meta = short_term.get_material_meta(cite_id)
#     # if not meta:
#     #     if material_path.exists():
#     #         with open(material_path, "r", encoding="utf-8") as f:
#     #             content = f.read()
#     #         short_term.save_material(
#     #             cite_id=cite_id,
#     #             content=content,
#     #             description="比亚迪首次覆盖报告（2024-06-24）",
#     #             source="测试材料",
#     #         )
#     #         print(f"材料 {cite_id} 已注册")
#     #     else:
#     #         print(f"警告：材料文件 {material_path} 不存在，请检查路径")
#     # else:
#     #     print(f"材料 {cite_id} 已在元数据中")
    
#     verifier = SegmentVerifier(short_term=short_term, long_term=long_term)

#     async def test():
#         issues = await verifier.verify(segment)
#         print("\n=== 验证结果 ===")
#         for iss in issues:
#             print(f"[{iss.severity}] {iss.type} → {iss.description}")
#             print(f"  建议: {iss.suggestion}")
#             print(f" 证据：{iss.evidence}")

#     asyncio.run(test())