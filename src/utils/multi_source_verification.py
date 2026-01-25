# -*- coding: utf-8 -*-

from dataclasses import dataclass
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import pandas as pd
from typing import Optional, List, Dict, Any
from ..utils.get_entity_info import get_entity_info
from ..agents.verifier import create_verifier_agent, build_verifier_toolkit
from ..memory.short_term import ShortTermMemoryStore, MaterialType
from ..memory.long_term import LongTermMemoryStore
from ..utils.instance import create_chat_model, create_agent_formatter
from ..utils.call_with_retry import call_agent_with_retry
from ..prompt import prompt_dict
from agentscope.message import Msg
from urllib.parse import urlparse


class VerifyTask(str, Enum):
    FACTUAL = "factual"          # 多源事实验证
    GROUNDING = "grounding"      # material -> text 是否正确
    CORROBORATE = "corroborate"  # 在排除指定来源后的事实验证（本质仍是 factual + exclude）
    CODE = "code"                # 代码正确性验证

@dataclass
class VerifyInput:
    text: Optional[str] = None                 # 待验证的自然语言文本
    claims: Optional[List[str]] = None         # 文本中提取的原子级陈述，如果传了就不用再提取了
    ref_id: Optional[str] = None               # 需要读取 material 内容时用

@dataclass
class ClaimResult:
    claim: str
    slots: Dict[str, Any]
    verdict: str                 # support/refute/unknown
    confidence: float            # 0~1
    evidence: List[str]  # 关键证据（已去重、已满足独立性）
    conflicts: List[str] # 反证/冲突证据（如有）
    rationale: str               # 解释

def _read_chat_text(res: Any) -> str:
    content = getattr(res, "content", None)
    if isinstance(content, list):
        out = "".join(
            blk.get("text", "")
            for blk in content
            if isinstance(blk, dict) and blk.get("type") == "text"
        ).strip()
    else:
        out = str(res).strip()
    return out

def _extract_json_block(out: str) -> str:
    out = out.strip()
    if not out.startswith("{") and not out.startswith("["):
        l_obj, r_obj = out.find("{"), out.rfind("}")
        l_arr, r_arr = out.find("["), out.rfind("]")
        # 优先截取数组，否则截取对象
        if l_arr != -1 and r_arr != -1 and r_arr > l_arr:
            return out[l_arr:r_arr + 1]
        if l_obj != -1 and r_obj != -1 and r_obj > l_obj:
            return out[l_obj:r_obj + 1]
    return out


def material_source_key(
    short_term,  # ShortTermMemoryStore
    ref_id: str,
) -> str:
    """
    给定 ref_id，返回“来源归一化 key”，用于判定两个 material 是否来自同一来源。

    规则：
      - search_engine：按 domain 聚合（同域名视为同来源）
      - 计算工具：每个 ref_id 视为独立来源
      - AKshare API：按api的数据来源
      - 其它：按 meta.source 归一化（source 为空则回退到 filename）

    返回示例：
      - "web:zhihu.com"
      - "calc:calculate_financial_ratio_result_1767927143"
      - "akshare:eastmoney"
      - "src:xxx"
      - "file:xxx.csv"
    """
    meta = short_term.get_material_meta(ref_id)
    if meta is None:
        raise ValueError(f"Material ref_id='{ref_id}' 不存在于 registry。")

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
    if ref_id.startswith("calculate_"):
        return f"calc:{ref_id}"

    # 3) Search engine：按 domain 聚合
    if ref_id.startswith("search_engine") or ("search engine" in src_lower):
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
            obj = short_term.load_material(ref_id)
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


async def extract_atomic_claims(text: str) -> List[str]:
    """
    text -> 原子级 claims
    返回：List[str]，长度<=max_claims
    """
    text = (text or "").strip()
    if not text:
        return []

    model = create_chat_model()
    res = await model(
        messages=[
            {"role": "system", "content": prompt_dict["claim_extract_sys_prompt"]},
            {"role": "user", "content": prompt_dict["claim_extract_prompt"].format(text=text)},
        ],
        temperature=0.0,
    )

    out = _read_chat_text(res)
    out = _extract_json_block(out)

    data = json.loads(out)
    # 允许模型返回 {"claims":[...]} 或直接 [...]
    if isinstance(data, dict):
        claims = data.get("claims", [])
    else:
        claims = data

    if not isinstance(claims, list):
        return []

    # 轻度清洗：去空、去重、截断
    cleaned: List[str] = []
    seen = set()
    for c in claims:
        if not isinstance(c, str):
            continue
        c = c.strip()
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        cleaned.append(c)

    return cleaned

async def extract_claim_slots(claim: str) -> Dict[str, Any]:
    """
    claim -> {subj, pred, obj, time}
    返回至少包含：subj, pred, obj, time 四个字段（time 可为 null/"")
    """
    claim = (claim or "").strip()
    if not claim:
        return {"subj": "", "pred": "", "obj": "", "time": ""}

    model = create_chat_model()
    res = await model(
        messages=[
            {"role": "system", "content": prompt_dict["slot_extract_sys_prompt"]},
            {"role": "user", "content": prompt_dict["slot_extract_prompt"].format(claim=claim)},
        ],
        temperature=0.0,
    )

    out = _read_chat_text(res)
    out = _extract_json_block(out)
    data = json.loads(out)

    if not isinstance(data, dict):
        return {"subj": "", "pred": "", "obj": "", "time": ""}

    # 避免下游 KeyError
    for k in ("subj", "pred", "obj", "time"):
        if k not in data or data[k] is None:
            data[k] = ""
        if not isinstance(data[k], str):
            data[k] = str(data[k])

    return data

async def judge_grounding(material_text: str, text: str) -> Dict[str, Any]:
    model = create_chat_model()
    
    res = await model(messages=[
        {"role": "system", "content": prompt_dict['grounding_sys_prompt']},
        {"role": "user", "content": prompt_dict['grounding_prompt'].format(material_text=material_text,text=text)},
    ], temperature=0.0)

    out = _read_chat_text(res)

    # 允许模型输出前后带少量解释：截取最外层 {...}
    if not out.startswith("{"):
        l = out.find("{")
        r = out.rfind("}")
        if l != -1 and r != -1 and r > l:
            out = out[l:r+1]
    data = json.loads(out)

    # 轻度容错
    data["confidence"] = float(data.get("confidence", 0.0))
    data["confidence"] = max(0.0, min(1.0, data["confidence"]))
    for k in ("evidence", "missing", "conflicts"):
        if not isinstance(data.get(k), list):
            data[k] = []
    if not isinstance(data.get("rationale"), str):
        data["rationale"] = ""
    data["entailed"] = bool(data.get("entailed", False))
    return data


def parse_verifier_output(text: str) -> Dict[str, Any]:
        """
        期望格式（重复多条）：
        1. ref_id:xxxx
        支持该陈述 / 不支持该陈述
        理由...

        没有来源时：无
        """
        raw = (text or "").strip()
        if not raw or raw == "无":
            return {"evidence_ref_ids": [], "conflict_ref_ids": [], "points": []}

        lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
        # 兼容两种 ref_id 行：
        # 1) "1. ref_id:xxx" / "1.ref_id：xxx"
        # 2) "[ref_id:xxx|...]"（兜底）
        pat1 = re.compile(r"^\s*(\d+)\s*[\.\)]\s*ref_id\s*[:：]\s*([A-Za-z0-9_\-]+)\s*$")
        pat2 = re.compile(r"\[ref_id:([A-Za-z0-9_\-]+)(?:\|[^\]]*)?\]")

        blocks: List[Dict[str, str]] = []
        cur: Optional[Dict[str, str]] = None

        def flush():
            nonlocal cur
            if cur and cur.get("ref_id"):
                blocks.append(cur)
            cur = None

        for ln in lines:
            m = pat1.match(ln)
            if m:
                flush()
                cur = {"idx": m.group(1), "ref_id": m.group(2), "label": "", "reason": ""}
                continue

            # 如果没有遇到 pat1，但遇到 [ref_id:xxx]，也尝试开新块
            if cur is None:
                m2 = pat2.search(ln)
                if m2:
                    flush()
                    cur = {"idx": str(len(blocks) + 1), "ref_id": m2.group(1), "label": "", "reason": ""}
                    continue

            if cur is None:
                continue

            if not cur["label"]:
                cur["label"] = ln.strip()
            else:
                cur["reason"] = (cur["reason"] + " " + ln.strip()).strip()

        flush()

        evidence_ref_ids: List[str] = []
        conflict_ref_ids: List[str] = []
        points: List[str] = []

        for b in blocks:
            rid = b.get("ref_id", "").strip()
            if not rid:
                continue

            label = (b.get("label") or "").strip()
            reason = (b.get("reason") or "").strip()

            # 分类：支持 / 不支持（反驳）
            is_conflict = ("不支持" in label) or ("反驳" in label)
            is_support = ("支持" in label) and (not is_conflict)

            if is_conflict:
                conflict_ref_ids.append(rid)
                tag = "不支持"
            elif is_support:
                evidence_ref_ids.append(rid)
                tag = "支持"
            else:
                # label 不符合预期就忽略该条
                continue

            idx = b.get("idx") or str(len(points) + 1)
            if reason:
                points.append(f"{idx}.找到 Material(ref_id={rid}) {tag}陈述：{reason}")
            else:
                points.append(f"{idx}.找到 Material(ref_id={rid}) {tag}陈述")

        # 去重但保持顺序
        def uniq(xs: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in xs:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out

        return {
            "evidence_ref_ids": uniq(evidence_ref_ids),
            "conflict_ref_ids": uniq(conflict_ref_ids),
            "points": points,
        }
async def multi_source_verification(input:VerifyInput, task:VerifyTask, short_term:ShortTermMemoryStore, long_term:LongTermMemoryStore) ->  List[ClaimResult]:
    
    if input.ref_id:
        material = short_term.load_material(input.ref_id)
        if material is None:
            raise ValueError(f"ref_id={input.ref_id} material 不存在或读取失败")
        meta = short_term.get_material_meta(input.ref_id) 
        if meta.m_type == MaterialType.TABLE:
            material = material.to_csv(index=False)
        elif meta.m_type == MaterialType.JSON:
            material = json.dumps(material, ensure_ascii=False, indent=2)
        material_text = str(material)


    if task in (VerifyTask.FACTUAL, VerifyTask.CORROBORATE):
        # 1) 获取待验证 claims（text中提取，material中提取，或者直接参数传入）
        # 2) 获取 slots
        # 3) retrieve evidence with exclude constraints
        # 4) judge + aggregate
        if input.claims:
            claims = [c.strip() for c in input.claims if isinstance(c, str) and c.strip()]
        elif input.text and input.text.strip():
            claims = await extract_atomic_claims(input.text)
        elif input.ref_id and material_text.strip():
            # 没有 text 时可对 material 做“材料正确性验证”
            claims = await extract_atomic_claims(material_text)
        else:
            raise ValueError("FACTUAL/CORROBORATE 需要 input.text 或 input.claims 或 input.ref_id(可读 material)")

        if not claims:
            return []

        slots_list = [await extract_claim_slots(c) for c in claims]

        verifier_toolkit = build_verifier_toolkit(
            short_term=short_term,
            long_term=long_term,
            multi_source_verification=True
        )
        verifier_toolkit.update_tool_groups(group_names=["multi_source_search"], active=True)

        model = create_chat_model()
        verifier = create_verifier_agent(model=model, formatter=create_agent_formatter(), toolkit=verifier_toolkit)
        results: List[ClaimResult] = []

        for claim, slots in zip(claims, slots_list):
            subj = slots.get("subj", "").strip()
            pred = slots.get("pred", "").strip()
            obj = slots.get("obj", "").strip()
            t = slots.get("time", "").strip()

            core_lines = []
            if subj: core_lines.append(f"- 主体：{subj}")
            if pred: core_lines.append(f"- 关系/谓词：{pred}")
            if obj:  core_lines.append(f"- 客体/对象：{obj}")
            if t: core_lines.append(f"- 时间：{t}")
            core_info = "\n".join(core_lines)

            verifier_input = Msg(name="User", content=prompt_dict["multi_source_verify_prompt"].format(claim=claim,core_info=core_info), role="user")
            verify_msg = await call_agent_with_retry(verifier, verifier_input)
            verdict_text = verify_msg.get_text_content()

            parsed = parse_verifier_output(verdict_text)
            evidence_ref_ids = parsed["evidence_ref_ids"]
            conflict_ref_ids = parsed["conflict_ref_ids"]

            if task == VerifyTask.CORROBORATE and input.ref_id:
                evidence_ref_ids = [rid for rid in evidence_ref_ids if rid != input.ref_id]
                base_key = material_source_key(short_term, input.ref_id)
                evidence_ref_ids = [
                    rid for rid in evidence_ref_ids
                    if material_source_key(short_term, rid) != base_key
                ]



            if len(conflict_ref_ids) > 0:
                verdict = "refute"
                confidence = 0.90
            elif len(evidence_ref_ids) >= 2:
                verdict = "support"
                confidence = min(1, len(evidence_ref_ids)*0.05+0.8)
            elif len(evidence_ref_ids) == 1:
                verdict = "support"
                confidence = 0.50
            else:
                verdict = "unknown"
                confidence = 0.0

            points = parsed.get("points", [])
            rationale = "\n".join(points).strip()

            results.append(ClaimResult(
                claim=claim,
                slots=slots,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence_ref_ids,
                conflicts=conflict_ref_ids,
                rationale=rationale
            ))
        return results
    elif task == VerifyTask.GROUNDING:
        result = await judge_grounding(material_text, input.text.strip())
        verdict = "support" if result["entailed"] else ("refute" if result.get("conflicts") else "unknown")

        return [ClaimResult(
            claim=input.text.strip(),
            slots={}, 
            verdict=verdict,
            confidence=result["confidence"],
            evidence=result["evidence"],
            conflicts=result["conflicts"],
            rationale=result["rationale"] + ((" | missing: " + "; ".join(result["missing"])) if result["missing"] else ""),
    )]

    elif task == VerifyTask.CODE:
        if not input.ref_id or not input.ref_id.startswith("calculate_or_analysis_by_python_code"):
            raise ValueError("task=CODE 传入无效Material，应为自定义计算工具的计算结果 material(ref_id 以 calculate_or_analysis_by_python_code 开头)")
            # todo
    else:
        raise ValueError(f"task={task}无效( {VerifyTask.FACTUAL}, {VerifyTask.CORROBORATE}, {VerifyTask.GROUNDING}, {VerifyTask.CODE} )")

