# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
import json
import math
import re
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from agentscope.message import Msg
from agentscope.tool._coding._python import execute_python_code

from ..utils.get_entity_info import get_entity_info
from ..agents.verifier import create_verifier_agent, build_verifier_toolkit
from ..memory.short_term import ShortTermMemoryStore, MaterialType
from ..memory.long_term import LongTermMemoryStore
from ..utils.instance import create_chat_model, create_agent_formatter
from ..utils.call_with_retry import call_agent_with_retry
from ..prompt import prompt_dict


class VerifyTask(str, Enum):
    FACTUAL = "factual"          # 多源事实验证
    GROUNDING = "grounding"      # material -> text 是否正确
    CORROBORATE = "corroborate"  # 在排除指定来源后的事实验证（本质仍是 factual + exclude）
    CODE = "code"                # 代码正确性验证

@dataclass
class VerifyInput:
    text: Optional[str] = None                 # 待验证的自然语言文本
    claims: Optional[List[str]] = None         # 文本中提取的原子级陈述，如果传了就不用再提取了
    cite_id: Optional[str] = None               # 需要读取 material 内容时用

@dataclass
class ClaimResult:
    claim: str
    slots: Dict[str, Any]
    verdict: str                 # support/refute/unknown
    confidence: float            # 0~1
    evidence: List[str]  # 关键证据（已去重、已满足独立性）
    conflicts: List[str] # 反证/冲突证据（如有）
    rationale: str               # 解释

CITE_RE = re.compile(r"\[\^cite_id:([A-Za-z0-9_\-]+)(?:\|[^\]]*)?\]")

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
        1. cite_id:xxxx
        支持该陈述 / 不支持该陈述
        理由...

        没有来源时：无
        """
        raw = (text or "").strip()
        if not raw or raw == "无":
            return {"evidence_cite_ids": [], "conflict_cite_ids": [], "points": []}

        lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
        # 兼容两种 cite_id 行：
        # 1) "1. cite_id:xxx" / "1.cite_id：xxx"
        # 2) "[^cite_id:xxx|...]"（兜底）
        pat1 = re.compile(r"^\s*(\d+)\s*[\.\)]\s*cite_id[:：]\s*([A-Za-z0-9_\-]+)\s*$")
        pat2 = re.compile(r"\[(?:\^cite_id:|\^|cite_id:)+([A-Za-z0-9_\-]+)(?:\|[^\]]*)?\]")

        blocks: List[Dict[str, str]] = []
        cur: Optional[Dict[str, str]] = None

        def flush():
            nonlocal cur
            if cur and cur.get("cite_id"):
                blocks.append(cur)
            cur = None

        for ln in lines:
            m = pat1.match(ln)
            if m:
                flush()
                cur = {"idx": m.group(1), "cite_id": m.group(2), "label": "", "reason": ""}
                continue

            # 如果没有遇到 pat1，但遇到 [^xxx]，也尝试开新块
            if cur is None:
                m2 = pat2.search(ln)
                if m2:
                    flush()
                    cur = {"idx": str(len(blocks) + 1), "cite_id": m2.group(1), "label": "", "reason": ""}
                    continue

            if cur is None:
                continue

            if not cur["label"]:
                cur["label"] = ln.strip()
            else:
                cur["reason"] = (cur["reason"] + " " + ln.strip()).strip()

        flush()

        evidence_cite_ids: List[str] = []
        conflict_cite_ids: List[str] = []
        points: List[str] = []

        for b in blocks:
            rid = b.get("cite_id", "").strip()
            if not rid:
                continue

            label = (b.get("label") or "").strip()
            reason = (b.get("reason") or "").strip()

            # 分类：支持 / 不支持（反驳）
            is_conflict = ("不支持" in label) or ("反驳" in label)
            is_support = ("支持" in label) and (not is_conflict)

            if is_conflict:
                conflict_cite_ids.append(rid)
                tag = "不支持"
            elif is_support:
                evidence_cite_ids.append(rid)
                tag = "支持"
            else:
                # label 不符合预期就忽略该条
                continue

            idx = b.get("idx") or str(len(points) + 1)
            if reason:
                points.append(f"{idx}.找到 Material(cite_id={rid}) {tag}陈述：{reason}")
            else:
                points.append(f"{idx}.找到 Material(cite_id={rid}) {tag}陈述")

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
            "evidence_cite_ids": uniq(evidence_cite_ids),
            "conflict_cite_ids": uniq(conflict_cite_ids),
            "points": points,
        }




def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _normalize_result(obj: Any) -> Any:
    obj = _to_builtin(obj)

    if isinstance(obj, dict):
        return {
            str(k): _normalize_result(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(obj, list):
        return [_normalize_result(v) for v in obj]

    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return str(obj)
        return round(float(obj), 8)

    return obj


def _compare_result(expected: Any, actual: Any) -> Tuple[bool, str]:
    exp = _normalize_result(expected)
    act = _normalize_result(actual)

    if exp == act:
        return True, "复核结果与原结果一致"

    return False, (
        "复核结果与原结果不一致。\n"
        f"expected={json.dumps(exp, ensure_ascii=False, sort_keys=True), 800}\n"
        f"actual={json.dumps(act, ensure_ascii=False, sort_keys=True), 800}"
    )


def _extract_code_material_record(short_term: ShortTermMemoryStore, cite_id: str) -> Dict[str, Any]:
    material = short_term.load_material(cite_id)
    if not isinstance(material, list) or not isinstance(material[0], dict):
        raise ValueError(f"cite_id={cite_id} 对应的 material 格式不正确")

    record = material[0]
    code = (record.get("code") or "").strip()
    if not code:
        raise ValueError(f"cite_id={cite_id} 缺少 code 字段")

    return {
        "description": record.get("description") or "",
        "code": code,
        "result": record.get("result"),
        "result_type": record.get("result_type") or "text_repr",
    }


async def _generate_independent_verify_code(
    code: str,
    original_result: Any,
    result_type: str,
    description: str = "",
) -> Dict[str, Any]:
    model = create_chat_model()

    res = await model(
        messages=[
            {"role": "system", "content": prompt_dict["code_verify_sys_prompt"]},
            {
                "role": "user",
                "content": prompt_dict["code_verify_prompt"].format(
                    description=description or "(无补充描述)",
                    code=code,
                    result_type=result_type or "unknown",
                    original_result=json.dumps(_to_builtin(original_result), ensure_ascii=False, indent=2, default=str),
                ),
            },
        ],
        temperature=0.0,
    )

    out = _extract_json_block(_read_chat_text(res))
    data = json.loads(out)

    if not isinstance(data, dict):
        raise ValueError("code_verify_prompt 输出不是 JSON 对象")
    if not isinstance(data.get("verify_code"), str) or not data["verify_code"].strip():
        raise ValueError("code_verify_prompt 没有返回有效的 verify_code")
    if not isinstance(data.get("independence_note"), str):
        data["independence_note"] = ""

    return data


async def _execute_verify_code_simple(
    code: str,
    timeout: int = 60,
) -> Dict[str, Any]:
    python_wrapper = f"""
import math
import statistics
import datetime
import numpy as np
import pandas as pd
import json

# ==== 复核代码开始 ====
{code}
# ==== 复核代码结束 ====

try:
    _r = result
except NameError:
    print("<result_error>变量 'result' 未定义，请在代码中将最终结果赋值给 result。</result_error>")
else:
    print("<result_json>")
    try:
        if isinstance(_r, pd.DataFrame):
            payload = _r.to_dict(orient="records")
            r_type = "DataFrame"
        elif isinstance(_r, pd.Series):
            payload = _r.to_dict()
            r_type = "Series"
        else:
            payload = _r
            r_type = type(_r).__name__

        print(json.dumps({{"result_type": r_type, "result": payload}}, ensure_ascii=False))
    except Exception as _e:
        print(json.dumps({{"result_type": "text_repr", "result": repr(_r), "error": str(_e)}}, ensure_ascii=False))
    print("</result_json>")
"""

    exec_resp = await execute_python_code(python_wrapper, timeout=timeout)
    text_blocks = [b for b in exec_resp.content if isinstance(b, dict) and b.get("type") == "text"]
    if not text_blocks:
        return {"ok": False, "error": "未返回任何文本输出"}

    raw_text = text_blocks[0].get("text", "")

    m_ret = re.search(r"<returncode>(.*?)</returncode>", raw_text, re.S)
    m_out = re.search(r"<stdout>(.*?)</stdout>", raw_text, re.S)
    m_err = re.search(r"<stderr>(.*?)</stderr>", raw_text, re.S)

    returncode = int(m_ret.group(1).strip()) if m_ret else -1
    stdout = m_out.group(1) if m_out else ""
    stderr = m_err.group(1) if m_err else ""

    if returncode != 0:
        return {
            "ok": False,
            "error": f"执行失败，returncode={returncode}",
            "stdout": stdout,
            "stderr": stderr,
        }

    m_err_tag = re.search(r"<result_error>(.*?)</result_error>", stdout, re.S)
    if m_err_tag:
        return {
            "ok": False,
            "error": m_err_tag.group(1).strip(),
            "stdout": stdout,
            "stderr": stderr,
        }

    m_json = re.search(r"<result_json>(.*?)</result_json>", stdout, re.S)
    if not m_json:
        return {
            "ok": False,
            "error": "未找到 <result_json> 输出",
            "stdout": stdout,
            "stderr": stderr,
        }

    payload = json.loads(m_json.group(1).strip())
    return {
        "ok": True,
        "result_type": payload.get("result_type", "text_repr"),
        "result": payload.get("result"),
        "stdout": stdout,
        "stderr": stderr,
    }

def is_high_risk_claim(text: str) -> bool:
    """
    判断一句话是否属于高风险陈述，适合进入 factual / corroborate 验证。

    高风险的典型特征：
    1) 含具体数字、比例、金额、估值指标
    2) 含明确时间锚点
    3) 含同比/环比/增长/下降等比较关系
    4) 含公告、发布、收购、中标、投产等可核验事件
    5) 含因果归因，如“主要由于/受益于/驱动”
    6) 含“第一/领先/最大/唯一”等绝对化表述
    7) 强事实句但没有 cite_id
    """

    text = (text or "").strip()
    if not text:
        return False

    def has(patterns):
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    numeric_patterns = [
        r"\d",
        r"\d+(\.\d+)?\s*%",
        r"\d+(\.\d+)?\s*倍",
        r"\d+(\.\d+)?\s*亿元?",
        r"\d+(\.\d+)?\s*万元?",
        r"\d+(\.\d+)?\s*万吨",
        r"\d+(\.\d+)?\s*万台",
        r"\d+(\.\d+)?\s*GWh",
    ]

    time_patterns = [
        r"20\d{2}年",
        r"20\d{2}Q[1-4]",
        r"20\d{2}H[12]",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{4}-\d{2}",
        r"一季度|二季度|三季度|四季度",
        r"上半年|下半年",
        r"本期|上期|同期|年末|期末",
    ]

    comparison_patterns = [
        r"同比", r"环比", r"较上年", r"较去年", r"较上期",
        r"增长", r"下滑", r"提升", r"下降",
        r"高于", r"低于", r"超过", r"不及", r"优于", r"弱于",
    ]

    event_patterns = [
        r"发布", r"披露", r"公告", r"签署", r"中标", r"获批",
        r"投产", r"开工", r"停产", r"收购", r"并购",
        r"增发", r"回购", r"分红", r"推出", r"上线", r"量产",
        r"销售", r"出货", r"交付", r"扩产", r"募资", r"诉讼", r"处罚",
    ]

    causality_patterns = [
        r"因为", r"由于", r"主要系", r"主要由于", r"受益于",
        r"驱动", r"拉动", r"导致", r"源于", r"得益于", r"拖累",
    ]

    superlative_patterns = [
        r"第一", r"领先", r"最高", r"最低", r"最大", r"最小",
        r"唯一", r"首个", r"龙头",
    ]

    metric_patterns = [
        r"\bPE\b", r"\bPB\b", r"\bPS\b", r"\bPEG\b",
        r"\bROE\b", r"\bROIC\b", r"\bEV/EBITDA\b",
        r"WACC", r"DCF",
        r"毛利率", r"净利率", r"收入", r"营收", r"利润",
        r"归母净利润", r"扣非", r"现金流", r"市占率",
        r"产能", r"销量", r"订单", r"ASP",
    ]

    prediction_patterns = [
        r"预计", r"有望", r"或将", r"预期", r"我们预计", r"展望",
    ]

    has_numeric = has(numeric_patterns)
    has_time = has(time_patterns)
    has_comparison = has(comparison_patterns)
    has_event = has(event_patterns)
    has_causality = has(causality_patterns)
    has_superlative = has(superlative_patterns)
    has_metric = has(metric_patterns)
    has_prediction = has(prediction_patterns)
    has_cite = bool(re.search(r"\[\^cite_id:[A-Za-z0-9_\-]+(?:\|[^\]]*)?\]", text))

    score = 0

    if has_numeric:
        score += 2
    if has_time:
        score += 1
    if has_comparison:
        score += 2
    if has_event:
        score += 2
    if has_causality:
        score += 2
    if has_superlative:
        score += 2
    if has_metric:
        score += 1
    if has_prediction and (has_numeric or has_metric or has_comparison):
        score += 1

    if not has_cite and (has_numeric or has_event or has_causality or has_superlative):
        score += 1
    return score >= 3


async def multi_source_verification(input:VerifyInput, task:VerifyTask, short_term:ShortTermMemoryStore, long_term:LongTermMemoryStore) ->  List[ClaimResult]:
    
    if input.cite_id:
        material = short_term.load_material(input.cite_id)
        if material is None:
            raise ValueError(f"cite_id={input.cite_id} material 不存在或读取失败")
        meta = short_term.get_material_meta(input.cite_id) 
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
        elif input.cite_id and material_text.strip():
            # 没有 text 时可对 material 做“材料正确性验证”
            claims = await extract_atomic_claims(material_text)
        else:
            raise ValueError("FACTUAL/CORROBORATE 需要 input.text 或 input.claims 或 input.cite_id(可读 material)")

        if not claims:
            return []

        slots_list = [await extract_claim_slots(c) for c in claims]

        verifier_toolkit = build_verifier_toolkit(
            short_term=short_term,
            long_term=long_term,
            multi_source_verification=True
        )

        model = create_chat_model()
        verifier = create_verifier_agent(model=model, formatter=create_agent_formatter(), toolkit=verifier_toolkit, verifier_type="multi_source")
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
            evidence_cite_ids = parsed["evidence_cite_ids"]
            conflict_cite_ids = parsed["conflict_cite_ids"]

            if task == VerifyTask.CORROBORATE and input.cite_id:
                evidence_cite_ids = [rid for rid in evidence_cite_ids if rid != input.cite_id]
                base_key = material_source_key(short_term, input.cite_id)
                evidence_cite_ids = [
                    rid for rid in evidence_cite_ids
                    if material_source_key(short_term, rid) != base_key
                ]



            if len(conflict_cite_ids) > 0:
                verdict = "refute"
                confidence = 0.90
            elif len(evidence_cite_ids) >= 2:
                verdict = "support"
                confidence = min(1, len(evidence_cite_ids)*0.05+0.8)
            elif len(evidence_cite_ids) == 1:
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
                evidence=evidence_cite_ids,
                conflicts=conflict_cite_ids,
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
        if not input.cite_id or "calculate_or_analysis_by_python_code" not in input.cite_id:
            raise ValueError("task=CODE 传入无效Material，应为自定义计算工具的计算结果 material(cite_id 中包含 calculate_or_analysis_by_python_code)")

        record = _extract_code_material_record(short_term, input.cite_id)

        plan = await _generate_independent_verify_code(
            code=record["code"],
            original_result=record["result"],
            result_type=record["result_type"],
            description=record["description"],
        )

        verify_run = await _execute_verify_code_simple(
            code=plan["verify_code"],
        )

        if not verify_run.get("ok"):
            return [ClaimResult(
                claim="",
                slots={},
                verdict="unknown",
                confidence=0.0,
                evidence=[],
                conflicts=[],
                rationale=(
                    f"独立实现生成成功，但执行失败。\n"
                    f"independence_note: {plan.get('independence_note', '')}\n"
                    f"error: {verify_run.get('error', '')}"
                ).strip(),
            )]

        is_match, compare_msg = _compare_result(record["result"], verify_run.get("result"))

        if is_match:
            verdict = "support"
            confidence = 0.9
        else:
            verdict = "refute"
            confidence = 0.9

        return [ClaimResult(
            claim="",
            slots={},
            verdict=verdict,
            confidence=confidence,
            evidence=[],
            conflicts=[],
            rationale=(
                f"independence_note: {plan.get('independence_note', '')}\n"
                f"{compare_msg}"
            ).strip(),
        )]

    else:
        raise ValueError(f"task={task}无效( {VerifyTask.FACTUAL}, {VerifyTask.CORROBORATE}, {VerifyTask.GROUNDING}, {VerifyTask.CODE} )")


def strip_citations(text: str) -> str:
    return CITE_RE.sub("", text or "").strip()

def extract_cite_ids(text: str) -> list[str]:
    return list(dict.fromkeys(CITE_RE.findall(text or "")))

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；\n])", text or "")
    return [p.strip() for p in parts if p.strip()]

async def verify_segment_content(segment, short_term, long_term):
    issues = []
    text = segment.content or ""
    if not text.strip():
        return issues

    # 1) grounding: 带 cite_id 的句子必须被至少一个对应 material 支持
    for sent in split_sentences(text):
        cite_ids = extract_cite_ids(sent)
        plain_sent = strip_citations(sent)
        if not cite_ids or not plain_sent:
            continue

        grounding_results = []
        for cid in cite_ids:
            res = await multi_source_verification(
                input=VerifyInput(text=plain_sent, cite_id=cid),
                task=VerifyTask.GROUNDING,
                short_term=short_term,
                long_term=long_term,
            )
            if res:
                grounding_results.append((cid, res[0]))

        if not any(r.verdict == "support" and r.confidence >= 0.7 for _, r in grounding_results):
            detail = "\n".join(
                f"- 引用材料 {cid} 的核验结果：{r.rationale}"
                for cid, r in grounding_results
            )
            issues.append(
                f"下面这句话与所引用的材料不完全一致，或无法从引用材料中直接推出，请修改：\n"
                f"原句：{plain_sent}\n"
                f"引用：{cite_ids}\n"
                f"核验说明：\n{detail}\n"
                f"修改要求：只保留能够被这些引用材料直接支持的内容；如果原句中有材料无法支持的判断、数字、时间或结论，请删除、弱化或改写，并补充合适的引用。"
            )
        # 2) code：自定义计算结果需要独立实现一遍
        if "calculate_or_analysis_by_python_code" not in cid:
            continue
        res = await multi_source_verification(
                    input=VerifyInput(cite_id=cid),
                    task=VerifyTask.CODE,
                    short_term=short_term,
                    long_term=long_term,
                )

        if not res:
            continue

        r = res[0]
        if r.verdict != "support":
            issues.append(
                f"你在当前段落中引用了自定义 Python 计算结果（cite_id={cid}），"
                "但该结果没有通过系统复核，暂时不能作为可靠依据。\n"
                f"具体原因：{r.rationale}\n"
                "请修改相关正文：删除、弱化或改写依赖该计算结果的结论，"
                "不要把这个结果继续写成确定事实。"
            )
    # 3) corroborate / factual: 只对高风险句子做
    for sent in split_sentences(text):
        plain_sent = strip_citations(sent)
        if not plain_sent or not is_high_risk_claim(plain_sent):
            continue

        cite_ids = extract_cite_ids(sent)

        if cite_ids:
            # 有引用时，优先做独立佐证
            for cid in cite_ids[:1]:   # 先用首个 cite_id，后续可扩成多 cite
                res = await multi_source_verification(
                    input=VerifyInput(text=plain_sent, cite_id=cid),
                    task=VerifyTask.CORROBORATE,
                    short_term=short_term,
                    long_term=long_term,
                )
                if res and res[0].verdict in ("refute", "unknown"):
                    issues.append(
                        f"下面这句话虽然有引用，但缺少足够的独立来源支持，或与其他来源存在冲突，请修改：\n"
                        f"原句：{plain_sent}\n"
                        f"当前引用：{cid}\n"
                        f"核验说明：{res[0].rationale}\n"
                        f"修改要求：如果该结论无法被独立来源支持，请删除、弱化或改写这句话；如果需要保留，请补充更可靠的独立来源，并避免保留没有充分依据的强结论。"
                    )
        else:
            # 无引用的高风险句子，直接做 factual
            res = await multi_source_verification(
                input=VerifyInput(text=plain_sent),
                task=VerifyTask.FACTUAL,
                short_term=short_term,
                long_term=long_term,
            )
            if res and res[0].verdict != "support":
                issues.append(
                    f"下面这句话经过核查，没有得到充分验证，请修改：\n"
                    f"原句：{plain_sent}\n"
                    f"核验说明：{res[0].rationale}\n"
                    f"修改要求：如果没有可靠依据，请删除这句话；如果只有部分内容可以确认，请只保留可证实部分。"
                )

    return issues