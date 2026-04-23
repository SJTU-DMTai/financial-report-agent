import json
import re
from json import JSONDecodeError
from typing import Tuple, List

from pydantic import BaseModel

from src.evaluation.extract_evidence import find_best_matches_by_similarity
from src.utils.call_with_retry import call_chatbot_with_retry
from src.utils.instance import llm_reasoning, formatter


class EvidenceComparison(BaseModel):
    evidence: str   # 论据的核心主题抽象描述
    value1: str     # 来源1的具体事实
    value2: str     # 来源2的具体事实
    is_same: bool   # value1 和 value2 是否事实一致


class EvidenceComparisons(BaseModel):
    result_list: List[EvidenceComparison]


async def judge_consistency_for_evidence_pairs(
    evidences: List[Tuple[str, str, str, str]],  # List of (key1, value1, key2, value2)
) -> List[bool]:
    """
    对若干论据对，直接根据已知的具体事实判断是否一致。

    Args:
        evidences: List of (key1, value1, key2, value2)
            key1/key2: 论据描述（抽象主题）
            value1/value2: 对应的具体事实或数据

    Returns:
        List of bool, one per evidence pair (True = consistent).
    """
    numbered_evidences = "\n".join(
        f"<论据{i+1}>\n主题描述1: {pair[0]}\n来源1具体事实: {pair[1]}\n主题描述2: {pair[2]}\n来源2具体事实: {pair[3]}\n</论据{i+1}>"
        for i, pair in enumerate(evidences)
    )

    example_json = json.dumps([
        {
            "evidence": "2024年营业收入",
            "value1": "2024年营业收入为52.3亿元",
            "value2": "2024年营收达到61.8亿元",
            "is_same": False
        },
        {
            "evidence": "毛利率水平",
            "value1": "毛利率为38.5%",
            "value2": "毛利率约38.5%",
            "is_same": True
        }
    ], ensure_ascii=False, indent=2)

    prompt = f"""你是一名严谨的金融事实核查员。你将收到若干论据对，每对包含两个来源的主题描述和具体事实。

**核查流程（对每一条论据对）：**
1. 如果两个主题描述的不是同一件事（配对错误），请直接跳过。如果论据描述的是主观判断，也请跳过。对于其他情况：
2. 根据 value1 和 value2 是否相符（数字、单位、时间范围、主体等核心要素都属于同一语义）赋值 `is_same`，如果value1包含更多信息并覆盖了value2，也可以为true。

**输出格式：**
将所有相关论据的结果汇总为一个JSON数组。
数组中每个元素包含字段：evidence（抽象描述该论据的核心主题）、value1、value2、is_same。

**输出示例（假设有2条论据）：**
{example_json}"""

    user_msg = (
        f"{numbered_evidences}\n\n"
        f"请按流程逐条分析，并在最后输出JSON数组。"
    )

    comparisons = await call_chatbot_with_retry(
        llm_reasoning, formatter, prompt, user_msg,
        structured_model=EvidenceComparisons, handle_hook_exceptions=(JSONDecodeError, ), max_retries=3,
    )
    print(comparisons, flush=True)
    return [c.is_same for c in comparisons.result_list]


async def evidence_coverage_and_accuracy(
    report_evidences: List[Tuple[str, str]],
    reference_evidences: List[Tuple[str, str]],
) -> Tuple[float, float]:
    """
    统计论据覆盖率和准确率。
    流程：
      1. find_best_matches_by_similarity 基于 embedding cosine 相似度进行配对，
         得到四元组 (report_ev_key, ref_ev_key, report_ev_value, ref_ev_value)。
      2. 将配对结果送入 LLM 判断一致性：
         - 如果两个主题描述的不是同一个论据（配对错误），LLM 会跳过
         - 否则根据 value（具体事实）判断是否一致
      3. 计算 accuracy。

    Args:
        report_evidences: List of (evidence_key, evidence_value) from AI report.
        reference_evidences: List of (evidence_key, evidence_value) from human reference.

    Returns:
        (coverage_ratio, accuracy_ratio)
    """
    print(f"  - Report 论据数量: {len(report_evidences)}")
    print(f"  - Reference 论据数量: {len(reference_evidences)}")

    if not report_evidences or not reference_evidences:
        raise RuntimeError("  - 警告: Report 或 Reference 中缺少论据，无法计算准确率。")

    # 步骤 1: 基于 embedding 相似度进行配对，得到四元组 (report_ev_key, ref_ev_key, report_ev_value, ref_ev_value)
    matched_pairs = await find_best_matches_by_similarity(report_evidences, reference_evidences)
    print(f"  - 共配对到 {len(matched_pairs)} 对论据。")

    coverage_ratio = len(matched_pairs) / len(reference_evidences)

    if not matched_pairs:
        raise RuntimeError("  - 未能构建任何有效的论据对。")

    # 步骤 2: 构建 (key1, value1, key2, value2) 列表，交由 LLM 判断一致性
    # matched_pairs 格式: (report_ev_key, ref_ev_key, report_ev_value, ref_ev_value)
    ev_quads = [
        (report_key, report_val, ref_key, ref_val)
        for report_key, ref_key, report_val, ref_val in matched_pairs
    ]

    print(f"  - 开始对 {len(ev_quads)} 对论据进行一致性判断...")
    results = await judge_consistency_for_evidence_pairs(ev_quads)

    consistent_count = sum(results)
    print(f"  - 一致性判断完成，{consistent_count} / {len(matched_pairs)} 对论据事实一致。")

    accuracy_ratio = consistent_count / len(matched_pairs)
    return coverage_ratio, accuracy_ratio