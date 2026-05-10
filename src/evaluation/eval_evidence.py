import json
from json import JSONDecodeError
from typing import List, Tuple

from pydantic import BaseModel

from src.evaluation.extract_evidence import find_best_matches_by_similarity


EvidenceTuple = Tuple[str, str]
EvidenceMatch = Tuple[str, str, str, str]


class EvidenceComparison(BaseModel):
    evidence: str
    value1: str
    value2: str
    is_same: bool


class EvidenceComparisons(BaseModel):
    result_list: List[EvidenceComparison]


def _normalize_evidences(evidences) -> List[EvidenceTuple]:
    normalized = []
    for item in evidences or []:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("evidence") or "").strip()
            value = str(item.get("value") or item.get("fact") or "").strip()
        elif isinstance(item, (list, tuple)) and item:
            text = str(item[0]).strip()
            value = str(item[1]).strip() if len(item) > 1 and item[1] is not None else ""
        else:
            text = str(item).strip()
            value = ""
        if text:
            normalized.append((text, value))
    return normalized


async def judge_consistency_for_evidence_pairs(evidences: List[EvidenceMatch]) -> List[bool]:
    numbered_evidences = "\n".join(
        f"<论据{i + 1}>\n"
        f"主题描述1: {pair[0]}\n"
        f"来源1具体事实: {pair[2] or pair[0]}\n"
        f"主题描述2: {pair[1]}\n"
        f"来源2具体事实: {pair[3] or pair[1]}\n"
        f"</论据{i + 1}>"
        for i, pair in enumerate(evidences)
    )

    example_json = json.dumps([
        {
            "evidence": "2024年营业收入",
            "value1": "2024年营业收入为52.3亿元",
            "value2": "2024年营收达到61.8亿元",
            "is_same": False,
        },
        {
            "evidence": "毛利率水平",
            "value1": "毛利率为38.5%",
            "value2": "毛利率约38.5%",
            "is_same": True,
        },
    ], ensure_ascii=False, indent=2)

    prompt = f"""你是一名严谨的金融事实核查员。你将收到若干论据对，每对包含两个来源的主题描述和具体事实。

核查流程：
1. 如果两个主题描述不是同一件事，或属于无法核验的主观判断，请将 is_same 置为 false。
2. 其他情况根据 value1 和 value2 是否相符判断。数字、单位、时间范围、主体等核心要素都必须一致；如果一方只是信息更完整且覆盖另一方，可以判为 true。

输出格式：
将所有论据对的判断汇总为 JSON 数组。数组中每个元素包含 evidence、value1、value2、is_same。

输出示例：
{example_json}"""

    user_msg = f"{numbered_evidences}\n\n请按流程逐条判断，并输出 JSON 数组。"
    from src.utils.call_with_retry import call_chatbot_with_retry
    from src.utils.instance import formatter, llm_reasoning

    comparisons = await call_chatbot_with_retry(
        llm_reasoning,
        formatter,
        prompt,
        user_msg,
        structured_model=EvidenceComparisons,
        handle_hook_exceptions=(JSONDecodeError,),
        max_retries=3,
    )
    return [comparison.is_same for comparison in comparisons.result_list]


async def evidence_coverage_and_accuracy(
    report_evidences: List[EvidenceTuple],
    reference_evidences: List[EvidenceTuple],
) -> Tuple[float, float]:
    report_evidences = _normalize_evidences(report_evidences)
    reference_evidences = _normalize_evidences(reference_evidences)

    print(f"  - Report 论据数量: {len(report_evidences)}")
    print(f"  - Reference 论据数量: {len(reference_evidences)}")

    if not report_evidences or not reference_evidences:
        raise RuntimeError("  - 警告: Report 或 Reference 中缺少论据，无法计算准确率。")

    matched_pairs = await find_best_matches_by_similarity(report_evidences, reference_evidences)
    print(f"  - 共配对到 {len(matched_pairs)} 对论据。")

    coverage_ratio = len(matched_pairs) / len(reference_evidences)
    if not matched_pairs:
        raise RuntimeError("  - 未能构建任何有效的论据对。")

    print(f"  - 开始对 {len(matched_pairs)} 对论据进行一致性判断...")
    results = await judge_consistency_for_evidence_pairs(matched_pairs)
    consistent_count = sum(results)
    print(f"  - 一致性判断完成，{consistent_count} / {len(matched_pairs)} 对论据事实一致。")

    accuracy_ratio = consistent_count / len(matched_pairs)
    return coverage_ratio, accuracy_ratio
