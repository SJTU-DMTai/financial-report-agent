import json
from json import JSONDecodeError
from typing import List, Tuple

from pydantic import BaseModel

from src.evaluation.extract_evidence import find_best_matches_by_similarity
from src.utils.call_with_retry import call_chatbot_with_retry

EvidenceTuple = Tuple[str, str]
EvidenceMatch = Tuple[str, str, str, str]
MAX_EVIDENCE_PAIRS_PER_JUDGE_CALL = 25


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
            value = str(item.get("fact") or "").strip()
        elif isinstance(item, (list, tuple)) and item:
            text = str(item[0]).strip()
            value = str(item[1]).strip() if len(item) > 1 and item[1] is not None else ""
        else:
            text = str(item).strip()
            value = ""
        if text:
            normalized.append((text, value))
    return normalized


async def evidence_coverage_and_accuracy(
    report_evidences: List[EvidenceTuple],
    reference_evidences: List[EvidenceTuple],
) -> Tuple[float, float, int]:
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

    evidence_quads: List[EvidenceMatch] = []
    missing_fact_count = 0
    for report_key, ref_key, report_value, ref_value in matched_pairs:
        if not report_value.strip() or not ref_value.strip():
            missing_fact_count += 1
            continue
        evidence_quads.append((report_key, report_value, ref_key, ref_value))

    if missing_fact_count:
        print(f"  - 跳过 {missing_fact_count} 对缺少具体事实的论据。")
    if not evidence_quads:
        print("  - 没有具备双方具体事实的有效论据对，accuracy 记为 0。")
        return coverage_ratio, 0.0, 0

    print(f"  - 开始对 {len(evidence_quads)} 对论据进行一致性判断...")
    comparisons: List[EvidenceComparison] = []
    total_batches = (len(evidence_quads) + MAX_EVIDENCE_PAIRS_PER_JUDGE_CALL - 1) // MAX_EVIDENCE_PAIRS_PER_JUDGE_CALL
    from src.utils.instance import formatter, llm_reasoning

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
1. 如果两个主题描述不是同一件事，或属于无法核验的主观判断，请直接跳过，不要在 result_list 中返回。
2. 其他情况根据 value1 和 value2 是否相符判断。数字、单位、时间范围、主体等核心要素都必须一致；如果一方只是信息更完整且覆盖另一方，可以判为 true。

输出格式：
输出一个 JSON 对象，字段 result_list 是数组。数组中每个元素包含 evidence、value1、value2、is_same。

输出示例：
{{"result_list": {example_json}}}"""

    for batch_idx in range(total_batches):
        start = batch_idx * MAX_EVIDENCE_PAIRS_PER_JUDGE_CALL
        end = start + MAX_EVIDENCE_PAIRS_PER_JUDGE_CALL
        batch = evidence_quads[start:end]
        print(
            f"  - 一致性判断分批 {batch_idx + 1}/{total_batches}: "
            f"{len(batch)} 对论据",
            flush=True,
        )
        numbered_evidences = "\n".join(
            f"<论据{i + 1}>\n"
            f"主题描述1: {pair[0]}\n"
            f"来源1具体事实: {pair[1]}\n"
            f"主题描述2: {pair[2]}\n"
            f"来源2具体事实: {pair[3]}\n"
            f"</论据{i + 1}>"
            for i, pair in enumerate(batch)
        )
        user_msg = f"{numbered_evidences}\n\n请按流程逐条判断，并输出包含 result_list 字段的 JSON 对象。"
        batch_comparisons = await call_chatbot_with_retry(
            llm_reasoning,
            formatter,
            prompt,
            user_msg,
            structured_model=EvidenceComparisons,
            handle_hook_exceptions=(JSONDecodeError,),
            max_retries=3,
        )
        if isinstance(batch_comparisons, dict):
            batch_comparisons = EvidenceComparisons(**batch_comparisons)
        comparisons.extend(batch_comparisons.result_list)

    valid_count = len(comparisons)
    skipped_by_judge = len(evidence_quads) - valid_count
    consistent_count = sum(1 for comparison in comparisons if comparison.is_same)
    if skipped_by_judge:
        print(f"  - LLM 跳过 {skipped_by_judge} 对错配或主观判断论据。")
    if valid_count == 0:
        print("  - LLM 未返回任何有效一致性判断，accuracy 记为 0。")
        return coverage_ratio, 0.0, 0

    print(f"  - 一致性判断完成，{consistent_count} / {valid_count} 对有效论据事实一致。")

    accuracy_ratio = consistent_count / valid_count
    return coverage_ratio, accuracy_ratio, consistent_count
