import json
import asyncio
import re
from collections import defaultdict
from json import JSONDecodeError
from typing import Tuple, List, Dict

from pydantic import BaseModel

from src.evaluation.extract_evidence import find_best_matches, build_seg_reference_map
from src.memory.working import Section
from src.utils.call_with_retry import call_chatbot_with_retry
from src.utils.instance import llm_reasoning, formatter


class EvidenceComparison(BaseModel):
    evidence: str   # Abstract description of what is being compared
    value1: str     # Specific information found in paragraph 1
    value2: str     # Specific information found in paragraph 2
    is_same: bool   # Whether value1 and value2 are factually identical


class EvidenceComparisons(BaseModel):
    result_list: List[EvidenceComparison]

async def judge_consistency_for_segment_pair(
    seg_text_1: str,
    seg_text_2: str,
    evidences: List[Tuple[str, str]],  # List of (evidence_from_1, evidence_from_2)
) -> List[bool]:
    """
    对同一个段落对中的若干论据，通过LLM在原始段落中定位具体信息并判断是否一致。

    Returns:
        List of bool, one per evidence pair (True = consistent).
    """
    numbered_evidences = "\n".join(
        f"<论据{i+1}>\n来源1: {pair[0]}\n来源2: {pair[1]}\n<论据{i+1}>"
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

    prompt = f"""你是一名严谨的金融事实核查员。你将收到两段研报原文，以及若干需要核查的"论据对"——每对论据分别来自两段原文，描述的是同一类信息。

**核查流程（对每一条论据对）：**
1. 如果"来源1论据"和"来源2论据"描述的不是同一件事（配对错误），请直接跳过。否则：
2. 在段落1中找到与"来源1论据"对应的具体数值/事实/信息，作为`value1`。
3. 在段落2中找到与"来源2论据"对应的具体数值/事实/信息，作为`value2`。
4. 判断`value1`和`value2`是否完全相符（数字、单位、时间范围、主体等核心要素必须完全一致），赋值为`is_same`。

**输出格式：**
如果任何一对"来源1论据"和"来源2论据"都匹配错误，并不指向同一件事，则最终结果请直接输出空数组。
否则，对于所有相关论据，注意判断其对应的value和is_same，将所有结果汇总为一个JSON数组。
数组中每个元素是一个对象，包含字段：evidence（抽象描述该论据的核心主题）、value1（段落1中找到的具体信息）、value2（段落2中找到的具体信息）、is_same（布尔值）。

**输出示例（假设有2条论据）：**
{example_json}"""

    user_msg = (
        f"<段落1>\n{seg_text_1}\n</段落1>\n\n"
        f"<段落2>\n{seg_text_2}\n</段落2>\n\n"
        f"{numbered_evidences}\n\n"
        f"请按流程逐条分析，并在最后输出JSON数组。"
    )

    def _parse(response: str) -> List[EvidenceComparison]:
        match = re.search(r"<ANSWER>\s*(\[.*?])\s*</ANSWER>", response, re.DOTALL)
        assert match is not None, "未找到<ANSWER>标签包裹的JSON数组。"
        raw_list = json.loads(match.group(1))
        assert isinstance(raw_list, list) and len(raw_list) == len(evidences), \
            f"输出数组长度 {len(raw_list)} 与论据数量 {len(evidences)} 不符。"
        return [EvidenceComparison(**item) for item in raw_list]

    comparisons = await call_chatbot_with_retry(
        llm_reasoning, formatter, prompt, user_msg,
        structured_model=EvidenceComparisons, handle_hook_exceptions=(JSONDecodeError, ), max_retries=3,
    )
    return [c["is_same"] for c in comparisons['result_list']]


async def evidence_coverage_and_accuracy(
    report_evidences: List[Tuple[str, str]],
    reference_evidences: List[Tuple[str, str]],
    report_section: Section,
    ref_section: Section,
) -> Tuple[float, float]:
    """
    统计论据覆盖率和准确率。
    流程：
      1. find_best_matches 得到配对后的 (report_ev, ref_ev, report_seg_id, ref_seg_id)。
      2. 按 (report_seg_id, ref_seg_id) 分组，同一段落对的论据合并到一个prompt。
      3. 对每个段落对调用 judge_consistency_for_segment_pair，传入原始段落文本。

    Args:
        report_evidences: List of (evidence_text, seg_id) from AI report.
        reference_evidences: List of (evidence_text, seg_id) from human reference.
        report_section: AI报告的Section对象（用于获取原始段落文本）。
        ref_section: 参考报告的Section对象（用于获取原始段落文本）。

    Returns:
        (coverage_ratio, accuracy_ratio)
    """
    print(f"  - Report 论据数量: {len(report_evidences)}")
    print(f"  - Reference 论据数量: {len(reference_evidences)}")

    if not report_evidences or not reference_evidences:
        raise RuntimeError("  - 警告: Report 或 Reference 中缺少论据，无法计算准确率。")

    # 步骤 1: 语义配对，得到四元组 (report_ev, ref_ev, report_seg_id, ref_seg_id)
    matched_pairs = await find_best_matches(report_evidences, reference_evidences)
    print(f"  - 共配对到 {len(matched_pairs)} 对论据。")

    coverage_ratio = len(matched_pairs) / len(reference_evidences)

    if not matched_pairs:
        raise RuntimeError("  - 未能构建任何有效的论据对。")

    # 步骤 2: 构建 seg_id → raw reference text 的映射
    report_seg_map = build_seg_reference_map(report_section)
    ref_seg_map = build_seg_reference_map(ref_section)

    # 步骤 3: 按 (report_seg_id, ref_seg_id) 分组
    groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for report_ev, ref_ev, report_seg_id, ref_seg_id in matched_pairs:
        groups[(report_seg_id, ref_seg_id)].append((report_ev, ref_ev))

    print(f"  - 共 {len(groups)} 个不同的段落对，开始逐对判断...")

    # 步骤 4: 并发对每个段落对调用 judge_consistency_for_segment_pair
    async def judge_group(seg_pair: Tuple[str, str], ev_pairs: List[Tuple[str, str]]) -> List[bool]:
        report_seg_id, ref_seg_id = seg_pair
        report_seg_text = report_seg_map[report_seg_id].replace("\n\n", "\n")
        ref_seg_text = ref_seg_map[ref_seg_id].replace("\n\n", "\n")
        print(f"    - 判断段落对 ({report_seg_id}, {ref_seg_id})，含 {len(ev_pairs)} 条论据...")
        return await judge_consistency_for_segment_pair(report_seg_text, ref_seg_text, ev_pairs)

    tasks = [judge_group(seg_pair, ev_pairs) for seg_pair, ev_pairs in groups.items()]
    group_results = await asyncio.gather(*tasks)

    # 步骤 5: 汇总结果，consistent_count 按配对顺序与 matched_pairs 对齐
    all_results: List[bool] = []
    for results in group_results:
        all_results.extend(results)

    consistent_count = sum(all_results)
    print(f"  - 一致性判断完成，{consistent_count} / {len(matched_pairs)} 对论据事实一致。")

    accuracy_ratio = consistent_count / len(matched_pairs)
    return coverage_ratio, accuracy_ratio