import sys
import json
import asyncio
from json import JSONDecodeError
from pathlib import Path
from typing import Tuple, List, Optional

from src.evaluation.extract_evidence import get_all_evidences_from_section, drop_duplicate_evidences, find_best_matches
from src.memory.working import Section
from src.utils.instance import create_chat_model
from src.utils.call_with_retry import call_chatbot_with_retry

from src.utils.instance import llm_reasoning, llm_instruct, formatter

async def evidence_coverage_and_accuracy(report_evidences: List[str], reference_evidences: List[str]) -> Tuple[float, float]:
    """
    统计覆盖到的论据中，有多少是正确的（与 reference 一致）。
    采用两阶段LLM调用：1. 精确配对；2. 批量判断。

    Args:
        report: AI生成的报告 Outline 对象。
        reference: 人类撰写的参考报告 Section 对象。
        ref_evidences: 参考报告的论据列表（可选，如不为None则使用此值，否则从reference提取）。

    Returns:
        一个元组 (coverage_ratio, accuracy_ratio)，代表覆盖率和准确率 (0.0 到 1.0)。
    """
    print(f"  - Report 论据数量: {len(report_evidences)}")
    print(f"  - Reference 论据数量: {len(reference_evidences)}")

    if not report_evidences or not reference_evidences:
        raise RuntimeError("  - 警告: Report 或 Reference 中缺少论据，无法计算准确率。")

    # 步骤 2: 找出共通论据 (以 report 中的表述为准)

    evidence_pairs_to_judge = await find_best_matches(report_evidences, reference_evidences)
    print(f"  - 构建的论据对 (evidence_pairs_to_judge): {evidence_pairs_to_judge}")

    # 最终共通论据的数量，是成功配对的数量
    final_common_count = len(evidence_pairs_to_judge)

    # 步骤 4: 计算比例
    coverage_ratio = final_common_count / len(reference_evidences)

    if not evidence_pairs_to_judge:
        raise RuntimeError("  - 未能构建任何有效的论据对。")

    print(f"\n  - 步骤 2/2: 将对 {len(evidence_pairs_to_judge)} 对论据进行批量事实一致性判断...")

    # --- [已修改] 第二轮LLM调用：批量判断 (增加分批逻辑) ---
    async def batch_judge_consistency(pairs: List[Tuple[str, str]], batch_size: int = 8) -> int:
        
        all_judgements = []
        # 将总任务切分成多个小批次
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            print(f"    - 正在处理批次 {i // batch_size + 1} / { -(-len(pairs) // batch_size) } (包含 {len(batch_pairs)} 对论据)...")

            judgement_list_formatted = "\n".join(
                f"  对 {j+1}:\n    - 论据A: {pair[0]}\n    - 论据B: {pair[1]}\n---"
                for j, pair in enumerate(batch_pairs)
            )

            prompt = f"""你是一名严谨、注重细节的事实核查员。你的任务是批量判断以下多个“论据对”中，“论据A”和“论据B”在事实上是否一致。

**任务指令:**
1.  逐一阅读下面提供的每一个“论据对”。
2.  对于**每一对**，独立思考并判断“论据A”的核心事实信息，是否与“论据B”中的信息一致。允许措辞和细节有所不同，但核心事实不能有矛盾。
3.  你的最终输出必须是一个合法的JSON数组（Array）。数组的长度必须与输入的“论据对”数量（本批次为 {len(batch_pairs)} 对）完全相同。
4.  数组中的第 N 个元素，对应你对第 N 个“论据对”的判断。判断结果只能是字符串 "一致" 或 "不一致"。

**重要规则:**
-   请严格按照顺序进行判断。
-   你的回答中**绝对不能包含**除了这个JSON数组之外的任何其他文字、解释或注释。"""
            judgement_results = await call_chatbot_with_retry(llm_instruct, formatter, prompt,
                                  f"**待判断的论据对列表:**\n---\n{judgement_list_formatted}\n---\n现在，请以单个JSON数组的格式，返回你对本批次所有论据对的判断结果列表。",
                                  hook=json.loads, handle_hook_exceptions=(JSONDecodeError, ))
            if isinstance(judgement_results, list):
                all_judgements.extend(judgement_results) # 将当前批次的结果汇总
            else:
                print(f"      ! 批次处理失败：模型返回的不是一个列表。")

        # 最终统计所有批次的结果
        consistent_count = sum(1 for result in all_judgements if isinstance(result, str) and "一致" in result)
        print(f"    -> 所有批次判断完成，总计有 {consistent_count} 对论据被判断为事实一致。")
        return consistent_count

    consistent_count = await batch_judge_consistency(evidence_pairs_to_judge)
    
    # 步骤 4: 计算准确率
    accuracy_ratio = consistent_count / len(evidence_pairs_to_judge)
    
    return coverage_ratio, accuracy_ratio
