import sys
import json
import asyncio
from json import JSONDecodeError
from pathlib import Path
from typing import Tuple, List, Optional

from evaluation.extract_evidence import get_all_evidences_from_section, drop_duplicate_evidences
from src.memory.working import Section
from src.utils.instance import create_chat_model
from utils.call_with_retry import call_chatbot_with_retry

from src.utils.instance import llm_reasoning, llm_instruct, formatter

async def evidence_coverage_and_accuracy(report: Section, reference: Section, ref_evidences: Optional[List[str]] = None) -> Tuple[float, float]:
    """
    统计覆盖到的论据中，有多少是正确的（与 reference 一致）。
    采用两阶段LLM调用：1. 精确配对；2. 批量判断。

    Args:
        report: AI生成的报告 Section 对象。
        reference: 人类撰写的参考报告 Section 对象。
        ref_evidences: 参考报告的论据列表（可选，如不为None则使用此值，否则从reference提取）。

    Returns:
        一个元组 (coverage_ratio, accuracy_ratio)，代表覆盖率和准确率 (0.0 到 1.0)。
    """
    print("  - 正在提取 report 的论据...")
    report_evidences = get_all_evidences_from_section(report)

    # 如果提供了 ref_evidences，使用它；否则从 reference 中提取
    if ref_evidences is not None:
        print(f"  - 使用缓存的 reference 论据...")
        reference_evidences = ref_evidences
    else:
        print(f"  - 正在提取 reference 的论据...")
        reference_evidences = get_all_evidences_from_section(reference)

    print(f"  - Report 论据数量: {len(report_evidences)}")
    print(f"  - Reference 论据数量: {len(reference_evidences)}")

    if not report_evidences or not reference_evidences:
        raise RuntimeError("  - 警告: Report 或 Reference 中缺少论据，无法计算准确率。")

    # 步骤 2: 找出共通论据 (以 report 中的表述为准)
    common_texts_from_report = await drop_duplicate_evidences(report_evidences + reference_evidences)
    
    if not common_texts_from_report:
        raise RuntimeError("  - 未找到任何共通论据。")

    print(f"\n  - 步骤 1/2: 找到 {len(common_texts_from_report)} 条共通论据，将为它们在 Reference 中寻找最佳匹配...")
    
    # 第一轮LLM调用：精确配对 ---

    evidence_pairs_to_judge = await find_best_matches(common_texts_from_report, reference_evidences)
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


async def find_best_matches(texts_to_match: List[str], source_texts: List[str]) -> List[Tuple[str, str]]:
    # 为每个列表的项加上唯一ID
    texts_to_match_formatted = "\n".join(f"  R_{i+1}: {text}" for i, text in enumerate(texts_to_match))
    source_texts_formatted = "\n".join(f"  S_{i+1}: {text}" for i, text in enumerate(source_texts))

    prompt = f"""
    你是一个精准的语义匹配引擎。你的任务是从“源列表(S)”中，为“待匹配列表(R)”里的每一项找到与之**完全匹配或语义上非常相似**的匹配项。

    **任务指令:**
    1.  仔细阅读下面提供的两个列表：“待匹配列表(R)”和“源列表(S)”。
    2.  对于“待匹配列表(R)”中的**每一项**（如 R_1, R_2），请在“源列表(S)”中找到**一个**与之意思最接近的匹配项（如 S_5, S_12）。表述不一定完全一致只需意思高度相似即可。
    3.  你的输出必须是一个合法的JSON对象。对象的键（key）是“待匹配列表”中的项的ID（如 "R_1"），值（value）是你在“源列表”中找到的最佳匹配项的ID（如 "S_5"）。

    **重要规则:**
    -   如果对于某个 "R_ID"，在 "S" 列表中找不到任何合适的匹配项，请在JSON对象中将其值设为 "NONE"。
    -   你的回答中**绝对不能包含**除了这个JSON对象之外的任何其他文字、解释或注释。


    """
    match_map_by_id = await call_chatbot_with_retry(llm_reasoning, formatter, prompt,
                                  f"**待匹配列表(R):**\n---\n{texts_to_match_formatted}\n---\n"
                                  f"**源列表(S):**\n---\n{source_texts_formatted}\n---\n"
                                  f"现在，请以单个JSON对象的格式返回所有R项的最佳匹配S项。",
                                  hook=json.loads, handle_hook_exceptions=(JSONDecodeError, ))
    evidence_pairs = []
    for i, r_text in enumerate(texts_to_match):
        r_id = f"R_{i+1}"
        s_id = match_map_by_id.get(r_id)

        if s_id and s_id != "NONE":
            try:
                s_idx = int(s_id.split('_')[1]) - 1
                if 0 <= s_idx < len(source_texts):
                    s_text = source_texts[s_idx]
                    evidence_pairs.append((r_text, s_text))
            except (ValueError, IndexError):
                continue # 如果ID格式错误，则跳过

    print(f"    -> 配对完成，成功构建了 {len(evidence_pairs)} 对可供判断的论据。")
    return evidence_pairs


# ==============================================================================
# 独立的测试函数
# ==============================================================================

async def run_single_test(test_function_name: str):
    print(f"\n========== 开始执行评估指标测试: {test_function_name} ==========")
    
    # 加载测试文件和模型 
    report_path = Path("./data/temp_batch_v3/300972_2025-08-24_拥抱极致性价比与下沉时代（折扣系列）：万辰集团：万店之上的成长空间_outline.json")
    reference_path = Path("./data/temp_batch_v3/300972_2025-10-21_首次覆盖：引领量贩零食模式，具备较高盈利潜力_outline.json")
    
    if not report_path.exists() or not reference_path.exists():
        print(f"错误：测试输入文件未找到！请确保 '{report_path.name}' 和 '{reference_path.name}' 都存在。"); return

    print(f"加载 Report 文件: {report_path.name}")
    print(f"加载 Reference 文件: {reference_path.name}")
    try:
        report_section = Section.from_json(report_path.read_text(encoding='utf-8'))
        reference_section = Section.from_json(reference_path.read_text(encoding='utf-8'))
        print("文件解析成功！")
    except Exception as e:
        print(f"解析JSON文件时出错: {e}"); return
        
    print("\n--- 正在创建大模型实例... ---")
    try:
        llm = create_chat_model()
        print(f"模型 '{llm.model_name}' 创建成功！")
    except Exception as e:
        print(f"创建大模型实例时出错: {e}"); return

    accuracy = await evidence_coverage_and_accuracy(report_section, reference_section)
    print("--- 测试结果 ---")
    print(f"论据准确性 (Evidence Accuracy): {accuracy:.2%}") # 格式化为百分比

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 可选项: "num_of_segment", "structure_score", "evidence_coverage"，"evidence_accuracy"
    test_to_run = "evidence_accuracy"
    
    asyncio.run(run_single_test(test_to_run))