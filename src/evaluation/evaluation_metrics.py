# 文件名: evaluation_metrics.py

import sys
import json
import asyncio
from pathlib import Path
from typing import Tuple, List

# --- 自动将项目根目录添加到系统路径 ---
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# --- 核心依赖导入 ---
try:
    from src.memory.working import Section
    from src.utils.instance import create_chat_model
    # 从你的批量脚本中导入 get_content_from_response，确保一致性
    from run_batch_comparison_v3 import get_content_from_response,find_semantic_common_evidences, get_all_evidences_from_section

except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在项目的根目录下运行此脚本，并且所有依赖已安装。")
    exit()


def num_of_segment(report: Section) -> Tuple[int, int]:
    """
    统计一篇研报中所有segment数量，以及每个section平均segment数量。

    Args:
        report: 代表整篇研报的根 Section 对象。

    Returns:
        一个元组 (total_segments, avg_segments_per_section)。
    """
    
    total_segments = 0
    total_sections = 0

    # 创建一个内部递归函数来遍历整个树状结构
    def _recursive_traverse(section: Section):
        nonlocal total_segments, total_sections
        
        # 只要它是一个Section/Subsection，计数器就+1
        # 根节点也算作一个section
        total_sections += 1
        
        # 累加当前section下的segment数量
        if section.segments:
            total_segments += len(section.segments)
            
        # 递归进入所有子章节
        if section.subsections:
            for subsection in section.subsections:
                _recursive_traverse(subsection)

    # 从根节点开始遍历
    if report:
        _recursive_traverse(report)
    
    # 计算平均值，避免除以零的错误
    if total_sections == 0:
        avg_segments_per_section = 0
    else:
        # 四舍五入取小数点后一位
        avg_segments_per_section = round(total_segments / total_sections, 1)
        
    return (total_segments, avg_segments_per_section)


async def structure_score(report: Section, llm_model) -> Tuple[int, int]:
    """
    调用LLM评估研报结构的完整性和逻辑性。

    Args:
        report: 代表整篇研报的根 Section 对象。
        llm_model: 用于评估的大模型实例。

    Returns:
        一个元组 (comprehensiveness_score, logicality_score)。出错时返回 (-1, -1)。
    """
    
    # 提取简洁的大纲文本
    outline = report.read(with_requirements=False, with_content=False, with_evidence=False, with_reference=False)
    prompt = f"""你是一名资深的金融分析师和首席编辑。你的任务是评估以下这份研究报告的大纲结构。

**评估维度:**
1.  **内容完整性 (Comprehensiveness)**: 大纲是否全面覆盖了分析一家公司所需的核心要素？例如：公司概况、行业分析、业务拆解、财务预测、估值、风险提示等。
2.  **逻辑连贯性 (Logicality)**: 大纲的章节和段落主题之间的组织是否合乎逻辑？论述是否层层递进，从宏观到微观，从现状到未来？

**评分标准:**
-   请在每个维度上给出 1-10 的整数分数。10分代表完美。

**输出格式:**
-   你的输出必须是一个合法的、不包含任何其他文字的JSON对象，格式如下：
    `{{"comprehensiveness": [分数], "logicality": [分数]}}`

**待评估的研报大纲:**
---
{outline}
---

现在，请给出你的评分。"""

    print("  - 正在调用大模型进行结构评估...")
    response_msg = await llm_model([{"role": "user", "content": prompt}])
    json_string = await get_content_from_response(response_msg)

    scores = json.loads(json_string)

    comprehensiveness = int(scores.get("comprehensiveness", -1))
    logicality = int(scores.get("logicality", -1))

    print("  - 评估完成。")
    return (comprehensiveness, logicality)

    

async def evidence_coverage(report: Section, reference: Section, llm_model) -> float:
    """
    统计 report 覆盖了 reference 多少比例的论据 (经过精确配对验证)。

    Args:
        report: AI生成的报告 Section 对象。
        reference: 人类撰写的参考报告 Section 对象 ("标准答案")。
        llm_model: 用于评估的大模型实例。

    Returns:
        一个浮点数，代表覆盖比例 (0.0 到 1.0)。出错时返回 -1.0。
    """
    print("\n--- [测试 3/4] 开始测试 evidence_coverage 函数 (精确配对模式) ---")
    
    # 步骤 1: 提取论据
    print("  - 正在从 report 和 reference 中提取论据...")
    report_evidences = get_all_evidences_from_section(report)
    reference_evidences = get_all_evidences_from_section(reference)
    
    print(f"  - Report 论据数量: {len(report_evidences)}")
    print(f"  - Reference 论据数量: {len(reference_evidences)}")
    
    if not reference_evidences:
        print("  - 警告: Reference (标准答案) 中没有论据，覆盖率计为 0.0。")
        return 0.0
        
    # 步骤 2: (粗筛) 找出 report 中可能与 reference 共通的论据
    print("  - 步骤 1/2: (粗筛) 找出共通论据...")
    common_texts_from_report = await find_semantic_common_evidences(report_evidences, reference_evidences, llm_model)

    if not common_texts_from_report:
        print("  - (粗筛) 未找到任何共通论据，覆盖率计为 0.0。")
        return 0.0
    
    # 步骤 3: (精炼) 为粗筛出的论据在 reference 中寻找唯一最佳匹配
    print(f"  - 步骤 2/2: (精炼) 为 {len(common_texts_from_report)} 条粗筛论据寻找精确匹配...")
    # [新逻辑] 调用独立的精确配对函数
    final_evidence_pairs = await find_best_matches(common_texts_from_report, reference_evidences, llm_model)
    
    # 最终共通论据的数量，是成功配对的数量
    final_common_count = len(final_evidence_pairs)
    
    # 步骤 4: 计算比例
    coverage_ratio = final_common_count / len(reference_evidences)
    
    return coverage_ratio

async def evidence_accuracy(report: Section, reference: Section, llm_model) -> float:
    """
    统计覆盖到的论据中，有多少是正确的（与 reference 一致）。
    采用两阶段LLM调用：1. 精确配对；2. 批量判断。

    Args:
        report: AI生成的报告 Section 对象。
        reference: 人类撰写的参考报告 Section 对象。
        llm_model: 用于评估的大模型实例。

    Returns:
        一个浮点数，代表准确率 (0.0 到 1.0)。出错时返回 -1.0。
    """
    print("\n--- [测试 4/4] 开始测试 evidence_accuracy 函数 (高效模式) ---")

    # 步骤 1: 提取论据
    print("  - 正在提取 report 和 reference 的论据...")
    report_evidences = get_all_evidences_from_section(report)
    reference_evidences = get_all_evidences_from_section(reference)

    print(f"  - Report 论据数量: {len(report_evidences)}")
    print(f"  - Reference 论据数量: {len(reference_evidences)}")

    if not report_evidences or not reference_evidences:
        print("  - 警告: Report 或 Reference 中缺少论据，无法计算准确率。")
        return 0.0

    # 步骤 2: 找出共通论据 (以 report 中的表述为准)
    common_texts_from_report = await find_semantic_common_evidences(report_evidences, reference_evidences, llm_model)
    
    if not common_texts_from_report:
        print("  - 未找到任何共通论据，无法计算准确率，计为 0.0。")
        return 0.0
        
    print(f"\n  - 步骤 1/2: 找到 {len(common_texts_from_report)} 条共通论据，将为它们在 Reference 中寻找最佳匹配...")
    
    # 第一轮LLM调用：精确配对 ---
    

    evidence_pairs_to_judge = await find_best_matches(common_texts_from_report, reference_evidences, llm_model)
    print(f"  - 构建的论据对 (evidence_pairs_to_judge): {evidence_pairs_to_judge}")


    if not evidence_pairs_to_judge:
        print("  - 未能构建任何有效的论据对，准确率计为 0.0。")
        return 0.0

    print(f"\n  - 步骤 2/2: 将对 {len(evidence_pairs_to_judge)} 对论据进行批量事实一致性判断...")

    # --- [已修改] 第二轮LLM调用：批量判断 (增加分批逻辑) ---
    async def batch_judge_consistency(pairs: List[Tuple[str, str]], llm, batch_size: int = 100) -> int:
        
        all_judgements = []
        # 将总任务切分成多个小批次
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            print(f"    - 正在处理批次 {i // batch_size + 1} / { -(-len(pairs) // batch_size) } (包含 {len(batch_pairs)} 对论据)...")

            judgement_list_formatted = "\n".join(
                f"  对 {j+1}:\n    - 论据A: {pair[0]}\n    - 论据B: {pair[1]}\n---"
                for j, pair in enumerate(batch_pairs)
            )

            prompt = f"""
            你是一名严谨、注重细节的事实核查员。你的任务是批量判断以下多个“论据对”中，“论据A”和“论据B”在事实上是否一致。

            **任务指令:**
            1.  逐一阅读下面提供的每一个“论据对”。
            2.  对于**每一对**，独立思考并判断“论据A”的核心事实信息，是否与“论据B”中的信息一致。允许措辞和细节有所不同，但核心事实不能有矛盾。
            3.  你的最终输出必须是一个合法的JSON数组（Array）。数组的长度必须与输入的“论据对”数量（本批次为 {len(batch_pairs)} 对）完全相同。
            4.  数组中的第 N 个元素，对应你对第 N 个“论据对”的判断。判断结果只能是字符串 "一致" 或 "不一致"。

            **重要规则:**
            -   请严格按照顺序进行判断。
            -   你的回答中**绝对不能包含**除了这个JSON数组之外的任何其他文字、解释或注释。

            **待判断的论据对列表:**
            ---
            {judgement_list_formatted}
            ---

            现在，请以单个JSON数组的格式，返回你对本批次所有论据对的判断结果列表。
            """
            try:
                response_msg = await llm([{"role": "user", "content": prompt}])
                json_string = await get_content_from_response(response_msg)
                if not json_string: continue # 如果某个批次失败，就跳过
                
                judgement_results = json.loads(json_string)
                
                if isinstance(judgement_results, list):
                    all_judgements.extend(judgement_results) # 将当前批次的结果汇总
                else:
                    print(f"      ! 批次处理失败：模型返回的不是一个列表。")

            except Exception as e:
                print(f"      ! 批次处理时发生严重错误: {e}")
        
        # 最终统计所有批次的结果
        consistent_count = sum(1 for result in all_judgements if isinstance(result, str) and "一致" in result)
        print(f"    -> 所有批次判断完成，总计有 {consistent_count} 对论据被判断为事实一致。")
        return consistent_count

    consistent_count = await batch_judge_consistency(evidence_pairs_to_judge, llm_model)
    
    # 步骤 4: 计算准确率
    accuracy_ratio = consistent_count / len(evidence_pairs_to_judge)
    
    return accuracy_ratio


async def find_best_matches(texts_to_match: List[str], source_texts: List[str], llm) -> List[Tuple[str, str]]:
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
        -   如果对于某个 "R_ID"，在 "S" 列表中找不到任何合适的匹配项，请在JSON对象中将其值设为 "not_found"。
        -   你的回答中**绝对不能包含**除了这个JSON对象之外的任何其他文字、解释或注释。

        **待匹配列表(R):**
        ---
        {texts_to_match_formatted}
        ---

        **源列表(S):**
        ---
        {source_texts_formatted}
        ---

        现在，请以单个JSON对象的格式返回所有R项的最佳匹配S项。
        """
        try:
            response_msg = await llm([{"role": "user", "content": prompt}])
            json_string = await get_content_from_response(response_msg)
            if not json_string: return []
            
            match_map_by_id = json.loads(json_string)
            
            evidence_pairs = []
            for i, r_text in enumerate(texts_to_match):
                r_id = f"R_{i+1}"
                s_id = match_map_by_id.get(r_id)
                
                if s_id and s_id != "not_found":
                    try:
                        s_idx = int(s_id.split('_')[1]) - 1
                        if 0 <= s_idx < len(source_texts):
                            s_text = source_texts[s_idx]
                            evidence_pairs.append((r_text, s_text))
                    except (ValueError, IndexError):
                        continue # 如果ID格式错误，则跳过
            
            print(f"    -> 配对完成，成功构建了 {len(evidence_pairs)} 对可供判断的论据。")
            return evidence_pairs
        except Exception as e:
            print(f"    ! 在寻找最佳匹配时发生错误: {e}")
            return []


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

    # 根据函数名调用不同的测试逻辑 
    if test_function_name == "num_of_segment":
        print("\n--- [测试 1/4] 开始测试 num_of_segment 函数 ---")
        total_segments, avg_segments = num_of_segment(report_section)
        print("--- 测试结果 ---")
        print(f"报告总段落 (Segment) 数量: {total_segments}")
        print(f"平均每个章节 (Section/Subsection) 包含的段落数量: {avg_segments}")
    
    elif test_function_name == "structure_score":
        print("\n--- [测试 2/4] 开始测试 structure_score 函数 ---")
        comp_score, logi_score = await structure_score(report_section, llm)
        print("--- 测试结果 ---")
        print(f"内容完整性 (Comprehensiveness) 得分: {comp_score} / 10")
        print(f"逻辑连贯性 (Logicality) 得分: {logi_score} / 10")

    elif test_function_name == "evidence_coverage":
        # `evidence_coverage` 函数内部已经包含了打印日志，这里直接调用即可
        coverage = await evidence_coverage(report_section, reference_section, llm)
        print("--- 测试结果 ---")
        print(f"论据覆盖率 (Evidence Coverage): {coverage:.2%}") # 格式化为百分比
    
    elif test_function_name == "evidence_accuracy":
        accuracy = await evidence_accuracy(report_section, reference_section, llm)
        print("--- 测试结果 ---")
        print(f"论据准确性 (Evidence Accuracy): {accuracy:.2%}") # 格式化为百分比

    else:
        print(f"错误: 未知的测试函数名称 '{test_function_name}'")

    print("--- 测试完成 ---")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 可选项: "num_of_segment", "structure_score", "evidence_coverage"，"evidence_accuracy"
    test_to_run = "evidence_coverage"
    
    asyncio.run(run_single_test(test_to_run))