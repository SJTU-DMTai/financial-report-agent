# 文件名: src/utils/defect_classifier.py

from __future__ import annotations
from typing import List, Dict
from collections import defaultdict

from src.memory.feedback_models import Feedback
from src.memory.working import Section
from src.utils.outline_navigator import find_segment_by_id

class DefectType:
    SINGLE_EVIDENCE_ERROR = "single_evidence_error"
    REGIONAL_SEGMENT_ERROR = "regional_segment_error"
    WRITER_ERROR = "writer_error"
    CROSS_SEGMENT_CONTRADICTION = "cross_segment_contradiction"
    MACRO_INCOMPLETENESS = "macro_incompleteness"

SINGLE_POINT_ERROR_THRESHOLD = 0.3

def classify_defects(feedback_list: List[Feedback], manuscript: Section) -> Dict[str, List[Feedback]]:
    """
    根据结构化的反馈列表，自动将 outline 的缺陷分类。
    这个函数不调用LLM，完全基于规则和统计。

    Args:
        feedback_list: 结构化 feedback 对象列表。
        manuscript: 当前的 manuscript (Section) 对象。

    Returns:
        一个按缺陷类型分类的字典，例如 {"single_evidence_error": [...], ...}
    """
    print(f"--- Defect Classifier: 开始对 {len(feedback_list)} 条反馈进行分类 ---")
    
    classified_defects = defaultdict(list)
    
    # 1. 优先处理宏观错误 (逻辑清晰，不受 segment_id 干扰)
    for fb in feedback_list:
        if fb.type == "cross_segment_contradiction":
            classified_defects[DefectType.CROSS_SEGMENT_CONTRADICTION].append(fb)
            print(f"  - 分类: 发现 '跨 Segment 逻辑矛盾' 问题")
        elif fb.type == "missing_analysis_dimension":
            classified_defects[DefectType.MACRO_INCOMPLETENESS].append(fb)
            print(f"  - 分类: 发现 '宏观完整性缺失' 问题")
    
    # 2. 处理段落级错误 (基于 segment_id)
    feedbacks_by_segment = defaultdict(list)
    for fb in feedback_list:
        if fb.segment_id and fb.type not in ["cross_segment_contradiction", "missing_analysis_dimension"]:
            feedbacks_by_segment[fb.segment_id].append(fb)

    for segment_id, segment_feedbacks in feedbacks_by_segment.items():
        target_segment = find_segment_by_id(manuscript, segment_id)
        if not target_segment:
            print(f"  ! 警告: 找不到 Segment '{segment_id}'，跳过。")
            continue

        # [修改] 将 Searcher 和 Writer 的失败完全分开处理
        searcher_failures = [fb for fb in segment_feedbacks if fb.agent_name == "Searcher"]
        writer_failures = [fb for fb in segment_feedbacks if fb.agent_name == "Writer"]
        
        # --- 处理 Searcher 相关的失败 ---
        if searcher_failures:
            total_evidences_in_segment = len(target_segment.evidences) if target_segment.evidences else 1
            failure_rate = len(searcher_failures) / total_evidences_in_segment
            
            if failure_rate <= SINGLE_POINT_ERROR_THRESHOLD:
                # 失败率低，归类为“单点 Evidence 错误”
                # 这里依然是将每个feedback单独放入，因为Planner需要逐个修正
                classified_defects[DefectType.SINGLE_EVIDENCE_ERROR].extend(searcher_failures)
                print(f"  - 分类: Segment '{segment_id}' 被诊断为 '单点 Evidence 错误' (失败率: {failure_rate:.1%})")
            else:
                # [修改] 失败率高，归类为“区域性 Segment 错误”
                # 输出结构改变：将这个 Segment 的所有反馈打包成一个对象
                regional_error_entry = {
                    "segment_id": segment_id,
                    "reason": f"该 Segment 的 {total_evidences_in_segment} 条论据中，有 {len(searcher_failures)} 条搜索失败，失败率高。",
                    "related_feedbacks": [fb.model_dump() for fb in searcher_failures] # 将相关的feedback打包
                }
                classified_defects[DefectType.REGIONAL_SEGMENT_ERROR].append(regional_error_entry)
                print(f"  - 分类: Segment '{segment_id}' 被诊断为 '区域性 Segment 错误' (失败率: {failure_rate:.1%})")

        # --- 处理 Writer 相关的失败 (独立处理) ---
        if writer_failures:
            # Writer 的失败通常与整个 Segment 的规划有关，
            # 直接保留 Feedback 对象，方便后续 Planner 直接读取 segment_id/message
            classified_defects[DefectType.WRITER_ERROR].extend(writer_failures)
            print(f"  - 分类: Segment '{segment_id}' 被诊断为 'Writer 撰写错误'")
            

    print("--- Defect Classifier: 分类完成 ---")
    return dict(classified_defects)