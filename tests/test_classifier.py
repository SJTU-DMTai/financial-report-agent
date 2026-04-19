# 文件名: tests/simple_test_classifier.py

import sys
from pathlib import Path
import json

# --- 自动将项目根目录添加到系统路径 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 导入 ---
try:
    from src.memory.working import Section
    from src.memory.feedback_models import Feedback
    from src.utils.defect_classifier import classify_defects, DefectType
    from src.utils.outline_navigator import find_segment_by_id
    from src.pipelines.planning import process_pdf_to_outline
except ImportError as e:
    print(f"!!! 导入失败: {e}")
    sys.exit(1)

# [TODO] 替换为你 outline 文件中真实存在的路径
TEST_OUTLINE_PATH = PROJECT_ROOT / "tests\\test_outputs\\simple_planning\\gpt-oss-120b/000333_2025-08-31_智慧家居全面发展，ToB业务多方突破_outline.json"


def run_classifier_test():
    print("\n" + "="*60)
    print("      开始执行 Defect Classifier 功能测试")
    print("="*60 + "\n")

    # --- 1. 准备测试数据 ---
    
    # 加载 manuscript
    if not TEST_OUTLINE_PATH.exists():
        print(f"!!! 错误: 找不到测试文件 {TEST_OUTLINE_PATH}"); return
    
    manuscript = Section.from_json(TEST_OUTLINE_PATH.read_text(encoding='utf-8'))
    print("✅ outline.json 加载成功。")

    # [TODO] 手动构造模拟的错误反馈列表
    # 确保这些 segment_id 在你的 JSON 中是真实存在的
    mock_feedback_list = [
        # 1. 模拟单点 Evidence 错误 (s1_s1_p1 中只有一个失败)
        Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p1", evidence_id="s0_s1_p1_e1", message="搜索失败"),
        Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p4", evidence_id="s0_s1_p4_e1", message="搜索失败"),
        
        # 2. 模拟区域性 Segment 错误 (s1_s2_p1 中大量失败)
        Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p2", evidence_id="s0_s1_p2_e1", message="搜索失败"),
        Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p2", evidence_id="s0_s1_p2_e2", message="搜索失败"),
        Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p2", evidence_id="s0_s1_p2_e3", message="搜索失败"),
        Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p3", evidence_id="s0_s1_p3_e1", message="搜索失败"),
        Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p3", evidence_id="s0_s1_p3_e2", message="搜索失败"),
        Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p3", evidence_id="s0_s1_p3_e3", message="搜索失败"),
        
        
        # 3. 模拟 Writer 撰写错误
        Feedback(agent_name="Writer", type="content_quality_low", segment_id="s0_s1_p3", message="内容质量得分低"),
        
        # 4. 模拟跨 Segment 逻辑矛盾
        Feedback(type="cross_segment_contradiction", message="Segment s1_p1 和 s2_p1 的结论存在矛盾"),
        
        # 5. 模拟宏观完整性缺失
        Feedback(
            type="missing_analysis_dimension", 
            segment_id="s0_s1_p4", # [请修改] 假设我们希望在 s0_s1_s4_p1 之后插入新内容
            message="报告缺少对公司未来风险的详细分析，建议补充。"
        )
    ]
    print(f"✅ 成功构造了 {len(mock_feedback_list)} 条模拟反馈。")

    # --- 2. 调用分类器 ---
    print("\n--- 正在调用 Defect Classifier... ---")
    try:
        classified_defects = classify_defects(mock_feedback_list, manuscript)
        print("\n--- 分类器执行完毕，结果如下 ---")
        # 美化打印结果
        print(classified_defects)
        pretty_print_defects(classified_defects)
    except Exception as e:
        print(f"!!! 分类器执行时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print("      Defect Classifier 功能测试执行结束")
    print("="*60 + "\n")


def pretty_print_defects(classified_defects):
    normalized = {}
    for defect_type, items in classified_defects.items():
        normalized[defect_type] = []
        for item in items:
            if isinstance(item, dict):
                normalized[defect_type].append(item)
            elif hasattr(item, "model_dump"):
                normalized[defect_type].append(item.model_dump())
            else:
                normalized[defect_type].append(item.__dict__ if hasattr(item, "__dict__") else str(item))
    print(json.dumps(normalized, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_classifier_test()