# 文件名: tests/test_replan.py

import sys
import json
import asyncio
from pathlib import Path

# --- 自动将项目根目录添加到系统路径 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 导入 ---
try:
    from src.utils.outline_executor import execute_instructions
    from src.memory.working import Section
    from src.memory.feedback_models import Feedback
    from src.utils.instance import create_chat_model, create_agent_formatter
    from src.utils.defect_classifier import classify_defects
    from src.pipelines.replan_generator_baseline import generate_replan_instructions
    from src.pipelines.planning import process_pdf_to_outline
except ImportError as e:
    print(f"!!! 导入失败: {e}")
    sys.exit(1)

# [TODO] 替换为你 outline 文件中真实存在的路径
TEST_OUTLINE_PATH = PROJECT_ROOT / "tests\\test_outputs\\simple_planning\\gpt-oss-120b/000333_2025-08-31_智慧家居全面发展，ToB业务多方突破_outline.json"
async def run_replan_generator_test():
    print("\n" + "="*60)
    print("      开始测试 Planner 重规划指令生成")
    print("="*60 + "\n")

    # 1. 加载测试数据
    if not TEST_OUTLINE_PATH.exists():
        print(f"!!! 错误: 找不到测试文件 {TEST_OUTLINE_PATH}"); return
    manuscript = Section.from_json(TEST_OUTLINE_PATH.read_text(encoding='utf-8'))
    print("✅ outline.json 加载成功。")

    # 2. 构造模拟的错误反馈 (模拟多种类型)
    # 请确保这里的 segment_id 和 evidence_id 在你的 outline.json 中真实存在！
    mock_feedback_list = [
        # 单点错误
        #Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_s1_p1", evidence_id="s0_s1_s1_p1_e1", message="搜索失败"),
        
        # [新] Writer 错误
        #Feedback(agent_name="Writer", type="template_unfillable", segment_id="s0_s1_p1", message="模板中撰写不好。")
    
        # [新] 区域性错误 (s0_s1_p2 中大量失败)
        # Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p2", evidence_id="s0_s1_p2_e1", message="搜索失败"),
        # Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p2", evidence_id="s0_s1_p2_e2", message="搜索失败"),
        # Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p2", evidence_id="s0_s1_p2_e3", message="搜索失败"),
        # Feedback(agent_name="Searcher", type="search_failed", segment_id="s0_s1_p2", evidence_id="s0_s1_p2_e4", message="搜索失败"),

        # [新] 宏观完整性缺失，并指定了插入点
        # Feedback(
        #     type="missing_analysis_dimension", 
        #     segment_id="s0_s1_p4", # [请修改] 假设我们希望在 s0_s1_s4_p1 之后插入新内容
        #     message="报告缺少对公司未来风险的详细分析，建议补充。"
        # )

        # [新] 跨 Segment 矛盾
        Feedback(
            type="cross_segment_contradiction", 
            segment_id="s0_s1_p1",      # [请修改] 矛盾的第一个 Segment
            segment_id_2="s0_s1_p2",   # [请修改] 矛盾的第二个 Segment
            message="段落 A 的营收预测为增长，但段落 B 的风险分析中却基于营收下滑的假设。"
        )
    ]
    print(f"✅ 构造了 {len(mock_feedback_list)} 条模拟反馈。")

    # 3. 初始化 LLM
    try:
        llm = create_chat_model()
        formatter = create_agent_formatter()
        print(f"✅ 模型 '{llm.model_name}' 初始化成功。")
    except Exception as e:
        print(f"!!! 创建LLM失败: {e}"); return

    # 4. 执行分类与生成指令
    print("\n--- 正在调用分类器与指令生成器 ---")
    classified_defects = classify_defects(mock_feedback_list, manuscript)
    print("\n--- 分类结果 ---")
    serializable_data = {
        k: [item.model_dump() for item in v] 
        for k, v in classified_defects.items()
    }
    print(json.dumps(serializable_data, indent=2, ensure_ascii=False))
    print("--- 分类结果结束 ---\n")
        
    # 获取指令
    instructions = await generate_replan_instructions(
        classified_defects, 
        manuscript, 
        llm, 
        formatter
    )

    # 5. 打印指令 (这是你最关心的部分)
    print("\n" + "*"*30 + " 生成的重规划指令 " + "*"*30)
    print(json.dumps(instructions, indent=2, ensure_ascii=False))
    print("*"*80 + "\n")
    
    if instructions:
        print(f"✅ 成功生成了 {len(instructions)} 条重规划指令。")
    else:
        print("⚠️ 未生成任何修正指令，请检查 Prompt 或测试数据。")

    # 真实应用指令到 outline
    # if instructions:
    #     updated_manuscript = execute_instructions(manuscript, instructions)
    #     print("✅ 指令已应用到 outline。")

    #     # 可选：保存更新后的 outline
    #     updated_path = TEST_OUTLINE_PATH.with_name(TEST_OUTLINE_PATH.stem + "_updated.json")
    #     updated_path.write_text(updated_manuscript.to_json(ensure_ascii=False, indent=4), encoding="utf-8")
    # else:
    #     print("⚠️ 未生成任何修正指令，未执行修改。")

if __name__ == "__main__":
    asyncio.run(run_replan_generator_test())