# 文件名: tests/test_executor.py

import sys
from pathlib import Path

# --- 自动将项目根目录添加到系统路径 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

# --- 导入 ---
try:
    from src.memory.working import Section
    from src.utils.outline_executor import execute_instructions, find_evidence_by_id
    from src.pipelines.planning import process_pdf_to_outline
except ImportError as e:
    print(f"!!! 导入失败: {e}")
    sys.exit(1)

TEST_OUTLINE_PATH = PROJECT_ROOT / "tests\\test_outputs\\simple_planning\\gpt-oss-120b/000333_2025-08-31_智慧家居全面发展，ToB业务多方突破_outline.json"

def run_executor_test():
    print("\n" + "="*60)
    print("      开始执行 Outline Executor 功能测试")
    print("="*60 + "\n")

    # 1. 加载数据
    if not TEST_OUTLINE_PATH.exists():
        print(f"!!! 错误: 找不到测试文件 {TEST_OUTLINE_PATH}"); return
    
    manuscript = Section.from_json(TEST_OUTLINE_PATH.read_text(encoding='utf-8'))
    print("✅ outline.json 加载成功。")

    # 2. 构造测试指令
    # 请确保这些ID在你的 JSON 中是真实存在的，且以 s0_... 开头
    mock_instructions = [
        {
            "type": "add_segment",
            "parent_segment_id": "s0_s1_p1", # [请修改] 在这个 Segment 之后插入
            "new_segment_data": {
                "topic": "【新增Segment】公司创新技术分析",
                "template": "公司在`[年份]`专注于`[技术领域]`的创新，取得了`[成果]`。",
                "requirements": "- 描述技术成果\n- 分析市场影响",
                "evidences": [
                    {"text": "查询公司最新技术报告", "is_static": False},
                    {"text": "评估技术市场潜力", "is_static": False}
                ]
            },
            "reason": "测试新增 Segment 功能"
        }
    ]


    mock_instructions_1 = [
        {
            "type": "add_evidence",
            "segment_id": "s0_s1_p1", # [请修改] 替换为一个真实存在的 segment_id
            "new_evidence": {
                "text": "【新增论据】这是一个由Planner新增的论据",
                "is_static": False,
                "value": None
            },
            "reason": "测试新增Evidence功能"
        },
        {
            "type": "delete_segment",
            "segment_id": "s0_s1_p2", # [请修改] 替换为一个你想要删除的真实 segment_id
            "reason": "测试删除Segment功能"
        },
        {
            "type": "delete_evidence",
            "evidence_id": "s0_s1_p1_e2",
        },
        # --- [新] 测试 modify_segment_field ---
        {
            "type": "modify_segment_field",
            "segment_id": "s0_s1_p1", # [请修改] 替换为一个真实存在的 segment_id
            "field": "topic",
            "value": "【已修正】这是一个新的段落主题",
            "reason": "测试修改topic"
        },
        {
            "type": "modify_segment_field",
            "segment_id": "s0_s1_p1", # [请修改] 同上
            "field": "template",
            "value": "【已修正】这是一个全新的写作模板，旧的已被替换。",
            "reason": "测试修改template"
        },
        {
            "type": "modify_segment_field",
            "segment_id": "s0_s1_p2", # [请修改] 替换为另一个真实存在的 segment_id
            "field": "evidences",
            # value 是一个字典列表，将被转换为 Evidence 对象列表
            "value": [
                {"text": "【新论据1】这是一个全新的论据", "is_static": False},
                {"text": "【新论据2】这是另一个全新的静态论据", "is_static": True, "value": "静态值示例"}
            ],
            "reason": "测试替换整个evidences列表"
        }
    ]

    # 3. 执行测试
    try:
        updated_manuscript = execute_instructions(manuscript, mock_instructions)
    
        # --- [新逻辑] 手动保存为 _outline2.json ---
        # 这里才是存放“业务操作”的地方
        new_path = TEST_OUTLINE_PATH.parent / (TEST_OUTLINE_PATH.stem.replace("_outline", "_outline2") + ".json")
        try:
            new_path.write_text(updated_manuscript.to_json(ensure_ascii=False, indent=4), encoding='utf-8')
            print(f"✅ 修正后的 outline 已保存至: {new_path.name}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

    except Exception as e:
        print(f"!!! 执行过程中发生异常: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("      Outline Executor 测试结束")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_executor_test()