# 文件名: tests/test_navigator.py

import sys
from pathlib import Path
import json
from typing import Optional

# --- 自动将项目根目录添加到系统路径 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 导入被测函数和数据结构 ---
try:
    from src.memory.working import Section, Segment, Evidence
    from src.utils.outline_navigator import find_segment_by_id, find_evidence_by_id
    from src.pipelines.planning import process_pdf_to_outline
    #from src.utils.instance import create_chat_model, create_agent_formatter
except ImportError as e:
    print(f"!!! 导入模块失败: {e}")
    print("!!! 请确保你已经从项目根目录运行此脚本（例如：python tests/simple_test_navigator.py）。")
    sys.exit(1)

# ==============================================================================
# 1. 测试配置
# ==============================================================================

# [TODO] 指定一个你已生成的、包含层级ID的 _outline.json 文件作为测试基础
TEST_OUTLINE_PATH = PROJECT_ROOT / "tests\\test_outputs\\simple_planning\\qwen3.5-flash/002594_2025-08-30_高端化战略：科技安全并举，重塑新能源豪华格局_outline.json"


# ==============================================================================
# 2. 测试函数 (不带断言)
# ==============================================================================

def test_find_segment_by_id_func(manuscript_root: Section):
    print("\n--- 开始测试 find_segment_by_id ---")
    
    # [TODO] 替换为一个你 outline 文件中真实存在的 segment_id
    # 确保这个ID是完整的、以 s0 开头的路径，例如 "s0_s1_s1_p1"
    existing_segment_id = "s0_s2_s2_p4" 
    
    print(f"尝试查找存在的 Segment ID: '{existing_segment_id}'")
    found_segment = find_segment_by_id(manuscript_root, existing_segment_id)
    
    if found_segment is not None:
        print(f"✅ 成功找到 Segment。其 ID 为: '{found_segment.segment_id}', topic 为: '{found_segment.topic}'")
    else:
        print(f"❌ 未能找到应该存在的 Segment '{existing_segment_id}'。请检查 ID 和文件内容。")

    non_existing_segment_id = "s0_s99_p99"
    print(f"尝试查找不存在的 Segment ID: '{non_existing_segment_id}'")
    not_found_segment = find_segment_by_id(manuscript_root, non_existing_segment_id)
    if not_found_segment is None:
        print("✅ 确认无法找到不存在的 Segment。")
    else:
        print(f"❌ 错误：找到了不应该存在的 Segment '{non_existing_segment_id}'。")


def test_find_evidence_by_id_func(manuscript_root: Section):
    print("\n--- 开始测试 find_evidence_by_id ---")
    
    # [TODO] 替换为一个你 outline 文件中真实存在的 evidence_id
    # 确保这个ID是完整的、以 s0 开头的路径，例如 "s0_s1_s1_p1_e1"
    existing_evidence_id = "s0_s2_s2_p4_e2"
    
    print(f"尝试查找存在的 Evidence ID: '{existing_evidence_id}'")
    found_evidence = find_evidence_by_id(manuscript_root, existing_evidence_id)
    
    if found_evidence is not None:
        print(f"✅ 成功找到 Evidence。其 ID 为: '{found_evidence.evidence_id}', text 为: '{found_evidence.text}'")
    else:
        print(f"❌ 未能找到应该存在的 Evidence '{existing_evidence_id}'。请检查 ID 和文件内容。")

    non_existing_evidence_id = "s0_s1_s1_p1_e99"
    print(f"尝试查找不存在的 Evidence ID: '{non_existing_evidence_id}'")
    not_found_evidence = find_evidence_by_id(manuscript_root, non_existing_evidence_id)
    if not_found_evidence is None:
        print("✅ 确认无法找到不存在的 Evidence。")
    else:
        print(f"❌ 错误：找到了不应该存在的 Evidence '{non_existing_evidence_id}'。")


# ==============================================================================
# 3. 主函数
# ==============================================================================

def main():
    
    print("\n" + "="*60)
    print("      开始执行 Outline Navigator 简单功能测试")
    print("="*60 + "\n")

    # --- 1. 准备工作：加载测试数据 ---
    print(f"--- 步骤 1: 加载测试用的 outline.json 文件 ---")
    if not TEST_OUTLINE_PATH.exists():
        print(f"!!! 错误：测试文件未找到！请确保 '{TEST_OUTLINE_PATH.name}' 存在于:")
        print(TEST_OUTLINE_PATH.parent)
        print("!!! 请先运行你的 Planner 脚本生成一个有效的 _outline.json 文件。")
        return
    
    manuscript_root: Optional[Section] = None
    try:
        json_text = TEST_OUTLINE_PATH.read_text(encoding='utf-8')
        manuscript_root = Section.from_json(json_text)
        print("✅ 测试用的 outline.json 文件加载成功。")
    except Exception as e:
        print(f"!!! 错误：解析 outline.json 文件时出错: {e}")
        print("!!! 请检查文件内容是否是合法的 JSON 格式，且符合 Section/Segment/Evidence 定义。")
        return
    
    if manuscript_root is None:
        print("!!! 错误：无法从文件加载 Section 对象。")
        return

    # --- 2. 调用测试函数 ---
    test_find_segment_by_id_func(manuscript_root)
    test_find_evidence_by_id_func(manuscript_root)

    print("\n" + "="*60)
    print("      Outline Navigator 简单功能测试执行结束")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()