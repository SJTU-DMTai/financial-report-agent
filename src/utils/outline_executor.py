# 文件名: src/utils/outline_executor.py

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

from src.utils.outline_navigator import find_segment_by_id, find_evidence_by_id, find_parent_section_and_index
from src.memory.working import Section

def execute_instructions(manuscript: Section, instructions: List[Dict[str, Any]], original_path: Path) -> Section:
    """
    解析并执行一系列修正指令，修改 manuscript 对象，并将结果保存为 _outline2.json。
    original_path: 原始 outline.json 文件的路径，用于确定输出 _outline2.json 的位置。
    """
    print(f"--- Outline Executor: 收到 {len(instructions)} 条修正指令 ---")
    
    # 操作 manuscript 的一个深拷贝，避免修改原始对象
    try:
        updated_manuscript = Section.from_json(manuscript.to_json())
    except Exception as e:
        print(f"  ! 错误: 无法深拷贝 manuscript 对象，操作将在原始对象上进行。错误: {e}")
        updated_manuscript = manuscript

    for i, instruction in enumerate(instructions):
        inst_type = instruction.get("type")
        print(f"  - 正在执行第 {i+1} 条指令: type='{inst_type}'")
        
        if inst_type == "modify_evidence_field":
            evidence_id = instruction.get("evidence_id")
            field_to_modify = instruction.get("field")
            new_value = instruction.get("value")
            
            if not all([evidence_id, field_to_modify, new_value is not None]):
                print(f"    ! 警告: 指令格式不完整，跳过。指令: {instruction}")
                continue

            target_evidence = find_evidence_by_id(updated_manuscript, evidence_id)
            if target_evidence:
                if hasattr(target_evidence, field_to_modify):
                    setattr(target_evidence, field_to_modify, new_value)
                    print(f"    ✅ 成功: Evidence '{evidence_id}' 的字段 '{field_to_modify}' 已更新。")
                else:
                    print(f"    ❌ 失败: Evidence '{evidence_id}' 不存在字段 '{field_to_modify}'。")
            else:
                print(f"    ❌ 失败: 无法在 outline 中找到 Evidence '{evidence_id}'。")

        elif inst_type == "delete_evidence":
            evidence_id = instruction.get("evidence_id")
            if not evidence_id:
                print(f"    ! 警告: 指令格式不完整，缺少 evidence_id，跳过。")
                continue

            # 找到它的父 Segment
            segment_path = "_".join(evidence_id.split('_')[:-1])
            parent_segment = find_segment_by_id(updated_manuscript, segment_path)

            if parent_segment and parent_segment.evidences:
                initial_len = len(parent_segment.evidences)
                # 过滤掉要删除的 evidence
                parent_segment.evidences = [ev for ev in parent_segment.evidences if ev.evidence_id != evidence_id]
                if len(parent_segment.evidences) < initial_len:
                    print(f"    ✅ 成功: Evidence '{evidence_id}' 已从 Segment '{segment_path}' 中删除。")
                else:
                    print(f"    ❌ 失败: 在 Segment '{segment_path}' 的论据列表中未找到 Evidence '{evidence_id}'。")
            else:
                print(f"    ❌ 失败: 无法找到 Evidence '{evidence_id}' 的父 Segment '{segment_path}'。")
        
        # [TODO] 在这里添加更多指令类型的处理逻辑，例如 modify_segment_field, delete_segment
        
        else:
            print(f"    ! 警告: 未知或当前版本不支持的指令类型 '{inst_type}'，已跳过。")

    # 如果原文件是 _outline.json 结尾
    file_stem = original_path.stem # 获取文件名，去掉 .json
    
    # 将 _outline 替换为 _outline2
    if file_stem.endswith("_outline"):
        new_file_name = file_stem.replace("_outline", "_outline2") + ".json"
    else:
        new_file_name = file_stem + "_outline2.json"
        
    output_path = original_path.parent / new_file_name
    
    try:
        json_output = updated_manuscript.to_json(ensure_ascii=False, indent=4)
        output_path.write_text(json_output, encoding='utf-8')
        print(f"✅ 修改后的 outline 已保存至: {output_path.name}")
    except Exception as e:
        print(f"❌ 保存 {new_file_name} 失败: {e}")
    
    print(f"--- Outline Executor: 所有指令执行完毕 ---")
    return updated_manuscript