# 文件名: src/utils/outline_navigator.py

from __future__ import annotations
from typing import Optional, Tuple

from src.memory.working import Section, Segment, Evidence


def find_evidence_by_id(manuscript_root: Section, evidence_id: str) -> Optional[Evidence]:
    """
    根据完整的路径ID (例如 "s0_s1_s2_p3_e1")，在 manuscript 树中查找并返回对应的 Evidence 对象。
    如果找不到，返回 None。
    """
    if not evidence_id or not isinstance(evidence_id, str):
        return None

    parts = evidence_id.split('_')
    if len(parts) < 2 or not parts[-1].startswith('e'):
        return None

    segment_path = "_".join(parts[:-1])
    parent_segment = find_segment_by_id(manuscript_root, segment_path)
    
    if not parent_segment or not parent_segment.evidences:
        return None
        
    for evidence in parent_segment.evidences:
        if evidence.evidence_id == evidence_id:
            return evidence
            
    return None


def find_segment_by_id(manuscript_root: Section, segment_id: str) -> Optional[Segment]:
    """
    根据完整的路径ID (例如 "s0_s1_s2_p3")，在 manuscript 树中查找并返回对应的 Segment 对象。
    能同时兼容 "s0_..." 和 "s1_..." 开头的路径。
    如果找不到，返回 None。
    """
    if not segment_id or not isinstance(segment_id, str):
        return None

    parts = segment_id.split('_')
    if not parts[-1].startswith('p'):
        return None
        
    section_path_parts = parts[:-1]
    current_section_node = manuscript_root

    # [新逻辑] 如果路径以 's0' 开头，则跳过第一个部分，直接从根节点的子节点开始查找
    start_index = 0
    if section_path_parts and section_path_parts[0] == 's0':
        # 检查根节点ID是否匹配 (可选，增加健壮性)
        if manuscript_root.section_id != 0:
            print(f"警告: 路径以 's0' 开头，但根节点的 section_id 不是 0。")
        start_index = 1 # 从 's1' 部分开始遍历

    # 遍历章节层级路径
    for i in range(start_index, len(section_path_parts)):
        part = section_path_parts[i]
        if not part.startswith('s'): return None
        try:
            target_section_id = int(part[1:])
        except ValueError:
            return None
            
        found_subsection = None
        if current_section_node.subsections:
            for sub in current_section_node.subsections:
                if sub.section_id == target_section_id:
                    found_subsection = sub
                    break
        
        if not found_subsection:
            return None
        current_section_node = found_subsection
        
    # 在最终找到的父 Section 内查找 Segment
    if current_section_node.segments:
        for segment in current_section_node.segments:
            if segment.segment_id == segment_id:
                return segment

    return None

def find_parent_section_and_index(manuscript_root: Section, segment_id_path: str) -> Tuple[Optional[Section], Optional[int]]:
    """
    根据 segment 的完整路径ID，找到其父 Section 和它在父 Section.segments 列表中的索引。
    """
    parts = segment_id_path.split('_')
    if not parts[-1].startswith('p'): return None, None
    
    parent_path_parts = parts[:-1]
    parent_section = manuscript_root
    
    # [新逻辑] 同样处理 's0' 开头的路径
    start_index = 0
    if parent_path_parts and parent_path_parts[0] == 's0':
        start_index = 1

    for i in range(start_index, len(parent_path_parts)):
        part = parent_path_parts[i]
        if not part.startswith('s'): return None, None
        target_section_id = int(part[1:])
        found_sub = None
        if parent_section.subsections:
            for sub in parent_section.subsections:
                if sub.section_id == target_section_id:
                    found_sub = sub; break
        if not found_sub: return None, None
        parent_section = found_sub
    
    if parent_section and parent_section.segments:
        for idx, seg in enumerate(parent_section.segments):
            if seg.segment_id == segment_id_path:
                return parent_section, idx
                
    return None, None