# 文件名: src/memory/feedback_models.py

from pydantic import BaseModel, Field
from typing import Optional

class Feedback(BaseModel):
    """
    一个结构化的反馈对象，用于Agent之间传递错误或建议信息。
    """
    # 错误类型，例如 'search_failed', 'content_quality_low', 'missing_analysis_dimension'
    type: str
    
    # [修改] agent_name 变为可选项，宏观错误可以不提供
    agent_name: Optional[str] = Field(None, description="反馈的Agent名称，例如 'Searcher', 'Writer'")
    
    # 发生错误的 Segment 的完整路径ID (如 's1_p2')
    segment_id: Optional[str] = Field(None)
    
    # [新字段] 用于表示与 segment_id 存在矛盾的第二个 Segment
    segment_id_2: Optional[str] = Field(None, description="当 type 为 'cross_segment_contradiction' 时，表示第二个矛盾的 Segment ID")

    # 如果错误与特定论据相关，提供论据的完整路径ID (如 's1_p2_e1')
    evidence_id: Optional[str] = Field(None)
    
    # [修改] reason 字段改为 message
    message: str = Field(description="错误的详细描述或建议，例如 '修改建议内容' 或 '事实不符'")
    