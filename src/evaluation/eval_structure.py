import sys
import json
import asyncio
from json import JSONDecodeError
from typing import Tuple, List

from src.memory.working import Section
from utils.call_with_retry import call_chatbot_with_retry

from src.utils.instance import llm_reasoning, formatter


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


async def structure_score(report: Section) -> Tuple[int, int]:
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
    sys_prompt = f"""你是一名资深的金融分析师和首席编辑。你的任务是评估以下这份研究报告的大纲结构。

**评估维度:**
1.  **内容完整性 (Comprehensiveness)**: 大纲是否全面覆盖了分析一家公司所需的核心要素？例如：公司概况、行业分析、业务拆解、财务预测、估值、风险提示等。
2.  **逻辑连贯性 (Logicality)**: 大纲的章节和段落主题之间的组织是否合乎逻辑？论述是否层层递进，从宏观到微观，从现状到未来？

**评分标准:**
-   请在每个维度上给出 1-10 的整数分数。10分代表完美。

**输出格式:**
-   你的输出必须是一个合法的、不包含任何其他文字的JSON对象，格式如下：
    `{{"comprehensiveness": [分数], "logicality": [分数]}}`



现在，请给出你的评分。"""
    scores = await call_chatbot_with_retry(llm_reasoning, formatter, sys_prompt,
                                           f"**待评估的研报大纲:**\n---\n{outline}\n---",
                                           hook=json.loads, handle_hook_exceptions=(JSONDecodeError, ))
    comprehensiveness = int(scores.get("comprehensiveness", -1))
    logicality = int(scores.get("logicality", -1))
    return (comprehensiveness, logicality)
