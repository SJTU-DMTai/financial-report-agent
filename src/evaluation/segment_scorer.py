# src/evaluation/segment_scorer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Tuple

from agentscope.message import Msg
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory

from src.memory.working import Section, Segment
from src.utils.call_agent_with_retry import call_agent_with_retry


@dataclass
class SegmentScore:
    segment_id: str
    comprehensiveness: float
    insight: float
    instruction_following: float
    readability: float
    sufficiency: float

    def to_dict(self) -> Dict:
        return {
            "segment_id": self.segment_id,
            "scores": {
                "comprehensiveness": self.comprehensiveness,
                "insight": self.insight,
                "instruction_following": self.instruction_following,
                "readability": self.readability,
                "sufficiency": self.sufficiency
            }
        }


class SegmentScorer:
    """
    改进版 SegmentScorer：
    1. 为每个维度创建专用的评分prompt（而不是用生成标准的prompt）
    2. 保持简单性和鲁棒性
    3. 每个维度单独评分，但使用正确的评分prompt
    """

    def __init__(self, model, formatter):
        self.model = model
        self.formatter = formatter
        self.scoring_agent = self._create_scoring_agent()
        
        # 五个维度的评分prompt（简化版，用于1-5分评分）
        self.dimension_prompts = {
            "comprehensiveness": """请对以下报告片段的全面性进行1-5分评分（1分最低，5分最高）：
全面性指信息覆盖的广度、深度和相关性。

报告内容：
{content}

写作要点：
{topic}

请只输出一个1-5的整数，不要有其他任何文字。""",
            
            "insight": """请对以下报告片段的洞察力进行1-5分评分（1分最低，5分最高）：
洞察力指分析的深度、独到性、逻辑性和结论价值。

报告内容：
{content}

写作要点：
{topic}

请只输出一个1-5的整数，不要有其他任何文字。""",
            
            "instruction_following": """请对以下报告片段的指令遵循能力进行1-5分评分（1分最低，5分最高）：
指令遵循能力指报告是否准确、完整地回应了任务的所有要求和限定条件。

报告内容：
{content}

写作要点：
{topic}

请只输出一个1-5的整数，不要有其他任何文字。""",
            
            "readability": """请对以下报告片段的可读性进行1-5分评分（1分最低，5分最高）：
可读性指结构清晰度、语言流畅度、数据呈现效果和整体易理解性。

报告内容：
{content}

写作要点：
{topic}

请只输出一个1-5的整数，不要有其他任何文字。""",
            
            "sufficiency": """请对以下报告片段的论据充分性进行1-5分评分（1分最低，5分最高）：
充分性指所有论据是否足够支持论点，论据是否相关、有力、全面。

报告内容：
{content}

写作要点：
{topic}

请只输出一个1-5的整数，不要有其他任何文字。"""
        }

    def _create_scoring_agent(self) -> ReActAgent:
        return ReActAgent(
            name="Segment-Scorer",
            sys_prompt="你是一个专业的金融研报质量评估专家。请根据要求给出1-5分的整数评分。",
            model=self.model,
            memory=InMemoryMemory(),
            formatter=self.formatter,
            toolkit=None,
            parallel_tool_calls=False,
            max_iters=2,
        )

    def _extract_score(self, text: str) -> float:
        """从文本中提取1-5的分数"""
        try:
            # 查找1-5的数字
            match = re.search(r'[1-5]', text)
            if match:
                score = float(match.group())
                return max(1.0, min(5.0, score))
            
            # 如果没找到1-5，找任何数字并限制范围
            match = re.search(r'\d+', text)
            if match:
                score = float(match.group())
                score = max(1.0, min(5.0, score))
                return score
            
            return 3.0  # 默认值
        except:
            return 3.0

    async def _score_dimension(self, segment: Segment, dimension: str, segment_id: str) -> float:
        """对单个维度进行评分"""
        prompt_template = self.dimension_prompts[dimension]
        prompt = prompt_template.format(
            content=segment.content,
            topic=segment.topic
        )
        
        try:
            response = await call_agent_with_retry(
                self.scoring_agent,
                Msg("user", prompt, "user")
            )
            
            text = response.get_text_content()
            score = self._extract_score(text)
            
            # 清空记忆，避免上下文影响
            await self.scoring_agent.memory.clear()
            return score
            
        except Exception as e:
            print(f"[Scorer] {segment_id} - {dimension} 评分失败: {e}")
            await self.scoring_agent.memory.clear()
            return 3.0

    async def score_segment(self, segment: Segment, segment_id: str) -> SegmentScore:
        """对单个segment进行五个维度的评分"""
        # 并行执行五个维度的评分以提高效率
        tasks = [
            self._score_dimension(segment, "comprehensiveness", segment_id),
            self._score_dimension(segment, "insight", segment_id),
            self._score_dimension(segment, "instruction_following", segment_id),
            self._score_dimension(segment, "readability", segment_id),
            self._score_dimension(segment, "sufficiency", segment_id)
        ]
        
        scores = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_scores = []
        for i, score in enumerate(scores):
            if isinstance(score, Exception):
                print(f"[Scorer] 维度 {i} 评分异常: {score}")
                final_scores.append(3.0)
            else:
                final_scores.append(score)
        
        return SegmentScore(
            segment_id=segment_id,
            comprehensiveness=final_scores[0],
            insight=final_scores[1],
            instruction_following=final_scores[2],
            readability=final_scores[3],
            sufficiency=final_scores[4]
        )

    def calculate_average_scores(self, all_segment_scores: List[SegmentScore]) -> Tuple[float, float, float, float, float]:
        """计算五个维度的平均分"""
        if not all_segment_scores:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        n = len(all_segment_scores)
        
        return (
            sum(s.comprehensiveness for s in all_segment_scores) / n,
            sum(s.insight for s in all_segment_scores) / n,
            sum(s.instruction_following for s in all_segment_scores) / n,
            sum(s.readability for s in all_segment_scores) / n,
            sum(s.sufficiency for s in all_segment_scores) / n,
        )

    def content_score(self, report: Section, all_segment_scores: List[SegmentScore]) -> Tuple[float, float, float, float, float]:
        """对外接口：计算平均分"""
        return self.calculate_average_scores(all_segment_scores)