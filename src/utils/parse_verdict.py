# -*- coding: utf-8 -*-
import re

def parse_verifier_verdict(verdict_text: str):
    """
    解析 Verifier 的结构化输出。
    返回:
    - passed: bool
    - problems: str | None
    - reason: str | None    # 未通过原因，固定由 TASK_COMPLETION / CORRECTNESS 生成
    """

    # 1. 解析第一部分：总体结论
    def _parse_bool(field_name: str):
        # 匹配形如 "- PASSED: YES" 的行，忽略大小写和多余空格
        m = re.search(
            rf"-\s*{field_name}\s*:\s*(YES|NO)",
            verdict_text,
            flags=re.IGNORECASE
        )
        if not m:
            return None
        return m.group(1).strip().upper() == "YES"

    passed = _parse_bool("PASSED")
    task_completion = _parse_bool("TASK_COMPLETION")
    correctness = _parse_bool("CORRECTNESS")

    # 如果没解析到 PASSED，就一律当作未通过
    if passed is None:
        passed = False

    # 2. 抽取第二部分：PROBLEMS 段落（如果存在）
    problems = None
    m_problems = re.search(
        r"PROBLEMS:\s*(.*)$",
        verdict_text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if m_problems:
        problems = m_problems.group(1).strip()


    reason = None
    if not passed:
        if task_completion is False and correctness is True:
            reason = "任务未完全按照outline要求完成"
        elif task_completion is True and correctness is False:
            reason = "事实或引用正确性存在问题"
        elif task_completion is False and correctness is False:
            reason = "任务未完全按照outline要求完成，且存在事实正确性问题"

    return passed, problems, reason

