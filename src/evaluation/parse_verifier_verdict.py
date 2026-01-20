# src/evaluation/parse_verifier_verdict.py
import re
from typing import Dict, Any, List, Optional

def parse_verdict_common(text: str) -> Dict[str, Any]:
    """解析通用部分: PASSED / 原始文本"""
    result = {
        "passed": False,
        "problems": [],
        "raw_text": text,
    }
    # 查找 PASSED / FINAL_PASSED
    m = re.search(r"(?:PASSED|FINAL_PASSED):\s*(YES|NO)", text, re.I)
    if m:
        result["passed"] = m.group(1).upper() == "YES"
    return result

def extract_problem_blocks(text: str) -> List[str]:
    """提取 PROBLEMS 的所有问题块"""
    if "PROBLEMS:" not in text:
        return []
    problems_text = text.split("PROBLEMS:")[-1]
    # 按 1. 2. 3. 等分块
    blocks = [b.strip() for b in re.split(r'\n(?=\d+\.)', problems_text) if b.strip()]
    return blocks

def parse_numeric_verdict(text: str) -> Dict[str, Any]:
    result = parse_verdict_common(text)
    problems = []
    for block in extract_problem_blocks(text):
        if block.lower().startswith("无"):
            continue
        problem_data = {
            "type": "numeric",
            "description": "",
            "location": "",
            "expected": "",
            "actual": "",
            "material_id": None,
            "suggestion": ""
        }
        lines = block.split("\n")
        if lines:
            problem_data["description"] = lines[0].strip()
        for line in lines:
            # material_id 提取
            match = re.search(r'material_id[:\s]*\[?([^\],\s]+)\]?', line, re.I)
            if match:
                problem_data["material_id"] = match.group(1)
            if "原文:" in line:
                m = re.search(r'原文:\s*"([^"]+)"', line)
                if m:
                    problem_data["location"] = m.group(1)
            if "预期值:" in line:
                problem_data["expected"] = line.split("预期值:")[-1].strip()
            if "当前值:" in line:
                problem_data["actual"] = line.split("当前值:")[-1].strip()
            if "建议:" in line:
                problem_data["suggestion"] = line.split("建议:")[-1].strip()
        problems.append(problem_data)
    result["problems"] = problems
    return result

def parse_reference_verdict(text: str) -> Dict[str, Any]:
    result = parse_verdict_common(text)
    problems = []
    for block in extract_problem_blocks(text):
        if block.lower().startswith("无"):
            continue
        problem_data = {
            "type": "reference",
            "description": "",
            "location": "",
            "expected": None,
            "actual": None,
            "suggestion": ""
        }
        lines = block.split("\n")
        if lines:
            problem_data["description"] = lines[0].strip()
        for line in lines:
            match = re.search(r'material_id[:\s]*\[?([^\],\s]+)\]?', line, re.I)
            if match:
                problem_data["expected"] = match.group(1)
            if "位置:" in line:
                m = re.search(r'位置:\s*"([^"]+)"', line)
                if m:
                    problem_data["location"] = m.group(1)
            if "问题描述:" in line:
                problem_data["actual"] = line.split("问题描述:")[-1].strip()
            if "建议:" in line:
                problem_data["suggestion"] = line.split("建议:")[-1].strip()
        problems.append(problem_data)
    result["problems"] = problems
    return result

def parse_logic_verdict(text: str) -> Dict[str, Any]:
    result = parse_verdict_common(text)
    result["scores"] = {"logic": None, "language": None}
    m_logic = re.search(r"LOGICAL_CONSISTENCY[:_ ]?\s*([1-5])", text, re.I)
    if m_logic:
        result["scores"]["logic"] = int(m_logic.group(1))
    m_lang = re.search(r"LANGUAGE_QUALITY[:_ ]?\s*([1-5])", text, re.I)
    if m_lang:
        result["scores"]["language"] = int(m_lang.group(1))
    problems = []
    for block in extract_problem_blocks(text):
        if block.lower().startswith("无"):
            continue
        problem_data = {
            "type": "logic",
            "description": "",
            "location": "",
            "expected": "",
            "actual": "",
            "suggestion": ""
        }
        lines = block.split("\n")
        if lines:
            problem_data["description"] = lines[0].strip()
        for line in lines:
            if "位置:" in line:
                m = re.search(r'位置:\s*"([^"]+)"', line)
                if m:
                    problem_data["location"] = m.group(1)
            if "问题:" in line:
                problem_data["actual"] = line.split("问题:")[-1].strip()
            if "建议:" in line:
                problem_data["suggestion"] = line.split("建议:")[-1].strip()
        problems.append(problem_data)
    result["problems"] = problems
    return result

def parse_quality_verdict(text: str) -> Dict[str, Any]:
    result = parse_verdict_common(text)
    result["scores"] = {}
    result["average_score"] = None
    score_patterns = {
        "信息深度": r"信息深度[: ]\s*(\d+(\.\d+)?)",
        "洞察力": r"洞察力[: ]\s*(\d+(\.\d+)?)",
        "组织清晰度": r"组织清晰度[: ]\s*(\d+(\.\d+)?)",
        "表达精炼度": r"表达精炼度[: ]\s*(\d+(\.\d+)?)",
        "整体效果": r"整体效果[: ]\s*(\d+(\.\d+)?)",
    }
    for key, pattern in score_patterns.items():
        m = re.search(pattern, text)
        if m:
            result["scores"][key] = float(m.group(1))
    avg_match = re.search(r"平均分[: ]\s*([\d.]+)", text)
    if avg_match:
        result["average_score"] = float(avg_match.group(1))
    problems = []
    for block in extract_problem_blocks(text):
        if block.lower().startswith("无"):
            continue
        problem_data = {
            "type": "quality",
            "description": "",
            "location": "",
            "expected": "",
            "actual": "",
            "suggestion": ""
        }
        lines = block.split("\n")
        if lines:
            m = re.search(r'\[未达到参考的方面\]:\s*(.+)', lines[0])
            if m:
                problem_data["description"] = m.group(1)
        for line in lines:
            if "当前:" in line:
                m = re.search(r'当前:\s*"([^"]+)"', line)
                if m:
                    problem_data["actual"] = m.group(1)
            if "参考:" in line:
                m = re.search(r'参考:\s*"([^"]+)"', line)
                if m:
                    problem_data["expected"] = m.group(1)
            if "差距:" in line:
                problem_data["suggestion"] = line.split("差距:")[-1].strip()
        problems.append(problem_data)
    result["problems"] = problems
    return result

def parse_final_verdict(text: str) -> Dict[str, Any]:
    result = parse_verdict_common(text)
    m_conf = re.search(r"OVERALL_CONFIDENCE[: ]\s*([1-5])", text, re.I)
    result["confidence"] = int(m_conf.group(1)) if m_conf else None
    notes_match = re.search(r"NOTES[: ]\s*(.+)", text, re.S)
    result["notes"] = notes_match.group(1).strip() if notes_match else ""
    # 同时解析PROBLEMS
    problems = []
    for block in extract_problem_blocks(text):
        problem_data = {
            "type": "final",
            "description": block,
            "location": "",
            "expected": "",
            "actual": "",
            "suggestion": ""
        }
        problems.append(problem_data)
    result["problems"] = problems
    return result

VERDICT_PARSERS = {
    "numeric": parse_numeric_verdict,
    "reference": parse_reference_verdict,
    "logic": parse_logic_verdict,
    "quality": parse_quality_verdict,
    "final": parse_final_verdict,
}

def parse_verdict(verdict_type: str, text: str) -> Dict[str, Any]:
    parser = VERDICT_PARSERS.get(verdict_type)
    if not parser:
        raise ValueError(f"未知的verdict类型: {verdict_type}")
    return parser(text)
