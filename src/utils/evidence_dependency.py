# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from agentscope.formatter import FormatterBase
from agentscope.model import ChatModelBase
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.memory.evidence_registry import EvidenceRecord, EvidenceRegistry
from src.utils.call_with_retry import call_chatbot_with_retry


DEPENDENCY_SYS_PROMPT = """
你负责判断金融研报中论据（evidence）之间是否存在检索依赖关系。

定义：
若 evidence B 的搜索、解释或计算需要使用 evidence A 的结果，则认为 B 依赖 A，记为：
A -> B

判定原则：
1. 仅当 B 明显需要 A 的结果时，才建立依赖关系。
2. 背景信息、弱相关参考或仅有帮助但非必要的信息，不构成依赖。
3. 依赖关系必须形成有向无环图（DAG），不允许产生循环依赖。若发现双向依赖，应仅保留依赖更强的一条边。
4. 只允许判断输入 candidate_pairs 中列出的候选对，不要自行组合其它 evidence。
5. 输出符合要求的JSON格式。
"""


DEPENDENCY_TERMS = ["原因", "驱动", "影响", "风险", "假设", "预测", "解读", "优势", "机制", "对比", "估值"]

SUPPORTING_DATA_TERMS = ["数据", "规模", "增速", "数量", "月活", "成交额", "收入", "费用", "现金流", "净利润", "合同负债", "政策", "文件", "排名"]

TOPIC_GROUPS = [
    ["监管", "政策", "风险", "文件"],
    ["用户", "月活", "APP", "流量", "排名"],
    ["市场", "行业", "规模", "增速", "成交额", "投资者", "活跃度"],
    ["收入", "费用", "现金流", "净利润", "合同负债", "财务", "预测", "估值", "业绩"],
    ["AI", "产品", "研发", "技术", "功能", "合作"],
    ["竞争", "对手", "排名", "优势"],
]


def build_dependency_candidates(
    records: list[EvidenceRecord],
    target_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            # Keep the LLM dependency check small and conservative.
            if target_ids is not None and left.evidence_id not in target_ids and right.evidence_id not in target_ids:
                continue
            if should_check_dependency(left, right):
                candidates.append(_dependency_pair(left, right))
    return candidates


def should_check_dependency(left: EvidenceRecord, right: EvidenceRecord) -> bool:
    if left.canonical_key == right.canonical_key:
        return False
    left_entity = str(left.fields.get("entity") or "")
    right_entity = str(right.fields.get("entity") or "")
    if left_entity and right_entity and left_entity != right_entity:
        return False
    left_text = f"{left.description} {left.fields.get('aspect', '')}"
    right_text = f"{right.description} {right.fields.get('aspect', '')}"
    if evidence_semantic_similarity(left_text, right_text) >= 0.35:
        return True
    if _looks_like_explanation(left_text) != _looks_like_explanation(right_text):
        if _shares_financial_topic(left_text, right_text):
            return True
    if _has_dependency_signal(left_text, right_text):
        return True
    return False


def evidence_semantic_similarity(text_a: str, text_b: str) -> float:
    cleaned_a = clean_for_similarity(text_a)
    cleaned_b = clean_for_similarity(text_b)
    word_score = tfidf_similarity(cleaned_a, cleaned_b, "word")
    char_score = tfidf_similarity(cleaned_a, cleaned_b, "char")
    sequence_score = SequenceMatcher(None, cleaned_a, cleaned_b).ratio()
    return 0.55 * word_score + 0.35 * char_score + 0.10 * sequence_score


def clean_for_similarity(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"\[\^cite_id[:=][^\]]+\]", "", cleaned)
    cleaned = re.sub(r"(查询|获取|确认|计算|了解|分析)", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.strip()


def tokenize_for_similarity(text: str) -> list[str]:
    return [token.strip() for token in jieba.cut(text or "", HMM=True) if token.strip()]


def tfidf_similarity(text_a: str, text_b: str, analyzer: str) -> float:
    if not text_a or not text_b:
        return 0.0
    try:
        if analyzer == "word":
            vectorizer = TfidfVectorizer(tokenizer=tokenize_for_similarity, token_pattern=None)
        else:
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def _dependency_pair(left: EvidenceRecord, right: EvidenceRecord) -> dict[str, str]:
    return {
        "left_id": left.evidence_id,
        "left_description": left.description,
        "right_id": right.evidence_id,
        "right_description": right.description,
    }


def _looks_like_explanation(text: str) -> bool:
    return any(word in text for word in ["原因", "驱动", "影响因素", "归因", "解释"])


def _shares_financial_topic(left_text: str, right_text: str) -> bool:
    keywords = ["营业收入", "营收", "利润", "净利润", "毛利率", "增长", "费用率", "现金流"]
    return any(keyword in left_text and keyword in right_text for keyword in keywords)


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _shares_topic_group(left_text: str, right_text: str) -> bool:
    for group in TOPIC_GROUPS:
        if _contains_any(left_text, group) and _contains_any(right_text, group):
            return True
    return False


def _has_dependency_signal(left_text: str, right_text: str) -> bool:
    if not _shares_topic_group(left_text, right_text):
        return False
    left_dependent = _contains_any(left_text, DEPENDENCY_TERMS)
    right_dependent = _contains_any(right_text, DEPENDENCY_TERMS)
    left_supporting = _contains_any(left_text, SUPPORTING_DATA_TERMS)
    right_supporting = _contains_any(right_text, SUPPORTING_DATA_TERMS)
    return (left_dependent and right_supporting) or (right_dependent and left_supporting)


def build_dependency_user_prompt(candidates: list[dict[str, str]]) -> str:
    payload = {
        "candidate_pairs": candidates,
        "instruction": "只判断 candidate_pairs 中的候选对。若存在依赖，输出方向；若无明确依赖，不输出。",
        "output_format": {
            "dependencies": [
                {
                    "evidence_id": "后置 evidence_id",
                    "depends_on": "前置 evidence_id",
                    "reason": "简短原因",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_dependency_response(text: str) -> list[dict[str, str]]:
    raw = str(text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    payload = json.loads(raw)
    dependencies = payload.get("dependencies") or []
    parsed = []
    for item in dependencies:
        evidence_id = str(item.get("evidence_id") or "").strip()
        depends_on = str(item.get("depends_on") or "").strip()
        if evidence_id and depends_on:
            parsed.append({"evidence_id": evidence_id, "depends_on": depends_on})
    return parsed


def apply_dependency_response(registry: EvidenceRegistry, dependencies: list[dict[str, str]]) -> int:
    applied = 0
    for item in dependencies:
        if registry.add_dependency(item["evidence_id"], item["depends_on"]):
            applied += 1
    return applied


async def build_evidence_dependencies(
    registry: EvidenceRegistry,
    model: ChatModelBase,
    formatter: FormatterBase,
    target_ids: set[str] | None = None,
) -> int:
    records = registry.active_records()
    if len(records) < 2:
        return 0
    candidates = build_dependency_candidates(records, target_ids=target_ids)
    if not candidates:
        return 0
    user_prompt = build_dependency_user_prompt(candidates)
    response = await call_chatbot_with_retry(
        model,
        formatter,
        DEPENDENCY_SYS_PROMPT,
        user_prompt,
        hook=parse_dependency_response,
        handle_hook_exceptions=(json.JSONDecodeError, KeyError, TypeError),
    )
    return apply_dependency_response(registry, response)
