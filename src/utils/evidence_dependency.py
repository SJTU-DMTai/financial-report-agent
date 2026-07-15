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
from src.prompt import prompt_dict
from src.utils.call_with_retry import call_chatbot_with_retry


FINANCIAL_TOPIC_KEYWORDS = [
    "营业收入", "营业总收入", "营收", "收入", "利润", "净利润", "归母净利润", "扣非净利润",
    "盈利", "业绩", "毛利", "毛利率", "净利率", "成本", "费用", "费用率", "销售费用",
    "管理费用", "研发费用", "财务费用", "现金流", "现金流量", "经营活动现金流", "合同负债",
    "增长", "增速", "EPS", "每股收益", "PE", "市盈率", "PB", "市净率", "ROE", "ROA",
    "估值", "目标价",
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
    return any(keyword in left_text and keyword in right_text for keyword in FINANCIAL_TOPIC_KEYWORDS)


def build_dependency_user_prompt(candidates: list[dict[str, str]]) -> str:
    payload = {
        "candidate_pairs": candidates,
        "instruction": "只判断 candidate_pairs 中的候选对。若存在依赖，输出方向；若无明确依赖，不输出。",
        "output_format": {
            "dependencies": [
                {
                    "evidence_id": "后置 evidence_id",
                    "depends_on": "前置 evidence_id",
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
        prompt_dict["evidence_dependency_sys_prompt"],
        user_prompt,
        hook=parse_dependency_response,
        handle_hook_exceptions=(json.JSONDecodeError, KeyError, TypeError),
    )
    return apply_dependency_response(registry, response)
