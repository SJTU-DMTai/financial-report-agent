# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, TYPE_CHECKING

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.memory.working import Evidence, Section, Segment
from src.utils.get_entity_info import get_entity_info

if TYPE_CHECKING:
    from src.memory.long_term import LongTermMemoryStore


@dataclass
class EvidenceMergeItem:
    evidence: Evidence
    entity_key: str


def _normalize_company_key(entity: Optional[dict], default_entity: Optional[dict]) -> str:
    resolved = entity or default_entity or {}
    code = (resolved.get("code") or "").strip()
    if code:
        return code.zfill(6)
    name = (resolved.get("name") or "").strip()
    if name:
        return name
    return "__default_company__"


def _resolve_evidence_entity_key(
    long_term: Optional[LongTermMemoryStore],
    text: str,
    default_entity: Optional[dict],
) -> str:
    entity = None
    if long_term is not None:
        entity = get_entity_info(long_term, text)
    return _normalize_company_key(entity, default_entity)


def _tokenize_for_similarity(text: str) -> list[str]:
    return [token.strip() for token in jieba.cut(text or "", HMM=True) if token.strip()]


def _clean_for_similarity(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"\[\^cite_id[:=][^\]]+\]", "", cleaned)
    cleaned = re.sub(r"(查询|获取|确认|计算|了解|分析)", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.strip()


def _tfidf_similarity(text_a: str, text_b: str, analyzer: str) -> float:
    texts = [text_a, text_b]
    if not text_a or not text_b:
        return 0.0
    try:
        if analyzer == "word":
            vectorizer = TfidfVectorizer(
                tokenizer=_tokenize_for_similarity,
                token_pattern=None,
            )
        else:
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def evidence_semantic_similarity(text_a: str, text_b: str) -> float:
    cleaned_a = _clean_for_similarity(text_a)
    cleaned_b = _clean_for_similarity(text_b)
    word_score = _tfidf_similarity(cleaned_a, cleaned_b, "word")
    char_score = _tfidf_similarity(cleaned_a, cleaned_b, "char")
    sequence_score = SequenceMatcher(None, cleaned_a, cleaned_b).ratio()
    return 0.55 * word_score + 0.35 * char_score + 0.10 * sequence_score


def _build_merge_items(
    evidences: list[Evidence],
    long_term: Optional[LongTermMemoryStore],
    default_entity: Optional[dict],
) -> list[EvidenceMergeItem]:
    items = []
    for evidence in evidences:
        entity_key = _resolve_evidence_entity_key(long_term, evidence.text, default_entity)
        items.append(EvidenceMergeItem(evidence=evidence, entity_key=entity_key))
    return items


def _find_best_merge_pair(items: list[EvidenceMergeItem]) -> tuple[int, int] | None:
    best_pair = None
    best_score = -1.0
    for left_index, left_item in enumerate(items):
        if left_item.evidence.is_static:
            continue
        for right_index in range(left_index + 1, len(items)):
            right_item = items[right_index]
            if right_item.evidence.is_static:
                continue
            if left_item.entity_key != right_item.entity_key:
                continue
            score = evidence_semantic_similarity(left_item.evidence.text, right_item.evidence.text)
            if score > best_score:
                best_score = score
                best_pair = (left_index, right_index)
    return best_pair


def _merge_evidence_text(left_text: str, right_text: str) -> str:
    left = (left_text or "").strip()
    right = (right_text or "").strip()
    if not left:
        return right
    if not right:
        return left
    if left in right:
        return right
    if right in left:
        return left
    return f"{left}；{right}"


def merge_segment_evidences(
    segment: Segment,
    max_evidences: int,
    long_term: Optional[LongTermMemoryStore] = None,
    default_entity: Optional[dict] = None,
) -> bool:
    if max_evidences <= 0 or not segment.evidences:
        return False
    if len(segment.evidences) <= max_evidences:
        return False

    original_evidences = [evidence.text for evidence in segment.evidences]
    items = _build_merge_items(segment.evidences, long_term, default_entity)
    changed = False
    while len(items) > max_evidences:
        pair = _find_best_merge_pair(items)
        if pair is None:
            break
        left_index, right_index = pair
        left_item = items[left_index]
        right_item = items[right_index]
        merged_text = _merge_evidence_text(left_item.evidence.text, right_item.evidence.text)
        merged_evidence = Evidence(
            text=merged_text,
            is_static=False,
        )
        items[left_index] = EvidenceMergeItem(
            evidence=merged_evidence,
            entity_key=left_item.entity_key,
        )
        items.pop(right_index)
        changed = True

    if changed:
        segment.evidences = [item.evidence for item in items]
        merged_evidences = [evidence.text for evidence in segment.evidences]
        print(f"[Evidence Merge] before: {original_evidences}", flush=True)
        print(f"[Evidence Merge] after: {merged_evidences}", flush=True)
    return changed


def merge_outline_evidences(
    section: Section,
    max_evidences: int,
    long_term: Optional[LongTermMemoryStore] = None,
    default_entity: Optional[dict] = None,
) -> int:
    changed_count = 0
    for segment in section.segments or []:
        if merge_segment_evidences(segment, max_evidences, long_term, default_entity):
            changed_count += 1
    for subsection in section.subsections or []:
        changed_count += merge_outline_evidences(
            subsection,
            max_evidences,
            long_term,
            default_entity,
        )
    return changed_count
