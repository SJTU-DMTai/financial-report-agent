# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any

from src.memory.working import Evidence


_SYNONYM_REPLACEMENTS = [
    ("营收", "营业收入"),
    ("境外业务", "海外业务"),
    ("海外市场业务", "海外业务"),
    ("毛利率变动", "毛利率变化"),
]


def normalize_evidence_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "", text)
    for old, new in _SYNONYM_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def normalize_evidence_period(value: Any) -> str:
    text = normalize_evidence_text(value)
    if not text:
        return ""
    fy_match = re.search(r"(20\d{2})(?:年)?(?:全年|年度|年报|FY|fy)", text)
    if fy_match:
        return f"{fy_match.group(1)}FY"
    quarter_match = re.search(r"(20\d{2})(?:年)?(?:第?([一二三四1-4])季(?:度|报)?|Q([1-4]))", text, re.IGNORECASE)
    if quarter_match:
        quarter = quarter_match.group(2) or quarter_match.group(3)
        quarter = {"一": "1", "二": "2", "三": "3", "四": "4"}.get(quarter, quarter)
        return f"{quarter_match.group(1)}Q{quarter}"
    return text


def evidence_to_fields(evidence: Evidence | dict[str, Any], default_entity: str | None = None) -> dict[str, str | bool]:
    if isinstance(evidence, Evidence):
        raw = evidence.model_dump()
    else:
        raw = dict(evidence or {})

    description = str(raw.get("description") or raw.get("text") or raw.get("evidence") or "").strip()
    entity = str(raw.get("entity") or default_entity or "").strip()
    aspect = str(raw.get("aspect") or description).strip()
    period = str(raw.get("period") or "").strip()
    scope = str(raw.get("scope") or "公司整体").strip()
    if not period:
        period = _infer_period_from_text(description)

    return {
        "description": description,
        "entity": entity,
        "aspect": aspect,
        "period": period,
        "scope": scope,
        "required": bool(raw.get("required", True)),
        "is_static": bool(raw.get("is_static", False)),
        "fact": str(raw.get("fact") or "").strip(),
    }


def _infer_period_from_text(text: str) -> str:
    normalized = normalize_evidence_period(text)
    if re.fullmatch(r"20\d{2}(?:FY|Q[1-4])", normalized):
        return normalized
    return ""


def build_canonical_key(fields: dict[str, Any]) -> str:
    entity = normalize_evidence_text(fields.get("entity"))
    aspect = normalize_evidence_text(fields.get("aspect") or fields.get("description"))
    period = normalize_evidence_period(fields.get("period"))
    scope = normalize_evidence_text(fields.get("scope") or "公司整体")
    return "|".join([entity, aspect, period, scope])


def canonical_key_for_evidence(evidence: Evidence | dict[str, Any], default_entity: str | None = None) -> str:
    fields = evidence_to_fields(evidence, default_entity)
    return build_canonical_key(fields)
