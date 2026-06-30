# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from src.memory.evidence_registry import EvidenceRegistry
from src.memory.tracking_board import SegmentIssue
from src.utils.format import _strip_section_number_prefix, extract_tagged_text

if TYPE_CHECKING:
    from src.memory.evidence_registry import EvidenceRecord
    from src.memory.short_term import ShortTermMemoryStore


CITE_ID_RE = re.compile(r"\[\^\s*(?:(?:cite_id)\s*[:=]\s*)?([^\]\|\s]+)(\|[^\]]*)?\]", re.IGNORECASE)


def extract_cite_ids(text: str) -> list[str]:
    cite_ids = []
    for match in CITE_ID_RE.finditer(text or ""):
        cite_id = match.group(1).strip()
        if cite_id not in cite_ids:
            cite_ids.append(cite_id)
    return cite_ids


def _normalize_cite_marker_match(match: re.Match) -> str:
    cite_id = match.group(1).strip()
    suffix = match.group(2) or ""
    return f"[^cite_id:{cite_id}{suffix}]"


def normalize_cite_markers(text: str) -> str:
    return CITE_ID_RE.sub(_normalize_cite_marker_match, text or "")


def is_unavailable_marker(text: str) -> bool:
    return str(text or "").strip().upper() == "UNAVAILABLE"


def apply_search_result_to_evidence_record(record: "EvidenceRecord", result_text: str, status: str = "") -> bool:
    normalized_status = str(status or "").strip().upper()
    stripped_text = str(result_text or "").strip()
    if normalized_status in {"SKIPPED", "SKIP"} or stripped_text.upper() == "SKIP":
        record.search_result = ""
        record.state = "SKIPPED"
        return True
    if normalized_status == "UNAVAILABLE" or is_unavailable_marker(stripped_text):
        record.search_result = ""
        record.cite_ids = []
        record.state = "UNAVAILABLE"
        return False

    normalized_text = normalize_cite_markers(stripped_text)
    cite_ids = extract_cite_ids(normalized_text)
    record.cite_ids = []
    for cite_id in cite_ids:
        if cite_id and cite_id not in record.cite_ids:
            record.cite_ids.append(cite_id)
    record.search_result = normalized_text
    record.state = "RESOLVED" if record.cite_ids else "UNAVAILABLE"
    return record.state == "RESOLVED"


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence_match:
        raw = fence_match.group(1).strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        raw = raw[start:end + 1]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_writer_issue(text: str) -> tuple[SegmentIssue, list[dict[str, Any]]] | None:
    payload = extract_json_object(text)
    if not payload:
        return None
    evidences = payload.get("evidences")
    if not isinstance(evidences, list):
        return None
    descriptions = []
    for evidence in evidences:
        if not isinstance(evidence, dict):
            continue
        description = str(evidence.get("description") or evidence.get("text") or "").strip()
        if description:
            descriptions.append(description)
    issue = SegmentIssue(
        type="EVIDENCE_GAP",
        detail="缺少：" + "；".join(descriptions) if descriptions else "缺少关键证据。",
        action="RETRIEVE",
        evidences=evidences,
    )
    return issue, issue.evidences


def parse_replan_response(text: str) -> dict[str, Any]:
    payload = extract_json_object(text)
    if payload is None:
        raise ValueError("replan response is not a JSON object")
    return payload


def parse_section_polish_response(text: str):
    title = extract_tagged_text(text, "title")
    content = extract_tagged_text(text, "content")
    assert title is not None and content is not None, (
        "输出格式错误，答案没有被正确的标签包裹。请严格按以下格式重新输出：\n"
        "<title>润色后的章节标题</title>\n"
        "<content>\n"
        "完整章节正文。正文中必须保留原有引用标记[^cite_id:xxx]和图表标记![...](chart:chart_xxx)。\n"
        "</content>"
    )
    return _strip_section_number_prefix(title.strip().strip("#").strip()), content


def _format_resolved_evidence_record(
    record: EvidenceRecord,
    index: int,
    short_term: "ShortTermMemoryStore" | None = None,
) -> str:
    lines = [
        f"{index}. {record.description}",
    ]
    if record.search_result and record.cite_ids:
        lines.append(f"   搜索结果: {record.search_result}")
    elif record.search_result:
        lines.append("   note: 无 cite_id，仅用于理解，不要作为可核验事实写入正文")
        lines.append(f"   搜索结果: {record.search_result}")
    if record.cite_ids:
        lines.append(f"   cite_ids: {', '.join(record.cite_ids)}")
        if not record.search_result and short_term is not None:
            previews = []
            for cite_id in record.cite_ids[:4]:
                preview = short_term.load_material_preview(cite_id, max_chars=700)
                if preview:
                    previews.append(f"      [^cite_id:{cite_id}] 摘要：\n{preview}")
            if previews:
                lines.append("   material_previews:")
                lines.extend(previews)
    return "\n".join(lines)


def build_evidence_context(
    registry: EvidenceRegistry,
    segment_id: str,
    short_term: "ShortTermMemoryStore",
) -> str:
    blocks = []
    for record in registry.records_for_segment(segment_id):
        if record.state != "RESOLVED":
            continue
        blocks.append(_format_resolved_evidence_record(record, len(blocks) + 1, short_term))
    return "\n\n".join(blocks).strip()


def build_known_evidence_context(registry: EvidenceRegistry, evidence_ids: list[str] | None = None) -> str:
    lines = []
    records = (
        [registry.records[evidence_id] for evidence_id in evidence_ids or [] if evidence_id in registry.records]
        if evidence_ids is not None
        else list(registry.records.values())
    )
    for record in records:
        if record.state != "RESOLVED":
            continue
        if record.search_result:
            lines.append(_format_resolved_evidence_record(record, len(lines) + 1))
    if not lines:
        return ""
    return "当前 evidence 可能依赖的证据：\n" + "\n\n".join(lines) + "\n"
