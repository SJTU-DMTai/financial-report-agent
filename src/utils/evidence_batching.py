# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from xml.sax.saxutils import escape

from src.memory.evidence_registry import EvidenceRecord, EvidenceRegistry


@dataclass(frozen=True)
class EvidenceBatchResult:
    evidence_id: str
    status: str
    answer: str


def cluster_ready_evidence_records(
    records: list[EvidenceRecord],
    registry: EvidenceRegistry,
    max_batch_size: int,
    segment_ids_in_order: list[str] | None = None,
) -> list[list[EvidenceRecord]]:
    batch_size = max(int(max_batch_size), 1)
    segment_order = {
        segment_id: index
        for index, segment_id in enumerate(segment_ids_in_order or [])
    }
    clusters: list[list[EvidenceRecord]] = []
    cluster_segments: list[set[str]] = []

    for record in sorted(records, key=lambda item: _record_sort_key(item, segment_order)):
        if (
            clusters
            and _can_join_cluster(record, clusters[-1], cluster_segments[-1], registry, segment_order)
            and len(clusters[-1]) < batch_size
        ):
            clusters[-1].append(record)
            primary_segment = _primary_segment_id(record, segment_order)
            if primary_segment:
                cluster_segments[-1].add(primary_segment)
            continue

        clusters.append([record])
        primary_segment = _primary_segment_id(record, segment_order)
        cluster_segments.append({primary_segment} if primary_segment else set())

    return clusters


def _can_join_cluster(
    record: EvidenceRecord,
    cluster: list[EvidenceRecord],
    cluster_segment_ids: set[str],
    registry: EvidenceRegistry,
    segment_order: dict[str, int],
) -> bool:
    primary_segment = _primary_segment_id(record, segment_order)
    if _section_key(primary_segment) != _section_key(_primary_segment_id(cluster[0], segment_order)):
        return False
    if _entity_key(record) != _entity_key(cluster[0]):
        return False
    if any(
        registry.has_dependency_path(record.evidence_id, item.evidence_id)
        or registry.has_dependency_path(item.evidence_id, record.evidence_id)
        for item in cluster
    ):
        return False
    return _segment_ids_are_adjacent(cluster_segment_ids | ({primary_segment} if primary_segment else set()))


def _record_sort_key(record: EvidenceRecord, segment_order: dict[str, int]) -> tuple[str, str, int, str]:
    primary_segment = _primary_segment_id(record, segment_order)
    return (
        _section_key(primary_segment),
        _entity_key(record),
        segment_order.get(primary_segment, len(segment_order)),
        record.evidence_id,
    )


def _entity_key(record: EvidenceRecord) -> str:
    entity = str(record.fields.get("entity") or "").strip()
    entity_payload = record.fields.get("entity_info")
    if not entity and isinstance(entity_payload, dict):
        name = str(entity_payload.get("name") or "").strip()
        code = str(entity_payload.get("code") or "").strip()
        entity = f"{name}:{code}" if name or code else ""
    return entity


def _primary_segment_id(record: EvidenceRecord, segment_order: dict[str, int]) -> str:
    if not record.used_by_segments:
        return ""
    return min(
        record.used_by_segments,
        key=lambda segment_id: (segment_order.get(segment_id, len(segment_order)), segment_id),
    )


def _section_key(segment_id: str) -> str:
    if ".s" not in segment_id:
        return segment_id
    return segment_id.rsplit(".s", 1)[0]


def _segment_index(segment_id: str) -> int | None:
    match = re.search(r"\.s(\d+)$", segment_id)
    return int(match.group(1)) if match else None


def _segment_ids_are_adjacent(segment_ids: set[str]) -> bool:
    indexes = sorted(
        index
        for index in (_segment_index(segment_id) for segment_id in segment_ids)
        if index is not None
    )
    return not indexes or indexes[-1] - indexes[0] + 1 == len(set(indexes))


def format_evidence_batch_xml(
    records: list[EvidenceRecord],
    known_evidence_by_id: dict[str, str],
) -> str:
    lines = ["<evidences>"]
    for record in records:
        lines.extend(
            [
                "  <evidence>",
                f"    <evidence_id>{escape(record.evidence_id)}</evidence_id>",
                f"    <description>{escape(record.description or '')}</description>",
                f"    <known_evidence>{escape(known_evidence_by_id.get(record.evidence_id, ''))}</known_evidence>",
                "  </evidence>",
            ]
        )
    lines.append("</evidences>")
    return "\n".join(lines)


def parse_batch_search_xml(text: str) -> dict[str, EvidenceBatchResult]:
    raw_text = str(text or "").strip()
    fence_match = re.search(r"```(?:xml)?\s*(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        raw_text = fence_match.group(1).strip()
    root = _parse_batch_xml_root(raw_text)
    result_nodes = root.findall(".//evidence_result")
    if root.tag == "evidence_result":
        result_nodes = [root]

    results: dict[str, EvidenceBatchResult] = {}
    for node in result_nodes:
        evidence_id = _child_text(node, "evidence_id")
        if not evidence_id:
            continue
        results[evidence_id] = EvidenceBatchResult(
            evidence_id=evidence_id,
            status=_child_text(node, "status").upper(),
            answer=_child_text(node, "answer"),
        )
    return results


def _parse_batch_xml_root(text: str) -> ET.Element:
    parse_error = None
    for candidate in _batch_xml_candidates(text):
        try:
            return ET.fromstring(candidate)
        except ET.ParseError as exc:
            parse_error = exc
    if parse_error is not None:
        raise parse_error
    return ET.fromstring(text)


def _batch_xml_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stripped = text.strip()

    result_blocks = [
        match.group(0).strip()
        for match in re.finditer(r"<results\b.*?</results>", text, re.DOTALL | re.IGNORECASE)
    ]
    candidates.extend(reversed(result_blocks))

    result_nodes = re.findall(
        r"<evidence_result\b.*?</evidence_result>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if result_nodes:
        candidates.append("<results>\n" + "\n".join(result_nodes) + "\n</results>")

    if stripped:
        candidates.append(stripped)

    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _child_text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()
