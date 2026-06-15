# -*- coding: utf-8 -*-
from __future__ import annotations

import json
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
) -> list[list[EvidenceRecord]]:
    clusters_by_key: dict[tuple[str, str], list[list[EvidenceRecord]]] = {}
    batch_size = max(int(max_batch_size), 1)
    for record in records:
        segment_id = record.used_by_segments[0] if record.used_by_segments else ""
        entity = str(record.fields.get("entity") or "").strip()
        entity_payload = record.fields.get("entity_info")
        if not entity and isinstance(entity_payload, dict):
            name = str(entity_payload.get("name") or "").strip()
            code = str(entity_payload.get("code") or "").strip()
            entity = f"{name}:{code}" if name or code else ""
        key = (segment_id, entity)
        candidate_clusters = clusters_by_key.setdefault(key, [])
        placed = False
        for cluster in candidate_clusters:
            if len(cluster) >= batch_size:
                continue
            if any(
                registry.has_dependency_path(record.evidence_id, item.evidence_id)
                or registry.has_dependency_path(item.evidence_id, record.evidence_id)
                for item in cluster
            ):
                continue
            cluster.append(record)
            placed = True
            break
        if not placed:
            candidate_clusters.append([record])

    clusters: list[list[EvidenceRecord]] = []
    for key in sorted(clusters_by_key):
        clusters.extend(clusters_by_key[key])
    return clusters


def format_evidence_batch_xml(
    records: list[EvidenceRecord],
    known_evidence_by_id: dict[str, str],
) -> str:
    lines = ["<evidences>"]
    for record in records:
        fields_json = json.dumps(record.fields or {}, ensure_ascii=False, sort_keys=True)
        lines.extend(
            [
                "  <evidence>",
                f"    <evidence_id>{escape(record.evidence_id)}</evidence_id>",
                f"    <description>{escape(record.description or '')}</description>",
                f"    <fields_json>{escape(fields_json)}</fields_json>",
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
    start = raw_text.find("<results")
    end = raw_text.rfind("</results>")
    if start >= 0 and end >= start:
        raw_text = raw_text[start:end + len("</results>")]
    root = ET.fromstring(raw_text)
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


def _child_text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()
