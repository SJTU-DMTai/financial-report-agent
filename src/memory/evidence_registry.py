# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.memory.working import Evidence
from src.utils.evidence_canonical import build_canonical_key, evidence_to_fields


EvidenceState = Literal["NEW", "WAITING", "PLANNED", "SEARCHING", "RESOLVED", "UNAVAILABLE", "SKIPPED"]


@dataclass
class EvidenceRecord:
    # Evidence records describe information needs; cite_ids point to material registry entries.
    evidence_id: str
    description: str
    canonical_key: str
    state: EvidenceState = "NEW"
    required: bool = True
    used_by_segments: list[str] = field(default_factory=list)
    cite_ids: list[str] = field(default_factory=list)
    search_plan: str = ""
    search_result: str = ""
    depends_on: list[str] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceRegistry:
    path: Path
    records: dict[str, EvidenceRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "EvidenceRegistry":
        if not path.exists():
            return cls(path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = {}
        for evidence_id, raw_record in (payload.get("evidences") or {}).items():
            records[evidence_id] = EvidenceRecord(
                evidence_id=evidence_id,
                description=raw_record.get("description", ""),
                canonical_key=raw_record.get("canonical_key", ""),
                state=raw_record.get("state", "NEW"),
                required=bool(raw_record.get("required", True)),
                used_by_segments=list(raw_record.get("used_by_segments") or []),
                cite_ids=list(raw_record.get("cite_ids") or []),
                search_plan=raw_record.get("search_plan", ""),
                search_result=raw_record.get("search_result", ""),
                depends_on=list(raw_record.get("depends_on") or []),
                fields=dict(raw_record.get("fields") or {}),
            )
        return cls(path=path, records=records)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"evidences": {key: asdict(value) for key, value in self.records.items()}}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_or_reuse(
        self,
        evidence: Evidence | dict[str, Any],
        segment_id: str,
        default_entity: str | None = None,
    ) -> EvidenceRecord:
        fields = evidence_to_fields(evidence, default_entity)
        canonical_key = build_canonical_key(fields)
        # SAME evidence is represented by one canonical record shared by segments.
        existing = self.find_by_canonical_key(canonical_key)
        if existing is not None:
            if segment_id not in existing.used_by_segments:
                existing.used_by_segments.append(segment_id)
            return existing

        evidence_id = self.next_evidence_id()
        record = EvidenceRecord(
            evidence_id=evidence_id,
            description=str(fields.get("description") or ""),
            canonical_key=canonical_key,
            state="NEW",
            required=bool(fields.get("required", True)),
            used_by_segments=[segment_id],
            fields=dict(fields),
        )
        self.records[evidence_id] = record
        return record

    def active_records(self) -> list[EvidenceRecord]:
        return [
            record
            for record in self.records.values()
            if record.used_by_segments
        ]

    def prune_unlinked_records(self) -> int:
        unlinked_ids = {
            evidence_id
            for evidence_id, record in self.records.items()
            if not record.used_by_segments
        }
        if not unlinked_ids:
            return 0
        for evidence_id in unlinked_ids:
            self.records.pop(evidence_id, None)
        for record in self.records.values():
            record.depends_on = [
                prerequisite_id
                for prerequisite_id in record.depends_on
                if prerequisite_id not in unlinked_ids
            ]
        return len(unlinked_ids)

    def find_by_canonical_key(self, canonical_key: str) -> EvidenceRecord | None:
        for record in self.records.values():
            if record.canonical_key == canonical_key:
                return record
        return None

    def next_evidence_id(self) -> str:
        max_index = 0
        for evidence_id in self.records:
            match = re.fullmatch(r"ev_(\d+)", evidence_id)
            if match:
                max_index = max(max_index, int(match.group(1)))
        return f"ev_{max_index + 1:03d}"

    def mark_resolved(
        self,
        evidence_id: str,
        cite_ids: list[str],
        search_plan: str = "",
        search_result: str = "",
    ) -> None:
        record = self.records[evidence_id]
        for cite_id in cite_ids:
            if cite_id and cite_id not in record.cite_ids:
                record.cite_ids.append(cite_id)
        record.search_plan = search_plan or record.search_plan
        record.search_result = search_result or record.search_result
        record.state = "RESOLVED" if record.cite_ids else "UNAVAILABLE"

    def add_dependency(self, evidence_id: str, prerequisite_id: str) -> bool:
        if evidence_id == prerequisite_id:
            return False
        if evidence_id not in self.records or prerequisite_id not in self.records:
            return False
        if not self.records[evidence_id].used_by_segments or not self.records[prerequisite_id].used_by_segments:
            return False
        if self.has_dependency_path(prerequisite_id, evidence_id):
            return False
        record = self.records[evidence_id]
        if prerequisite_id not in record.depends_on:
            record.depends_on.append(prerequisite_id)
        if record.state == "NEW":
            record.state = "WAITING"
        return True

    def has_dependency_path(self, start_id: str, target_id: str) -> bool:
        stack = [start_id]
        seen = set()
        while stack:
            current_id = stack.pop()
            if current_id == target_id:
                return True
            if current_id in seen:
                continue
            seen.add(current_id)
            current = self.records.get(current_id)
            if current is not None:
                stack.extend(current.depends_on)
        return False

    def dependencies_resolved(self, evidence_id: str) -> bool:
        record = self.records[evidence_id]
        for prerequisite_id in record.depends_on:
            prerequisite = self.records.get(prerequisite_id)
            if prerequisite is None:
                continue
            if prerequisite.state not in {"RESOLVED", "SKIPPED", "UNAVAILABLE"}:
                return False
        return True

    def ready_to_search_records(self) -> list[EvidenceRecord]:
        return [
            record
            for record in self.records.values()
            if record.used_by_segments
            if record.state in {"NEW", "WAITING", "PLANNED"}
            and self.dependencies_resolved(record.evidence_id)
        ]

    def records_for_segment(self, segment_id: str) -> list[EvidenceRecord]:
        return [
            record
            for record in self.records.values()
            if segment_id in record.used_by_segments
        ]

    def unresolved_required_for_segment(self, segment_id: str) -> list[EvidenceRecord]:
        return [
            record
            for record in self.records_for_segment(segment_id)
            if record.required and record.state not in {"RESOLVED", "SKIPPED"}
        ]


def repair_registry_citation_states(registry: EvidenceRegistry) -> int:
    from src.utils.tracking_board_format import extract_cite_ids, normalize_cite_markers

    repaired = 0
    for record in registry.active_records():
        if not record.search_result:
            continue
        normalized_result = normalize_cite_markers(record.search_result)
        cite_ids = extract_cite_ids(normalized_result)
        if not cite_ids:
            continue
        if normalized_result != record.search_result:
            record.search_result = normalized_result
            repaired += 1
        for cite_id in cite_ids:
            if cite_id not in record.cite_ids:
                record.cite_ids.append(cite_id)
                repaired += 1
        if record.state == "UNAVAILABLE":
            record.state = "RESOLVED"
            repaired += 1
    return repaired


def apply_static_reference_citation(record: EvidenceRecord, reference_cite_id: str) -> None:
    static_fact = str(record.fields.get("fact") or "").strip()
    if not bool(record.fields.get("is_static", False)) or not static_fact:
        return

    marker = f"[^cite_id:{reference_cite_id}]"
    description = record.description or str(record.fields.get("description") or "").strip()
    record.search_result = f"{description}：{static_fact}{marker}"
    if reference_cite_id not in record.cite_ids:
        record.cite_ids.append(reference_cite_id)
    record.state = "RESOLVED"


def is_static_fact_record(record: EvidenceRecord) -> bool:
    return bool(record.fields.get("is_static", False)) and bool(str(record.fields.get("fact") or "").strip())


def evidence_record_to_evidence(record: EvidenceRecord) -> Evidence:
    fields = dict(record.fields or {})
    description = str(fields.get("description") or record.description or "").strip()
    return Evidence(
        text=description,
        description=description,
        entity=str(fields.get("entity") or "").strip() or None,
        aspect=str(fields.get("aspect") or "").strip() or None,
        period=str(fields.get("period") or "").strip() or None,
        scope=str(fields.get("scope") or "").strip() or None,
        required=bool(fields.get("required", record.required)),
        fact=str(fields.get("fact") or "").strip() or None,
        is_static=bool(fields.get("is_static", False)),
    )


def sync_segment_evidences_from_record(
    record: "SegmentRecord",
    binding: "SegmentBinding",
    registry: EvidenceRegistry,
) -> None:
    binding.segment.evidences = [
        evidence_record_to_evidence(registry.records[evidence_id])
        for evidence_id in record.evidences
        if evidence_id in registry.records
    ] or None
