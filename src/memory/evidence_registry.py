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
        static_fact = str(fields.get("fact") or "").strip()
        static_resolved = bool(fields.get("is_static", False)) and bool(static_fact)
        record = EvidenceRecord(
            evidence_id=evidence_id,
            description=str(fields.get("description") or ""),
            canonical_key=canonical_key,
            state="RESOLVED" if static_resolved else "NEW",
            required=bool(fields.get("required", True)),
            used_by_segments=[segment_id],
            search_result=f"{fields.get('description')}：{static_fact}" if static_resolved else "",
            fields=dict(fields),
        )
        self.records[evidence_id] = record
        return record

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
            if prerequisite is None or prerequisite.state not in {"RESOLVED", "SKIPPED"}:
                return False
        return True

    def has_unavailable_dependency(self, evidence_id: str) -> bool:
        record = self.records[evidence_id]
        for prerequisite_id in record.depends_on:
            prerequisite = self.records.get(prerequisite_id)
            if prerequisite is None or prerequisite.state == "UNAVAILABLE":
                return True
        return False

    def mark_records_blocked_by_unavailable_dependencies(self) -> int:
        changed = 0
        for record in self.records.values():
            if record.state not in {"NEW", "WAITING", "PLANNED"}:
                continue
            if self.has_unavailable_dependency(record.evidence_id):
                record.state = "UNAVAILABLE"
                changed += 1
        return changed

    def ready_to_search_records(self) -> list[EvidenceRecord]:
        return [
            record
            for record in self.records.values()
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
