# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from src.memory.working import Section, Segment


SegmentState = Literal[
    "EMPTY",
    "PLANNED",
    "RETRIEVING",
    "EVIDENCE_READY",
    "WRITING",
    "DRAFTED",
    "VERIFYING",
    "VERIFIED",
    "FINALIZED",
    "BLOCKED",
]


@dataclass
class SegmentIssue:
    type: str
    detail: str
    action: str
    evidences: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DraftVersion:
    version: int
    content: str
    reason: str


@dataclass
class SegmentRecord:
    segment_id: str
    state: SegmentState = "EMPTY"
    topic: str = ""
    template: str = ""
    requirements: list[str] = field(default_factory=list)
    evidences: list[str] = field(default_factory=list)
    draft_versions: list[DraftVersion] = field(default_factory=list)
    issue: Optional[SegmentIssue] = None
    issue_seen: bool = False
    attempts: dict[str, int] = field(default_factory=dict)

    def latest_draft(self) -> str:
        if not self.draft_versions:
            return ""
        return self.draft_versions[-1].content

    def add_draft(self, content: str, reason: str) -> None:
        version = len(self.draft_versions) + 1
        self.draft_versions.append(
            DraftVersion(version=version, content=content, reason=reason)
        )

    def increment_attempt(self, action: str) -> int:
        key = str(action or "").upper()
        self.attempts[key] = self.attempts.get(key, 0) + 1
        return self.attempts[key]


@dataclass
class SegmentBinding:
    segment_id: str
    segment: Segment
    parent: Section


@dataclass
class TrackingBoard:
    records: dict[str, SegmentRecord] = field(default_factory=dict)

    def unfinished_records(self) -> list[SegmentRecord]:
        return [
            record
            for record in self.records.values()
            if record.state not in {"FINALIZED", "BLOCKED"}
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"segments": {key: asdict(value) for key, value in self.records.items()}}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrackingBoard":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = {}
        for segment_id, raw_record in (payload.get("segments") or {}).items():
            drafts = [
                DraftVersion(**draft)
                for draft in raw_record.get("draft_versions", [])
            ]
            issue = raw_record.get("issue")
            records[segment_id] = SegmentRecord(
                segment_id=segment_id,
                state=raw_record.get("state", "EMPTY"),
                topic=raw_record.get("topic", ""),
                template=raw_record.get("template", ""),
                requirements=list(raw_record.get("requirements") or []),
                evidences=list(raw_record.get("evidences") or []),
                draft_versions=drafts,
                issue=SegmentIssue(**issue) if issue else None,
                issue_seen=bool(raw_record.get("issue_seen", False) or issue),
                attempts=dict(raw_record.get("attempts") or {}),
            )
        return cls(records=records)


def split_requirements(requirements: str | None) -> list[str]:
    if not requirements:
        return []
    lines = [line.strip() for line in requirements.replace("\r\n", "\n").split("\n")]
    return [line for line in lines if line]


def make_section_path(parent_id: str | None, section: Section) -> str:
    section_id = str(section.section_id)
    if not parent_id:
        return section_id
    return f"{parent_id}.{section_id}"


def make_segment_id(section_path: str, segment_index: int) -> str:
    return f"{section_path}.s{segment_index}"


def segment_record_from_segment(segment_id: str, segment: Segment) -> SegmentRecord:
    state: SegmentState = "FINALIZED" if segment.finished else "PLANNED"
    record = SegmentRecord(
        segment_id=segment_id,
        state=state,
        topic=segment.topic or "",
        template=segment.template or "",
        requirements=split_requirements(segment.requirements),
    )
    if segment.content:
        record.add_draft(segment.content, "loaded_from_manuscript")
    return record


def collect_segment_bindings(section: Section, parent_id: str | None = None) -> dict[str, SegmentBinding]:
    bindings: dict[str, SegmentBinding] = {}
    add_segment_bindings(section, parent_id, bindings)
    return bindings


def add_segment_bindings(section: Section, parent_id: str | None, bindings: dict[str, SegmentBinding]) -> None:
    section_path = make_section_path(parent_id, section)
    for index, segment in enumerate(section.segments or [], start=1):
        segment_id = make_segment_id(section_path, index)
        bindings[segment_id] = SegmentBinding(segment_id=segment_id, segment=segment, parent=section)
    for subsection in section.subsections or []:
        add_segment_bindings(subsection, section_path, bindings)


def build_tracking_board(section: Section) -> tuple[TrackingBoard, dict[str, SegmentBinding]]:
    bindings = collect_segment_bindings(section)
    board = TrackingBoard()
    for segment_id, binding in bindings.items():
        board.records[segment_id] = segment_record_from_segment(segment_id, binding.segment)
    return board, bindings
