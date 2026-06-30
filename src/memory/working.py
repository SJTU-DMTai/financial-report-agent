# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    is_static: bool = False
    fact: Optional[str] = None
    description: Optional[str] = None
    entity: Optional[str] = None
    aspect: Optional[str] = None
    period: Optional[str] = None
    scope: Optional[str] = None
    required: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class Segment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    finished: bool = False
    topic: Optional[str] = None
    requirements: Optional[str] = None
    reference: Optional[str] = None
    content: Optional[str] = None
    template: Optional[str] = None
    evidences: Optional[List[Evidence]] = None

    @field_validator("evidences", mode="before")
    @classmethod
    def _coerce_evidences(cls, value):
        return _normalize_evidences(value)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def __str__(self, with_requirements=True, with_reference=False, with_content=True, with_evidence=True):
        ctx = ""
        if with_reference and self.reference is not None:
            ref_text = "\n".join(
                ["\t\t> " + l for l in self.reference.splitlines()]
            )
            ctx += f"\t+ > **原文**\n{ref_text}\n\n"

        if with_content:
            if self.content is not None:
                ctx += f"{self.content}\n\n"
            elif self.template is not None:
                ctx += f"\t+ **示例**\n\t{self.template}\n\n"

        if with_requirements and self.requirements is not None:
            requirements = "\n".join(
                [
                    ("\t\t" if r.strip()[:2] in ["- ", "* "] else "") + r
                    for r in self.requirements.split("\n")
                ]
            )
            ctx += f"\t+ **写作要求**\n{requirements}\n\n"
        if with_evidence and self.evidences is not None:
            evidence_text = "\n".join(
                "\t\t- " + _format_evidence_for_display(e).replace("\n\n", "\n")
                for e in self.evidences if e and e.text
            )
            ctx += f"\t+ **论据材料**\n{evidence_text}\n\n"

        return ctx


class Section(BaseModel):
    model_config = ConfigDict(extra="ignore")

    section_id: int
    level: int
    title: str
    segments: List[Segment]
    subsections: List["Section"]
    content: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Section":
        return cls.model_validate(_upgrade_section_payload(data))

    @classmethod
    def from_json(cls, text: str) -> "Section":
        return load_section_from_json_text(text)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, ensure_ascii: bool = False, indent: int | None = None) -> str:
        return self.model_dump_json(ensure_ascii=ensure_ascii, indent=indent)

    def read(self, with_requirements=True, with_reference=False, with_content=False, with_evidence=False,
             fold_other=True, fold_all=False, read_subsections=False) -> str:
        ctx = f"{'#' * max(self.level, 1)} {self.title}\n"
        unfinished = [i for i, s in enumerate(self.segments) if not s.finished]
        if with_content and self.content is not None:
            ctx += self.content + '\n\n'
        else:
            for i, s in enumerate(self.segments):
                ctx += f"* [{'x' if s.finished else ' '}] {s.topic}\n"
                ctx += s.__str__(with_requirements=with_requirements, with_reference=with_reference,
                                 with_content=with_content, with_evidence=with_evidence)
        if read_subsections:
            for sec in self.subsections:
                ctx += sec.read(with_requirements=with_requirements, with_evidence=with_evidence,
                                with_reference=with_reference, with_content=with_content,
                                fold_other=fold_other, fold_all=fold_all, read_subsections=True) + "\n\n"
        return ctx

    def load_with_prev_sections(self, section_id, with_requirements=True, with_reference=False, with_content=False) -> str:
        ctx = ""
        for i in range(section_id - 1):
            ctx += self.subsections[i].read(with_requirements=with_requirements, with_reference=with_reference, with_content=with_content, fold_all=True)
        return ctx + self.read(with_requirements=with_requirements, with_reference=with_reference, with_content=with_content, fold_other=True)

    @staticmethod
    def parse(contents: str) -> Segment:
        keys = ['requirement', 'template', 'evidence', 'topic']
        cnts = [contents.count(f"<{k}>") for k in keys]
        for c1 in cnts:
            for c2 in cnts:
                assert c1 == c2 > 0, "Incomplete answer. You must give <evidence>, </evidence>, <template>, </template>, <requirement>, </requirement>, <topic> and </topic> for each item. Please Retry."
        contents = contents.replace("\r\n", "\n")
        print(contents, flush=True)
        res = re.findall(r"<evidence>(.+?)(?:</evidence>)?\s*<template>(.+?)(?:</template>)?\s*<requirement>(.+?)(?:</requirement>)?\s*<topic>(.+?)</topic>", contents, re.DOTALL)
        assert len(res) > 0, "Format error. You did not correctly warp template, evidence, requirement, or topic with the corresponding blocks and put them in order. Please Retry."
        evidences_text, template, requirements, topic = [s.strip() for s in res[0]]
        evidences = _parse_evidences(evidences_text)
        return Segment(template=template, requirements=requirements, topic=topic, evidences=evidences)

    @staticmethod
    def parse_evidence(contents: str) -> Segment | str:
        json_payload = _extract_json_payload(contents)
        assert json_payload is not None, "Format error. Evidence extraction must output a JSON object."
        return _parse_json_evidence_response(json_payload)


def _format_evidence_for_display(evidence: Evidence) -> str:
    if evidence.fact:
        return f"{evidence.text}：{evidence.fact}"
    return evidence.text


def _parse_evidence_item(raw_text: str) -> Optional[Evidence]:
    text = raw_text.strip()
    if not text:
        return None
    is_static = False
    static_match = re.search(r"\s*\((static|静态)\)\s*$", text, re.IGNORECASE)
    if static_match:
        is_static = True
        text = re.sub(r"\s*\((static|静态)\)\s*$", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return None
    return Evidence(text=text, is_static=is_static)


def _parse_evidences(evidences_text: str) -> Optional[List[Evidence]]:
    json_payload = _extract_json_like_payload(evidences_text)
    if json_payload is not None:
        try:
            payload = json.loads(json_payload)
            if isinstance(payload, dict) and "evidences" in payload:
                return _normalize_evidences(payload.get("evidences"))
            return _normalize_evidences(payload)
        except Exception:
            pass

    parsed_evidences = []
    raw_items = evidences_text.replace("\n", "").replace(";", "；").split("；")
    for raw_item in raw_items:
        evidence = _parse_evidence_item(raw_item)
        if evidence is None:
            continue
        if any(existing.text == evidence.text for existing in parsed_evidences):
            continue
        parsed_evidences.append(evidence)
    if not parsed_evidences:
        return None
    return parsed_evidences


def _build_structured_evidence_text(mapping: dict) -> str:
    parts = [
        mapping.get("entity"),
        mapping.get("period"),
        mapping.get("scope"),
        mapping.get("aspect"),
    ]
    return " ".join(str(part).strip() for part in parts if str(part or "").strip())


def _coerce_evidence_mapping(mapping: dict) -> List[Evidence]:
    structured_keys = {"description", "entity", "aspect", "period", "scope", "required"}
    text = str(
        mapping.get("text")
        or mapping.get("description")
        or mapping.get("evidence")
        or _build_structured_evidence_text(mapping)
        or ""
    ).strip()
    if text:
        fact = mapping.get("fact")
        return [
            Evidence(
                text=text,
                description=str(mapping.get("description") or text).strip(),
                entity=str(mapping.get("entity")).strip() if mapping.get("entity") is not None else None,
                aspect=str(mapping.get("aspect")).strip() if mapping.get("aspect") is not None else None,
                period=str(mapping.get("period")).strip() if mapping.get("period") is not None else None,
                scope=str(mapping.get("scope")).strip() if mapping.get("scope") is not None else None,
                required=bool(mapping.get("required", True)),
                fact=str(fact).strip() if fact is not None else None,
                is_static=bool(mapping.get("is_static", False)),
            )
        ]

    if any(key in mapping for key in structured_keys):
        return []

    evidences = []
    for key, fact in mapping.items():
        if key in {"text", "description", "evidence", "is_static", "fact"}:
            return []
        text = str(key).strip()
        if text:
            evidences.append(Evidence(text=text, fact=str(fact).strip() if fact is not None else None))
    return evidences


def _extract_json_payload(contents: str) -> str | None:
    if not isinstance(contents, str):
        return None
    text = contents.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return None


def _extract_json_like_payload(contents: str) -> str | None:
    if not isinstance(contents, str):
        return None
    text = contents.strip()
    fenced_match = re.search(r"```(?:json)?\s*([\[{].+?[\]}])\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        return text
    object_start = text.find("{")
    object_end = text.rfind("}")
    array_start = text.find("[")
    array_end = text.rfind("]")
    candidates = []
    if object_start >= 0 and object_end > object_start:
        candidates.append((object_start, object_end + 1))
    if array_start >= 0 and array_end > array_start:
        candidates.append((array_start, array_end + 1))
    if not candidates:
        return None
    start, end = min(candidates, key=lambda item: item[0])
    return text[start:end]


def _parse_json_evidence_response(json_payload: str) -> Segment | str:
    payload = json.loads(json_payload)
    if payload.get("skip") is True:
        return "<skip>true</skip>"
    evidences = _parse_fact_evidences(payload.get("evidences"))
    topic = str(payload.get("topic") or "").strip()
    assert topic, "Format error. JSON evidence response must include topic."
    assert evidences, "Format error. JSON evidence response must include non-empty evidences."
    return Segment(template=None, requirements=None, topic=topic, evidences=evidences)


def _parse_fact_evidences(items) -> Optional[List[Evidence]]:
    evidences = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        fact = str(item.get("fact") or "").strip()
        if description and fact:
            evidences.append(Evidence(text=description, description=description, fact=fact))
    return evidences or None


def _coerce_evidence_sequence_item(item) -> Optional[Evidence]:
    if isinstance(item, Evidence):
        return item
    if isinstance(item, str):
        return _parse_evidence_item(item)
    if isinstance(item, dict):
        coerced = _coerce_evidence_mapping(item)
        return coerced[0] if len(coerced) == 1 else None
    if isinstance(item, (list, tuple)) and item:
        text = str(item[0]).strip()
        fact = str(item[1]).strip() if len(item) > 1 and item[1] is not None else None
        return Evidence(text=text, fact=fact) if text else None
    return None


def _normalize_evidences(evidences) -> Optional[List[Evidence]]:
    if evidences is None:
        return None
    if isinstance(evidences, dict):
        normalized = _coerce_evidence_mapping(evidences)
    else:
        normalized = []
        for item in evidences:
            evidence = _coerce_evidence_sequence_item(item)
            if evidence is not None:
                normalized.append(evidence)

    deduped = []
    for evidence in normalized:
        if not evidence.text:
            continue
        if any(existing.text == evidence.text and existing.fact == evidence.fact for existing in deduped):
            continue
        deduped.append(evidence)
    if not deduped:
        return None
    return deduped


def section_has_unfinished(section: Section) -> bool:
    if section.segments:
        for segment in section.segments:
            if not segment.finished:
                return True
    if section.subsections:
        for subsection in section.subsections:
            if section_has_unfinished(subsection):
                return True
    return False


def count_section_segments(section: Section) -> tuple[int, int]:
    total = len(section.segments or [])
    finalized = sum(1 for segment in section.segments or [] if segment.finished)
    for subsection in section.subsections or []:
        child_total, child_finalized = count_section_segments(subsection)
        total += child_total
        finalized += child_finalized
    return total, finalized


def replace_unfinished_segments_with_outline(manuscript: Section, outline: Section) -> None:
    section_pairs = [(manuscript, outline)]
    while section_pairs:
        manuscript_section, outline_section = section_pairs.pop()
        replaced_segment = False
        if manuscript_section.segments and outline_section.segments:
            for index, segment in enumerate(manuscript_section.segments):
                if not segment.finished and index < len(outline_section.segments):
                    manuscript_section.segments[index] = outline_section.segments[index]
                    replaced_segment = True
        if replaced_segment:
            manuscript_section.content = None
        if manuscript_section.subsections and outline_section.subsections:
            for index, subsection in enumerate(manuscript_section.subsections):
                if index < len(outline_section.subsections):
                    section_pairs.append((subsection, outline_section.subsections[index]))


def evidence_texts(evidences: Optional[List[Evidence]]) -> List[str]:
    if not evidences:
        return []
    return [e.text for e in evidences if e and e.text]


def evidence_pairs(evidences: Optional[List[Evidence]]) -> List[tuple[str, str]]:
    if not evidences:
        return []
    return [(e.text, e.fact or "") for e in evidences if e and e.text]


def _upgrade_section_payload(payload):
    segments = payload.get("segments") or []
    for segment in segments:
        evidences = segment.get("evidences")
        if evidences:
            segment["evidences"] = [e.model_dump() for e in _normalize_evidences(evidences) or []]
    subsections = payload.get("subsections") or []
    for subsection in subsections:
        _upgrade_section_payload(subsection)
    return payload


def load_section_from_json_text(json_text: str) -> Section:
    payload = json.loads(json_text)
    upgraded_payload = _upgrade_section_payload(payload)
    return Section.model_validate(upgraded_payload)


def _get_outline_cache_path(
    pdf_path: Path,
    save_dir: Path,
    only_evidence: bool,
    model_name: str | None = None,
) -> Path:
    from src.utils.instance import cfg

    stem = pdf_path.stem
    suffix = "_outline_only_evidence.json" if only_evidence else "_outline.json"
    cache_model_name = model_name or cfg.llm_name
    return save_dir / cache_model_name / f"{stem}{suffix}"


def _get_outline_cache_candidates(
    pdf_path: Path,
    save_dir: Path,
    only_evidence: bool,
    reuse_other_model_cache: bool = False,
    cache_model_name: str | None = None,
) -> list[Path]:
    cache_candidates = [_get_outline_cache_path(pdf_path, save_dir, only_evidence, cache_model_name)]
    if only_evidence:
        full_model_cache_path = _get_outline_cache_path(pdf_path, save_dir, False, cache_model_name)
        if full_model_cache_path not in cache_candidates:
            cache_candidates.append(full_model_cache_path)

    if reuse_other_model_cache and save_dir.exists():
        stem = pdf_path.stem
        suffixes = ["_outline_only_evidence.json"] if only_evidence else ["_outline.json"]
        if only_evidence:
            suffixes.append("_outline.json")
        for child in save_dir.iterdir():
            if not child.is_dir():
                continue
            for suffix in suffixes:
                candidate = child / f"{stem}{suffix}"
                if candidate not in cache_candidates:
                    cache_candidates.append(candidate)
    return cache_candidates


def _load_cached_outline(
    pdf_path: Path,
    save_dir: Path,
    only_evidence: bool,
    reuse_other_model_cache: bool = False,
    cache_model_name: str | None = None,
) -> Section | None:
    for cache_path in _get_outline_cache_candidates(
        pdf_path,
        save_dir,
        only_evidence,
        reuse_other_model_cache,
        cache_model_name,
    ):
        if cache_path.exists():
            return load_section_from_json_text(cache_path.read_text(encoding="utf-8"))
    return None


def _parse_segment_response(contents: str, only_evidence: bool) -> Segment | str:
    if isinstance(contents, str) and "<skip>true</skip>" in contents.lower():
        return contents
    if only_evidence:
        return Section.parse_evidence(contents)
    return Section.parse(contents)


Section.model_rebuild()
