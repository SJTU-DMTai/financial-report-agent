# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pathlib import Path
from typing import List

@dataclass_json
@dataclass
class Segment:
    finished: bool = False
    topic: str = None
    requirements: str = None
    reference: str = None
    content: str = None
    template: str = None
    evidences: List[str] = None

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
                "\t\t- " + e.replace("\n\n", "\n")
                for e in self.evidences if e
            )
            ctx += f"\t+ **论据材料**\n{evidence_text}\n\n"

        return ctx

@dataclass_json
@dataclass
class Section:
    section_id: int
    level: int
    title: str
    segments: List[Segment]
    subsections: List[Section]
    content: str = None

    def read(self, with_requirements=True, with_reference=False, with_content=False, with_evidence=False,
             fold_other=True, fold_all=False, read_subsections=False) -> str:
        ctx = f"{'#' * self.level} {self.title}\n"
        unfinished = [i for i, s in enumerate(self.segments) if not s.finished]
        # if len(unfinished) == 0:
        #     return "All finished."
        if with_content and self.content is not None:
            ctx += self.content + '\n\n'
        else:
            for i, s in enumerate(self.segments):
                ctx += f"* [{'x' if s.finished else ' '}] {s.topic}\n"
                # if fold_all or i != unfinished[0] and fold_other:
                #     continue
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
        # cnts += [contents.count(f"</{k}>") for k in keys]
        for c1 in cnts:
            for c2 in cnts:
                assert c1 == c2 > 0, "Incomplete answer. You must give <evidence>, </evidence>, <template>, </template>, <requirement>, </requirement>, <topic> and </topic> for each item. Please Retry."
        contents = contents.replace("\r\n", "\n")
        print(contents, flush=True)
        res = re.findall(r"<evidence>(.+?)(?:</evidence>)?\s*<template>(.+?)(?:</template>)?\s*<requirement>(.+?)(?:</requirement>)?\s*<topic>(.+?)</topic>", contents, re.DOTALL)
        assert len(res) > 0, "Format error. You did not correctly warp template, evidence, requirement, or topic with the corresponding blocks and put them in order. Please Retry."
        evidences, template, requirements, topic = [s.strip() for s in res[0]]
        evidences = evidences.replace("\n", "").replace(";", "；").split("；")
        evidences = [e.strip() for e in evidences if e.strip() != ""]
        _evidences = []
        for e in evidences:
            if e not in _evidences:
                _evidences.append(e)
        evidences = None if len(_evidences) == 0 else _evidences
        return Segment(template=template, requirements=requirements, topic=topic, evidences=evidences)

    @staticmethod
    def parse_evidence(contents: str) -> Segment:
        keys = ['evidence', 'topic']
        cnts = [contents.count(f"<{k}>") for k in keys]
        # cnts += [contents.count(f"</{k}>") for k in keys]
        for c1 in cnts:
            for c2 in cnts:
                assert c1 == c2 > 0, "Incomplete answer. You must give <evidence>, </evidence>, <topic> and </topic> for each item. Please Retry."
        contents = contents.replace("\r\n", "\n")
        print(contents, flush=True)
        res = re.findall(r"<evidence>(.+?)(?:</evidence>)?\s*<topic>(.+?)</topic>", contents, re.DOTALL)
        assert len(res) > 0, "Format error. You did not correctly warp evidence or topic with the corresponding blocks and put them in order. Please Retry."
        evidences, topic = [s.strip() for s in res[0]]
        evidences = evidences.replace("\n", "").replace(";", "；").split("；")
        evidences = [e.strip() for e in evidences if e.strip() != ""]
        _evidences = []
        for e in evidences:
            if e not in _evidences:
                _evidences.append(e)
        evidences = None if len(_evidences) == 0 else _evidences
        return Segment(template=None, requirements=None, topic=topic, evidences=evidences)


def _get_outline_cache_paths(pdf_path: Path, save_dir: Path, only_evidence: bool) -> tuple[Path, Path]:
    from src.utils.instance import cfg
    
    stem = pdf_path.stem
    suffix = "_outline_only_evidence.json" if only_evidence else "_outline.json"
    model_cache_path = save_dir / cfg.llm_name / f"{stem}{suffix}"
    legacy_cache_path = save_dir / f"{stem}{suffix}"
    return model_cache_path, legacy_cache_path


def _get_outline_cache_candidates(pdf_path: Path, save_dir: Path, only_evidence: bool) -> list[Path]:
    model_cache_path, legacy_cache_path = _get_outline_cache_paths(pdf_path, save_dir, only_evidence)
    cache_candidates = [model_cache_path, legacy_cache_path]
    if only_evidence:
        full_model_cache_path, full_legacy_cache_path = _get_outline_cache_paths(pdf_path, save_dir, False)
        cache_candidates.extend([full_model_cache_path, full_legacy_cache_path])
    return cache_candidates


def _load_cached_outline(pdf_path: Path, save_dir: Path, only_evidence: bool) -> Section | None:
    for cache_path in _get_outline_cache_candidates(pdf_path, save_dir, only_evidence):
        if cache_path.exists():
            return Section.from_json(cache_path.read_text(encoding="utf-8"))
    return None


def _parse_segment_response(contents: str, only_evidence: bool) -> Segment | str:
    if isinstance(contents, str) and "<skip>true</skip>" in contents.lower():
        return contents
    if only_evidence:
        return Section.parse_evidence(contents)
    return Section.parse(contents)

