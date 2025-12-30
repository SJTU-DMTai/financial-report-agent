# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pathlib import Path
from typing import List, Tuple

@dataclass_json
@dataclass
class Element:
    finished: bool = False
    summary: str = None
    requirements: str = None
    reference: str = None
    content: str = None
    template: List[str] = None
    # ref_uri: List[str] = None

@dataclass_json
@dataclass
class Section:
    section_id: int
    level: int
    title: str
    elements: List[Element]
    subsections: List[Section]

    def read(self, with_requirements=True, with_reference=False, with_content=False,
             fold_other=True, fold_all=False, read_subsections=False) -> str:
        ctx = f"{'#' * self.level} {self.title}\n"
        unfinished = [i for i, e in enumerate(self.elements) if not e.finished]
        # if len(unfinished) == 0:
        #     return "All finished."
        for i, e in enumerate(self.elements):
            ctx += f"* [{'x' if e.finished else ' '}] {e.summary}\n"
            if fold_all or i != unfinished[0] and fold_other:
                continue
            if with_reference and e.reference is not None:
                ctx += f"\t- > **Reference**\n{'\n'.join(['\t\t> ' + l for l in e.reference.splitlines()])}\n\n"
            if with_content and e.content is not None:
                ctx += f"\t- **Template**\n\t{e.content}\n\n"
            if with_requirements and e.requirements is not None:
                requirements = "\n".join([("\t\t" if r.strip()[:2] in ["- ", "* "] else "") + r
                                          for r in e.requirements.split("\n")])
                ctx += f"\t- **Requirements**\n{requirements}\n\n"
        if read_subsections:
            for sec in self.subsections:
                ctx += sec.read(with_requirements=with_requirements,
                                with_reference=with_reference, with_content=with_content,
                                fold_other=fold_other, fold_all=fold_all, read_subsections=True) + "\n\n"
        return ctx

    def load_with_prev_sections(self, section_id, with_requirements=True, with_reference=False, with_content=False) -> str:
        ctx = ""
        for i in range(section_id - 1):
            ctx += self.subsections[i].read(with_requirements=with_requirements, with_reference=with_reference, with_content=with_content, fold_all=True)
        return ctx + self.read(with_requirements=with_requirements, with_reference=with_reference, with_content=with_content, fold_other=True)

    @staticmethod
    def parse(contents: str) -> Element:
        keys = ['requirement', 'template', 'summary']
        cnts = [contents.count(f"<{k}>") for k in keys]
        cnts += [contents.count(f"</{k}>") for k in keys]
        for c1 in cnts:
            for c2 in cnts:
                assert c1 == c2 > 0, "Incomplete answer. You must give <template>, </template>, <requirement>, </requirement>, <summary> and </summary> for each item. Please Retry."
        contents = contents.replace("\r\n", "\n")
        print(contents, flush=True)
        res = re.findall(r"<template>(.+?)</template>.*<requirement>(.+?)</requirement>.*<summary>(.+?)</summary>", contents, re.DOTALL)
        assert len(res) > 0, "Format error. You did not give template, requirement, and summary in order. Please Retry."
        res = [s.strip() for s in res[0]]
        return Element(content=res[0], requirements=res[1], summary=res[2])

