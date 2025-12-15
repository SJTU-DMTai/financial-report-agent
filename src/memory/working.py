# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import json
from io import StringIO
import shutil

@dataclass
class Element:
    finished: bool = False
    summary: str = None
    requirements: str = None
    example: str = None
    content: str = None
    # ref_uri: List[str] = None

@dataclass
class Section:
    section_id: int
    title: str
    elements: List[Element]
    subsections: List[Section]

    def read_all(self, level=1, **kwargs) -> str:
        return ("\n\n".join(sec.read(level=level + 1, **kwargs) for sec in self.subsections) +
                '\n\n' + self.read(level=level, **kwargs))

    def read(self, level=1, with_requirements=True, with_example=False, with_content=False,
             fold_other=True, fold_all=False) -> str:
        ctx = f"{'#' * level} {self.title}\n"
        unfinished = [i for i, e in enumerate(self.elements) if not e.finished]
        if len(unfinished) == 0:
            return "All finished."
        for i, e in enumerate(self.elements):
            ctx += f"- [{'x' if e.finished else ' '}] {e.summary}\n"
            if fold_all or i != unfinished[0] and fold_other:
                continue
            if with_requirements and e.requirements is not None:
                ctx += f"\t- Requirements: {e.requirements}\n"
            if with_example and e.example is not None:
                ctx += f"\t- Example: {e.example}\n"
            if with_content and e.content is not None:
                ctx += f"\t- Content: {e.content}\n"
        return ctx

    def load_with_prev_sections(self, section_id, with_requirements=True, with_example=False, with_content=False) -> str:
        ctx = ""
        for i in range(section_id - 1):
            ctx += self.read(i, with_requirements=with_requirements, with_example=with_example, with_content=with_content, fold_all=True)
        return ctx + self.read(section_id, with_requirements=with_requirements, with_example=with_example, with_content=with_content, fold_other=True)

    @staticmethod
    def parse(contents: str) -> List[Element]:
        keys = ['example', 'requirement', 'template', 'summary']
        cnts = [contents.count(f"<{k}>") for k in keys]
        cnts += [contents.count(f"</{k}>") for k in keys]
        for c1 in cnts:
            for c2 in cnts:
                assert c1 == c2 > 0, "Incomplete answer. You must give example, requirement, and summary for each item. Please Retry."
        contents = contents.replace("\r\n", "\n")
        res = re.findall(r"<example>(.+?)</example>\n*<template>(.+?)</template>\n*<requirement>(.+?)</requirement>\n*<summary>(.+?)</summary>", contents, re.DOTALL)
        assert len(res) == cnts[0], "Format error. You did not give example, template, requirement, and summary in order. Please Retry."
        res = [[s.strip() for s in _res] for _res in res]
        return [Element(example=_res[0], content=_res[1], requirements=_res[2], summary=_res[3]) for _res in res]

