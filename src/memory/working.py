# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pathlib import Path
from typing import List, Tuple


@dataclass_json
@dataclass
class Segment:
    finished: bool = False
    topic: str = None
    requirements: str = None
    reference: str = None
    content: str = None
    template: List[str] = None
    evidences: List[str] = None

    def __str__(
        self,
        with_requirements=True,
        with_reference=True,
        with_content=True,
        with_evidence=True,
    ):
        ctx = ""

        if with_reference and self.reference is not None:
            ref_block = "\n".join(
                "\t\t> " + l for l in self.reference.splitlines()
            )
            ctx += f"\t+ > **原文**\n{ref_block}\n\n"

        if with_content:
            if self.content is not None:
                ctx += f"\t+ **内容**\n\t{self.content}\n\n"
            elif self.template is not None:
                ctx += f"\t+ **示例**\n\t{self.template}\n\n"

        if with_requirements and self.requirements is not None:
            requirements = "\n".join(
                ("\t\t" if r.strip()[:2] in ["- ", "* "] else "") + r
                for r in self.requirements.split("\n")
            )
            ctx += f"\t+ **写作要求**\n{requirements}\n\n"

        if with_evidence and self.evidences is not None:
            evidence_block = "\n".join(
                "\t\t- " + e.replace("\n\n", "\n") for e in self.evidences
            )
            ctx += f"\t+ **论据材料**\n{evidence_block}\n\n"

        return ctx


@dataclass_json
@dataclass
class Section:
    section_id: int
    level: int
    title: str
    segments: List[Segment]
    subsections: List["Section"]
    is_leaf: bool = False

    def read(
        self,
        with_requirements=True,
        with_reference=False,
        with_content=False,
        with_evidence=False,
        fold_other=True,
        fold_all=False,
        read_subsections=False,
    ) -> str:
        ctx = f"{'#' * self.level} {self.title}\n"
        unfinished = [i for i, s in enumerate(self.segments) if not s.finished]

        for i, s in enumerate(self.segments):
            ctx += f"* [{'x' if s.finished else ' '}] {s.topic}\n"
            if fold_all or (unfinished and i != unfinished[0] and fold_other):
                continue
            ctx += s.__str__(
                with_requirements=with_requirements,
                with_reference=with_reference,
                with_content=with_content,
                with_evidence=with_evidence,
            )

        if read_subsections:
            for sec in self.subsections:
                ctx += (
                    sec.read(
                        with_requirements=with_requirements,
                        with_reference=with_reference,
                        with_content=with_content,
                        with_evidence=with_evidence,
                        fold_other=fold_other,
                        fold_all=fold_all,
                        read_subsections=True,
                    )
                    + "\n\n"
                )
        return ctx

    def load_with_prev_sections(
        self,
        section_id,
        with_requirements=True,
        with_reference=False,
        with_content=False,
    ) -> str:
        ctx = ""
        for i in range(section_id - 1):
            ctx += self.subsections[i].read(
                with_requirements=with_requirements,
                with_reference=with_reference,
                with_content=with_content,
                fold_all=True,
            )
        return ctx + self.read(
            with_requirements=with_requirements,
            with_reference=with_reference,
            with_content=with_content,
            fold_other=True,
        )

    @staticmethod
    def parse(contents: str) -> Segment:
        keys = ["requirement", "template", "evidence", "topic"]
        counts = [contents.count(f"<{k}>") for k in keys] + [
            contents.count(f"</{k}>") for k in keys
        ]

        for c1 in counts:
            for c2 in counts:
                assert c1 == c2 > 0, (
                    "Incomplete answer. You must give "
                    "<template>, </template>, <requirement>, </requirement>, "
                    "<topic> and </topic>."
                )

        contents = contents.replace("\r\n", "\n")
        print(contents, flush=True)

        res = re.findall(
            r"<template>(.+?)</template>.*<requirement>(.+?)</requirement>.*<topic>(.+?)</topic>",
            contents,
            re.DOTALL,
        )
        assert res, "Format error."

        template, requirements, topic = (s.strip() for s in res[0])

        evidences_match = re.search(
            r"<evidence>(.+?)</evidence>", contents, re.DOTALL
        )
        evidences = None
        if evidences_match:
            evidences = (
                evidences_match.group(1)
                .replace("\n", "")
                .replace(";", "；")
                .split("；")
            )
            evidences = [e.strip() for e in evidences]

        return Segment(
            template=template,
            requirements=requirements,
            topic=topic,
            evidences=evidences,
        )
