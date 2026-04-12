# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass, field # [修改] 导入 field
from dataclasses_json import dataclass_json
from pathlib import Path
from typing import List, Tuple, Optional # [修改] 导入 Optional


# [新] 定义一个专门的 Evidence 数据类
@dataclass_json
@dataclass
class Evidence:
    evidence_id: str = None # 例如: "s1_s2_p3_e1"
    text: str = None
    is_static: bool = False
    value: Optional[str] = None # 存储静态证据的具体值

@dataclass_json
@dataclass
class Segment:
    segment_id: str = None # 例如: "s1_s2_p3"
    finished: bool = False
    topic: str = None
    requirements: str = None
    reference: str = None
    content: str = None
    template: str = None
    # [修改] evidences 现在是一个 Evidence 对象的列表
    evidences: Optional[List[Evidence]] = field(default_factory=list)

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
            # [修改] 从 Evidence 对象中提取 text
            evidence_text = "\n".join(
                f"\t\t- {e.text} (静态: {e.is_static})" # 可以在打印时显示 is_static
                for e in self.evidences if e and e.text
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
        contents = contents.strip()
        if contents == "<skip>true</skip>":
            return None

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
        evidences_text, template, requirements, topic = [s.strip() for s in res[0]]
        
        # 解析带有 is_static 标志的 evidences
        parsed_evidences: List[Evidence] = []
        raw_evidences = evidences_text.replace("\n", "").replace(";", "；").split("；")
        for e_str in raw_evidences:
            e_str = e_str.strip()
            if not e_str: continue
            
            is_static = False
            static_value = None # 默认无值
            # 检查是否有 (static)[value] 标记
            static_match = re.search(r'\s*\((static|静态)\)\s*\[(.*?)\]\s*$', e_str, flags=re.IGNORECASE)
            if static_match:
                is_static = True
                static_value = static_match.group(2).strip() # 提取方括号里的值
                e_str = re.sub(r'\s*\((static|静态)\)\s*\[.*?\]\s*$', '', e_str, flags=re.IGNORECASE).strip()
            # 如果只有 (static) 没有值，也标记为静态
            elif e_str.lower().endswith("(static)") or e_str.endswith("(静态)"):
                is_static = True
                e_str = re.sub(r'\s*\((static|静态)\)\s*$', '', e_str, flags=re.IGNORECASE).strip()

            # 避免重复添加完全相同的文本
            if e_str and not any(e.text == e_str for e in parsed_evidences):
                # [修改] 传入 static_value
                parsed_evidences.append(Evidence(text=e_str, is_static=is_static, value=static_value))

        return Segment(template=template, requirements=requirements, topic=topic, evidences=parsed_evidences)
    
    @staticmethod
    def parse_evidence(contents: str) -> Segment:
        contents = contents.strip()
        if contents == "<skip>true</skip>":
            return None

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
        evidences_text, topic = [s.strip() for s in res]
    
        parsed_evidences: List[Evidence] = []
        raw_evidences = evidences_text.replace("\n", "").replace(";", "；").split("；")
        for e_str in raw_evidences:
            e_str = e_str.strip()
            if not e_str: continue
            is_static = False
            static_value = None

            static_match = re.search(r'\s*\((static|静态)\)\s*\[(.*?)\]\s*$', e_str, flags=re.IGNORECASE)
            if static_match:
                is_static = True
                static_value = static_match.group(2).strip()
                e_str = re.sub(r'\s*\((static|静态)\)\s*\[.*?\]\s*$', '', e_str, flags=re.IGNORECASE).strip()
            elif e_str.lower().endswith("(static)") or e_str.endswith("(静态)"):
                is_static = True
                e_str = re.sub(r'\s*\((static|静态)\)\s*$', '', e_str, flags=re.IGNORECASE).strip()

            if e_str and not any(e.text == e_str for e in parsed_evidences):
                parsed_evidences.append(Evidence(text=e_str, is_static=is_static, value=static_value))

        return Segment(template=None, requirements=None, topic=topic, evidences=parsed_evidences)
