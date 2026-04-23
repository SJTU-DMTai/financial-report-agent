# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass
from itertools import count
from typing import Dict, List, Literal, Optional, Type

from agentscope.formatter import FormatterBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase
from openai import RateLimitError
from pydantic import BaseModel, Field

from src.memory.working import Evidence, Section, Segment
from src.prompt import prompt_dict

MAX_REQUIREMENTS_CHARS = 360
MAX_TEMPLATE_CHARS = 480
MAX_TOPIC_CHARS = 80
MAX_EVIDENCE_CHARS = 120


@dataclass
class SectionRef:
    ref_id: str
    section: Section
    parent: Optional[Section]


@dataclass
class SegmentRef:
    ref_id: str
    segment: Segment
    parent: Section


@dataclass
class OutlineRefineContext:
    snapshot: Dict
    section_refs: Dict[str, SectionRef]
    segment_refs: Dict[str, SegmentRef]


class OutlineEvidencePayload(BaseModel):
    text: str


class OutlineSegmentPayload(BaseModel):
    topic: str
    requirements: Optional[str] = None
    template: Optional[str] = None
    evidences: Optional[List[OutlineEvidencePayload]] = None


class OutlineNewSegmentPayload(BaseModel):
    topic: str
    requirements: str
    evidences: List[OutlineEvidencePayload]


class OutlineSectionPayload(BaseModel):
    title: str
    segments: List[OutlineNewSegmentPayload] = Field(default_factory=list)


class OutlinePatch(BaseModel):
    title: Optional[str] = None
    topic: Optional[str] = None
    requirements: Optional[str] = None
    template: Optional[str] = None
    evidences: Optional[List[OutlineEvidencePayload]] = None


class OutlineRefineOperation(BaseModel):
    action: Literal[
        "modify_section",
        "add_section",
        "delete_section",
        "modify_segment",
        "add_segment",
        "delete_segment",
        "merge_segments",
        "split_segment",
    ]
    section_id: Optional[str] = None
    parent_section_id: Optional[str] = None
    segment_id: Optional[str] = None
    segment_ids: List[str] = Field(default_factory=list)
    insert_index: Optional[int] = None
    new_section: Optional[OutlineSectionPayload] = None
    new_segment: Optional[OutlineNewSegmentPayload] = None
    new_segments: List[OutlineNewSegmentPayload] = Field(default_factory=list)
    updates: Optional[OutlinePatch] = None


class OutlineRefineResponse(BaseModel):
    operations: List[OutlineRefineOperation] = Field(default_factory=list)


def _model_validate(model_cls, payload):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _normalize_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = text.strip()
    if not normalized:
        return None
    return normalized


def _truncate_text(text: Optional[str], max_chars: int) -> Optional[str]:
    normalized = _normalize_text(text)
    if normalized is None:
        return None
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


def _serialize_evidences_for_refine(evidences: Optional[List[Evidence]]) -> List[Dict]:
    serialized: List[Dict] = []
    if not evidences:
        return serialized
    for evidence in evidences:
        if evidence is None or not evidence.text:
            continue
        serialized.append(
            {
                "text": _truncate_text(evidence.text, MAX_EVIDENCE_CHARS),
            }
        )
    return serialized


def _serialize_segment_for_refine(segment: Segment, segment_id: str) -> Dict:
    return {
        "id": segment_id,
        "topic": _truncate_text(segment.topic, MAX_TOPIC_CHARS),
        "requirements": _truncate_text(segment.requirements, MAX_REQUIREMENTS_CHARS),
        "template": _truncate_text(segment.template, MAX_TEMPLATE_CHARS),
        "evidences": _serialize_evidences_for_refine(segment.evidences),
        "reference_available": bool(segment.reference),
        "finished": bool(segment.finished),
    }


def _serialize_section_for_refine(
    section: Section,
    parent_id: Optional[str],
    section_counter,
    segment_counter,
    section_refs: Dict[str, SectionRef],
    segment_refs: Dict[str, SegmentRef],
) -> Dict:
    section_id = f"SEC_{next(section_counter):03d}"
    section_refs[section_id] = SectionRef(ref_id=section_id, section=section, parent=parent_id and section_refs[parent_id].section)

    serialized_segments: List[Dict] = []
    for segment in section.segments or []:
        segment_id = f"SEG_{next(segment_counter):03d}"
        segment_refs[segment_id] = SegmentRef(ref_id=segment_id, segment=segment, parent=section)
        serialized_segments.append(_serialize_segment_for_refine(segment, segment_id))

    serialized_subsections: List[Dict] = []
    for subsection in section.subsections or []:
        serialized_subsections.append(
            _serialize_section_for_refine(
                subsection,
                section_id,
                section_counter,
                segment_counter,
                section_refs,
                segment_refs,
            )
        )

    return {
        "id": section_id,
        "parent_id": parent_id,
        "title": _normalize_text(section.title),
        "level": section.level,
        "segments": serialized_segments,
        "subsections": serialized_subsections,
    }


def build_outline_refine_context(outline: Section) -> OutlineRefineContext:
    section_refs: Dict[str, SectionRef] = {}
    segment_refs: Dict[str, SegmentRef] = {}
    section_counter = count(1)
    segment_counter = count(1)
    snapshot = _serialize_section_for_refine(
        outline,
        None,
        section_counter,
        segment_counter,
        section_refs,
        segment_refs,
    )
    return OutlineRefineContext(
        snapshot=snapshot,
        section_refs=section_refs,
        segment_refs=segment_refs,
    )


def serialize_outline_for_refine(outline: Section) -> str:
    context = build_outline_refine_context(outline)
    payload = {
        "stats": {
            "section_count": len(context.section_refs),
            "segment_count": len(context.segment_refs),
        },
        "outline": context.snapshot,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _clone_evidences(evidences: Optional[List[Evidence]]) -> Optional[List[Evidence]]:
    if evidences is None:
        return None
    cloned: List[Evidence] = []
    seen = set()
    for evidence in evidences:
        if evidence is None or not evidence.text:
            continue
        key = evidence.text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cloned.append(Evidence(text=key, is_static=bool(evidence.is_static)))
    if not cloned:
        return None
    return cloned


def _build_evidences(payload_evidences: Optional[List[OutlineEvidencePayload]]) -> Optional[List[Evidence]]:
    if payload_evidences is None:
        return None
    built: List[Evidence] = []
    seen = set()
    for payload in payload_evidences:
        text = _normalize_text(payload.text)
        if text is None or text in seen:
            continue
        seen.add(text)
        built.append(Evidence(text=text, is_static=False))
    if not built:
        return None
    return built


def _build_required_evidences(payload_evidences: List[OutlineEvidencePayload]) -> List[Evidence]:
    evidences = _build_evidences(payload_evidences)
    if evidences is None:
        raise ValueError("新生成 segment 的 evidences 不能为空")
    return evidences


def _collect_combined_reference(segments: List[Segment]) -> Optional[str]:
    references: List[str] = []
    seen = set()
    for segment in segments:
        reference = _normalize_text(segment.reference)
        if reference is None or reference in seen:
            continue
        seen.add(reference)
        references.append(reference)
    if not references:
        return None
    return "\n\n".join(references)


def _collect_combined_evidences(segments: List[Segment]) -> Optional[List[Evidence]]:
    evidences: List[Evidence] = []
    seen = set()
    for segment in segments:
        for evidence in segment.evidences or []:
            if evidence is None or not evidence.text:
                continue
            key = evidence.text.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            evidences.append(Evidence(text=key, is_static=bool(evidence.is_static)))
    if not evidences:
        return None
    return evidences


def _build_segment_from_payload(
    payload: OutlineSegmentPayload | OutlineNewSegmentPayload,
    fallback_reference: Optional[str] = None,
    fallback_evidences: Optional[List[Evidence]] = None,
) -> Segment:
    template = None
    if isinstance(payload, OutlineNewSegmentPayload):
        evidences = _build_required_evidences(payload.evidences)
    else:
        evidences = _build_evidences(payload.evidences)
        if evidences is None:
            evidences = _clone_evidences(fallback_evidences)
        template = _normalize_text(payload.template)
    return Segment(
        finished=False,
        topic=_normalize_text(payload.topic),
        requirements=_normalize_text(payload.requirements),
        reference=_normalize_text(fallback_reference),
        content=None,
        template=template,
        evidences=evidences,
    )


def _build_section_from_payload(payload: OutlineSectionPayload, parent_level: int) -> Section:
    segments = [_build_segment_from_payload(segment_payload) for segment_payload in payload.segments]
    return Section(
        section_id=0,
        level=parent_level + 1,
        title=_normalize_text(payload.title) or "未命名章节",
        segments=segments,
        subsections=[],
        content=None,
    )


def _find_child_section_index(parent: Section, section: Section) -> int:
    for index, child in enumerate(parent.subsections or []):
        if child is section:
            return index
    return -1


def _find_child_segment_index(parent: Section, segment: Segment) -> int:
    for index, child in enumerate(parent.segments or []):
        if child is segment:
            return index
    return -1


def _insert_item(items: List, item, insert_index: Optional[int]) -> None:
    if insert_index is None:
        items.append(item)
        return
    bounded_index = max(0, min(insert_index, len(items)))
    items.insert(bounded_index, item)


def _apply_section_patch(section: Section, patch: OutlinePatch) -> bool:
    changed = False
    if patch.title is not None:
        title = _normalize_text(patch.title)
        if title and title != section.title:
            section.title = title
            changed = True
    return changed


def _apply_segment_patch(segment: Segment, patch: OutlinePatch) -> bool:
    changed = False
    if patch.topic is not None:
        topic = _normalize_text(patch.topic)
        if topic and topic != segment.topic:
            segment.topic = topic
            changed = True
    if patch.requirements is not None:
        requirements = _normalize_text(patch.requirements)
        if requirements != segment.requirements:
            segment.requirements = requirements
            changed = True
    if patch.template is not None:
        template = _normalize_text(patch.template)
        if template != segment.template:
            segment.template = template
            changed = True
    if patch.evidences is not None:
        evidences = _build_evidences(patch.evidences)
        if evidences != segment.evidences:
            segment.evidences = evidences
            changed = True
    return changed


def _apply_modify_section_operation(operation: OutlineRefineOperation, context: OutlineRefineContext) -> bool:
    if operation.section_id is None or operation.updates is None:
        return False
    section_ref = context.section_refs.get(operation.section_id)
    if section_ref is None:
        warnings.warn(f"[Outline Refine] 未找到 section_id={operation.section_id}，跳过 modify_section")
        return False
    return _apply_section_patch(section_ref.section, operation.updates)


def _apply_add_section_operation(operation: OutlineRefineOperation, context: OutlineRefineContext) -> bool:
    if operation.parent_section_id is None or operation.new_section is None:
        return False
    parent_ref = context.section_refs.get(operation.parent_section_id)
    if parent_ref is None:
        warnings.warn(f"[Outline Refine] 未找到 parent_section_id={operation.parent_section_id}，跳过 add_section")
        return False
    if parent_ref.section.subsections is None:
        parent_ref.section.subsections = []
    new_section = _build_section_from_payload(operation.new_section, parent_ref.section.level or 1)
    _insert_item(parent_ref.section.subsections, new_section, operation.insert_index)
    return True


def _apply_delete_section_operation(operation: OutlineRefineOperation, context: OutlineRefineContext) -> bool:
    if operation.section_id is None:
        return False
    section_ref = context.section_refs.get(operation.section_id)
    if section_ref is None or section_ref.parent is None:
        warnings.warn(f"[Outline Refine] 无法删除 section_id={operation.section_id}，跳过 delete_section")
        return False
    parent = section_ref.parent
    child_index = _find_child_section_index(parent, section_ref.section)
    if child_index < 0:
        warnings.warn(f"[Outline Refine] section_id={operation.section_id} 已不存在，跳过 delete_section")
        return False
    parent.subsections.pop(child_index)
    return True


def _apply_modify_segment_operation(operation: OutlineRefineOperation, context: OutlineRefineContext) -> bool:
    if operation.segment_id is None or operation.updates is None:
        return False
    segment_ref = context.segment_refs.get(operation.segment_id)
    if segment_ref is None:
        warnings.warn(f"[Outline Refine] 未找到 segment_id={operation.segment_id}，跳过 modify_segment")
        return False
    if _find_child_segment_index(segment_ref.parent, segment_ref.segment) < 0:
        warnings.warn(f"[Outline Refine] segment_id={operation.segment_id} 已不存在，跳过 modify_segment")
        return False
    return _apply_segment_patch(segment_ref.segment, operation.updates)


def _apply_add_segment_operation(operation: OutlineRefineOperation, context: OutlineRefineContext) -> bool:
    if operation.section_id is None or operation.new_segment is None:
        return False
    section_ref = context.section_refs.get(operation.section_id)
    if section_ref is None:
        warnings.warn(f"[Outline Refine] 未找到 section_id={operation.section_id}，跳过 add_segment")
        return False
    if section_ref.section.segments is None:
        section_ref.section.segments = []
    new_segment = _build_segment_from_payload(operation.new_segment)
    _insert_item(section_ref.section.segments, new_segment, operation.insert_index)
    return True


def _apply_delete_segment_operation(operation: OutlineRefineOperation, context: OutlineRefineContext) -> bool:
    if operation.segment_id is None:
        return False
    segment_ref = context.segment_refs.get(operation.segment_id)
    if segment_ref is None:
        warnings.warn(f"[Outline Refine] 未找到 segment_id={operation.segment_id}，跳过 delete_segment")
        return False
    child_index = _find_child_segment_index(segment_ref.parent, segment_ref.segment)
    if child_index < 0:
        warnings.warn(f"[Outline Refine] segment_id={operation.segment_id} 已不存在，跳过 delete_segment")
        return False
    segment_ref.parent.segments.pop(child_index)
    return True


def _resolve_merge_segments(operation: OutlineRefineOperation, context: OutlineRefineContext) -> List[SegmentRef]:
    resolved: List[SegmentRef] = []
    for segment_id in operation.segment_ids:
        segment_ref = context.segment_refs.get(segment_id)
        if segment_ref is None:
            warnings.warn(f"[Outline Refine] 未找到 segment_id={segment_id}，跳过 merge_segments")
            return []
        if _find_child_segment_index(segment_ref.parent, segment_ref.segment) < 0:
            warnings.warn(f"[Outline Refine] segment_id={segment_id} 已不存在，跳过 merge_segments")
            return []
        resolved.append(segment_ref)
    return resolved


def _apply_merge_segments_operation(operation: OutlineRefineOperation, context: OutlineRefineContext) -> bool:
    if len(operation.segment_ids) < 2 or operation.new_segment is None:
        return False
    segment_refs = _resolve_merge_segments(operation, context)
    if not segment_refs:
        return False

    parent = segment_refs[0].parent
    for segment_ref in segment_refs[1:]:
        if segment_ref.parent is not parent:
            warnings.warn("[Outline Refine] merge_segments 仅支持同一 section 下的 segments，已跳过")
            return False

    current_indices = [_find_child_segment_index(parent, segment_ref.segment) for segment_ref in segment_refs]
    sorted_indices = sorted(current_indices)
    expected_indices = list(range(sorted_indices[0], sorted_indices[0] + len(sorted_indices)))
    if sorted_indices != expected_indices:
        warnings.warn("[Outline Refine] merge_segments 仅支持合并当前 section 内相邻 segments，已跳过")
        return False

    segments_to_merge = [parent.segments[index] for index in sorted_indices]
    merged_segment = _build_segment_from_payload(
        operation.new_segment,
        fallback_reference=_collect_combined_reference(segments_to_merge),
        fallback_evidences=_collect_combined_evidences(segments_to_merge),
    )
    start_index = sorted_indices[0]
    parent.segments[start_index : sorted_indices[-1] + 1] = [merged_segment]
    return True


def _apply_split_segment_operation(operation: OutlineRefineOperation, context: OutlineRefineContext) -> bool:
    if operation.segment_id is None or len(operation.new_segments) < 2:
        return False
    segment_ref = context.segment_refs.get(operation.segment_id)
    if segment_ref is None:
        warnings.warn(f"[Outline Refine] 未找到 segment_id={operation.segment_id}，跳过 split_segment")
        return False
    child_index = _find_child_segment_index(segment_ref.parent, segment_ref.segment)
    if child_index < 0:
        warnings.warn(f"[Outline Refine] segment_id={operation.segment_id} 已不存在，跳过 split_segment")
        return False

    replacement_segments: List[Segment] = []
    for payload in operation.new_segments:
        replacement_segments.append(
            _build_segment_from_payload(
                payload,
                fallback_reference=segment_ref.segment.reference,
            )
        )
    segment_ref.parent.segments[child_index : child_index + 1] = replacement_segments
    return True


def _apply_outline_refine_operation(operation: OutlineRefineOperation, context: OutlineRefineContext) -> bool:
    if operation.action == "modify_section":
        return _apply_modify_section_operation(operation, context)
    if operation.action == "add_section":
        return _apply_add_section_operation(operation, context)
    if operation.action == "delete_section":
        return _apply_delete_section_operation(operation, context)
    if operation.action == "modify_segment":
        return _apply_modify_segment_operation(operation, context)
    if operation.action == "add_segment":
        return _apply_add_segment_operation(operation, context)
    if operation.action == "delete_segment":
        return _apply_delete_segment_operation(operation, context)
    if operation.action == "merge_segments":
        return _apply_merge_segments_operation(operation, context)
    if operation.action == "split_segment":
        return _apply_split_segment_operation(operation, context)
    warnings.warn(f"[Outline Refine] 未知操作类型={operation.action}，已跳过")
    return False


def _renumber_child_sections(parent: Section) -> None:
    if not parent.subsections:
        return
    for index, subsection in enumerate(parent.subsections, start=1):
        subsection.section_id = index
        subsection.level = (parent.level or 1) + 1
        _renumber_child_sections(subsection)


def renumber_outline_structure(outline: Section) -> None:
    outline.section_id = 0
    outline.level = outline.level or 1
    _renumber_child_sections(outline)


def apply_outline_refine_operations(
    outline: Section,
    context: OutlineRefineContext,
    operations: List[OutlineRefineOperation],
) -> bool:
    changed = False
    for operation in operations:
        try:
            changed = _apply_outline_refine_operation(operation, context) or changed
        except Exception as exc:
            warnings.warn(
                f"[Outline Refine] 应用操作失败 action={operation.action} "
                f"reason={type(exc).__name__}: {exc}"
            )
    if changed:
        renumber_outline_structure(outline)
    return changed


def _build_outline_refine_review_user_prompt(task_desc: str, cur_date: Optional[str], outline_json: str) -> str:
    return (
        f"{prompt_dict['outline_refine_review']}\n\n"
        f"当前任务：{task_desc}\n"
        f"当前写作日期：{cur_date or '未提供'}\n\n"
        "请只基于下面的初始 outline JSON 进行诊断。\n"
        f"```json\n{outline_json}\n```"
    )


def _build_outline_refine_apply_user_prompt() -> str:
    return prompt_dict["outline_refine"]


async def _call_chatbot_with_message_history_retry(
    model: ChatModelBase,
    formatter: FormatterBase,
    messages: List[Msg],
    structured_model: Optional[Type[BaseModel]] = None,
    max_retries: int = 5,
):
    from src.utils.call_with_retry import endpoints
    from src.utils.global_semaphore import get_global_semaphore
    from src.utils.instance import cfg

    semaphore = get_global_semaphore()
    result = ""
    async with semaphore:
        exceed_tpm_models = set()
        for _ in range(max_retries):
            try:
                formatted_messages = await formatter.format(messages)
                response = await model(formatted_messages, structured_model=structured_model)
                if structured_model is not None:
                    result = response.metadata
                else:
                    result = Msg(role="assistant", content=response.content, name="assistant").get_text_content()
            except RateLimitError as exc:
                if cfg.get_model_cfg()["provider"] == "ark" and os.getenv("LLM_NAME") == "deepseek-v3.2":
                    print(exc)
                    exceed_tpm_models.add(model.model_name)
                    if len(exceed_tpm_models) >= len(endpoints):
                        await asyncio.sleep(60)
                        exceed_tpm_models = set()
                    model.model_name = list(endpoints - exceed_tpm_models)[0]
                    print("切换为", model.model_name, flush=True)
                    continue
                raise
            except Exception as exc:
                warnings.warn(f"[Outline Refine] 多轮调用失败，第 {_} 次异常：{type(exc).__name__}: {exc}")
                continue
            if result:
                return result
        raise Exception("Outline refine 多轮调用多次失败，放弃重试。")


def _log_outline_refine_issues(issues_markdown: str) -> None:
    print("[Outline Refine] round1 issues:", flush=True)
    print(issues_markdown, flush=True)


def _print_outline_refine_debug_block(title: str, content) -> None:
    print(f"[Outline Refine Debug] {title}", flush=True)
    if isinstance(content, str):
        print(content, flush=True)
        return
    print(json.dumps(content, ensure_ascii=False, indent=2), flush=True)


def _log_outline_refine_operations(response: OutlineRefineResponse) -> List[OutlineRefineOperation]:
    operations = response.operations
    print(
        f"[Outline Refine] operations={len(operations)}",
        flush=True,
    )
    for index, operation in enumerate(operations, start=1):
        print(
            f"[Outline Refine] op{index}: action={operation.action}",
            flush=True,
        )
    return operations


async def refine_outline(
    outline: Section,
    task_desc: str,
    cur_date: Optional[str],
    model: ChatModelBase,
    formatter: FormatterBase,
    debug_print: bool = False,
) -> Section:
    outline_copy = deepcopy(outline)
    context = build_outline_refine_context(outline_copy)
    outline_json = json.dumps(
        {
            "stats": {
                "section_count": len(context.section_refs),
                "segment_count": len(context.segment_refs),
            },
            "outline": context.snapshot,
        },
        ensure_ascii=False,
        indent=2,
    )

    try:
        conversation = [
            Msg("system", prompt_dict["outline_refine_sys_prompt"], "system"),
            Msg(
                "user",
                _build_outline_refine_review_user_prompt(task_desc, cur_date, outline_json),
                "user",
            ),
        ]
        if debug_print:
            _print_outline_refine_debug_block("round1 system prompt", prompt_dict["outline_refine_sys_prompt"])
            _print_outline_refine_debug_block("round1 user prompt", conversation[1].content)
        issues_markdown = await _call_chatbot_with_message_history_retry(
            model=model,
            formatter=formatter,
            messages=conversation,
        )
        if debug_print:
            _print_outline_refine_debug_block("round1 assistant response", issues_markdown)
        _log_outline_refine_issues(issues_markdown)
        conversation.append(Msg("assistant", issues_markdown, "assistant"))
        conversation.append(
            Msg(
                "user",
                _build_outline_refine_apply_user_prompt(),
                "user",
            )
        )
        if debug_print:
            _print_outline_refine_debug_block("round2 user prompt", conversation[-1].content)
        raw_response = await _call_chatbot_with_message_history_retry(
            model=model,
            formatter=formatter,
            messages=conversation,
            structured_model=OutlineRefineResponse,
        )
        if debug_print:
            _print_outline_refine_debug_block("round2 assistant response", raw_response)
        response = _model_validate(OutlineRefineResponse, raw_response)
        operations = _log_outline_refine_operations(response)
        apply_outline_refine_operations(outline_copy, context, operations)
        return outline_copy
    except Exception as exc:
        warnings.warn(
            f"[Outline Refine] 调用或应用 refine 失败，回退原 outline。"
            f" reason={type(exc).__name__}: {exc}"
        )
        return outline
