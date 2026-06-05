# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from agentscope.message import Msg

import config
from src.agents.searcher import build_searcher_toolkit, create_searcher_agent
from src.agents.writer import build_writer_toolkit, create_writer_agent
from src.evaluation.eval_content import _extract_score_suggestion
from src.memory.evidence_registry import EvidenceRecord, EvidenceRegistry
from src.memory.long_term import LongTermMemoryStore
from src.memory.short_term import ShortTermMemoryStore
from src.memory.tracking_board import (
    SegmentBinding,
    SegmentIssue,
    SegmentRecord,
    TrackingBoard,
    build_tracking_board,
)
from src.memory.working import Section, load_section_from_json_text, _normalize_evidences
from src.pipelines.planning import process_pdf_to_outline
from src.prompt import prompt_dict
from src.utils.call_with_retry import call_agent_with_retry, call_chatbot_with_retry
from src.utils.evidence_dependency import build_evidence_dependencies
from src.utils.file_converter import md_to_pdf, section_to_markdown
from src.utils.format import (
    _infer_report_title,
    _normalize_report_title,
    _normalize_section_titles,
    extract_writer_content,
    print_section_reference_warning,
)
from src.utils.get_entity_info import get_entity_info
from src.utils.instance import (
    create_agent_formatter,
    formatter,
    llm_instruct,
    llm_judge,
    llm_outline_refine,
    llm_reasoning,
    outline_refine_formatter,
)
from src.utils.local_file import STOCK_REPORT_PATHS
from src.utils.multi_types_verification import (
    SegmentVerifier,
    append_verifier_trace,
    set_verifier_trace_path,
)
from src.utils.outline_refine import refine_outline
from src.utils.tracking_board_format import (
    build_evidence_context,
    build_known_evidence_context,
    extract_cite_ids,
    parse_replan_response,
    parse_section_polish_response,
    parse_writer_issue,
)


SAVE_LOCK = asyncio.Lock()


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


def create_evidence_searcher(short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore):
    searcher_toolkit = build_searcher_toolkit(short_term=short_term, long_term=long_term)
    return create_searcher_agent(model=llm_reasoning, formatter=formatter, toolkit=searcher_toolkit)


def create_segment_agents(short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore):
    writer_toolkit = build_writer_toolkit(short_term=short_term, long_term=long_term)
    writer = create_writer_agent(
        model=llm_reasoning,
        formatter=formatter,
        toolkit=writer_toolkit,
    )
    verifier = SegmentVerifier(short_term, long_term)
    return writer, verifier


def initialize_registry_for_board(
    registry: EvidenceRegistry,
    board: TrackingBoard,
    bindings: dict[str, SegmentBinding],
    company_name: str,
) -> None:
    current_segment_ids = set(board.records)
    # Drop stale segment links before rebuilding the current outline bindings.
    for registry_record in registry.records.values():
        registry_record.used_by_segments = [
            segment_id
            for segment_id in registry_record.used_by_segments
            if segment_id not in current_segment_ids
        ]
    for segment_id, record in board.records.items():
        binding = bindings[segment_id]
        record.evidences = []
        for evidence in binding.segment.evidences or []:
            # add_or_reuse is the SAME-evidence gate based on canonical_key.
            evidence_record = registry.add_or_reuse(evidence, segment_id, default_entity=company_name)
            if evidence_record.evidence_id not in record.evidences:
                record.evidences.append(evidence_record.evidence_id)


def merge_loaded_board(
    current_board: TrackingBoard,
    loaded_board: TrackingBoard | None,
) -> TrackingBoard:
    if loaded_board is None:
        return current_board
    for segment_id, record in current_board.records.items():
        loaded_record = loaded_board.records.get(segment_id)
        if loaded_record is None:
            continue
        loaded_record.topic = record.topic
        loaded_record.template = record.template
        loaded_record.requirements = record.requirements
        current_board.records[segment_id] = loaded_record
    return current_board


def sync_board_records_to_segments(
    board: TrackingBoard,
    bindings: dict[str, SegmentBinding],
) -> None:
    for segment_id, record in board.records.items():
        binding = bindings.get(segment_id)
        if binding is None:
            continue
        # Restore persisted drafts into the manuscript tree for resume runs.
        latest_draft = record.latest_draft()
        if latest_draft:
            binding.segment.content = latest_draft
        if record.state == "FINALIZED":
            binding.segment.finished = True


def apply_replan_payload(
    payload: dict[str, Any],
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    company_name: str,
) -> None:
    topic = str(payload.get("topic") or record.topic or "").strip()
    raw_requirements = payload.get("requirements")
    if isinstance(raw_requirements, list):
        requirements_list = [str(item).strip() for item in raw_requirements if str(item).strip()]
    else:
        requirements_text = str(raw_requirements or "\n".join(record.requirements)).strip()
        requirements_list = [line.strip() for line in requirements_text.splitlines() if line.strip()]
    template = str(payload.get("template") or record.template or "").strip()
    evidences_payload = payload.get("evidences") if "evidences" in payload else None
    evidences = _normalize_evidences(evidences_payload) if evidences_payload is not None else binding.segment.evidences

    if topic:
        record.topic = topic
        binding.segment.topic = topic
    if requirements_list:
        record.requirements = requirements_list
        binding.segment.requirements = "\n".join(requirements_list)
    if template:
        record.template = template
        binding.segment.template = template
    if evidences_payload is not None:
        binding.segment.evidences = evidences

    # Replan replaces this segment's evidence links without deleting shared records.
    for evidence_id in record.evidences:
        evidence_record = registry.records.get(evidence_id)
        if evidence_record and record.segment_id in evidence_record.used_by_segments:
            evidence_record.used_by_segments.remove(record.segment_id)

    record.evidences = []
    for evidence in binding.segment.evidences or []:
        evidence_record = registry.add_or_reuse(evidence, record.segment_id, default_entity=company_name)
        record.evidences.append(evidence_record.evidence_id)

    record.issue = None
    record.state = "PLANNED"


async def replan_segment_after_not_found(
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    task_desc: str,
    cur_date: str,
    company_name: str,
) -> bool:
    unavailable = [
        registry.records[evidence_id]
        for evidence_id in record.evidences
        if registry.records.get(evidence_id) and registry.records[evidence_id].state == "UNAVAILABLE"
    ]
    unavailable_text = "\n".join(
        f"- {item.description}"
        for item in unavailable
    )
    user_prompt = prompt_dict["replan_segment_user_prompt"].format(
        task_desc=task_desc,
        cur_date=cur_date,
        topic=record.topic,
        requirements="\n".join(record.requirements),
        template=record.template,
        unavailable_evidences=unavailable_text or (record.issue.detail if record.issue else ""),
    )
    payload = await call_chatbot_with_retry(
        llm_instruct,
        formatter,
        prompt_dict["replan_segment_sys_prompt"],
        user_prompt,
        hook=parse_replan_response,
        handle_hook_exceptions=(ValueError,),
    )
    apply_replan_payload(payload, record, binding, registry, company_name)
    return True


def select_evidence_segment_context(
    record: EvidenceRecord,
    bindings: dict[str, SegmentBinding],
) -> tuple[str, str | None]:
    for segment_id in record.used_by_segments:
        binding = bindings.get(segment_id)
        if binding is not None:
            return binding.segment.topic or record.description, binding.segment.reference
    return record.description, None


async def search_tracking_evidence(
    record: EvidenceRecord,
    known_evidence: str,
    task_desc: str,
    demo_date: str,
    segment_topic: str,
    searcher,
    reference: str | None = None,
) -> str:
    reference_context = (
        f"{demo_date}发布了一份历史研报，以下片段可能包含所需材料：\n{reference}\n"
        f"如果该片段包含所需材料，并且一定不会因时间变化，在当前撰写时间依然成立，"
        f"可以摘取相关两三句话作为论据，并加上 [^cite_id:{demo_date}_reference_report]。"
        if reference
        else "无可用参考片段。"
    )
    searcher_input = Msg(
        name="user",
        content=prompt_dict["search_tracking_evidence_user_prompt"].format(
            task_desc=task_desc,
            segment_topic=segment_topic,
            known_evidence=known_evidence,
            evidence_description=record.description,
            reference_context=reference_context,
        ),
        role="user",
    )
    print(searcher_input.content)
    msg = await call_agent_with_retry(searcher, searcher_input)
    print(f"[Searcher] Finished searching: {record.description[:20]}...")
    return msg.get_text_content()


async def resolve_global_evidence_record(
    record: EvidenceRecord,
    registry: EvidenceRegistry,
    bindings: dict[str, SegmentBinding],
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    task_desc: str,
    demo_date: str,
) -> bool:
    if record.state == "RESOLVED":
        return True
    if record.state == "SKIPPED":
        return True
    if record.state == "UNAVAILABLE":
        return False
    if record.state == "SEARCHING":
        # Another concurrent segment is already resolving the shared evidence.
        for _ in range(120):
            await asyncio.sleep(1)
            if record.state != "SEARCHING":
                return record.state == "RESOLVED"
        return False
    if not registry.dependencies_resolved(record.evidence_id):
        return False
    static_fact = str(record.fields.get("fact") or "").strip()
    if bool(record.fields.get("is_static", False)) and static_fact:
        # Static evidence with a concrete planned fact needs no additional search.
        record.search_result = record.search_result or f"{record.description}：{static_fact}"
        record.state = "RESOLVED"
        return True

    registry.records[record.evidence_id].state = "SEARCHING"
    segment_topic, reference = select_evidence_segment_context(record, bindings)
    known_evidence = build_known_evidence_context(registry, exclude_id=record.evidence_id)
    searcher = create_evidence_searcher(short_term, long_term)
    try:
        result_text = await search_tracking_evidence(
            record,
            known_evidence,
            task_desc,
            demo_date,
            segment_topic,
            searcher,
            reference=reference,
        )
    except Exception:
        record.state = "PLANNED"
        raise
    if str(result_text or "").strip().upper() == "SKIP":
        record.search_result = ""
        record.state = "SKIPPED"
        return True
    cite_ids = extract_cite_ids(result_text)
    registry.mark_resolved(
        record.evidence_id,
        cite_ids=cite_ids,
        search_result=result_text,
    )
    return registry.records[record.evidence_id].state == "RESOLVED"


async def resolve_global_evidence_registry(
    registry: EvidenceRegistry,
    bindings: dict[str, SegmentBinding],
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    task_desc: str,
    demo_date: str,
    evidence_concurrency: int,
) -> bool:
    changed = False
    concurrency = max(int(evidence_concurrency), 1)
    for _ in range(max(len(registry.records), 1)):
        changed = registry.mark_records_blocked_by_unavailable_dependencies() > 0 or changed
        ready_records = registry.ready_to_search_records()
        if not ready_records:
            break
        batch = ready_records[:concurrency]
        results = await asyncio.gather(
            *[
                resolve_global_evidence_record(
                    record,
                    registry,
                    bindings,
                    short_term,
                    long_term,
                    task_desc,
                    demo_date,
                )
                for record in batch
            ],
            return_exceptions=True,
        )
        for record, result in zip(batch, results):
            if isinstance(result, Exception):
                traceback.print_exception(type(result), result, result.__traceback__)
                record.state = "PLANNED"
                continue
            changed = True
    return changed


async def check_segment_evidences_ready(
    record: SegmentRecord,
    registry: EvidenceRegistry,
) -> bool:
    unresolved = registry.unresolved_required_for_segment(record.segment_id)
    unavailable = [item for item in unresolved if item.state == "UNAVAILABLE"]
    if unavailable:
        detail = "；".join(item.description for item in unavailable)
        record.issue = SegmentIssue(
            type="EVIDENCE_NOT_FOUND",
            detail=f"关键证据未找到或不可用：{detail}",
            action="REPLAN",
        )
        return False
    if unresolved:
        record.issue = None
        record.state = "RETRIEVING"
        return False
    record.issue = None
    record.state = "EVIDENCE_READY"
    return True


async def write_segment_draft(
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    short_term: ShortTermMemoryStore,
    writer,
    task_desc: str,
    reason: str,
) -> bool:
    evidence_context = build_evidence_context(registry, record.segment_id, short_term)
    outline_context_intro = "写作要求和可用论据材料如下："
    segment_text = f"{binding.segment.__str__(with_evidence=False)}\n\n可用论据材料：\n{evidence_context}"
    writer_input = Msg(
        name="user",
        content=prompt_dict["segment_writer_user_prompt"].format(
            task_desc=task_desc,
            segment_topic=record.topic or "未指定",
            outline_context_intro=outline_context_intro,
            segment_text=segment_text,
        ),
        role="user",
    )
    draft_msg = await call_agent_with_retry(writer, writer_input)
    raw_text = draft_msg.get_text_content()
    content = extract_writer_content(raw_text)
    if content:
        binding.segment.content = content
        binding.parent.content = None
        record.add_draft(content, reason)
        record.issue = None
        record.state = "DRAFTED"
        return True

    record.issue = SegmentIssue(
        type="EXECUTION_ERROR",
        detail="Writer 输出为空，未生成正文。",
        action="RETRY",
    )
    return False


async def check_writer_evidence_sufficiency(
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    short_term: ShortTermMemoryStore,
    task_desc: str,
    cur_date: str,
) -> bool:
    evidence_context = build_evidence_context(registry, record.segment_id, short_term)
    segment_text = binding.segment.__str__(with_evidence=False)
    user_prompt = prompt_dict["segment_evidence_check_user_prompt"].format(
        task_desc=task_desc,
        cur_date=cur_date,
        segment_text=segment_text,
        evidence_context=evidence_context,
    )
    response = await call_chatbot_with_retry(
        llm_instruct,
        formatter,
        prompt_dict["writer_evidence_check_sys_prompt"],
        user_prompt,
    )
    if str(response or "").strip().upper() == "SKIP":
        record.issue = None
        return True
    parsed_issue = parse_writer_issue(response)
    if parsed_issue is None:
        record.issue = None
        return True
    issue, evidences = parsed_issue
    issue.evidences = evidences
    record.issue = issue
    return False


async def revise_segment_draft(
    record: SegmentRecord,
    binding: SegmentBinding,
    writer,
    feedback: str,
    reason: str,
) -> bool:
    writer_input = Msg(
        name="user",
        content=(
            f"以下是当前正文的修改意见：\n{feedback}\n\n"
            "请只基于已有证据修改正文。修改后的正文必须使用<content>和</content>包裹，"
            "不要输出额外说明。"
        ),
        role="user",
    )
    draft_msg = await call_agent_with_retry(writer, writer_input)
    content = extract_writer_content(draft_msg.get_text_content())
    if not content:
        return False
    binding.segment.content = content
    binding.parent.content = None
    record.add_draft(content, reason)
    return True


async def evaluate_segment_quality(record: SegmentRecord, binding: SegmentBinding) -> str | None:
    evaluate_user_prompt = prompt_dict["evaluate_segment_user_prompt"].format(
        segment_topic=record.topic or "未指定",
        segment_reference=binding.segment.reference,
        segment_requirements="\n".join(record.requirements),
        segment_content=binding.segment.content,
    )
    return await call_chatbot_with_retry(
        llm_judge,
        create_agent_formatter(),
        prompt_dict["compare_content_with_ref"],
        evaluate_user_prompt,
        hook=_extract_score_suggestion,
        handle_hook_exceptions=(AssertionError, KeyError),
    )


async def verify_segment_fact(
    record: SegmentRecord,
    binding: SegmentBinding,
    verifier: SegmentVerifier,
    company_name: str,
    cur_date: str,
    round_idx: int,
) -> str | None:
    current_text = binding.segment.content or ""
    verify_issues = await verifier.verify(
        current_text,
        company_name=company_name,
        report_date=cur_date,
    )
    major_issues = [
        issue
        for issue in verify_issues
        if issue.severity in ("critical", "major")
    ]
    if not major_issues:
        await append_verifier_trace(
            topic=record.topic,
            round_idx=round_idx,
            checked_text=current_text,
            issue_count=0,
            status="passed",
        )
        return None

    lines = []
    for index, issue in enumerate(major_issues[:8], 1):
        lines.append(
            f"{index}. [{issue.severity.upper()}] {issue.description}\n"
            f"   建议: {issue.suggestion}\n"
            f"   证据: {issue.evidence}"
        )
    feedback = "\n\n".join(lines)[:1500]
    await append_verifier_trace(
        topic=record.topic,
        round_idx=round_idx,
        checked_text=current_text,
        verify_feedback=feedback,
        issue_count=len(major_issues),
    )
    return feedback


def add_issue_evidences_to_registry(
    record: SegmentRecord,
    registry: EvidenceRegistry,
    company_name: str,
) -> set[str]:
    added_ids: set[str] = set()
    if record.issue is None:
        return added_ids
    for evidence_payload in record.issue.evidences:
        evidence_record = registry.add_or_reuse(evidence_payload, record.segment_id, default_entity=company_name)
        if evidence_record.evidence_id not in record.evidences:
            record.evidences.append(evidence_record.evidence_id)
        added_ids.add(evidence_record.evidence_id)
    record.issue = None
    record.state = "RETRIEVING"
    return added_ids


async def handle_segment_issue(
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    writer,
    task_desc: str,
    cur_date: str,
    company_name: str,
    tracking_cfg: dict[str, Any],
) -> bool:
    if record.issue is None:
        return False
    action = record.issue.action.upper()
    if action == "RETRIEVE":
        added_ids = add_issue_evidences_to_registry(record, registry, company_name)
        if added_ids:
            await build_evidence_dependencies(registry, llm_instruct, formatter, target_ids=added_ids)
        return True
    if action == "REPLAN":
        attempts = record.increment_attempt("REPLAN")
        if attempts > int(tracking_cfg["max_replan_attempts"]):
            record.state = "BLOCKED"
            return True
        return await replan_segment_after_not_found(
            record,
            binding,
            registry,
            task_desc,
            cur_date,
            company_name,
        )
    if action == "REVISE":
        attempts = record.increment_attempt("REVISE")
        if attempts > int(tracking_cfg["max_revise_attempts"]):
            record.state = "BLOCKED"
            return True
        ok = await revise_segment_draft(record, binding, writer, record.issue.detail, "revise_after_issue")
        record.issue = None
        record.state = "DRAFTED" if ok else "BLOCKED"
        return True
    if action == "RETRY":
        attempts = record.increment_attempt("RETRY")
        record.issue = None
        record.state = "EVIDENCE_READY" if attempts <= 1 else "BLOCKED"
        return True
    record.state = "BLOCKED"
    return True


async def process_segment_record(
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    short_term: ShortTermMemoryStore,
    writer,
    verifier: SegmentVerifier,
    task_desc: str,
    company_name: str,
    cur_date: str,
    tracking_cfg: dict[str, Any],
    multi_source_verification_enabled: bool,
    max_verify_rounds: int,
) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] Tracking segment start: {record.segment_id} {record.topic}", flush=True)
    for _ in range(40):
        if record.state in {"FINALIZED", "BLOCKED"}:
            return
        if record.issue is not None:
            action = record.issue.action.upper()
            await handle_segment_issue(record, binding, registry, writer, task_desc, cur_date, company_name, tracking_cfg)
            if action == "RETRIEVE" and record.state == "RETRIEVING":
                return
            continue
        if record.state in {"EMPTY", "PLANNED"}:
            record.state = "RETRIEVING"
            if not await check_segment_evidences_ready(record, registry):
                return
            continue
        if record.state == "RETRIEVING":
            if not await check_segment_evidences_ready(record, registry):
                return
            continue
        if record.state == "EVIDENCE_READY":
            if not await check_writer_evidence_sufficiency(
                record,
                binding,
                registry,
                short_term,
                task_desc,
                cur_date,
            ):
                continue
            record.state = "WRITING"
            continue
        if record.state == "WRITING":
            reason = "initial_draft" if not record.draft_versions else "continue_after_gap"
            await write_segment_draft(record, binding, registry, short_term, writer, task_desc, reason)
            continue
        if record.state == "DRAFTED":
            suggestions = await evaluate_segment_quality(record, binding)
            if suggestions:
                record.issue = SegmentIssue(type="QUALITY_GAP", detail=suggestions, action="REVISE")
                continue
            record.state = "VERIFYING"
            continue
        if record.state == "VERIFYING":
            if not multi_source_verification_enabled:
                record.state = "VERIFIED"
                continue
            verified = await run_verifier_loop(record, binding, verifier, writer, company_name, cur_date, max_verify_rounds)
            record.state = "VERIFIED" if verified else "BLOCKED"
            continue
        if record.state == "VERIFIED":
            binding.segment.finished = True
            record.issue = None
            record.state = "FINALIZED"
            print(f"[{time.strftime('%H:%M:%S')}] Tracking segment finalized: {record.segment_id}", flush=True)
            return
    record.state = "BLOCKED"


async def run_verifier_loop(
    record: SegmentRecord,
    binding: SegmentBinding,
    verifier: SegmentVerifier,
    writer,
    company_name: str,
    cur_date: str,
    max_verify_rounds: int,
) -> bool:
    for round_idx in range(1, max_verify_rounds + 1):
        feedback = await verify_segment_fact(record, binding, verifier, company_name, cur_date, round_idx)
        if feedback is None:
            return True
        attempts = record.increment_attempt("VERIFY_REVISE")
        if attempts > max_verify_rounds:
            return False
        ok = await revise_segment_draft(record, binding, writer, feedback, "revise_after_verifier")
        if not ok:
            return False
    return False


async def process_tracking_board(
    board: TrackingBoard,
    bindings: dict[str, SegmentBinding],
    registry: EvidenceRegistry,
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    task_desc: str,
    company_name: str,
    demo_date: str,
    cur_date: str,
    tracking_cfg: dict[str, Any],
    multi_source_verification_enabled: bool,
    max_verify_rounds: int,
) -> None:
    semaphore = asyncio.Semaphore(int(tracking_cfg["segment_concurrency"]))
    for _ in range(50):
        # Reflect pending global evidence work on dependent segment states.
        for evidence_record in registry.records.values():
            if not evidence_record.required or evidence_record.state in {"RESOLVED", "SKIPPED"}:
                continue
            for segment_id in evidence_record.used_by_segments:
                segment_record = board.records.get(segment_id)
                if segment_record is not None and segment_record.state in {"EMPTY", "PLANNED"}:
                    segment_record.state = "RETRIEVING"
        await resolve_global_evidence_registry(
            registry,
            bindings,
            short_term,
            long_term,
            task_desc,
            demo_date,
            int(tracking_cfg["evidence_concurrency"]),
        )
        tasks = []
        for segment_id, record in board.records.items():
            if record.state in {"FINALIZED", "BLOCKED"}:
                continue
            task = process_tracking_segment_with_semaphore(
                semaphore,
                record,
                bindings[segment_id],
                registry,
                short_term,
                long_term,
                task_desc,
                company_name,
                cur_date,
                tracking_cfg,
                multi_source_verification_enabled,
                max_verify_rounds,
            )
            tasks.append(task)
        if not tasks:
            return
        before_states = {segment_id: record.state for segment_id, record in board.records.items()}
        await asyncio.gather(*tasks)
        after_states = {segment_id: record.state for segment_id, record in board.records.items()}
        if before_states == after_states and not registry.ready_to_search_records():
            return


async def process_tracking_segment_with_semaphore(
    semaphore: asyncio.Semaphore,
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    task_desc: str,
    company_name: str,
    cur_date: str,
    tracking_cfg: dict[str, Any],
    multi_source_verification_enabled: bool,
    max_verify_rounds: int,
) -> None:
    async with semaphore:
        writer, verifier = create_segment_agents(short_term, long_term)
        try:
            await process_segment_record(
                record,
                binding,
                registry,
                short_term,
                writer,
                verifier,
                task_desc,
                company_name,
                cur_date,
                tracking_cfg,
                multi_source_verification_enabled,
                max_verify_rounds,
            )
        except Exception as exc:
            traceback.print_exc()
            record.issue = SegmentIssue(
                type="EXECUTION_ERROR",
                detail=f"{type(exc).__name__}: {exc}",
                action="RETRY",
            )
            record.state = "BLOCKED"


async def polish_completed_sections(section: Section) -> None:
    for subsection in section.subsections or []:
        await polish_completed_sections(subsection)
    successful_segments = [segment for segment in section.segments or [] if segment.finished and segment.content]
    if not successful_segments:
        return
    section_text = "\n".join(segment.content for segment in successful_segments if segment.content)
    title, content = await call_chatbot_with_retry(
        llm_instruct,
        create_agent_formatter(),
        prompt_dict["section_polish_sys_prompt"],
        prompt_dict["section_polish_user_prompt"].format(
            section_text=section_text,
            section_title=section.title,
        ),
        parse_section_polish_response,
        handle_hook_exceptions=(AssertionError,),
    )
    print_section_reference_warning(section.title, section_text, content)
    section.title = title
    section.content = content


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


async def load_or_build_outline(
    output_pth: Path,
    filename: str,
    demo_pdf_path: Path,
    long_term_dir: Path,
    task_desc: str,
    cur_date: str,
    cfg: config.Config,
) -> Section:
    before_refine_outline_path = output_pth / f"{filename}_before_refine_outline.json"
    after_refine_outline_path = output_pth / f"{filename}_after_refine_outline.json"
    if after_refine_outline_path.exists():
        outline = load_section_from_json_text(after_refine_outline_path.read_text(encoding="utf-8"))
        print(f"[Outline Debug] loaded after_refine_outline: {after_refine_outline_path}", flush=True)
        return outline
    outline = await process_pdf_to_outline(
        demo_pdf_path,
        long_term_dir / "demonstration",
        llm_reasoning,
        llm_instruct,
        formatter,
    )
    _normalize_section_titles(outline)
    before_refine_outline_path.write_text(outline.to_json(ensure_ascii=False), encoding="utf-8")
    outline = await refine_outline(
        outline=outline,
        task_desc=task_desc,
        cur_date=cur_date,
        model=llm_outline_refine,
        formatter=outline_refine_formatter,
        model_cfg=cfg.get_outline_refine_model_cfg(),
    )
    after_refine_outline_path.write_text(outline.to_json(ensure_ascii=False), encoding="utf-8")
    return outline


async def run_workflow(task_desc: str, cur_date=None, demo_pdf_path=None):
    cur_date = cur_date or os.getenv("CUR_DATE", datetime.now().strftime("%Y%m%d"))
    project_root = Path(__file__).resolve().parent.parent.parent
    long_term_dir = project_root / "data" / "memory" / "long_term"
    long_term = LongTermMemoryStore(base_dir=long_term_dir)
    entity = get_entity_info(long_term, task_desc)
    if not entity or not entity.get("code"):
        raise ValueError(f"无法从 task_desc 解析股票实体/代码：{task_desc}")

    stock_symbol = entity["code"]
    company_name = str(entity.get("name") or "").strip()
    cfg = config.Config()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = open(f"log_tracking_{stock_symbol}_{now_str}.txt", "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    set_verifier_trace_path(project_root / f"verifier_trace_tracking_{stock_symbol}_{now_str}.txt")
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        filename = f"{stock_symbol}_{cur_date}"
        short_term_dir = project_root / "data" / "memory" / "short_term" / filename
        short_term = ShortTermMemoryStore(base_dir=short_term_dir, current_date=cur_date)
        if demo_pdf_path is None:
            demo_pdf_path = STOCK_REPORT_PATHS[stock_symbol][-1]
        demo_date = demo_pdf_path.name.split("_")[1]

        output_pth = project_root / "output" / "reports" / cfg.llm_name
        output_pth.mkdir(parents=True, exist_ok=True)
        manuscript_path = output_pth / f"{stock_symbol}_{cur_date}.json"
        manuscript = None
        if manuscript_path.exists():
            manuscript = load_section_from_json_text(manuscript_path.read_text(encoding="utf-8"))
            _normalize_section_titles(manuscript)

        outline = None
        if manuscript is None or section_has_unfinished(manuscript):
            outline = await load_or_build_outline(
                output_pth,
                filename,
                demo_pdf_path,
                long_term_dir,
                task_desc,
                cur_date,
                cfg,
            )
            _normalize_section_titles(outline)

        if manuscript is None:
            manuscript = outline
            short_term.save_material(
                cite_id=f"{demo_date}_reference_report",
                content=(long_term_dir / "demonstration" / f"{demo_pdf_path.stem}.md").read_text(encoding="utf-8"),
                description=demo_pdf_path.name,
                source="",
                entity=entity,
                time={"point": str(demo_date)},
            )
        if outline is not None and section_has_unfinished(manuscript):
            replace_unfinished_segments_with_outline(manuscript, outline)

        if section_has_unfinished(manuscript):
            board, bindings = build_tracking_board(manuscript)
            board_path = short_term.base_dir / f"{cfg.llm_name}_tracking_board.json"
            loaded_board = TrackingBoard.load(board_path) if board_path.exists() else None
            board = merge_loaded_board(board, loaded_board)
            sync_board_records_to_segments(board, bindings)
            registry = EvidenceRegistry.load(short_term.base_dir / f"{cfg.llm_name}_evidence_registry.json")
            initialize_registry_for_board(registry, board, bindings, company_name)
            await build_evidence_dependencies(registry, llm_instruct, formatter)
            registry.save()

            await process_tracking_board(
                board,
                bindings,
                registry,
                short_term,
                long_term,
                task_desc,
                company_name,
                demo_date,
                cur_date,
                cfg.get_tracking_board_cfg(),
                cfg.is_multi_source_verification_enabled(),
                cfg.get_max_verify_rounds(),
            )
            registry.save()
            board.save(board_path)
            async with SAVE_LOCK:
                manuscript_path.write_text(manuscript.to_json(ensure_ascii=False), encoding="utf-8")

            await polish_completed_sections(manuscript)

        _normalize_section_titles(manuscript)
        _normalize_report_title(manuscript, entity, task_desc)
        manuscript_path.write_text(manuscript.to_json(ensure_ascii=False), encoding="utf-8")
        markdown_text = section_to_markdown(manuscript)
        (output_pth / f"{filename}.md").write_text(markdown_text, encoding="utf-8")
        md_to_pdf(
            markdown_text,
            short_term=short_term,
            pdf_path=output_pth / f"{filename}.pdf",
            header_title=_infer_report_title(task_desc, entity),
        )
    finally:
        try:
            log_file.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()
            set_verifier_trace_path(None)
