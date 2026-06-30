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
from src.memory.evidence_registry import (
    EvidenceRecord,
    EvidenceRegistry,
    apply_static_reference_citation,
    repair_registry_citation_states,
    sync_segment_evidences_from_record,
)
from src.memory.long_term import LongTermMemoryStore
from src.memory.short_term import ShortTermMemoryStore
from src.memory.tracking_board import (
    SegmentBinding,
    SegmentIssue,
    SegmentRecord,
    TrackingBoard,
    board_has_actionable_issues,
    build_tracking_board,
    merge_loaded_board,
    set_segment_issue,
    set_segment_state,
    sync_board_records_to_segments,
    tracking_progress_snapshot,
)
from src.memory.working import (
    Section,
    _normalize_evidences,
    count_section_segments,
    load_section_from_json_text,
    replace_unfinished_segments_with_outline,
    section_has_unfinished,
)
from src.pipelines.planning import process_pdf_to_outline
from src.prompt import prompt_dict
from src.utils.call_with_retry import call_agent_with_retry, call_chatbot_with_retry
from src.utils.evidence_batching import (
    EvidenceBatchResult,
    build_batch_reference_context,
    build_batch_segment_context,
    cluster_ready_evidence_records,
    format_evidence_batch_xml,
    parse_batch_search_xml,
)
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
    format_verifier_feedback_for_writer,
    set_verifier_trace_path,
)
from src.utils.outline_refine import refine_outline
from src.utils.token_tracking import track_report_summary
from src.utils.tracking_board_format import (
    apply_search_result_to_evidence_record,
    build_evidence_context,
    build_known_evidence_context,
    is_unavailable_marker,
    parse_replan_response,
    parse_section_polish_response,
    parse_writer_issue,
)


SAVE_LOCK = asyncio.Lock()
BATCH_PARSE_REPAIR_ATTEMPTS = 2


async def save_tracking_progress(
    board: TrackingBoard,
    registry: EvidenceRegistry,
    board_path: Path,
) -> None:
    async with SAVE_LOCK:
        registry.save()
        board.save(board_path)


def writer_output_is_tool_call_only(text: str) -> bool:
    return (text or "").strip().startswith("<｜｜DSML｜｜tool_calls>")


def count_board_issue_segments(board: TrackingBoard | None) -> tuple[int, int]:
    if board is None:
        return 0, 0
    issue_segments = [
        record
        for record in board.records.values()
        if record.issue_seen
    ]
    recovered = [
        record
        for record in issue_segments
        if record.state == "FINALIZED"
    ]
    return len(issue_segments), len(recovered)


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def build_report_summary_metadata(
    manuscript: Section,
    board: TrackingBoard | None,
    markdown_text: str,
    task_desc: str,
    stock_symbol: str,
    company_name: str,
    cur_date: str,
) -> dict[str, Any]:
    total_segments, finalized_segments = count_section_segments(manuscript)
    issue_segments, recovered_issue_segments = count_board_issue_segments(board)
    return {
        "task_desc": task_desc,
        "stock_symbol": stock_symbol,
        "company_name": company_name,
        "cur_date": cur_date,
        "segment_total": total_segments,
        "segment_finalized": finalized_segments,
        "segment_success_rate": safe_rate(finalized_segments, total_segments),
        "issue_segment_total": issue_segments,
        "issue_segment_finalized": recovered_issue_segments,
        "issue_recovery_rate": safe_rate(recovered_issue_segments, issue_segments),
        "report_chars": len(markdown_text),
        "report_non_ws_chars": len("".join(markdown_text.split())),
        "report_lines": len(markdown_text.splitlines()),
    }


def evidence_searcher_max_iters(evidence_count: int) -> int:
    count = max(int(evidence_count), 1)
    return min(20, 4 + 2 * (count - 1))


def create_evidence_searcher(
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    evidence_count: int = 1,
):
    searcher_toolkit = build_searcher_toolkit(short_term=short_term, long_term=long_term)
    return create_searcher_agent(
        model=llm_reasoning,
        formatter=formatter,
        toolkit=searcher_toolkit,
        max_iters=evidence_searcher_max_iters(evidence_count),
    )


def create_segment_agents(
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
):
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
    reference_cite_id: str,
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
    removed_count = registry.prune_unlinked_records()
    if removed_count:
        print(f"[Evidence Registry] pruned stale evidence records: {removed_count}", flush=True)
    repaired_count = repair_registry_citation_states(registry)
    if repaired_count:
        print(f"[Evidence Registry] repaired citation states: {repaired_count}", flush=True)
    for evidence_record in registry.active_records():
        apply_static_reference_citation(evidence_record, reference_cite_id)


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
        unavailable_ids = [
            evidence_id
            for evidence_id in record.evidences
            if registry.records.get(evidence_id) and registry.records[evidence_id].state == "UNAVAILABLE"
        ]
        for evidence_id in unavailable_ids:
            evidence_record = registry.records.get(evidence_id)
            if evidence_record and record.segment_id in evidence_record.used_by_segments:
                evidence_record.used_by_segments.remove(record.segment_id)
        record.evidences = [
            evidence_id
            for evidence_id in record.evidences
            if evidence_id not in unavailable_ids
        ]
        for evidence in evidences or []:
            evidence_record = registry.add_or_reuse(evidence, record.segment_id, default_entity=company_name)
            if evidence_record.evidence_id not in record.evidences:
                record.evidences.append(evidence_record.evidence_id)
        registry.prune_unlinked_records()
        sync_segment_evidences_from_record(record, binding, registry)

    record.issue = None


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


async def search_tracking_evidence_batch(
    records: list[EvidenceRecord],
    registry: EvidenceRegistry,
    bindings: dict[str, SegmentBinding],
    task_desc: str,
    demo_date: str,
    searcher,
) -> str:
    topics, references = build_batch_segment_context(records, bindings)
    reference_context = build_batch_reference_context(references, demo_date)
    known_evidence_by_id = {
        record.evidence_id: build_known_evidence_context(registry, record.depends_on)
        for record in records
    }
    prompt = prompt_dict["search_tracking_evidence_batch_user_prompt"].format(
        task_desc=task_desc,
        segment_topic="；".join(topics) if topics else "未指定",
        reference_context=reference_context,
        evidences_xml=format_evidence_batch_xml(records, known_evidence_by_id),
    )
    searcher_input = Msg(name="user", content=prompt, role="user")
    print(searcher_input.content)
    msg = await call_agent_with_retry(searcher, searcher_input)
    batch_label = ", ".join(record.evidence_id for record in records)
    print(f"[Searcher] Finished batch searching: {batch_label}", flush=True)
    return msg.get_text_content()


def build_batch_parse_repair_prompt(
    records: list[EvidenceRecord],
) -> str:
    evidence_ids = ", ".join(record.evidence_id for record in records)
    return (
        "上一轮最终输出无法被程序解析为 XML。\n"
        "请只基于你刚才已经找到的材料和结论，把结果重新整理成合法 XML。\n\n"
        f"必须输出这些 evidence_id，且每个有且只有一个 evidence_result：{evidence_ids}\n"
        "最终只能输出 XML，不能输出 Markdown、解释文字或工具调用。格式如下：\n"
        "<results>\n"
        "  <evidence_result>\n"
        "    <evidence_id>ev_xxx</evidence_id>\n"
        "    <status>RESOLVED|SKIPPED|UNAVAILABLE</status>\n"
        "    <answer>简洁检索结果摘录；RESOLVED 时必须包含 [^cite_id:xxxxxx]</answer>\n"
        "  </evidence_result>\n"
        "</results>"
    )


async def repair_batch_search_xml(
    records: list[EvidenceRecord],
    searcher,
) -> str:
    repair_prompt = build_batch_parse_repair_prompt(records)
    repair_input = Msg(name="user", content=repair_prompt, role="user")
    batch_label = ", ".join(record.evidence_id for record in records)
    print(f"[Searcher] Repairing batch XML output: {batch_label}", flush=True)
    msg = await call_agent_with_retry(searcher, repair_input, max_retries=2)
    return msg.get_text_content()


async def parse_batch_search_xml_with_repair(
    records: list[EvidenceRecord],
    searcher,
    raw_text: str,
) -> dict[str, EvidenceBatchResult]:
    current_text = raw_text
    last_error: Exception | None = None
    for attempt in range(BATCH_PARSE_REPAIR_ATTEMPTS + 1):
        try:
            return parse_batch_search_xml(current_text)
        except Exception as exc:
            last_error = exc
            if attempt >= BATCH_PARSE_REPAIR_ATTEMPTS:
                break
            current_text = await repair_batch_search_xml(records, searcher)
    if last_error is not None:
        raise last_error
    return {}


async def resolve_global_evidence_cluster(
    records: list[EvidenceRecord],
    registry: EvidenceRegistry,
    bindings: dict[str, SegmentBinding],
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    task_desc: str,
    demo_date: str,
) -> bool:
    active_records = [
        record
        for record in records
        if record.state in {"NEW", "WAITING", "PLANNED"}
        and registry.dependencies_resolved(record.evidence_id)
    ]
    if not active_records:
        return False

    for record in active_records:
        record.state = "SEARCHING"

    searcher = create_evidence_searcher(short_term, long_term, len(active_records))
    try:
        raw_text = await search_tracking_evidence_batch(
            active_records,
            registry,
            bindings,
            task_desc,
            demo_date,
            searcher,
        )
        try:
            parsed_results = await parse_batch_search_xml_with_repair(
                active_records,
                searcher,
                raw_text,
            )
        except Exception:
            stripped_text = str(raw_text or "").strip()
            if len(active_records) == 1:
                parsed_results = {
                    active_records[0].evidence_id: EvidenceBatchResult(
                        evidence_id=active_records[0].evidence_id,
                        status="",
                        answer=stripped_text,
                    )
                }
            elif stripped_text.upper() == "SKIP":
                parsed_results = {
                    record.evidence_id: EvidenceBatchResult(record.evidence_id, "SKIPPED", "")
                    for record in active_records
                }
            elif is_unavailable_marker(stripped_text):
                parsed_results = {
                    record.evidence_id: EvidenceBatchResult(record.evidence_id, "UNAVAILABLE", "")
                    for record in active_records
                }
            else:
                raise
    except Exception:
        for record in active_records:
            record.state = "PLANNED"
        raise

    changed = False
    for record in active_records:
        result = parsed_results.get(record.evidence_id)
        if result is None:
            record.state = "PLANNED"
            print(
                f"[Searcher] Batch result missing evidence_id={record.evidence_id}; will retry later.",
                flush=True,
            )
            continue
        apply_search_result_to_evidence_record(
            record,
            result.answer,
            result.status,
        )
        changed = True
    return changed


def apply_ready_static_records(
    records: list[EvidenceRecord],
    demo_date: str,
) -> int:
    applied = 0
    for record in records:
        if is_static_fact_record(record):
            apply_static_reference_citation(record, f"{demo_date}_reference_report")
            applied += 1
    return applied


async def resolve_global_evidence_registry(
    registry: EvidenceRegistry,
    bindings: dict[str, SegmentBinding],
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    task_desc: str,
    demo_date: str,
    evidence_concurrency: int,
    evidence_batch_size: int,
) -> bool:
    changed = False
    concurrency = max(int(evidence_concurrency), 1)
    batch_size = max(int(evidence_batch_size), 1)
    for _ in range(max(len(registry.records), 1)):
        ready_records = registry.ready_to_search_records()
        if not ready_records:
            break
        static_count = apply_ready_static_records(ready_records, demo_date)
        if static_count:
            changed = True
            continue

        dynamic_records = [
            record
            for record in ready_records
            if not is_static_fact_record(record)
        ]
        clusters = cluster_ready_evidence_records(
            dynamic_records,
            registry,
            batch_size,
            segment_ids_in_order=list(bindings),
        )
        if not clusters:
            break
        batch = clusters[:concurrency]
        results = await asyncio.gather(
            *[
                resolve_global_evidence_cluster(
                    cluster,
                    registry,
                    bindings,
                    short_term,
                    long_term,
                    task_desc,
                    demo_date,
                )
                for cluster in batch
            ],
            return_exceptions=True,
        )
        for cluster, result in zip(batch, results):
            if isinstance(result, Exception):
                traceback.print_exception(type(result), result, result.__traceback__)
                for record in cluster:
                    if record.state == "SEARCHING":
                        record.state = "PLANNED"
                registry.save()
                continue
            changed = bool(result) or changed
            registry.save()
    return changed


async def check_segment_evidences_ready(
    record: SegmentRecord,
    registry: EvidenceRegistry,
    *,
    preserve_state: bool = False,
) -> bool:
    unresolved = registry.unresolved_required_for_segment(record.segment_id)
    unavailable = [item for item in unresolved if item.state == "UNAVAILABLE"]
    if unavailable:
        detail = "；".join(item.description for item in unavailable)
        set_segment_issue(
            record,
            SegmentIssue(
                type="EVIDENCE_NOT_FOUND",
                detail=f"关键证据未找到或不可用：{detail}",
                action="REPLAN",
            ),
        )
        return False
    if unresolved:
        record.issue = None
        if not preserve_state:
            set_segment_state(record, "RETRIEVING", "waiting_for_required_evidence")
        return False
    record.issue = None
    if not preserve_state:
        set_segment_state(record, "EVIDENCE_READY", "all_required_evidence_ready")
    return True


async def write_segment_draft(
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    short_term: ShortTermMemoryStore,
    writer,
    task_desc: str,
    reason: str,
    max_retry_attempts: int,
) -> bool:
    evidence_context = build_evidence_context(registry, record.segment_id, short_term)
    context_labels = []
    if binding.segment.template:
        context_labels.append("写作示例")
    if binding.segment.requirements:
        context_labels.append("写作要求")
    if evidence_context:
        context_labels.append("可用论据材料")
    outline_context_intro = (
        "、".join(context_labels) + "如下："
        if context_labels
        else "当前片段上下文如下："
    )
    segment_parts = [binding.segment.__str__(with_evidence=False).strip()]
    if evidence_context:
        segment_parts.append(f"可用论据材料：\n{evidence_context}")
    segment_text = "\n\n".join(part for part in segment_parts if part)
    continue_attempts = int(record.attempts.get("WRITER_TOOL_CALL_CONTINUE", 0))
    if continue_attempts <= 0:
        writer_content = prompt_dict["segment_writer_user_prompt"].format(
            task_desc=task_desc,
            segment_topic=record.topic or "未指定",
            outline_context_intro=outline_context_intro,
            segment_text=segment_text,
        )
    else:
        writer_content = (
            "请继续完成最终研报片段正文。\n\n"
            f"任务：{task_desc}\n"
            f"当前要点：{record.topic or '未指定'}\n\n"
            "最终答案必须使用<content>和</content>包裹；"
            "正文中如需引用图表，请使用 ![图表说明](chart:chart_id)。"
        )

    writer_input = Msg(name="user", content=writer_content, role="user")
    if continue_attempts <= 0 and hasattr(writer, "memory") and writer.memory is not None:
        await writer.memory.clear()
    draft_msg = await call_agent_with_retry(writer, writer_input)
    raw_text = draft_msg.get_text_content()
    content = extract_writer_content(raw_text)
    if writer_output_is_tool_call_only(content):
        attempts = record.increment_attempt("WRITER_TOOL_CALL_CONTINUE")
        print(
            f"[{time.strftime('%H:%M:%S')}] Writer draft produced tool call only: "
            f"{record.segment_id} continue_attempt={attempts}",
            flush=True,
        )
        if attempts <= max_retry_attempts:
            return False
        record.attempts["WRITER_TOOL_CALL_CONTINUE"] = 0
        set_segment_issue(
            record,
            SegmentIssue(
                type="EXECUTION_ERROR",
                detail="Writer 连续输出工具调用，未生成正文。",
                action="RETRY",
            ),
        )
        return False
    if content:
        record.attempts.pop("WRITER_TOOL_CALL_CONTINUE", None)
        draft_reason = reason
        if reason == "initial_draft" and continue_attempts > 0:
            draft_reason = "initial_draft_after_tool_call"
        binding.segment.content = content
        binding.parent.content = None
        record.add_draft(content, draft_reason)
        record.issue = None
        set_segment_state(record, "DRAFTED", draft_reason)
        return True

    set_segment_issue(
        record,
        SegmentIssue(
            type="EXECUTION_ERROR",
            detail="Writer 输出为空，未生成正文。",
            action="RETRY",
        ),
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
    segment_text = binding.segment.__str__(with_evidence=False).strip()
    evidence_context_section = (
        f"证据需求与可用材料摘录：\n{evidence_context}"
        if evidence_context
        else "证据需求与可用材料摘录：\n当前没有已检索到的 evidence。"
    )
    user_prompt = prompt_dict["segment_evidence_check_user_prompt"].format(
        task_desc=task_desc,
        cur_date=cur_date,
        segment_topic=record.topic or binding.segment.topic or "未指定",
        segment_text=segment_text,
        evidence_context_section=evidence_context_section,
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
    set_segment_issue(record, issue)
    return False


async def revise_segment_draft(
    record: SegmentRecord,
    binding: SegmentBinding,
    writer,
    task_desc: str,
    feedback: str,
    reason: str,
) -> bool | None:
    current_text = (binding.segment.content or record.latest_draft() or "").strip()
    segment_topic = record.topic or binding.segment.topic or "未指定"
    segment_requirements = "\n".join(record.requirements).strip() or "未指定"
    writer_input = Msg(
        name="user",
        content=(
            "你之前的写作内容没有通过审查，需要基于修改意见修改。\n\n"
            f"任务：{task_desc}\n"
            f"当前要点：{segment_topic}\n\n"
            f"写作要求：\n{segment_requirements}\n\n"
            f"当前正文：\n{current_text}\n\n"
            f"修改意见：\n{feedback}\n\n"
            "修改后的正文必须使用<content>和</content>包裹，"
            "不要输出额外说明。"
        ),
        role="user",
    )
    initial_draft_from_tool_call = (
        len(record.draft_versions) == 1
        and record.draft_versions[0].reason == "initial_draft_after_tool_call"
    )
    if (
        not initial_draft_from_tool_call
        and hasattr(writer, "memory")
        and writer.memory is not None
    ):
        await writer.memory.clear()
    draft_msg = await call_agent_with_retry(writer, writer_input)
    content = extract_writer_content(draft_msg.get_text_content())
    if not content:
        return False
    stripped_content = content.strip()
    if writer_output_is_tool_call_only(stripped_content):
        print(
            f"[{time.strftime('%H:%M:%S')}] Writer revise skipped: "
            f"{record.segment_id} produced tool call only; will retry without counting revise attempt",
            flush=True,
        )
        return None
    if len(stripped_content) < len(current_text) * 0.2:
        print(
            f"[{time.strftime('%H:%M:%S')}] Writer revise skipped: "
            f"{record.segment_id} content too short "
            f"({len(stripped_content)}/{len(current_text)})",
            flush=True,
        )
        return None
    binding.segment.content = content
    binding.parent.content = None
    record.add_draft(content, reason)
    return True


async def evaluate_segment_quality(record: SegmentRecord, binding: SegmentBinding) -> str | None:
    segment_reference = (binding.segment.reference or "").strip()
    prompt_key = "evaluate_segment_user_prompt"
    system_prompt_key = "evaluate_segment_quality_with_ref"
    prompt_values = {
        "segment_topic": record.topic or "未指定",
        "segment_reference": segment_reference,
        "segment_requirements": "\n".join(record.requirements),
        "segment_content": binding.segment.content,
    }
    if not segment_reference:
        prompt_key = "evaluate_segment_without_ref_user_prompt"
        system_prompt_key = "evaluate_segment_quality_without_ref"
    evaluate_user_prompt = prompt_dict[prompt_key].format(**prompt_values)
    return await call_chatbot_with_retry(
        llm_judge,
        create_agent_formatter(),
        prompt_dict[system_prompt_key],
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

    feedback = format_verifier_feedback_for_writer(major_issues)
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
    return added_ids


async def handle_segment_issue(
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    short_term: ShortTermMemoryStore,
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
            ignored_ids = [
                evidence_id
                for evidence_id in record.evidences
                if registry.records.get(evidence_id)
                and registry.records[evidence_id].state == "UNAVAILABLE"
            ]
            for evidence_id in ignored_ids:
                evidence_record = registry.records[evidence_id]
                if record.segment_id in evidence_record.used_by_segments:
                    evidence_record.used_by_segments.remove(record.segment_id)
            record.evidences = [
                evidence_id
                for evidence_id in record.evidences
                if evidence_id not in ignored_ids
            ]
            registry.prune_unlinked_records()
            sync_segment_evidences_from_record(record, binding, registry)
            record.issue = None
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
        is_verifier_revise = record.issue.type == "VERIFICATION_GAP"
        attempt_key = "VERIFY_REVISE" if is_verifier_revise else "REVISE"
        max_attempts = (
            int(tracking_cfg["max_verification_revise_attempts"])
            if is_verifier_revise
            else int(tracking_cfg["max_quality_revise_attempts"])
        )
        if int(record.attempts.get(attempt_key, 0)) >= max_attempts:
            set_segment_state(
                record,
                "BLOCKED",
                "max_verification_revise_attempts" if is_verifier_revise else "max_quality_revise_attempts",
            )
            return True
        reason = "revise_after_verifier_issue" if is_verifier_revise else "revise_after_issue"
        ok = await revise_segment_draft(
            record,
            binding,
            writer,
            task_desc,
            record.issue.detail,
            reason,
        )
        if ok is None:
            return True
        record.increment_attempt(attempt_key)
        if ok:
            record.issue = None
        return True
    if action == "RETRY":
        attempts = record.increment_attempt("RETRY")
        record.issue = None
        if attempts > int(tracking_cfg["max_retry_attempts"]):
            set_segment_state(record, "BLOCKED", "retry_after_issue")
        return True
    set_segment_state(record, "BLOCKED", f"unknown_issue_action_{action}")
    return True


async def process_segment_record(
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    board: TrackingBoard,
    board_path: Path,
    short_term: ShortTermMemoryStore,
    writer,
    verifier: SegmentVerifier,
    task_desc: str,
    company_name: str,
    cur_date: str,
    tracking_cfg: dict[str, Any],
    multi_source_verification_enabled: bool,
) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] Tracking segment start: {record.segment_id} {record.topic}", flush=True)
    for _ in range(40):
        if record.state in {"FINALIZED", "BLOCKED"}:
            return
        if record.issue is not None:
            action = record.issue.action.upper()
            await handle_segment_issue(
                record,
                binding,
                registry,
                short_term,
                writer,
                task_desc,
                cur_date,
                company_name,
                tracking_cfg,
            )
            await save_tracking_progress(board, registry, board_path)
            if action == "RETRIEVE":
                return
            continue
        if record.state in {"EMPTY", "PLANNED"}:
            set_segment_state(record, "RETRIEVING", "segment_needs_evidence")
            if not await check_segment_evidences_ready(record, registry):
                await save_tracking_progress(board, registry, board_path)
                return
            await save_tracking_progress(board, registry, board_path)
            continue
        if record.state == "RETRIEVING":
            if not await check_segment_evidences_ready(record, registry):
                await save_tracking_progress(board, registry, board_path)
                return
            await save_tracking_progress(board, registry, board_path)
            continue
        if record.state == "EVIDENCE_READY":
            if not await check_segment_evidences_ready(record, registry, preserve_state=True):
                await save_tracking_progress(board, registry, board_path)
                return
            if not await check_writer_evidence_sufficiency(
                record,
                binding,
                registry,
                short_term,
                task_desc,
                cur_date,
            ):
                await save_tracking_progress(board, registry, board_path)
                continue
            set_segment_state(record, "WRITING", "writer_evidence_sufficient")
            await save_tracking_progress(board, registry, board_path)
            continue
        if record.state == "WRITING":
            reason = "initial_draft" if not record.draft_versions else "continue_after_gap"
            await write_segment_draft(
                record,
                binding,
                registry,
                short_term,
                writer,
                task_desc,
                reason,
                int(tracking_cfg["max_retry_attempts"]),
            )
            await save_tracking_progress(board, registry, board_path)
            continue
        if record.state == "DRAFTED":
            suggestions = await evaluate_segment_quality(record, binding)
            if suggestions:
                set_segment_issue(
                    record,
                    SegmentIssue(type="QUALITY_GAP", detail=suggestions, action="REVISE"),
                )
                await save_tracking_progress(board, registry, board_path)
                continue
            set_segment_state(record, "VERIFYING", "quality_check_passed")
            await save_tracking_progress(board, registry, board_path)
            continue
        if record.state == "VERIFYING":
            if not multi_source_verification_enabled:
                set_segment_state(record, "VERIFIED", "verification_disabled")
                await save_tracking_progress(board, registry, board_path)
                continue
            verified = await run_verifier_check(
                record,
                binding,
                verifier,
                company_name,
                cur_date,
            )
            if verified:
                set_segment_state(record, "VERIFIED", "verifier_check_passed")
            else:
                await save_tracking_progress(board, registry, board_path)
                continue
            await save_tracking_progress(board, registry, board_path)
            continue
        if record.state == "VERIFIED":
            binding.segment.finished = True
            record.issue = None
            set_segment_state(record, "FINALIZED", "segment_finished")
            print(f"[{time.strftime('%H:%M:%S')}] Tracking segment finalized: {record.segment_id}", flush=True)
            await save_tracking_progress(board, registry, board_path)
            return
    set_segment_state(record, "BLOCKED", "state_loop_exhausted")
    await save_tracking_progress(board, registry, board_path)


async def run_verifier_check(
    record: SegmentRecord,
    binding: SegmentBinding,
    verifier: SegmentVerifier,
    company_name: str,
    cur_date: str,
) -> bool:
    verify_revise_attempts = record.attempts.get("VERIFY_REVISE", 0)
    round_idx = verify_revise_attempts + 1
    feedback = await verify_segment_fact(record, binding, verifier, company_name, cur_date, round_idx)
    if feedback is None:
        return True
    set_segment_issue(
        record,
        SegmentIssue(
            type="VERIFICATION_GAP",
            detail=feedback,
            action="REVISE",
        ),
    )
    return False


async def process_tracking_board(
    board: TrackingBoard,
    bindings: dict[str, SegmentBinding],
    registry: EvidenceRegistry,
    board_path: Path,
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    task_desc: str,
    company_name: str,
    demo_date: str,
    cur_date: str,
    tracking_cfg: dict[str, Any],
    multi_source_verification_enabled: bool,
) -> None:
    semaphore = asyncio.Semaphore(int(tracking_cfg["segment_concurrency"]))
    for _ in range(50):
        before_snapshot = tracking_progress_snapshot(board, registry)
        # Reflect pending global evidence work on dependent segment states.
        for evidence_record in registry.records.values():
            if not evidence_record.required or evidence_record.state in {"RESOLVED", "SKIPPED"}:
                continue
            for segment_id in evidence_record.used_by_segments:
                segment_record = board.records.get(segment_id)
                if segment_record is not None and segment_record.state in {"EMPTY", "PLANNED"}:
                    set_segment_state(segment_record, "RETRIEVING", "global_evidence_pending")
        await save_tracking_progress(board, registry, board_path)
        await resolve_global_evidence_registry(
            registry,
            bindings,
            short_term,
            long_term,
            task_desc,
            demo_date,
            int(tracking_cfg["evidence_concurrency"]),
            int(tracking_cfg["evidence_batch_size"]),
        )
        await save_tracking_progress(board, registry, board_path)
        tasks = []
        for segment_id, record in board.records.items():
            if record.state in {"FINALIZED", "BLOCKED"}:
                continue
            task = process_tracking_segment_with_semaphore(
                semaphore,
                record,
                bindings[segment_id],
                registry,
                board,
                board_path,
                short_term,
                long_term,
                task_desc,
                company_name,
                cur_date,
                tracking_cfg,
                multi_source_verification_enabled,
            )
            tasks.append(task)
        if not tasks:
            return
        await asyncio.gather(*tasks)
        registry.prune_unlinked_records()
        await save_tracking_progress(board, registry, board_path)
        after_snapshot = tracking_progress_snapshot(board, registry)
        if (
            before_snapshot == after_snapshot
            and not registry.ready_to_search_records()
            and not board_has_actionable_issues(board)
        ):
            return


async def process_tracking_segment_with_semaphore(
    semaphore: asyncio.Semaphore,
    record: SegmentRecord,
    binding: SegmentBinding,
    registry: EvidenceRegistry,
    board: TrackingBoard,
    board_path: Path,
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    task_desc: str,
    company_name: str,
    cur_date: str,
    tracking_cfg: dict[str, Any],
    multi_source_verification_enabled: bool,
) -> None:
    async with semaphore:
        writer, verifier = create_segment_agents(short_term, long_term)
        try:
            await process_segment_record(
                record,
                binding,
                registry,
                board,
                board_path,
                short_term,
                writer,
                verifier,
                task_desc,
                company_name,
                cur_date,
                tracking_cfg,
                multi_source_verification_enabled,
            )
        except Exception as exc:
            traceback.print_exc()
            set_segment_issue(
                record,
                SegmentIssue(
                    type="EXECUTION_ERROR",
                    detail=f"{type(exc).__name__}: {exc}",
                    action="RETRY",
                ),
            )
            await save_tracking_progress(board, registry, board_path)


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
    previous_llm_debug = os.environ.get("FRA_LLM_DEBUG")
    previous_llm_debug_file = os.environ.get("FRA_LLM_DEBUG_FILE")
    previous_usage_tracking_file = os.environ.get("FRA_USAGE_TRACKING_FILE")
    os.environ["FRA_LLM_DEBUG"] = "1"
    os.environ["FRA_LLM_DEBUG_FILE"] = str(project_root / f"llm_debug_tracking_{stock_symbol}_{now_str}.txt")
    os.environ["FRA_USAGE_TRACKING_FILE"] = str(project_root / f"usage_tracking_{stock_symbol}_{now_str}.jsonl")
    set_verifier_trace_path(project_root / f"verifier_trace_tracking_{stock_symbol}_{now_str}.txt")
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        board = None
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
            for evidence_record in registry.records.values():
                if evidence_record.state == "SEARCHING":
                    evidence_record.state = "PLANNED"
            initialize_registry_for_board(
                registry,
                board,
                bindings,
                company_name,
                f"{demo_date}_reference_report",
            )
            await build_evidence_dependencies(registry, llm_instruct, formatter)
            await save_tracking_progress(board, registry, board_path)

            await process_tracking_board(
                board,
                bindings,
                registry,
                board_path,
                short_term,
                long_term,
                task_desc,
                company_name,
                demo_date,
                cur_date,
                cfg.get_tracking_board_cfg(),
                cfg.is_multi_source_verification_enabled(),
            )
            await save_tracking_progress(board, registry, board_path)
            async with SAVE_LOCK:
                manuscript_path.write_text(manuscript.to_json(ensure_ascii=False), encoding="utf-8")

            await polish_completed_sections(manuscript)

        _normalize_section_titles(manuscript)
        _normalize_report_title(manuscript, entity, task_desc)
        manuscript_path.write_text(manuscript.to_json(ensure_ascii=False), encoding="utf-8")
        markdown_text = section_to_markdown(manuscript)
        (output_pth / f"{filename}.md").write_text(markdown_text, encoding="utf-8")
        track_report_summary(
            build_report_summary_metadata(
                manuscript,
                board,
                markdown_text,
                task_desc,
                stock_symbol,
                company_name,
                cur_date,
            )
        )
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
            if previous_llm_debug is None:
                os.environ.pop("FRA_LLM_DEBUG", None)
            else:
                os.environ["FRA_LLM_DEBUG"] = previous_llm_debug
            if previous_llm_debug_file is None:
                os.environ.pop("FRA_LLM_DEBUG_FILE", None)
            else:
                os.environ["FRA_LLM_DEBUG_FILE"] = previous_llm_debug_file
            if previous_usage_tracking_file is None:
                os.environ.pop("FRA_USAGE_TRACKING_FILE", None)
            else:
                os.environ["FRA_USAGE_TRACKING_FILE"] = previous_usage_tracking_file
            log_file.close()
            set_verifier_trace_path(None)

