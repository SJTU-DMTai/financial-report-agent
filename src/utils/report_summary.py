# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from src.memory.tracking_board import TrackingBoard
from src.memory.working import Section, count_section_segments


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
