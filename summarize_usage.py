# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def add_llm_event(row: dict[str, Any], event: dict[str, Any]) -> None:
    row["calls"] += 1
    row["input_est"] += int(event.get("input_tokens_est") or 0)
    row["output_est"] += int(event.get("output_tokens_est") or 0)
    row["input_api"] += int(event.get("input_tokens_api") or 0)
    row["output_api"] += int(event.get("output_tokens_api") or 0)
    row["max_input_est"] = max(row["max_input_est"], int(event.get("input_tokens_est") or 0))
    row["elapsed_ms"] += int(event.get("elapsed_ms") or 0)


def add_tool_event(row: dict[str, Any], event: dict[str, Any]) -> None:
    row["calls"] += 1
    row["tokens_est"] += int(event.get("text_tokens_est") or 0)
    row["chars"] += int(event.get("text_chars") or 0)
    row["max_tokens_est"] = max(row["max_tokens_est"], int(event.get("text_tokens_est") or 0))


def summarize_llm(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "calls": 0,
            "input_est": 0,
            "output_est": 0,
            "input_api": 0,
            "output_api": 0,
            "max_input_est": 0,
            "elapsed_ms": 0,
        },
    )
    for event in events:
        if event.get("event") != "llm_call":
            continue
        key = str(event.get("component") or "unscoped")
        add_llm_event(rows[key], event)
    return dict(rows)


def summarize_tools(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "calls": 0,
            "tokens_est": 0,
            "chars": 0,
            "max_tokens_est": 0,
        },
    )
    for event in events:
        if event.get("event") != "tool_result":
            continue
        component = str(event.get("component") or "unscoped")
        tool_name = str(event.get("tool_name") or "unknown_tool")
        key = f"{component} :: {tool_name}"
        add_tool_event(rows[key], event)
    return dict(rows)


def summarize_api_calls(events: list[dict[str, Any]]) -> dict[str, int]:
    rows: dict[str, int] = defaultdict(int)
    for event in events:
        if event.get("event") == "api_call":
            rows[str(event.get("api_type") or "unknown_api")] += 1
            continue
        if event.get("event") != "tool_result":
            continue
        if event.get("tool_name") == "financial_data_tool":
            rows["financial_api"] += 1
    return dict(rows)


def count_tool_calls(events: list[dict[str, Any]], tool_name: str) -> int:
    return sum(
        1
        for event in events
        if event.get("event") == "tool_result"
        and event.get("tool_name") == tool_name
    )


def summarize_reports(events: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [
        event.get("metadata") or {}
        for event in events
        if event.get("event") == "report_summary"
    ]
    report_count = len(reports)
    llm_calls = sum(1 for event in events if event.get("event") == "llm_call")
    input_api = sum(int(event.get("input_tokens_api") or 0) for event in events if event.get("event") == "llm_call")
    output_api = sum(int(event.get("output_tokens_api") or 0) for event in events if event.get("event") == "llm_call")
    input_est = sum(int(event.get("input_tokens_est") or 0) for event in events if event.get("event") == "llm_call")
    output_est = sum(int(event.get("output_tokens_est") or 0) for event in events if event.get("event") == "llm_call")
    api_calls = summarize_api_calls(events)
    search_engine_tool_calls = count_tool_calls(events, "search_engine")
    total_segments = sum(int(report.get("segment_total") or 0) for report in reports)
    finalized_segments = sum(int(report.get("segment_finalized") or 0) for report in reports)
    issue_segments = sum(int(report.get("issue_segment_total") or 0) for report in reports)
    recovered_issue_segments = sum(int(report.get("issue_segment_finalized") or 0) for report in reports)
    report_chars = sum(int(report.get("report_chars") or 0) for report in reports)
    report_non_ws_chars = sum(int(report.get("report_non_ws_chars") or 0) for report in reports)
    token_api = input_api + output_api
    token_est = input_est + output_est
    return {
        "reports": report_count,
        "segment_success_rate": safe_div(finalized_segments, total_segments),
        "issue_recovery_rate": safe_div(recovered_issue_segments, issue_segments),
        "tokens_api_per_report": safe_div(token_api, report_count),
        "tokens_est_per_report": safe_div(token_est, report_count),
        "llm_calls_per_report": safe_div(llm_calls, report_count),
        "search_engine_calls_per_report": safe_div(search_engine_tool_calls, report_count),
        "financial_api_calls_per_report": safe_div(api_calls.get("financial_api", 0), report_count),
        "report_chars_avg": safe_div(report_chars, report_count),
        "report_non_ws_chars_avg": safe_div(report_non_ws_chars, report_count),
        "segment_total": total_segments,
        "segment_finalized": finalized_segments,
        "issue_segment_total": issue_segments,
        "issue_segment_finalized": recovered_issue_segments,
    }


def safe_div(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_llm_table(rows: dict[str, dict[str, int]]) -> None:
    print("\nLLM calls by component")
    print("component,calls,input_est,output_est,total_est,input_api,output_api,total_api,max_input_est,elapsed_ms")
    for key, row in sorted(rows.items(), key=lambda item: item[1]["input_est"] + item[1]["output_est"], reverse=True):
        total_est = row["input_est"] + row["output_est"]
        total_api = row["input_api"] + row["output_api"]
        print(
            f"{key},{row['calls']},{row['input_est']},{row['output_est']},{total_est},"
            f"{row['input_api']},{row['output_api']},{total_api},{row['max_input_est']},{row['elapsed_ms']}",
        )


def print_tool_table(rows: dict[str, dict[str, int]]) -> None:
    print("\nTool results by component/tool")
    print("component_tool,calls,tokens_est,chars,max_tokens_est")
    for key, row in sorted(rows.items(), key=lambda item: item[1]["tokens_est"], reverse=True):
        print(f"{key},{row['calls']},{row['tokens_est']},{row['chars']},{row['max_tokens_est']}")


def print_api_table(rows: dict[str, int]) -> None:
    print("\nExternal API calls")
    print("api_type,calls")
    for key, calls in sorted(rows.items()):
        print(f"{key},{calls}")


def print_report_summary(row: dict[str, Any]) -> None:
    print("\nReport summary")
    print(
        "reports,segment_success_rate,tokens_api_per_report,tokens_est_per_report,"
        "llm_calls_per_report,search_engine_calls_per_report,financial_api_calls_per_report,"
        "issue_recovery_rate,report_chars_avg,report_non_ws_chars_avg,"
        "segment_finalized,segment_total,issue_segment_finalized,issue_segment_total"
    )
    keys = [
        "reports",
        "segment_success_rate",
        "tokens_api_per_report",
        "tokens_est_per_report",
        "llm_calls_per_report",
        "search_engine_calls_per_report",
        "financial_api_calls_per_report",
        "issue_recovery_rate",
        "report_chars_avg",
        "report_non_ws_chars_avg",
        "segment_finalized",
        "segment_total",
        "issue_segment_finalized",
        "issue_segment_total",
    ]
    print(",".join(format_metric(row.get(key)) for key in keys))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="usage_tracking_*.jsonl files")
    args = parser.parse_args()
    events = []
    for raw_path in args.paths:
        events.extend(read_events(Path(raw_path)))
    print_report_summary(summarize_reports(events))
    print_api_table(summarize_api_calls(events))
    print_llm_table(summarize_llm(events))
    print_tool_table(summarize_tools(events))


if __name__ == "__main__":
    main()
