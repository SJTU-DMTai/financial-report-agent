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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="token_usage_tracking_*.jsonl files")
    args = parser.parse_args()
    events = []
    for raw_path in args.paths:
        events.extend(read_events(Path(raw_path)))
    print_llm_table(summarize_llm(events))
    print_tool_table(summarize_tools(events))


if __name__ == "__main__":
    main()
