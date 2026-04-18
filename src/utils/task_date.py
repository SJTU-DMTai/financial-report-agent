from __future__ import annotations

import os
import re
from datetime import datetime

import pandas as pd

_TASK_DATE_HINT_RE = re.compile(
    r"(?:当前日期|报告日期|截止日期|截至|截止|日期是|日期为)[^\d]{0,8}"
    r"(?P<date>(?:\d{8}|\d{4}(?:[-./]\d{1,2}[-./]\d{1,2}|年\d{1,2}月\d{1,2}日?)))"
)
_DATE_TOKEN_RE = re.compile(
    r"(?P<date>(?:\d{8}|\d{4}(?:[-./]\d{1,2}[-./]\d{1,2}|年\d{1,2}月\d{1,2}日?)))"
)


def normalize_compact_date(value: str | None) -> str | None:
    """Normalize a date-like string into YYYYMMDD."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    matched = _DATE_TOKEN_RE.search(text)
    if matched:
        text = matched.group("date")
        text = text.replace("年", "-").replace("月", "-").replace("日", "")

    try:
        return pd.to_datetime(text).strftime("%Y%m%d")
    except Exception as exc:
        raise ValueError(f"无法解析日期: {value}") from exc


def extract_task_date(task_desc: str | None) -> str | None:
    """Extract the task date from task_desc and normalize it to YYYYMMDD."""
    text = (task_desc or "").strip()
    if not text:
        return None

    hinted_match = _TASK_DATE_HINT_RE.search(text)
    if hinted_match:
        return normalize_compact_date(hinted_match.group("date"))

    any_match = _DATE_TOKEN_RE.search(text)
    if any_match:
        return normalize_compact_date(any_match.group("date"))

    return None


def resolve_cur_date(task_desc: str | None, cur_date: str | None = None) -> str:
    """Resolve task date with priority: explicit arg > task_desc > env > runtime."""
    return (
        normalize_compact_date(cur_date)
        or extract_task_date(task_desc)
        or normalize_compact_date(os.getenv("CUR_DATE"))
        or datetime.now().strftime("%Y%m%d")
    )
