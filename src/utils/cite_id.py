# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import re
import time
from typing import Any
from urllib.parse import unquote, urlparse

_SAFE_PART_RE = re.compile(r"[^a-z0-9_]+")

SEARCH_PREFIX = "search_"
CALC_PREFIX = "calculate_"


def short_time_token() -> str:
    """Six-character base36 timestamp token, compact enough for LLM-visible cite IDs."""
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    num = int(time.time())
    out = ""
    while num:
        num, rem = divmod(num, 36)
        out = chars[rem] + out
    return (out or "0")[-6:]


def id_part(value: Any, max_len: int = 32, fallback: str = "") -> str:
    text = "" if value is None else str(value).strip().lower()
    text = _SAFE_PART_RE.sub("-", text).strip("-_")
    if not text:
        return fallback
    return text[:max_len].strip("-_") or fallback


def url_part(url: str, max_len: int = 48, fallback: str = "page") -> str:
    parsed = urlparse(url or "")
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    domain_part = id_part(domain, max_len=30)

    path = unquote(parsed.path or "")
    path = path.rsplit("/", 1)[-1] or path.strip("/")
    path = re.sub(r"\.[a-z0-9]{1,6}$", "", path.lower())
    path_part = id_part(path, max_len=24)

    combined = "-".join(part for part in (domain_part, path_part) if part)
    return id_part(combined, max_len=max_len, fallback=fallback)


def cite_id(
    prefix: str,
    *parts: Any,
    hash_parts: tuple[Any, ...] = (),
    unique: bool = False,
    max_part_len: int = 32,
) -> str:
    cleaned = [id_part(prefix, max_len=24)]
    cleaned.extend(part for part in (id_part(item, max_len=max_part_len) for item in parts) if part)

    if hash_parts:
        raw = "|".join("" if part is None else str(part) for part in hash_parts)
        cleaned.append(f"h{hashlib.blake2s(raw.encode('utf-8'), digest_size=6).hexdigest()[:6]}")
    if unique:
        cleaned.append(short_time_token())

    return "_".join(cleaned)


def is_search_cite_id(cite_id_value: str | None) -> bool:
    return isinstance(cite_id_value, str) and cite_id_value.startswith(SEARCH_PREFIX)


def is_calc_cite_id(cite_id_value: str | None) -> bool:
    return isinstance(cite_id_value, str) and cite_id_value.startswith(CALC_PREFIX)
