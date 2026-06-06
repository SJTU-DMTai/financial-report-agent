# -*- coding: utf-8 -*-
from __future__ import annotations

from bisect import bisect_right
from typing import Any, Callable
import copy
import json
import re

import pandas as pd
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from ..memory.short_term import ShortTermMemoryStore, MaterialType
from ..memory.long_term import LongTermMemoryStore
from ..utils.cite_id import is_calc_cite_id, is_search_cite_id
from ..utils.retrieve_in_memory import retrieve_in_memory

def _normalize_search_keywords(
    keywords: list[str],
    min_keyword_len: int = 1,
    ignore_case: bool = True,
) -> list[str]:
    """规范化关键词列表，并按大小写规则去重。"""
    normalized_keywords: list[str] = []
    seen_keywords: set[str] = set()
    for kw in keywords:
        kw = (kw or "").strip()
        if len(kw) < min_keyword_len:
            continue
        dedupe_key = kw.casefold() if ignore_case else kw
        if dedupe_key in seen_keywords:
            continue
        seen_keywords.add(dedupe_key)
        normalized_keywords.append(kw)
    return normalized_keywords

def _find_keyword_matches(
    text: str,
    keywords: list[str],
    min_keyword_len: int = 1,
    ignore_case: bool = True,
) -> list[dict[str, Any]]:
    """查找文本中所有关键词命中位置。"""
    if not text:
        return []

    normalized_keywords = _normalize_search_keywords(
        keywords=keywords,
        min_keyword_len=min_keyword_len,
        ignore_case=ignore_case,
    )
    if not normalized_keywords:
        return []

    matches: list[dict[str, Any]] = []
    for kw in normalized_keywords:
        flags = re.IGNORECASE if ignore_case else 0
        for match in re.finditer(re.escape(kw), text, flags):
            matches.append({
                "start": match.start(),
                "end": match.end(),
                "keyword": kw,
                "matched_text": match.group(0),
            })

    matches.sort(key=lambda item: (item["start"], item["end"], item["keyword"]))
    return matches


class _BoundQueryTool:
    def __init__(self, func, name: str, doc: str, *args) -> None:
        self._func = func
        self._args = args
        self.__name__ = name
        self.__doc__ = doc

    def __call__(self, query: str) -> ToolResponse:
        return self._func(*self._args, query)


def bind_query_tool(func, name: str, doc: str, *args):
    return _BoundQueryTool(func, name, doc, *args)


def bind_async_query_tool(func, name: str, doc: str, *args):
    async def tool(query: str) -> ToolResponse:
        return await func(*args, query)

    tool.__name__ = name
    tool.__doc__ = doc
    return tool


def grep_file_with_context(
    filepath: str,
    keyword: str,
    context_lines: int = 40,
    case_sensitive: bool = False,
    regex: bool = False,
) -> list[dict[str, Any]]:
    """
    Search for keyword in a file with surrounding context lines (like grep -aX -bX).

    Args:
        filepath: Path to the file to search
        keyword: The keyword or regex pattern to search for
        context_lines: Number of lines to show before and after match (default 100)
        case_sensitive: Whether search should be case sensitive
        regex: Whether keyword is a regex pattern

    Returns:
        List of dicts containing:
        - 'line_num': First matched line number in the merged window (0-based)
        - 'matched_line': First matched line text in the merged window
        - 'context': All lines in the merged context window
        - 'match_line_numbers': All matched line numbers covered by this window
        - 'context_start': Starting line number of context window
        - 'context_end': Ending line number of context window
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        return [{"error": f"Failed to read file: {str(e)}"}]

    if not lines:
        return []

    matched_line_indices: list[int] = []

    if regex:
        try:
            pattern = re.compile(keyword, 0 if case_sensitive else re.IGNORECASE)
        except Exception as e:
            return [{"error": f"Invalid regex pattern: {str(e)}"}]

        for line_idx, line in enumerate(lines):
            if pattern.search(line):
                matched_line_indices.append(line_idx)
    else:
        full_text = "".join(lines)
        line_start_offsets: list[int] = []
        current_offset = 0
        for line in lines:
            line_start_offsets.append(current_offset)
            current_offset += len(line)

        matches = _find_keyword_matches(
            text=full_text,
            keywords=[keyword],
            min_keyword_len=1,
            ignore_case=not case_sensitive,
        )

        unique_line_indices: set[int] = set()
        for match in matches:
            line_idx = bisect_right(line_start_offsets, match["start"]) - 1
            if line_idx >= 0:
                unique_line_indices.add(line_idx)
        matched_line_indices = sorted(unique_line_indices)

    if not matched_line_indices:
        return []

    merged_results: list[dict[str, Any]] = []
    for line_idx in matched_line_indices:
        start_idx = max(0, line_idx - context_lines)
        end_idx = min(len(lines), line_idx + context_lines + 1)

        if merged_results and start_idx <= merged_results[-1]["context_end"]:
            previous = merged_results[-1]
            previous["context_end"] = max(previous["context_end"], end_idx)
            previous["match_line_numbers"].append(line_idx)
            continue

        merged_results.append({
            "context_start": start_idx,
            "context_end": end_idx,
            "match_line_numbers": [line_idx],
            "context_lines_count": context_lines,
        })

    results = []
    for item in merged_results:
        context = [
            (i, lines[i].rstrip('\n'))
            for i in range(item["context_start"], item["context_end"])
        ]
        match_line_numbers = sorted(set(item["match_line_numbers"]))
        first_match_line = match_line_numbers[0]
        results.append({
            "line_num": first_match_line,
            "matched_line": lines[first_match_line].rstrip('\n'),
            "context": context,
            "match_line_numbers": match_line_numbers,
            "context_start": item["context_start"],
            "context_end": item["context_end"],
            "context_lines_count": context_lines,
        })

    return results


def print_grep_results(
    results: list[dict[str, Any]],
    max_results: int = 10,
    show_line_numbers: bool = True,
) -> str:
    """
    Format grep results for display.

    Args:
        results: List of search results from grep_file_with_context
        max_results: Maximum number of results to display
        show_line_numbers: Whether to show line numbers

    Returns:
        Formatted string for display
    """
    if not results:
        return "No matches found."

    if results[0].get('error'):
        return f"Error: {results[0]['error']}"

    output = []
    for i, result in enumerate(results[:max_results]):
        if i > 0:
            output.append("\n" + "="*80 + "\n")

        match_line_numbers = result.get("match_line_numbers") or [result["line_num"]]
        line_label = ", ".join(str(line_num) for line_num in match_line_numbers)
        output.append(f"Match #{i+1} at line {line_label}:\n")

        context = result.get("context")
        if context is None:
            context = []
            before = result.get("before", [])
            after = result.get("after", [])
            context.extend(before)
            context.append((result["line_num"], result["matched_line"]))
            context.extend(after)

        matched_line_set = set(match_line_numbers)
        for line_num, line_text in context:
            prefix = f"{line_num:5d}: " if show_line_numbers else ""
            marker = ">" if line_num in matched_line_set else " "
            output.append(f"{marker}{prefix}{line_text}\n")

    if len(results) > max_results:
        output.append(f"\n... and {len(results) - max_results} more matches (showing first {max_results})")

    return "".join(output)


def extract_keyword_context_snippets(
    text: str,
    keywords: list[str],
    context_chars: int = 100,
    min_keyword_len: int = 1,
    ignore_case: bool = True,
    merge_gap_chars: int = 20,
    highlight: bool = False,
    max_snippets: int | None = None,
) -> list[dict[str, Any]]:
    """提取关键词命中的上下文片段，并对重复/重叠命中做合并去重。"""
    if not text:
        return []

    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return []

    matches = _find_keyword_matches(
        text=clean_text,
        keywords=keywords,
        min_keyword_len=min_keyword_len,
        ignore_case=ignore_case,
    )

    intervals: list[list[Any]] = []
    for match in matches:
        start_idx = max(0, match["start"] - context_chars)
        end_idx = min(len(clean_text), match["end"] + context_chars)
        intervals.append([start_idx, end_idx, {match["keyword"]}])

    if not intervals:
        return []

    intervals.sort(key=lambda item: item[0])
    merged_intervals = [intervals[0]]
    for current in intervals[1:]:
        previous = merged_intervals[-1]
        if current[0] <= previous[1] + merge_gap_chars:
            previous[1] = max(previous[1], current[1])
            previous[2].update(current[2])
        else:
            merged_intervals.append(current)

    snippets: list[dict[str, Any]] = []
    seen_snippets: set[str] = set()
    for start_idx, end_idx, matched_keywords in merged_intervals:
        prefix = "..." if start_idx > 0 else ""
        suffix = "..." if end_idx < len(clean_text) else ""
        snippet = prefix + clean_text[start_idx:end_idx] + suffix

        if highlight:
            for kw in sorted(matched_keywords, key=len, reverse=True):
                snippet = re.sub(
                    re.escape(kw),
                    lambda m: f"【{m.group(0)}】",
                    snippet,
                    flags=re.IGNORECASE if ignore_case else 0,
                )

        snippet_key = snippet.casefold() if ignore_case else snippet
        if snippet_key in seen_snippets:
            continue
        seen_snippets.add(snippet_key)

        snippets.append({
            "snippet": snippet,
            "keywords": sorted(matched_keywords),
        })
        if max_snippets is not None and len(snippets) >= max_snippets:
            break

    return snippets


def retrieve_local_material(
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    query: str,
) -> ToolResponse:
    """
    在已保存的本地材料中按关键词搜索和query最相关的前若干条材料，返回cite_id、来源、简短描述和部分预览内容。
    Args:
        query (str):
            搜索内容。
    """
    candidates = retrieve_in_memory(
        short_term=short_term,
        long_term=long_term,
        query=query,
    )
    if candidates:
        lines = [
            f"你需要检索的信息：{query}",
            "",
            "用户本地保存的候选材料（materials）及其部分预览如下:",
            "",
        ]

        for meta in candidates:
            cite_id = meta.get("cite_id", "")
            desc = meta.get("description", "")
            src = meta.get("source", "")
            m_type = meta.get("m_type", "")
            lines.append(f"<material>\ncite_id={cite_id}")
            if desc:
                lines.append(f"简短描述: {desc}")
            if src:
                lines.append(f"来源: {src}")

            try:
                content = short_term.load_material(cite_id) if short_term is not None else None
            except Exception:
                content = None

            if is_search_cite_id(cite_id):
                preview = short_term.load_material_preview(cite_id=cite_id)
                if preview:
                    lines.append("    部分内容预览：")
                    lines.append(f"   {preview}")

            elif is_calc_cite_id(cite_id):
                params = None
                result = None
                if isinstance(content, list) and content:
                    first = content[0] if isinstance(content[0], dict) else None
                    if isinstance(first, dict):
                        params = first.get("parameters", None)
                        result = first.get("result", None)
                if params:
                    lines.append(f"计算参数: {params}")
                lines.append(f"计算结果: {result}")

            elif m_type == "table":
                preview = ""
                columns_preview = ""
                if isinstance(content, pd.DataFrame) and not content.empty:
                    df_preview = content.head(3).copy()
                    max_cell_chars = 32
                    suffix = "...[内容过长，已截断]"
                    for col in df_preview.columns:
                        df_preview[col] = df_preview[col].apply(
                            lambda value: (
                                value[:max_cell_chars] + suffix
                                if isinstance(value, str) and len(value) > max_cell_chars
                                else value
                            )
                        )
                    columns_preview = ", ".join(content.columns)
                    preview = df_preview.to_csv(index=False)

                lines.append("    列名:")
                lines.append(f"    {columns_preview}")
                if preview:
                    lines.append("    前3行预览:")
                    for ln in preview.splitlines():
                        lines.append(f"    {ln}")

            lines.append("</material>\n")

        lines.append("如果以上无合适材料，请重新调用工具获取数据。")
        res = "\n".join(lines)
    else:
        res = f"对于{query}，本地尚未保存相关材料。请调用合适的工具获取数据。"
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=res,
            ),
        ],
    )


def get_retrieve_fn(short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore) -> Callable[str]:
    return bind_query_tool(
        retrieve_local_material,
        "retrieve_local_material",
        retrieve_local_material.__doc__ or "",
        short_term,
        long_term,
    )


class MaterialTools:
    def __init__(self, short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore | None = None) -> None:
        self.short_term = short_term
        self.long_term = long_term
    
    def read_material(
        self,
        cite_id: str,
        query_key: str | None = None,
        context_lines: int = 20,
    ) -> ToolResponse:
        """
        读取已保存的 Material 内容。
        - 表格：query_key 为空时返回完整表格；query_key 不为空时按列名筛选，多个列名用英文逗号分隔。
        - 文本：query_key 为空时返回完整文本；query_key 不为空时按关键词搜索并返回上下文。
        - JSON：query_key 为空时返回完整 JSON；query_key 不为空时按点分字段路径提取内容，例如 "title" 或 "metadata.source"。

        Args:
            cite_id (str): Material 的唯一标识 ID。
            query_key (str | None):
                可选查询条件。表格中表示列名列表，文本中表示关键词，JSON 中表示字段路径。
            context_lines (int):
                文本关键词搜索时的上下文行数，关键词前后各显示该行数；默认为 20。
                仅在读取文本且 query_key 不为空时有效。
        """
        meta = self.short_term.get_material_meta(cite_id) 

        if not meta:
            return ToolResponse(
                content=[TextBlock(type="text", text=f"[read_material]未找到该 ID 对应的 Material")],
                metadata={"cite_id": cite_id}
            )

        try:
            if meta.m_type == MaterialType.TABLE:
                return self._read_table_impl(cite_id, query_key)
            elif meta.m_type == MaterialType.TEXT:
                # 如果提供了关键词，使用关键词搜索
                if query_key:
                    return self._read_with_keyword(cite_id, query_key, context_lines, meta)
                else:
                    return self._read_text_impl(cite_id)
            elif meta.m_type == MaterialType.JSON:
                return self._read_json_impl(cite_id, query_key)
            else:  
                return ToolResponse(
                    content=[TextBlock(type="text", text=f"[read_material]不支持的文件类型: {meta.m_type}")],
                    metadata={"cite_id": cite_id}
                )
        except Exception as e:
            return ToolResponse(
            content=[TextBlock(type="text", text=f"[read_material]读取失败: {str(e)}")],
            metadata={"cite_id": cite_id}
            )

    # ========== 辅助函数 ==========

    def _read_with_keyword(self, cite_id: str, keyword: str, context_lines: int, meta) -> ToolResponse:
        """
        使用关键词搜索Material内容。
        根据Material类型，使用grep工具进行搜索并返回结果。

        Args:
            cite_id: Material ID
            keyword: 搜索关键词
            context_lines: 上下文行数
            meta: Material元数据

        Returns:
            ToolResponse with search results
        """
        try:
            # 获取material文件路径
            material_file = self.short_term.material_dir / meta.filename

            if not material_file.exists():
                return ToolResponse(
                    content=[TextBlock(type="text", text=f"[read_material] Material文件不存在: {meta.filename}")],
                    metadata={"cite_id": cite_id}
                )

            # 使用grep_file_with_context进行搜索
            results = grep_file_with_context(
                filepath=str(material_file),
                keyword=keyword,
                context_lines=context_lines,
                case_sensitive=False,
                regex=False
            )

            if not results:
                return ToolResponse(
                    content=[
                        TextBlock(type="text", text=f"[read_material] 关键词'{keyword}'未在Material中找到任何匹配")],
                    metadata={"cite_id": cite_id, "keyword": keyword, "matches": 0}
                )

            # 格式化搜索结果
            formatted_results = print_grep_results(results, max_results=10, show_line_numbers=True)
            match_line_count = sum(len(result.get("match_line_numbers", [])) for result in results)

            text = (
                f"[read_material] ID: {cite_id}\n"
                f"关键词: '{keyword}'\n"
                f"共找到 {match_line_count} 处命中，合并为 {len(results)} 段上下文\n"
                f"上下文如下: \n\n"
                f"{formatted_results}"
            )

            return ToolResponse(
                content=[TextBlock(type="text", text=text)],
                metadata={
                    "cite_id": cite_id,
                    "keyword": keyword,
                    "matches": len(results),
                    "match_lines": match_line_count,
                    "type": "keyword_search",
                    "context_lines": context_lines
                }
            )

        except Exception as e:
            return ToolResponse(
                content=[TextBlock(type="text", text=f"[read_material] 关键词搜索失败: {str(e)}")],
                metadata={"cite_id": cite_id, "error": str(e)}
            )

    def _read_table_impl(self, cite_id, cols):
        df = self.short_term.load_material(cite_id)

        # 1. 列筛选 (Horizontal Slicing)
        if cols:
            col_list = [c.strip() for c in cols.split(",")]
            # 容错处理：只保留存在的列
            valid_cols = [c for c in col_list if c in df.columns]
            if valid_cols:
                df = df[valid_cols]

        # 2. 行筛选 (Vertical Slicing)
        total_rows = len(df)

        sliced_df = df.copy()

        if cite_id.startswith("disclosure_") and ("公告" in sliced_df.columns):
            MAX_TOTAL_CHARS = 20000
            MIN_PER_CELL = 200
            SUFFIX = "\n...... [内容过长，已截断，如需要完整阅读请对此条结果单独使用read_material工具]"

            # 只统计当前页（切片范围内）“公告”列的字符数
            ann_series = sliced_df["公告"]

            # 只对 str 进行统计/截断；非字符串保持原样
            valid_indices = []
            total_len = 0

            for idx, v in ann_series.items():
                if isinstance(v, str) and v:
                    l = len(v)
                    valid_indices.append(idx)
                    total_len += l
            if total_len > MAX_TOTAL_CHARS and valid_indices:
                n_items = len(valid_indices)
                per_limit = max(MAX_TOTAL_CHARS // n_items, MIN_PER_CELL)

                for idx in valid_indices:
                    v = sliced_df.at[idx, "公告"]
                    if isinstance(v, str) and v:
                        if len(v) > per_limit:
                            sliced_df.at[idx, "公告"] = v[:per_limit] + SUFFIX

        # preview_str = sliced_df.to_markdown(index=False, disable_numparse=True)
        preview_str = sliced_df.to_csv(index=False)
        text = (f"[read_material] ID: {cite_id}\n"
                f"完整 material 共 {total_rows} 行。\n"
                f"内容:\n{preview_str}")

        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"cite_id": cite_id, "type": "table", "rows": len(sliced_df)}
        )

    def _read_text_impl(self, cite_id):
        # 适用于 .txt, .md
        content = self.short_term.load_material(cite_id)  # 返回 str
        lines = content.split('\n')
        total_lines = len(lines)

        # 截取
        sliced_lines = lines
        preview_str = "\n".join(sliced_lines)
        assert len(preview_str) < 5000, f"{cite_id}完整内容过长，建议根据关键词读取上下文。"

        text = (f"[read_material] ID: {cite_id}\n"
                f"完整 material 共 {total_lines} 行。\n"
                f"内容:\n{preview_str}")

        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"cite_id": cite_id, "type": "text", "lines": len(sliced_lines)}
        )

    def _read_json_impl(
            self,
            cite_id: str,
            key_path: str | None,
    ) -> ToolResponse:
        data = self.short_term.load_material(cite_id)
        # if isinstance(data, list):
        sliced = copy.deepcopy(data)

        if is_search_cite_id(cite_id):
            if isinstance(sliced, list):
                for item in sliced:
                    if isinstance(item, dict):
                        # 删除 relevance 字段
                        item.pop("relevance", None)

        # 如果有 key_path，则提取每条对应字段，否则展示整个条目
        if key_path:
            extracted = []
            for item in sliced:
                cur = item
                for key in key_path.split("."):
                    if isinstance(cur, dict) and key in cur:
                        cur = cur[key]
                    else:
                        cur = None
                        break
                extracted.append(cur)
            sliced = extracted

        # 序列化成 JSON 字符串
        json_str = json.dumps(sliced, ensure_ascii=False, indent=2)

        text = (
            f"[read_material] ID: {cite_id}\n"
            f"内容:\n{json_str}"
        )

        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"cite_id": cite_id, "type": "json_list"}
        )

