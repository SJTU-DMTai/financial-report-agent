# -*- coding: utf-8 -*-
from __future__ import annotations

import traceback
from typing import Optional, Callable, Any, Dict, Union

import os
import io
import time as time_module
from datetime import datetime
import akshare as ak
import pandas as pd
import re
from urllib.parse import urlparse, parse_qs
import requests
import fitz
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit
from ..memory.short_term import ShortTermMemoryStore, MaterialType
from ..memory.long_term import LongTermMemoryStore
from ..utils.get_entity_info import get_entity_info
import json
import copy


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
        - 'line_num': Line number of the match (0-based)
        - 'matched_line': The line containing the match
        - 'before': Lines before the match (list of tuples: (line_num, line_text))
        - 'after': Lines after the match (list of tuples: (line_num, line_text))
        - 'context_start': Starting line number of context window
        - 'context_end': Ending line number of context window
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        return [{"error": f"Failed to read file: {str(e)}"}]

    results = []

    # Compile regex if needed
    if regex:
        try:
            pattern = re.compile(keyword, 0 if case_sensitive else re.IGNORECASE)
        except Exception as e:
            return [{"error": f"Invalid regex pattern: {str(e)}"}]
    else:
        if not case_sensitive:
            keyword_lower = keyword.lower()

    # Search through lines
    for line_idx, line in enumerate(lines):
        match_found = False

        if regex:
            match_found = bool(pattern.search(line))
        else:
            match_found = keyword_lower in line.lower() if not case_sensitive else keyword in line

        if match_found:
            # Calculate context window
            start_idx = max(0, line_idx - context_lines)
            end_idx = min(len(lines), line_idx + context_lines + 1)

            # Gather before and after lines
            before = [(i, lines[i].rstrip('\n')) for i in range(start_idx, line_idx)]
            after = [(i, lines[i].rstrip('\n')) for i in range(line_idx + 1, end_idx)]

            results.append({
                'line_num': line_idx,
                'matched_line': line.rstrip('\n'),
                'before': before,
                'after': after,
                'context_start': start_idx,
                'context_end': end_idx,
                'context_lines_count': context_lines,
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

        output.append(f"Match #{i+1} at line {result['line_num']}:\n")

        # Before context
        if result['before']:
            output.append("--- BEFORE ---\n")
            for line_num, line_text in result['before']:
                prefix = f"{line_num:5d}: " if show_line_numbers else ""
                output.append(f"{prefix}{line_text}\n")

        # Matched line
        output.append(">>> MATCHED <<<\n")
        prefix = f"{result['line_num']:5d}: " if show_line_numbers else ""
        output.append(f"{prefix}{result['matched_line']}\n")

        # After context
        if result['after']:
            output.append("--- AFTER ---\n")
            for line_num, line_text in result['after']:
                prefix = f"{line_num:5d}: " if show_line_numbers else ""
                output.append(f"{prefix}{line_text}\n")

    if len(results) > max_results:
        output.append(f"\n... and {len(results) - max_results} more matches (showing first {max_results})")

    return "".join(output)


def _preview_df(df: pd.DataFrame, max_rows: int | None = None) -> tuple[str, int, int, list[str]]:
    """生成 DataFrame 文本预览及相关统计。"""
    total_rows = len(df)

    if max_rows is None:
        preview_df = df  # 全部数据
        preview_rows = total_rows
    else:
        preview_df = df.head(max_rows)
        preview_rows = min(max_rows, total_rows)

    preview_str = preview_df.to_string(index=False)
    columns_names = list(df.columns)
    return preview_str, total_rows, preview_rows, columns_names


def _build_tool_response_from_df(
        df: pd.DataFrame,
        ref_id: str,
        header: str,
        preview_rows: int = 10,
        extra_meta: Optional[Dict[str, Any]] = None,
) -> ToolResponse:
    """统一构造 ToolResponse，包含预览文本和基础 metadata。"""
    preview_str, total_rows, used_rows, columns_names = _preview_df(df, preview_rows)  # 预览的时候返回前10行
    text = (
        f"{header} 共 {total_rows} 条记录，"
        f"Material 已写入 ref_id='{ref_id}'（CSV 格式）。\n\n"
        f"以下为全部列名：\n"
        f"{columns_names}\n"
        f"以下为前 {used_rows} 行预览：\n"
        f"{preview_str}"
    )
    meta: Dict[str, Any] = {"ref_id": ref_id, "row_count": total_rows}
    if extra_meta:
        meta.update(extra_meta)
    return ToolResponse(
        content=[TextBlock(type="text", text=text)],
        metadata=meta,
    )

def _build_tool_response_from_dict(
        data: dict,
        ref_id: str,
        header: str,
        preview_rows: int = 10,
        extra_meta: Optional[Dict[str, Any]] = None,
) -> ToolResponse:
    """统一构造 ToolResponse，包含预览文本和基础 metadata。"""
    preview_str = "\n".join([f"{k}: {v}" for k, v in data.items()])
    text = (
        f"{header} 共 {len(data)} 条属性，"
        f"Material 已写入 ref_id='{ref_id}'（CSV 格式）。\n\n"
        f"{preview_str}"
    )
    meta: Dict[str, Any] = {"ref_id": ref_id}
    if extra_meta:
        meta.update(extra_meta)
    return ToolResponse(
        content=[TextBlock(type="text", text=text)],
        metadata=meta,
    )


def add_exchange_prefix(symbol: str, type: str) -> str:
    """
    根据股票代码判断交易所，并按大小写返回带前缀的 symbol。
    由于akshare的stock_gdfx_free_top_10_em和stock_gdfx_top_10_em
    和stock_zygc_em需要传带交易所的股票代码
    """
    code = symbol.strip()
    t = type.lower()
    # 前缀映射
    prefix_upper = {"SH": "SH", "SZ": "SZ", "BJ": "BJ", "HK": "HK"}
    prefix_lower = {"SH": "sh", "SZ": "sz", "BJ": "bj", "HK": "hk"}

    if code.isdigit() and len(code) == 5:
        exchange = "HK"
    else:
        exchange = None

    # 其他交易所规则
    if exchange is None:
        # 上交所
        if code.startswith(("600", "601", "603", "605", "688", "900", "730", "700")):
            exchange = "SH"

        # 深交所
        elif code.startswith(("001", "002", "003", "200", "080")) or code.startswith("30"):
            exchange = "SZ"

        # 北交所
        elif code.startswith(("43", "83", "87", "920")):
            exchange = "BJ"

        else:
            return code

    # 返回大小写
    if t == "upper":
        return prefix_upper[exchange] + code
    elif t == "lower":
        return prefix_lower[exchange] + code
    else:
        raise ValueError("type 需为 'upper' 或 'lower'。")

def fmt_yyyymmdd(s: str) -> str:
        s = (s or "").strip()
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
        return s  # 已是 YYYY-MM-DD 或其他格式则原样返回

class MaterialTools:
    def __init__(self, short_term: Optional[ShortTermMemoryStore] = None, long_term: Optional[LongTermMemoryStore] = None) -> None:
        self.short_term = short_term
        self.long_term = long_term

    def read_material(
        self,
        ref_id: str,
        start_index: int | None = None,
        end_index: int | None = None,
        query_key: str | None = None,
        context_lines: int = 50,
    ) -> ToolResponse:
        """
        统一读取material。支持读取全文、通过参数、按行/条目截取其中部分，以及关键词搜索。
        - 表格：按行号切片，可选按名为query_key的列筛选。
        - 文本：可通过关键词 query_key 进行内容搜索。
        - JSON：可通过 query_key 获取特定条目内容。

        如果start_index, end_index, query_key 都为空表示读取全文。

        Args:
            ref_id (str): Material 的唯一标识 ID。
            start_index (int | None): 
                - 对于表格：起始行号（包含）。
                - 如果为空表示从第0行开始。
            end_index (int | None): 
                - 对于表格：结束行号（不包含）。
                - 如果为空表示到最后一条结束。
            query_key (str | None):
                - 对于 JSON list：可选，用于对每个条目提取该字段。
                - 对于表格：可选，用于筛选特定列（如 "Date,Close"）。
                - 对于文本：可选，用于在内容中搜索特定关键词，返回匹配结果及若干行上下文。不适用于表格和JSON
            context_lines (int):
                - 关键词搜索时的上下文行数（关键词前后各显示该行数）。
                - 默认为50行。只在使用keyword参数时有效。
        """
        if start_index is not None:
            start_index = int(start_index)
        if end_index is not None:
            end_index = int(end_index)

        meta = self.short_term.get_material_meta(ref_id) 

        if not meta:
            return ToolResponse(
                content=[TextBlock(type="text", text=f"[read_material]未找到该 ID 对应的 Material")],
                metadata={"ref_id": ref_id}
            )

        try:
            # ...existing code...
            if meta.m_type == MaterialType.TABLE:
                return self._read_table_impl(ref_id, start_index, end_index, query_key)
            elif meta.m_type == MaterialType.TEXT:
                # 如果提供了关键词，使用关键词搜索
                if query_key:
                    return self._read_with_keyword(ref_id, query_key, context_lines, meta)
                else:
                    return self._read_text_impl(ref_id, start_index, end_index)
            elif meta.m_type == MaterialType.JSON:
                return self._read_json_impl(ref_id, query_key)
            else:  
                return ToolResponse(
                    content=[TextBlock(type="text", text=f"[read_material]不支持的文件类型: {meta.m_type}")],
                    metadata={"ref_id": ref_id}
                )
        except Exception as e:
            print(traceback.print_exc())
            return ToolResponse(
            content=[TextBlock(type="text", text=f"[read_material]读取失败: {str(e)}")],
            metadata={"ref_id": ref_id}
            )


    # ========== 内部函数（不注册为 tool） ==========

    def _read_with_keyword(self, ref_id: str, keyword: str, context_lines: int, meta) -> ToolResponse:
        """
        使用关键词搜索Material内容。
        根据Material类型，使用grep工具进行搜索并返回结果。

        Args:
            ref_id: Material ID
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
                    metadata={"ref_id": ref_id}
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
                    content=[TextBlock(type="text", text=f"[read_material] 关键词'{keyword}'未在Material中找到任何匹配")],
                    metadata={"ref_id": ref_id, "keyword": keyword, "matches": 0}
                )

            # 格式化搜索结果
            formatted_results = print_grep_results(results, max_results=10, show_line_numbers=True)

            text = (
                f"[read_material] ID: {ref_id}\n"
                f"关键词: '{keyword}'\n"
                f"共找到 {len(results)} 个匹配\n"
                f"带有{context_lines}的上下文如下: \n\n"
                f"{formatted_results}"
            )

            return ToolResponse(
                content=[TextBlock(type="text", text=text)],
                metadata={
                    "ref_id": ref_id,
                    "keyword": keyword,
                    "matches": len(results),
                    "type": "keyword_search",
                    "context_lines": context_lines
                }
            )

        except Exception as e:
            return ToolResponse(
                content=[TextBlock(type="text", text=f"[read_material] 关键词搜索失败: {str(e)}")],
                metadata={"ref_id": ref_id, "error": str(e)}
            )

    def _read_table_impl(self, ref_id, start, end, cols):
        df = self.short_term.load_material(ref_id) 
        
        # 1. 列筛选 (Horizontal Slicing)
        if cols:
            col_list = [c.strip() for c in cols.split(",")]
            # 容错处理：只保留存在的列
            valid_cols = [c for c in col_list if c in df.columns]
            if valid_cols:
                df = df[valid_cols]

        # 2. 行筛选 (Vertical Slicing)
        total_rows = len(df)
        start = start if start is not None else 0
        end = end if end is not None else total_rows
        
        # 边界保护
        if start < 0: start = 0
        if end > total_rows: end = total_rows
        
        sliced_df = df.iloc[start:end].copy()

        if ("_disclosure_" in ref_id) and ("公告" in sliced_df.columns):
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
        text = (f"[read_material] ID: {ref_id}\n"
                f"完整 material 共 {total_rows} 行。已读取范围: 行 [{start}, {end})。\n"
                f"内容:\n{preview_str}")
                
        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"ref_id": ref_id, "type": "table", "rows": len(sliced_df)}
        )


    def _read_json_impl(
        self,
        ref_id: str,
        key_path: str | None,
    ) -> ToolResponse:
        data = self.short_term.load_material(ref_id)  
        # if isinstance(data, list):
        sliced = copy.deepcopy(data)

        if (ref_id.startswith("search_engine")):
            if isinstance(sliced, list):
                for item in sliced:
                    if isinstance(item, dict):
                        # 删除 relevance 字段
                        item.pop("relevance", None)


        # 如果有 key_path，则提取每条对应字段，否则展示整个条目
        if key_path:
            def extract(obj, path):
                cur = obj
                for k in path.split("."):
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        return None
                return cur

            sliced = [extract(item, key_path) for item in sliced]

        # 序列化成 JSON 字符串
        json_str = json.dumps(sliced, ensure_ascii=False, indent=2)

        text = (
            f"[read_material] ID: {ref_id}\n"
            f"内容:\n{json_str}"
        )

        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"ref_id": ref_id, "type": "json_list"}
        )

    def _save_df_to_material(
            self,
            df: pd.DataFrame,
            ref_id: str,
            description: str,
            source: str,
            entity:Dict[str,str] | None = None,
            time:Dict[str,str] | None = None,
    ) -> int:
        """DataFrame/Dict 存入 short-term material（CSV/JSON），返回行数。"""
        if self.short_term is not None:
            self.short_term.save_material(ref_id=ref_id, 
                                          content=df, 
                                          description=description,
                                          source=source,
                                        #   source="AKshare API",
                                          entity=entity,
                                          time=time)
        return len(df)

    # ===================== 股价数据 =====================

    async def fetch_realtime_price_material(
        self,
        symbol: str | None = None
    ) -> ToolResponse:
        """获取沪深京 A 股实时行情数据，并保存表格结果到Material当中，返回Material标识ref_id。
        适用场景：需要查询某只 A 股当前价格、涨跌幅、成交量等实时指标；需要一次性拉取全市场实时行情，作为选股或打分模型的输入。

        Args:
            symbol (Optional[str]):
                沪深京 A 股股票代码（不带市场标识），例如 "000001"。
                - 为 None 时：保留全部 A 股的实时行情数据；
                - 不为 None 时：仅保留 DataFrame 中 "代码" 列等于该值的记录。
        """
        time_point = datetime.now().strftime("%Y-%m-%d %H:%M")

        df = ak.stock_zh_a_spot_em()
        entity = None
        if symbol is not None:
            # 文档中 "代码" 列为股票代码
            df = df[df["代码"] == symbol]
            entity = get_entity_info(long_term=self.long_term, text=str(symbol))
        ref_id = f"{symbol or 'all'}_realtime_price_{int(time_module.time())}"
        if symbol:
            description = f"{entity['name']}（{entity['code']}）股票实时行情数据（获取时间：{time_point}）"
        else:
            description = f"A股全市场股票实时行情数据（获取时间：{time_point}）"

        self._save_df_to_material(df=df,
                                ref_id=ref_id,
                                description=description,
                                source="AKshare API:eastmoney",
                                entity=entity,
                                time={"point":time_point},
                                )
        header = f"[fetch_realtime_price_material] 股价实时行情（symbol={symbol or 'ALL'}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )

    async def fetch_history_price_material(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
        period: str = "daily",
        adjust: str = "",
    ) -> ToolResponse:
        """获取指定 A 股股票的历史行情（日/周/月），并保存表格结果到Material当中，返回Material标识ref_id。
        拉取指定股票在给定时间区间和周期上的历史行情数据（开盘价、收盘价、成交量、涨跌幅等），
        支持不复权、前复权和后复权数据，并将结果保存。
        适用场景：生成个股 K 线、收益率曲线、回测信号等历史行情分析；作为生成研报中"股价表现""历史走势"等章节的基础数据。

        Args:
            symbol (str):
                沪深京 A 股股票代码（不带市场标识），例如 "000001"。
            start_date (str):
                历史行情起始日期，格式为 "YYYYMMDD"，例如 "20000101"。必需参数。
            end_date (str | None):
                历史行情结束日期，格式为 "YYYYMMDD"，例如 "20251231"。
                如果为 None，则使用环境变量 CUR_DATE 或当前日期。
            period (str):
                支持的取值（默认 "daily"）：
                - "daily": 日频数据；
                - "weekly": 周频数据；
                - "monthly": 月频数据。
            adjust (str):
                复权方式，支持的取值（默认 ""）：
                - "": 不复权；
                - "qfq": 前复权；
                - "hfq": 后复权。

        """
        cur_date = os.getenv('CUR_DATE') or datetime.now().strftime("%Y%m%d")
        end_date = min(end_date, cur_date) if end_date else cur_date
        assert pd.to_datetime(start_date, format="%Y%m%d") <= pd.to_datetime(end_date, format="%Y%m%d")

        cur_date = os.getenv('CUR_DATE') or datetime.now().strftime("%Y%m%d")
        end_date = min(end_date, cur_date) if end_date else cur_date
        assert pd.to_datetime(start_date, format="%Y%m%d") <= pd.to_datetime(end_date, format="%Y%m%d")

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )


        ref_id = f"{symbol}_history_price_{start_date}_{end_date}_period{period}_adjust{adjust}"
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        time_range = {"start": fmt_yyyymmdd(start_date),"end":fmt_yyyymmdd(end_date)}
        description = f"{entity['name']}（{entity['code']}）股票历史行情数据（{fmt_yyyymmdd(start_date)}~{fmt_yyyymmdd(end_date)}）"

        self._save_df_to_material(df=df,
                                    ref_id=ref_id,
                                    description=description,
                                    source="AKshare API:eastmoney",
                                    entity=entity,
                                    time=time_range)

        header = (
            f"[fetch_history_price_material] {symbol} {period} 股价历史行情 "
            f"{start_date}~{end_date} adjust='{adjust or '无'}'"
        )
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={
                "symbol": symbol,
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "adjust": adjust,
            },
        )

    # ===================== 金融新闻 =====================
    async def fetch_stock_news_material(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
        keyword: str = "",
        latest_num: int = 100,
    ) -> ToolResponse:
        """获取指定个股的新闻资讯数据，并保存表格结果到Material当中，返回Material标识ref_id。
        相关的最新新闻资讯（默认为限定时间范围内最近约 100 条），包括新闻标题、内容摘要、发布时间、来源和链接等，
        适用场景：为个股研报生成“新闻动态”“舆情分析”等部分提供原始素材；需要快速获取近期与某股票相关的新闻列表。

        Args:
            symbol (str):
                沪深京 A 股股票代码（不带市场标识），例如 "000001"；为空字符串时不做股票过滤。
            keyword (str):
                新闻检索关键词，例如 "新能源 储能 政策"；为空字符串时不做关键词过滤。
            start_date (str):
                过滤限定日期之后的新闻，格式为 "YYYYMMDD"，例如 "20230101"。必需参数。
            end_date (str | None):
                过滤限定日期之前的新闻，格式为 "YYYYMMDD"，例如 "20231231"。如果为 None，则使用当前日期。
            latest_num (int):
                限定时间范围内最近的新闻条数，默认50
        """
        cur_date = os.getenv('CUR_DATE') or datetime.now().strftime("%Y%m%d")
        end_date = min(end_date, cur_date) if end_date else cur_date
        assert pd.to_datetime(start_date, format="%Y%m%d") <= pd.to_datetime(end_date, format="%Y%m%d")
        ref_id = f"{symbol}_{keyword}_news_daterange_{start_date}-{end_date}_num{latest_num}"
        start_date, end_date = pd.to_datetime(start_date, format="%Y%m%d"), pd.to_datetime(end_date, format="%Y%m%d")

        if symbol is None or symbol == "":
            entity = None
            description = f"股票新闻资讯 {keyword}"
        else:
            entity = get_entity_info(long_term=self.long_term, text=symbol)
            keyword = entity['name'] + ((" " + keyword) if keyword else "")
            description = (f"{entity['name']}（{entity['code']}）" if entity else "") + f"股票新闻资讯 {keyword}"
        try:
            dfs = []
            for page_idx in range(1, 25):
                df = stock_news_em(keyword=keyword, page_idx=page_idx)
                dfs.append(df[(pd.to_datetime(df['发布时间']) >= start_date) & (pd.to_datetime(df['发布时间']) <= end_date)])
                if sum([len(_df) for _df in dfs]) > latest_num:
                    break
        except Exception as e:
            traceback.print_exc()
            raise e
        df = pd.concat(dfs)
        df.sort_values("发布时间", inplace=True, ascending=False)

        self._save_df_to_material(df=df, ref_id=ref_id,source="AKshare API:eastmoney",entity=entity,description=description)
        header = f"[fetch_stock_news_material] 股票新闻资讯（新闻内容大于156字的部分被省略，请根据url搜索）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol} if symbol else {"keyword": keyword},
        )

    async def fetch_disclosure_material(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
        market: str = "沪深京",
        keyword: str = "",
        category: str = "",
    ) -> ToolResponse:
        """获取指定股票的信息披露公告，并保存表格结果到Material当中，返回Material标识ref_id。
        抓取指定 symbol 在给定时间区间内的各类信息披露公告，
        可按市场、公告类别和关键词进行过滤，并将结果保存为表格。
        适用场景：生成研报中的“公司公告梳理”“信息披露情况”章节；快速定位某段时间内的年报、季报、重大事项、股权变动等公告列表。

        Args:
            symbol (str):
                股票代码，例如 "000001"。
            start_date (str):
                公告起始日期，格式为 "YYYYMMDD"，例如 "20230618"。必需参数。
            end_date (str | None):
                公告结束日期，格式为 "YYYYMMDD"，例如 "20231219"。
                如果为 None，则使用环境变量 CUR_DATE 或当前日期。
            market (str):
                市场类型，支持的取值包括：
                - "沪深京"（默认）、"港股"、"三板"、"基金"、"债券"、"监管"、"预披露"。
            keyword (str):
                公告搜索关键词，例如 "股权激励"、"增发"；为空字符串时不做关键词过滤。
            category (str):
                公告类别，支持的取值包括（示例，默认 ""）：
                - "年报"、"半年报"、"一季报"、"三季报"、"业绩预告"、"权益分派"、
                "董事会"、"监事会"、"股东大会"、"日常经营"、"公司治理"、"中介报告"、
                "首发"、"增发"、"股权激励"、"配股"、"解禁"、"公司债"、"可转债"、
                "其他融资"、"股权变动"、"补充更正"、"澄清致歉"、"风险提示"、
                "特别处理和退市"、"退市整理期"。
                - 为空字符串时：不按类别过滤，返回全部信息披露公告。

        """
        cur_date = os.getenv('CUR_DATE') or datetime.now().strftime("%Y%m%d")
        end_date = min(end_date, cur_date) if end_date else cur_date
        assert pd.to_datetime(start_date, format="%Y%m%d") <= pd.to_datetime(end_date, format="%Y%m%d")


            # 内部工具函数：从公告链接 + 公告时间 拼出 PDF URL
        def _build_pdf_url(link: str, announce_date: str) -> str | None:
            if not isinstance(link, str) or not link:
                return None

            announcement_id = None
            try:
                parsed = urlparse(link)
                qs = parse_qs(parsed.query)
                if "announcementId" in qs and qs["announcementId"]:
                    announcement_id = qs["announcementId"][0]
                else:
                    # 兜底：用正则从 URL 中提取
                    m = re.search(r"announcementId=(\d+)", link)
                    if m:
                        announcement_id = m.group(1)
            except Exception:
                pass

            if not announcement_id:
                return None

            # 公告时间列一般形如 "2023-12-09" 或 "2023-12-09 00:00:00"
            if not isinstance(announce_date, str):
                announce_date = str(announce_date)
            date_str = announce_date.split()[0]  # 去掉可能的时间部分
            # 替换可能存在的分隔符为 "-"
            date_str = (
                date_str.replace(".", "-")
                        .replace("/", "-")
            )

            return f"https://static.cninfo.com.cn/finalpage/{date_str}/{announcement_id}.PDF"

        # 内部工具函数：下载 PDF 并用 PyMuPDF 抽取文本
        def _fetch_pdf_text(pdf_url: str, referer: str | None = None) -> str:
            if not pdf_url:
                return ""
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0",
                }
                if referer:
                    headers["Referer"] = referer

                resp = requests.get(pdf_url, headers=headers, timeout=20)
                resp.raise_for_status()

                with fitz.open(stream=resp.content, filetype="pdf") as doc:
                    texts: list[str] = []
                    for page in doc:
                        t = page.get_text().strip()
                        if t:
                            texts.append(t)
                return "\n".join(texts)
            except Exception:
                # 出错就返回空字符串，避免整个流程中断；具体错误可按需改成日志记录
                return ""

        try:
            df = ak.stock_zh_a_disclosure_report_cninfo(
                symbol=symbol,
                market=market,
                keyword=keyword,
                category=category,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
        # 捕获 akshare 在内部筛选、字段缺失等造成的异常
            df = None
            text = (
                f"[fetch_disclosure_material] Error: {e}"
                f"建议修改或放宽搜索条件（symbol={symbol}, market={market}, keyword={keyword}, "
                f"category={category}, start_date={start_date}, end_date={end_date}）"
            )
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=text,
                    ),
                ],
            )

        
        # df = df.head(10) # 避免获取的公告数量过多取前10条，后续可以改成按照某些条件排序取前10条
        if df is not None and len(df) > 10:
            df = df.sample(n=10)

        # 2. 遍历 df 行，构造 PDF URL 并抽文本，为每个公告保存为独立文件
        disclosure_ref_ids: list[str] = []
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        base_timestamp = int(time_module.time())

        for idx, row in df.iterrows():
            link = row.get("公告链接")
            announce_date = row.get("公告时间")
            announce_title = row.get("公告标题", "未命名公告")
            pdf_url = _build_pdf_url(link, announce_date)
            text = _fetch_pdf_text(pdf_url, referer=link)

            if text:  # 只保存有内容的公告
                # 为每个公告创建唯一的 ref_id
                disclosure_ref_id = f"{symbol}_disclosure_{announce_date}"

                # 创建详细的描述，包含工具名、源URL等信息
                description_parts = [
                    "fetch_disclosure_material",  # 工具名
                    f"source={entity['name']}({entity['code']})",
                    f"title={announce_title}",
                    f"date={announce_date}",
                    f"market={market}",
                ]
                if keyword:
                    description_parts.append(f"keyword={keyword}")
                if category:
                    description_parts.append(f"category={category}")
                if pdf_url:
                    description_parts.append(f"url={pdf_url}")

                description = " | ".join(description_parts)

                # 保存公告文本为单独的文件
                self.short_term.save_material(
                    ref_id=disclosure_ref_id,
                    content=text,
                    description=description,
                    source="AKshare API - stock_zh_a_disclosure_report_cninfo",
                    entity=entity,
                    time={"announce_date": str(announce_date)},
                )
                disclosure_ref_ids.append(disclosure_ref_id)

        # 3. 将 ref_ids 列表添加到 dataframe 中
        df["disclosure_ref_id"] = [disclosure_ref_ids[i] if i < len(disclosure_ref_ids) else "" for i in range(len(df))]
        if "公告链接" in df.columns:
            df = df.drop(columns=["公告链接"])

        # 保存包含 ref_ids 的元数据表格
        ref_id = f"{symbol}_disclosure_{category or 'all'}_{base_timestamp}"
        description = f"{entity['name']}（{entity['code']}）股票"
        parts = [x for x in (keyword, category) if x]
        if parts:
            description += " " + " ".join(parts)
        description = description + "信息披露公告"
        self._save_df_to_material(df=df, ref_id=ref_id, source="AKshare API:CNINFO", entity=entity, description=description)
        header = (
            f"[fetch_disclosure_material] 信息披露公告（symbol={symbol}, market={market}, "
            f"category={category or '全部'}, {start_date}~{end_date}）"
        )
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={
                "symbol": symbol,
                "market": market,
                "keyword": keyword,
                "category": category,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    # ===================== 财务报表 =====================
    async def fetch_balance_sheet_material(
        self,
        symbol: str,
        indicator: str = "按报告期",
    ) -> ToolResponse:
        """获取指定股票的资产负债表数据，并保存表格结果到Material当中，返回Material标识ref_id。
        抓取企业历年或各报告期的资产负债表数据，并将结果保存。
        适用场景：研报中的“资产结构分析”“杠杆水平”“偿债能力”相关章节；对比不同报告期的资产、负债、所有者权益变化情况。

        Args:
            symbol (str):
                股票代码，例如 "000063"。
            indicator (str):
                数据展示方式，持的取值：
                - "按报告期"（默认）：按季度 / 半年 / 年度等报告期展示；
                - "按年度"：按年度汇总展示；
                - "按单季度"：按单个季度拆分展示。
        """

        df = ak.stock_financial_debt_ths(symbol=symbol, indicator=indicator)


        safe_indicator = indicator.replace(" ", "")
        ref_id = f"{symbol}_balance_{safe_indicator}_{int(time_module.time())}"
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）{indicator}资产负债表"
        self._save_df_to_material(
            df=df, ref_id=ref_id,
            description=description,
            entity=entity,
            source="AKshare API:Hithink",
            )
        header = f"[fetch_balance_sheet_material] 资产负债表（symbol={symbol}, indicator={indicator}）"

        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol, "indicator": indicator},
        )

    async def fetch_profit_table_material(
        self,
        symbol: str,
        indicator: str = "按报告期",
    ) -> ToolResponse:
        """获取指定股票的利润表数据，并保存表格结果到Material当中，返回Material标识ref_id。
        抓取企业历年或各报告期的利润表数据，并保存。
        适用场景：盈利能力分析、收入与成本结构分析；生成研报中的“利润表分析”“盈利预测校验”等部分。

        Args:
            symbol (str):
                股票代码，例如 "000063"。
            indicator (str):
                数据展示方式，支持的取值：
                - "按报告期"（默认）；
                - "按年度"；
                - "按单季度"。
        """

        df = ak.stock_financial_benefit_ths(symbol=symbol, indicator=indicator)


        safe_indicator = indicator.replace(" ", "")
        ref_id = f"{symbol}_profit_{safe_indicator}"
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）{indicator}利润表"
        self._save_df_to_material(
            df=df, ref_id=ref_id,
            description=description,
            source="AKshare API:Hithink",
            entity=entity,
            )
        header = f"[fetch_profit_table_material] 利润表（symbol={symbol}, indicator={indicator}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol, "indicator": indicator},
        )

    async def fetch_cashflow_table_material(
        self,
        symbol: str,
        indicator: str = "按报告期",
    ) -> ToolResponse:
        """获取指定股票的现金流量表数据，并保存表格结果到Material当中，返回Material标识ref_id。
        抓取企业历年或各报告期的现金流量表数据（约 75 个字段），并将结果保存。
        适用场景：研报中对经营活动、投资活动、筹资活动现金流的分析；评估企业现金创造能力、分红支付能力和资本开支压力。

        Args:
            symbol (str):
                股票代码，例如 "000063"。
            indicator (str):
                数据展示方式，支持的取值：
                - "按报告期"（默认）；
                - "按年度"；
                - "按单季度"。
        """
        df = ak.stock_financial_cash_ths(symbol=symbol, indicator=indicator)


        safe_indicator = indicator.replace(" ", "")
        ref_id = f"{symbol}_cashflow_{safe_indicator}"
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）{indicator}现金流量表"
        self._save_df_to_material(
            df=df, ref_id=ref_id,
            description=description,
            source="AKshare API:Hithink",
            entity=entity,
            )
        header = f"[fetch_cashflow_table_material] 现金流量表（symbol={symbol}, indicator={indicator}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol, "indicator": indicator},
        )

    # ===================== 股东信息 =====================

    async def fetch_top10_float_shareholders_material(
        self,
        symbol: str,
        date: str,
    ) -> ToolResponse:
        """获取指定股票在某一报告期的十大流通股东，并保存表格结果到Material当中，返回Material标识ref_id。

        抓取指定 symbol 和 date 对应的所有流通股东信息，
        包括股东名称、股东性质、持股数量、持股比例、增减变动等。
        适用场景：分析个股流通股东结构、筹码集中度；跟踪机构、重要股东持股变动情况。

        Args:
            symbol (str):
                股票代码，例如 "000063"。
            date (str):
                财报发布季度最后一日，格式为 "YYYYMMDD"。
                2024 年的季度最后日分别为：20240331、20240630、20240930、20241231。
                2025 年的季度最后日分别为：20250331、20250630、20250930、20251231。
        """

        df = ak.stock_gdfx_free_top_10_em(symbol=add_exchange_prefix(symbol, "lower"), date=date)

        ref_id = f"{symbol}_top10_free_{date}_{int(time_module.time())}"
        time_point = fmt_yyyymmdd(date)
        time = {"point":time_point}
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）报告期{time_point}的十大流通股东"
        self._save_df_to_material(
            df=df, ref_id=ref_id,
            description=description,
            source="AKshare API:eastmoney",
            entity=entity,
            time=time,
            )

        header = f"[fetch_top10_float_shareholders_material] 十大流通股东（symbol={symbol}, date={date}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol, "date": date},
        )

    async def fetch_top10_shareholders_material(
        self,
        symbol: str,
        date: str,
    ) -> ToolResponse:
        """获取指定股票在某一报告期的十大股东（总股本口径），并保存表格结果到Material当中，返回Material标识ref_id。
        抓取指定 symbol 和 date 对应的股东名称、股份类型、
        持股数、持股比例及增减变动等信息，并保存。
        适用场景：分析公司控制权和股权结构；对比不同报告期十大股东持股变动情况。

        Args:
            symbol (str):
                股票代码，例如 "000063"。
            date (str):
                财报发布季度最后一日，格式为 "YYYYMMDD"。
                2024 年的季度最后日分别为：20240331、20240630、20240930、20241231。
                2025 年的季度最后日分别为：20250331、20250630、20250930、20251231。
        """
        df = ak.stock_gdfx_top_10_em(symbol=add_exchange_prefix(symbol, "lower"), date=date)
        ref_id = f"{symbol}_top10_{date}_{int(time_module.time())}"
        time_point = fmt_yyyymmdd(date)
        time = {"point":time_point}
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）报告期{time_point}的十大股东"
        self._save_df_to_material(
            df=df, ref_id=ref_id,
            description=description,
            source="AKshare API:eastmoney",
            entity=entity,
            time=time,
            )
        header = f"[fetch_top10_shareholders_material] 十大股东（symbol={symbol}, date={date}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol, "date": date},
        )

    async def fetch_main_shareholders_material(
        self,
        symbol: str,
    ) -> ToolResponse:
        """获取指定股票的主要股东信息，并保存表格结果到Material当中，返回Material标识ref_id。
        抓取所有历史披露的主要股东信息，
        包括股东名称、持股数量、持股比例、股本性质、截至日期、公告日期等。
        适用场景：分析公司历史上的主要股东变化；辅助研报中“股权结构与股东情况”章节的撰写。

        Args:
            symbol (str):
                股票代码，例如 "600004"。
        """
        df = ak.stock_main_stock_holder(stock=symbol)
        ref_id = f"{symbol}_main_holders_{int(time_module.time())}"

        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）的主要股东"
        self._save_df_to_material(
            df=df, ref_id=ref_id,
            description=description,
            source="AKshare API:Sina Finance",
            entity=entity,
            )
        header = f"[fetch_main_shareholders_material] 主要股东（symbol={symbol}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )

    async def fetch_shareholder_count_detail_material(
        self,
        symbol: str,
    ) -> ToolResponse:
        """获取指定股票的股东户数详情，并保存表格结果到Material当中，返回Material标识ref_id。
        获取指定 symbol 的全部历史数据，包括股东户数统计截止日、区间涨跌幅、股东户数本次/上次/增减及其比例、户均持股市值与数量、
        总市值、总股本及股本变动原因等。
        适用场景：分析股东户数与股价表现的关系；评估筹码集中度变化和市场参与者结构。

        Args:
            symbol (str):
                股票代码，例如 "000001"。
        """
        df = ak.stock_zh_a_gdhs_detail_em(symbol=symbol)

        ref_id = f"{symbol}_shareholder_count_detail_{int(time_module.time())}"

        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）股东户数详情"
        self._save_df_to_material(
            df=df, ref_id=ref_id,
            description=description,
            source="AKshare API:eastmoney",
            entity=entity,
            )
        header = f"[fetch_shareholder_count_detail_material] 股东户数详情（symbol={symbol}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )

    async def fetch_shareholder_change_material(
        self,
        symbol: str,
    ) -> ToolResponse:
        """获取指定股票的股东持股变动统计信息，并保存表格结果到Material当中，返回Material标识ref_id。
        抓取所有披露的股东持股变动记录，包括公告日期、变动股东、变动数量、交易均价、剩余股份总数、变动期间和变动途径等。
        适用场景： 跟踪重要股东和机构的减持 / 增持行为；分析股价波动背后的股东行为因素。

        Args:
            symbol (str):
                股票代码，例如 "688981"。
        """

        df = ak.stock_shareholder_change_ths(symbol=symbol)


        ref_id = f"{symbol}_shareholder_change_{int(time_module.time())}"

        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）股东持股变动"
        self._save_df_to_material(
            df=df, ref_id=ref_id,
            description=description,
            source="AKshare API:Hithink",
            entity=entity,
            )
        header = f"[fetch_shareholder_change_material] 股东持股变动（symbol={symbol}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )

    # ===================== 业务范围 =====================
    async def fetch_business_description_material(
        self,
        symbol: str,
    ) -> ToolResponse:
        """获取指定股票的主营业务介绍，并保存表格结果到Material当中，返回Material标识ref_id。
        抓取公司主营业务、产品类型、产品名称及经营范围等字段，
        适用场景：生成研报中“公司简介”“主营业务与商业模式”章节的基础描述；快速了解公司业务结构和核心产品。

        Args:
            symbol (str):
                股票代码，例如 "000066"。
        """
        df = ak.stock_zyjs_ths(symbol=symbol)
        data = df.iloc[0].to_dict()

        ref_id = f"{symbol}_business_description"
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）主营业务、产品介绍"
        self._save_df_to_material(
            df=data, ref_id=ref_id,
            description=description,
            source="AKshare API:Hithink",
            entity=entity,
            )

        header = f"[fetch_business_description_material] 主营介绍（symbol={symbol}）"
        return _build_tool_response_from_dict(
            data,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )

    async def fetch_business_composition_material(
        self,
        symbol: str,
    ) -> ToolResponse:
        """获取指定股票的主营构成数据，并保存表格结果到Material当中，返回Material标识ref_id。
        抓取按产品、地区等维度划分的主营收入、成本、利润、及对应比例和毛利率等历史数据。
        适用场景：分析公司按产品/地区划分的收入与利润结构；研报中“业务结构分析”“毛利率拆解”等章节的数据来源。

        Args:
            symbol (str):
                股票代码，例如 "000063"。
        """

        df = ak.stock_zygc_em(symbol=add_exchange_prefix(symbol, "upper"))

        ref_id = f"{symbol}_business_composition"

        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）主营构成、业务的收入与利润结构"
        self._save_df_to_material(
            df=df, ref_id=ref_id,
            description=description,
            source="AKshare API:eastmoney",
            entity=entity,
            )
        header = f"[fetch_business_composition_material] 主营构成（symbol={symbol}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )


def stock_news_em(keyword: str = "603777", page_idx=1) -> pd.DataFrame:
    """
    东方财富-个股新闻-最近 100 条新闻
    https://so.eastmoney.com/news/s?keyword=603777
    :param symbol: 股票代码
    :type symbol: str
    :return: 个股新闻
    :rtype: pandas.DataFrame
    """
    url = "http://search-api-web.eastmoney.com/search/jsonp"
    params = {
        "cb": "cb",
        "param": '{"uid":"",'
        + f'"keyword":"{keyword}"'
        + ',"type":["cmsArticleWebOld"],"client":"web","clientType":"web","clientVersion":"curr",'
        + '"param":{"cmsArticleWebOld":{"searchScope":"default","sort":"default","pageIndex":' + str(page_idx) + ','
        + '"pageSize":100,"preTag":"<em>","postTag":"</em>"}}}',
    }
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    r: requests.Response = requests.get(url, params=params, headers=headers)
    data_text = r.text
    data_json = json.loads(
        re.search(r'^\w+\((.*)\)$', data_text).group(1)
    )
    temp_df = pd.DataFrame(data_json["result"]["cmsArticleWebOld"])
    assert len(temp_df) > 0, "未获取到相关新闻，请调整关键词后重试。"
    temp_df["url"] = "http://finance.eastmoney.com/a/" + temp_df["code"] + ".html"
    temp_df.rename(
        columns={
            "date": "发布时间",
            "mediaName": "文章来源",
            "code": "-",
            "title": "新闻标题",
            "content": "新闻内容",
            "url": "新闻链接",
            "image": "-",
        },
        inplace=True,
    )
    temp_df["关键词"] = keyword
    temp_df = temp_df[
        [
            "关键词",
            "新闻标题",
            "新闻内容",
            "发布时间",
            "文章来源",
            "新闻链接",
        ]
    ]
    temp_df["新闻标题"] = (
        temp_df["新闻标题"]
        .str.replace(r"\(<em>", "", regex=True)
        .str.replace(r"</em>\)", "", regex=True)
    )
    temp_df["新闻标题"] = (
        temp_df["新闻标题"]
        .str.replace(r"<em>", "", regex=True)
        .str.replace(r"</em>", "", regex=True)
    )
    temp_df["新闻内容"] = (
        temp_df["新闻内容"]
        .str.replace(r"\(<em>", "", regex=True)
        .str.replace(r"</em>\)", "", regex=True)
    )
    temp_df["新闻内容"] = (
        temp_df["新闻内容"]
        .str.replace(r"<em>", "", regex=True)
        .str.replace(r"</em>", "", regex=True)
    )
    temp_df["新闻内容"] = temp_df["新闻内容"].str.replace(r"\u3000", "", regex=True)
    temp_df["新闻内容"] = temp_df["新闻内容"].str.replace(r"\r\n", " ", regex=True)
    return temp_df


