# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import akshare as ak
import fitz
import pandas as pd
import requests
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from htmldate import find_date

from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import LongTermMemoryStore
from ..utils.format import fmt_yyyymmdd
from ..utils.get_entity_info import get_entity_info
from ..utils.cite_id import cite_id as make_cite_id, id_part, url_part
from ..utils.task_date import normalize_compact_date
from ..utils.web_scraping import fetch_page_html, extract_text_and_images
from .material_tools import extract_keyword_context_snippets

_INDICATOR_CODE = {
    "按报告期": "report_period",
    "按年度": "annual",
    "按单季度": "single_quarter",
}

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
    cite_id: str,
    header: str,
    preview_rows: int = 10,
    extra_meta: Optional[Dict[str, Any]] = None,
    save: bool = True,
) -> ToolResponse:
    """统一构造 ToolResponse，包含预览文本和基础 metadata。"""
    preview_str, total_rows, used_rows, columns_names = _preview_df(df, preview_rows)  # 预览的时候返回前10行
    text = (
        f"{header} 共 {total_rows} 条记录，"
        f"以下为全部列名：\n"
        f"{columns_names}\n"
        f"以下为前 {used_rows} 行预览：\n"
        f"{preview_str}\n"
    )
    if save:
        text += f"Material 已写入 cite_id='{cite_id}'，可以通过 read_material 读取全部内容。\n\n"
    meta: Dict[str, Any] = {"cite_id": cite_id, "row_count": total_rows}
    if extra_meta:
        meta.update(extra_meta)
    return ToolResponse(
        content=[TextBlock(type="text", text=text)],
        metadata=meta,
    )


def _build_multi_material_response(
    tool_name: str,
    header: str,
    materials: list[dict[str, Any]],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> ToolResponse:
    """构造一次工具调用返回多个 material 的响应，便于 LLM agent 选择后续读取对象。"""
    lines = [
        f"[{tool_name}] {header}",
        "已分别写入以下 Materials；需要完整内容时，请使用 read_material(cite_id=...) 读取对应材料。",
    ]
    cite_ids: dict[str, str] = {}
    row_counts: dict[str, int] = {}

    for idx, item in enumerate(materials, 1):
        name = item["name"]
        cite_id = item["cite_id"]
        df = item["df"]
        preview_rows = item.get("preview_rows", 5)
        preview_str, total_rows, used_rows, columns_names = _preview_df(df, preview_rows)
        cite_ids[name] = cite_id
        row_counts[name] = total_rows
        lines.extend([
            "",
            f"{idx}. {name}",
            f"   cite_id: {cite_id}",
            f"   行数: {total_rows}",
            f"   列名: {columns_names}",
            f"   前 {used_rows} 行预览:",
            preview_str,
        ])

    meta: Dict[str, Any] = {
        "cite_ids": cite_ids,
        "row_counts": row_counts,
        "cite_id": next(iter(cite_ids.values()), ""),
    }
    if extra_meta:
        meta.update(extra_meta)
    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(lines))],
        metadata=meta,
    )


def _entity_label(symbol: str, entity: Dict[str, str] | None) -> str:
    if entity:
        return f"{entity['name']}（{entity['code']}）"
    return symbol


def add_exchange_prefix(symbol: str, type: str, with_dot: bool = False) -> str:
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
        elif code.startswith(("000", "001", "002", "003", "200", "080")) or code.startswith("30"):
            exchange = "SZ"

        # 北交所
        elif code.startswith(("43", "83", "87", "920")):
            exchange = "BJ"

        else:
            return code

    # 返回大小写
    if t == "upper":
        if with_dot:
            return prefix_upper[exchange] +"."+ code
        else:
            return prefix_upper[exchange] + code
    elif t == "lower":
        if with_dot:
            return prefix_lower[exchange] +"."+ code
        else:    
            return prefix_lower[exchange] + code
    else:
        raise ValueError("type 需为 'upper' 或 'lower'。")


def _normalize_tool_date(value: str | None, *, fallback: str | None = None) -> str:
    normalized = normalize_compact_date(value)
    if normalized:
        return normalized
    normalized_fallback = normalize_compact_date(fallback)
    if normalized_fallback:
        return normalized_fallback
    return datetime.now().strftime("%Y%m%d")

def _clip_end_date(end_date: str | None, cur_date: str) -> str:
    normalized_end = normalize_compact_date(end_date)
    return min(normalized_end, cur_date) if normalized_end else cur_date


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
            match = re.search(r"announcementId=(\d+)", link)
            if match:
                announcement_id = match.group(1)
    except Exception:
        pass

    if not announcement_id:
        return None

    if not isinstance(announce_date, str):
        announce_date = str(announce_date)
    date_str = announce_date.split()[0]
    date_str = date_str.replace(".", "-").replace("/", "-")
    return f"https://static.cninfo.com.cn/finalpage/{date_str}/{announcement_id}.PDF"


def _fetch_pdf_text(pdf_url: str, referer: str | None = None) -> str:
    if not pdf_url:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        if referer:
            headers["Referer"] = referer

        resp = requests.get(pdf_url, headers=headers, timeout=20)
        resp.raise_for_status()

        with fitz.open(stream=resp.content, filetype="pdf") as doc:
            texts: list[str] = []
            for page in doc:
                page_text = page.get_text().strip()
                if page_text:
                    texts.append(page_text)
        return "\n".join(texts)
    except Exception:
        return ""


async def _fetch_news_page_context(row, url_col: str, search_kws: list[str]) -> dict[str, str]:
    context = ""
    page_text = ""
    url = row.get(url_col) if url_col in row else None
    if isinstance(url, str) and url.startswith("http"):
        try:
            html_bytes = fetch_page_html(url)
            page_text, img_urls = extract_text_and_images(html_bytes, url)
            if page_text and search_kws:
                snippet_items = extract_keyword_context_snippets(
                    text=page_text,
                    keywords=search_kws,
                    context_chars=150,
                    merge_gap_chars=20,
                    ignore_case=True,
                )
                if snippet_items:
                    found_contexts = []
                    for item in snippet_items:
                        kws_str = ", ".join(item["keywords"])
                        found_contexts.append(f"[{kws_str}] {item['snippet']}")
                    context = " ｜ ".join(found_contexts)
        except Exception:
            pass
    return {
        "网页全文": page_text,
        "关键词上下文": context,
    }


class FinancialDataTools:
    def __init__(self, short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore) -> None:
        self.short_term = short_term
        self.long_term = long_term

    def _save_df_to_material(
            self,
            df: pd.DataFrame,
            cite_id: str,
            description: str,
            source: str,
            entity: Dict[str, str] | None = None,
            time: Dict[str, str] | None = None,
    ) -> int:
        """DataFrame/Dict 存入 short-term material（CSV/JSON），返回行数。"""
        if self.short_term is not None:
            self.short_term.save_material(cite_id=cite_id,
                                          content=df,
                                          description=description,
                                          source=source,
                                          entity=entity,
                                          time=time)
        return len(df)

    # ===================== 股价数据 =====================
    async def fetch_realtime_price_material(
        self,
        symbol: str | None = None
    ) -> ToolResponse:
        """获取沪深京 A 股实时行情数据，并保存表格结果到Material当中，返回Material标识cite_id。
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
        cite_id = make_cite_id("price_realtime", symbol or "all", unique=True)
        if symbol:
            description = f"{entity['name']}（{entity['code']}）股票实时行情数据（获取时间：{time_point}）"
        else:
            description = f"A股全市场股票实时行情数据（获取时间：{time_point}）"

        self._save_df_to_material(df=df,
                                cite_id=cite_id,
                                description=description,
                                source="AKshare API:eastmoney",
                                entity=entity,
                                time={"point":time_point},
                                )
        header = f"[fetch_realtime_price_material] 股价实时行情（symbol={symbol or 'ALL'}）"
        return _build_tool_response_from_df(
            df,
            cite_id=cite_id,
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
        """获取指定 A 股股票的历史行情（日/周/月），并保存表格结果到Material当中，返回Material标识cite_id。
        拉取指定股票在给定时间区间和周期上的历史行情数据（开盘价、收盘价、成交量、涨跌幅等），
        支持不复权、前复权和后复权数据。
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
        cur_date = _normalize_tool_date(self.short_term.current_date, fallback=os.getenv("CUR_DATE"))
        start_date = _normalize_tool_date(start_date)
        end_date = _clip_end_date(end_date, cur_date)
        assert pd.to_datetime(start_date, format="%Y%m%d") <= pd.to_datetime(end_date, format="%Y%m%d")
        source_info = "AKshare API:eastmoney"
        df = pd.DataFrame()

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )

            if df is None or df.empty:
                raise ValueError("akshare 返回数据为空")

        except Exception as e:
            # 2. akshare 失败，切换至 baostock 备用方案
            import baostock as bs
            source_info = "Baostock API"
            
            # --- 参数格式转换 ---
            # 转换股票代码 (baostock 需要 sh./sz./bj. 前缀)
            bs_symbol = add_exchange_prefix(symbol=symbol, type="lower",with_dot=True)
            # 转换日期格式 (YYYYMMDD -> YYYY-MM-DD)
            bs_start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            bs_end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

            # 转换周期参数
            period_map = {"daily": "d", "weekly": "w", "monthly": "m"}
            bs_period = period_map.get(period, "d")

            # 转换复权参数 (3:不复权, 2:前复权, 1:后复权)
            adjust_map = {"": "3", "qfq": "2", "hfq": "1"}
            bs_adjust = adjust_map.get(adjust, "3")

            # --- 请求 baostock ---
            bs.login()
            rs = bs.query_history_k_data_plus(
                bs_symbol,
                "date,open,high,low,close,volume,amount,pctChg,turn", # 提取核心字段
                start_date=bs_start,
                end_date=bs_end,
                frequency=bs_period,
                adjustflag=bs_adjust
            )
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            bs.logout()
            
            df = pd.DataFrame(data_list, columns=rs.fields)

            # --- 数据清洗与列名对齐 ---
            if not df.empty:
                # 强转数值类型 (baostock 默认返回全字符串)
                numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pctChg", "turn"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 重命名列名，与 akshare 的标准中文列名保持完全一致
                rename_map = {
                    "date": "日期", "open": "开盘", "high": "最高", "low": "最低",
                    "close": "收盘", "volume": "成交量", "amount": "成交额",
                    "pctChg": "涨跌幅", "turn": "换手率"
                }
                df.rename(columns=rename_map, inplace=True)            

                # ================= 新增：利用Baostock数据反推历史流通市值 =================
                mask = df["换手率"] > 0
                # 注意单位差异：Baostock成交量单位为“股”，换手率为百分比
                floating_shares = df.loc[mask, "成交量"] / (df.loc[mask, "换手率"] / 100)
                df.loc[mask, "流通市值(元)"] = floating_shares * df.loc[mask, "收盘"]
                df.loc[mask, "流通市值(亿元)"] = (df.loc[mask, "流通市值(元)"] / 1e8).round(2)
                # =========================================================================
                

        # ================= 新增：自动计算统计摘要（最高/最低极值） =================
        summary_info = ""
        if not df.empty and "最低" in df.columns and "最高" in df.columns and "日期" in df.columns:
            period_min = df["最低"].min()
            period_max = df["最高"].max()
            
            # 找到极值对应的日期 (iloc[0] 防止有多个相同极值天数)
            min_date = df.loc[df["最低"] == period_min, "日期"].iloc[0]
            max_date = df.loc[df["最高"] == period_max, "日期"].iloc[0]
            
            summary_info = (
                f"\n【系统自动计算的行情摘要】\n"
                f"在此数据区间（{start_date} ~ {end_date}）内：\n"
                f"期间最高价: {period_max} 元 (发生在 {max_date})\n"
                f"期间最低价: {period_min} 元 (发生在 {min_date})\n"
                f"提示：如果你需要寻找过去这段时间的最高/最低价，请直接使用上述精确数据，无需再去阅读长表格。\n"
            )
        # =========================================================================

        period_code = id_part(period, max_len=12, fallback="daily")
        adjust_code = id_part(adjust, max_len=8, fallback="raw")
        cite_id = make_cite_id("price_history", symbol, f"{start_date}-{end_date}", period_code, adjust_code)
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        time_range = {"start": fmt_yyyymmdd(start_date),"end":fmt_yyyymmdd(end_date)}
        if entity:
            description = f"{entity['name']}（{entity['code']}）股票历史行情数据（{fmt_yyyymmdd(start_date)}~{fmt_yyyymmdd(end_date)}）"
        else:
            description = f"{symbol} 股票历史行情数据（{fmt_yyyymmdd(start_date)}~{fmt_yyyymmdd(end_date)}）"
        description += summary_info
        self._save_df_to_material(df=df,
                                    cite_id=cite_id,
                                    description=description,
                                    source=source_info,
                                    entity=entity,
                                    time=time_range)

        header = (
            f"{summary_info}"
            f"[fetch_history_price_material] {symbol} {period} 股价历史行情 "
            f"{start_date}~{end_date} adjust='{adjust or '无'}'"
        )
        return _build_tool_response_from_df(
            df,
            cite_id=cite_id,
            header=header,
            extra_meta={
                "symbol": symbol,
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "adjust": adjust,
            },
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
        """获取指定股票的信息披露公告，并保存表格结果到Material当中，返回Material标识cite_id。
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
                公告类别，取值仅限于如下选择之一：
                - "年报"、"半年报"、"一季报"、"三季报"、"业绩预告"、"权益分派"、
                "董事会"、"监事会"、"股东大会"、"日常经营"、"公司治理"、"中介报告"、
                "首发"、"增发"、"股权激励"、"配股"、"解禁"、"公司债"、"可转债"、
                "其他融资"、"股权变动"、"补充更正"、"澄清致歉"、"风险提示"、
                "特别处理和退市"、"退市整理期"。
                - 为空字符串时：不按类别过滤，返回全部信息披露公告。

        """
        cur_date = _normalize_tool_date(self.short_term.current_date, fallback=os.getenv("CUR_DATE"))
        start_date = _normalize_tool_date(start_date)
        end_date = _clip_end_date(end_date, cur_date)

        try:
            assert pd.to_datetime(start_date) <= pd.to_datetime(end_date), "start_date 晚于当前时间，请重新设置"
            assert category in {'年报', '半年报', '一季报', '三季报', '业绩预告', '权益分派',
                                '董事会', '监事会', '股东大会', '日常经营', '公司治理', '中介报告',
                                '首发', '增发', '股权激励', '配股', '解禁', '公司债', '可转债', '其他融资',
                                '股权变动', '补充更正', '澄清致歉', '风险提示', '特别处理和退市', '退市整理期', '',
                                None}, f'category 设置错误，不支持"{category}"'
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
                f"[fetch_disclosure_material] {'检索结果为空' if isinstance(e, KeyError) else e}。\n"
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

        # 2. 遍历 df 行，构造 PDF URL 并抽文本，为每个公告保存为独立文件
        df["cite_id"] = ""
        entity = get_entity_info(long_term=self.long_term, text=symbol)

        for idx, row in df.iterrows():
            link = row.get("公告链接")
            announce_date = row.get("公告时间")
            announce_title = row.get("公告标题", "未命名公告")
            pdf_url = _build_pdf_url(link, announce_date)
            text = _fetch_pdf_text(pdf_url, referer=link)

            if text:  # 只保存有内容的公告
                # 为每个公告创建唯一的 cite_id
                announce_date = fmt_yyyymmdd(str(announce_date))
                date_only = announce_date.replace("-", "")
                disclosure_cite_id = make_cite_id("disclosure", symbol, date_only, f"{idx + 1:02d}")

                # 创建详细的描述，包含工具名、源URL等信息
                description_parts = [
                    "fetch_disclosure_material",  # 工具名
                    f"source={entity['name']} ({entity['code']}) 信息披露公告",
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
                    cite_id=disclosure_cite_id,
                    content=text,
                    description=description,
                    source="AKshare API:CNINFO",
                    entity=entity,
                    time={"point": str(announce_date)},
                )
                df.at[idx, "cite_id"] = disclosure_cite_id

        # 3. 保留每行公告对应的 cite_id，PDF 提取失败的行保持为空
        if "公告链接" in df.columns:
            df = df.drop(columns=["公告链接"])
        df = df[['cite_id', '公告标题']].set_index("cite_id")

        description = f"{entity['name']}（{entity['code']}）股票"
        parts = [x for x in (keyword, category) if x]
        if parts:
            description += " " + " ".join(parts)
        description = description + "信息披露公告"
        header = (
            f"[fetch_disclosure_material] 信息披露公告（symbol={symbol}, market={market}, "
            f"category={category or '全部'}, {start_date}~{end_date}）。"
            f"各disclosure按cite_id单独保存到了本地，请根据需要使用retrieve_local_material或read_material工具根据对应cite_id读取。"
        )
        return _build_tool_response_from_df(
            df,
            cite_id="",
            header=header,
            preview_rows=len(df),
            extra_meta={
                "symbol": symbol,
                "market": market,
                "keyword": keyword,
                "category": category,
                "start_date": start_date,
                "end_date": end_date,
            },
            save=False,
        )
    # ===================== 财务报表 =====================
    async def fetch_balance_sheet_material(
            self,
            symbol: str,
            indicator: str = "按报告期",
    ) -> ToolResponse:
        """    # 获取指定股票的资产负债表数据，并保存表格结果到Material当中，返回Material标识cite_id。
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

        indicator_code = _INDICATOR_CODE.get(indicator, id_part(indicator, max_len=16, fallback="report_period"))
        cite_id = make_cite_id("financial_balance", symbol, indicator_code)
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）{indicator}资产负债表"
        self._save_df_to_material(
            df=df, cite_id=cite_id,
            description=description,
            entity=entity,
            source="AKshare API:Hithink",
        )
        header = f"[fetch_balance_sheet_material] 资产负债表（symbol={symbol}, indicator={indicator}）"

        return _build_tool_response_from_df(
            df,
            cite_id=cite_id,
            header=header,
            extra_meta={"symbol": symbol, "indicator": indicator},
        )

    async def fetch_profit_table_material(
            self,
            symbol: str,
            indicator: str = "按报告期",
    ) -> ToolResponse:
        """获取指定股票的利润表数据，并保存表格结果到Material当中，返回Material标识cite_id。
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

        indicator_code = _INDICATOR_CODE.get(indicator, id_part(indicator, max_len=16, fallback="report_period"))
        cite_id = make_cite_id("financial_profit", symbol, indicator_code)
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）{indicator}利润表"
        self._save_df_to_material(
            df=df, cite_id=cite_id,
            description=description,
            source="AKshare API:Hithink",
            entity=entity,
        )
        header = f"[fetch_profit_table_material] 利润表（symbol={symbol}, indicator={indicator}）"
        return _build_tool_response_from_df(
            df,
            cite_id=cite_id,
            header=header,
            extra_meta={"symbol": symbol, "indicator": indicator},
        )

    async def fetch_cashflow_table_material(
            self,
            symbol: str,
            indicator: str = "按报告期",
    ) -> ToolResponse:
        """获取指定股票的现金流量表数据，并保存表格结果到Material当中，返回Material标识cite_id。
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

        indicator_code = _INDICATOR_CODE.get(indicator, id_part(indicator, max_len=16, fallback="report_period"))
        cite_id = make_cite_id("financial_cashflow", symbol, indicator_code)
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        description = f"{entity['name']}（{entity['code']}）{indicator}现金流量表"
        self._save_df_to_material(
            df=df, cite_id=cite_id,
            description=description,
            source="AKshare API:Hithink",
            entity=entity,
        )
        header = f"[fetch_cashflow_table_material] 现金流量表（symbol={symbol}, indicator={indicator}）"
        return _build_tool_response_from_df(
            df,
            cite_id=cite_id,
            header=header,
            extra_meta={"symbol": symbol, "indicator": indicator},
        )

    # ===================== 股东信息 =====================

    async def fetch_top_shareholders_material(
            self,
            symbol: str,
            date: str,
    ) -> ToolResponse:
        """获取指定股票在某一报告期的十大股东和十大流通股东，并分别保存为本地 Material。
        返回两类股东口径：
        - 十大股东：总股本口径，适合分析控制权和整体股权结构。
        - 十大流通股东：流通股本口径，适合分析流通筹码和二级市场持股结构。

        Args:
            symbol (str):
                股票代码，例如 "000063"。
            date (str):
                财报发布季度最后一日，格式为 "YYYYMMDD"。
                2024 年的季度最后日分别为：20240331、20240630、20240930、20241231。
                2025 年的季度最后日分别为：20250331、20250630、20250930、20251231。
        """
        prefixed_symbol = add_exchange_prefix(symbol, "lower")
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        time_point = fmt_yyyymmdd(date)
        time = {"point": time_point}
        entity_text = _entity_label(symbol, entity)
        top10_df = ak.stock_gdfx_top_10_em(symbol=prefixed_symbol, date=date)
        top10_cite_id = make_cite_id("shareholder_top10", symbol, date)
        self._save_df_to_material(
            df=top10_df,
            cite_id=top10_cite_id,
            description=f"{entity_text}报告期{time_point}的十大股东（总股本口径）",
            source="AKshare API:eastmoney",
            entity=entity,
            time=time,
        )

        top10_float_df = ak.stock_gdfx_free_top_10_em(symbol=prefixed_symbol, date=date)
        top10_float_cite_id = make_cite_id("shareholder_float_top10", symbol, date)
        self._save_df_to_material(
            df=top10_float_df,
            cite_id=top10_float_cite_id,
            description=f"{entity_text}报告期{time_point}的十大流通股东（流通股本口径）",
            source="AKshare API:eastmoney",
            entity=entity,
            time=time,
        )

        materials = [
            {
                "name": "十大股东（总股本口径）",
                "cite_id": top10_cite_id,
                "df": top10_df,
            },
            {
                "name": "十大流通股东（流通股本口径）",
                "cite_id": top10_float_cite_id,
                "df": top10_float_df,
            },
        ]
        return _build_multi_material_response(
            tool_name="fetch_top_shareholders_material",
            header=f"{entity_text} {time_point} 十大股东与十大流通股东",
            materials=materials,
            extra_meta={
                "symbol": symbol,
                "date": date,
                "time": time,
            },
        )

    async def fetch_shareholder_material(
            self,
            symbol: str,
    ) -> ToolResponse:
        """获取指定股票的主要股东、股东户数详情、股东持股变动，并分别保存为本地 Material。
        返回三类股东信息：
        - 主要股东：历史主要股东明细，适合分析股权结构。
        - 股东户数详情：股东户数、户均持股、市值等时间序列，适合分析筹码集中度。
        - 股东持股变动：重要股东增减持事件流水，适合分析股东行为变化。

        Args:
            symbol (str):
                股票代码，例如 "600004"。
        """
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        entity_text = _entity_label(symbol, entity)
        main_df = ak.stock_main_stock_holder(stock=symbol)
        main_cite_id = make_cite_id("shareholder_main", symbol)
        self._save_df_to_material(
            df=main_df,
            cite_id=main_cite_id,
            description=f"{entity_text}主要股东",
            source="AKshare API:Sina Finance",
            entity=entity,
        )

        count_df = ak.stock_zh_a_gdhs_detail_em(symbol=symbol)
        count_cite_id = make_cite_id("shareholder_count", symbol)
        self._save_df_to_material(
            df=count_df,
            cite_id=count_cite_id,
            description=f"{entity_text}股东户数详情",
            source="AKshare API:eastmoney",
            entity=entity,
        )

        change_df = ak.stock_shareholder_change_ths(symbol=symbol)
        change_cite_id = make_cite_id("shareholder_change", symbol)
        self._save_df_to_material(
            df=change_df,
            cite_id=change_cite_id,
            description=f"{entity_text}股东持股变动",
            source="AKshare API:Hithink",
            entity=entity,
        )

        materials = [
            {
                "name": "主要股东",
                "cite_id": main_cite_id,
                "df": main_df,
            },
            {
                "name": "股东户数详情",
                "cite_id": count_cite_id,
                "df": count_df,
            },
            {
                "name": "股东持股变动",
                "cite_id": change_cite_id,
                "df": change_df,
            },
        ]
        return _build_multi_material_response(
            tool_name="fetch_shareholder_material",
            header=f"{entity_text} 股东信息汇总",
            materials=materials,
            extra_meta={"symbol": symbol},
        )

    # ===================== 业务范围 =====================
    async def fetch_business_material(
            self,
            symbol: str,
    ) -> ToolResponse:
        """获取指定股票的主营业务介绍和主营构成，并分别保存为本地 Material。
        返回两类业务信息：
        - 主营业务介绍：公司主营业务、产品类型、产品名称及经营范围。
        - 主营构成：按产品、地区等维度划分的收入、成本、利润和毛利率。

        Args:
            symbol (str):
                股票代码，例如 "000066"。
        """
        entity = get_entity_info(long_term=self.long_term, text=symbol)
        entity_text = _entity_label(symbol, entity)

        description_df = ak.stock_zyjs_ths(symbol=symbol)
        description_cite_id = make_cite_id("business_description", symbol)
        self._save_df_to_material(
            df=description_df,
            cite_id=description_cite_id,
            description=f"{entity_text}主营业务、产品介绍",
            source="AKshare API:Hithink",
            entity=entity,
        )

        composition_df = ak.stock_zygc_em(symbol=add_exchange_prefix(symbol, "upper"))
        composition_cite_id = make_cite_id("business_composition", symbol)
        self._save_df_to_material(
            df=composition_df,
            cite_id=composition_cite_id,
            description=f"{entity_text}主营构成、业务收入与利润结构",
            source="AKshare API:eastmoney",
            entity=entity,
        )

        materials = [
            {
                "name": "主营业务介绍",
                "cite_id": description_cite_id,
                "df": description_df,
            },
            {
                "name": "主营构成",
                "cite_id": composition_cite_id,
                "df": composition_df,
            },
        ]
        return _build_multi_material_response(
            tool_name="fetch_business_material",
            header=f"{entity_text} 主营业务介绍与主营构成",
            materials=materials,
            extra_meta={"symbol": symbol},
        )
    # 适用场景：为个股研报生成“新闻动态”“舆情分析”等部分提供原始素材；需要快速获取近期与某股票相关的新闻列表。
    # ===================== 金融新闻 =====================
    async def fetch_stock_news_material(
                self,
                symbol: str,
                start_date: str,
                end_date: str | None = None,
                query: str = "",
                latest_num: int = 50,
        ) -> ToolResponse:
        """获取指定个股的新闻资讯数据，并保存表格结果到Material当中，返回Material标识cite_id，以及关键词附近的上下文。
        相关的最新新闻资讯（默认为限定时间范围内最近约 50 条），包括新闻标题、内容摘要、发布时间、来源和链接等，
        适用场景：为个股研报生成“新闻动态”“舆情分析”等部分提供原始素材；需要快速获取近期与某股票相关的新闻列表。

        Args:
            symbol (str):
                沪深京 A 股股票代码（不带市场标识），例如 "000001"；为空字符串时不做股票过滤。
            query (str):
                新闻检索词，例如 "营业收入 2024"、"AI存储 产能"；可以为空。
            start_date (str):
                过滤限定日期之后的新闻，格式为 "YYYYMMDD"，例如 "20230101"。必需参数。
            end_date (str | None):
                过滤限定日期之前的新闻，格式为 "YYYYMMDD"，例如 "20231231"。如果为 None，则使用当前日期。
            latest_num (int):
                限定时间范围内最近的新闻条数，默认100
        """
        cur_date = _normalize_tool_date(self.short_term.current_date, fallback=os.getenv("CUR_DATE"))
        start_date = _normalize_tool_date(start_date)
        end_date = _clip_end_date(end_date, cur_date)
        assert pd.to_datetime(start_date, format="%Y%m%d") <= pd.to_datetime(end_date, format="%Y%m%d")
        query = (query or "").strip()
        query_part = id_part(query, max_len=24, fallback="all")
        cite_id = make_cite_id(
            "stock_news",
            symbol or "all",
            query_part,
            f"{start_date}-{end_date}",
            f"n{latest_num}",
            hash_parts=(query,) if query else (),
        )
        start_date, end_date = pd.to_datetime(start_date, format="%Y%m%d"), pd.to_datetime(end_date, format="%Y%m%d")

        query_terms = [term.strip() for term in re.split(r"\s+", query) if term.strip()]
        keyword = query
        if symbol is None or symbol == "":
            entity = None
            description = f"股票新闻资讯 {query}"
        else:
            entity = get_entity_info(long_term=self.long_term, text=symbol)
            base_keyword = entity["name"] if entity else symbol
            keyword = " ".join([base_keyword] + query_terms).strip()
            description = (f"{entity['name']}（{entity['code']}）" if entity else f"{symbol}") + f"股票新闻资讯 {keyword}"
        dfs = []
        for page_idx in range(1, 25):
            df = stock_news_em(keyword=keyword, page_idx=page_idx)
            dfs.append(
                df[(pd.to_datetime(df['发布时间']) >= start_date) & (pd.to_datetime(df['发布时间']) <= end_date)])
            if sum([len(_df) for _df in dfs]) > latest_num:
                break
        df = pd.concat(dfs)
        df.sort_values("发布时间", inplace=True, ascending=False)

        # =====================================================================
        # === 新增：1. 获取网页内容； 2. 检索关键词获取上下文 ===
        # =====================================================================
        fetched_results: list[dict[str, str]] = []
        if not df.empty:
            # 确定检索词：优先使用传入的原始 keyword，若无则使用股票名称
            search_target = keyword
            search_kws = search_target.split() if search_target else []
            
            url_col = '新闻链接' if '新闻链接' in df.columns else ('文章链接' if '文章链接' in df.columns else 'url')

            # 并发执行以加速网页抓取和检索流程
            tasks = [
                _fetch_news_page_context(row, url_col, search_kws)
                for _, row in df.iterrows()
            ]
            fetched_results = await asyncio.gather(*tasks)
            df['网页全文'] = [item.get("网页全文", "") for item in fetched_results]
            
        # =====================================================================

        self._save_df_to_material(
            df=df,
            cite_id=cite_id,
            source="AKshare API:eastmoney",
            entity=entity,
            description=description,
        )
        header = f"[fetch_stock_news_material] 股票新闻资讯"
        df_for_response = df.drop(columns=["网页全文"], errors="ignore")
        response = _build_tool_response_from_df(
            df=df_for_response,
            cite_id=cite_id,
            header=header,
            extra_meta={"symbol": symbol} if symbol else {"keyword": keyword},
            save=True
        )
        if fetched_results:
            context_lines = ["", "以下为关键词上下文摘录："]
            for idx, item in enumerate(fetched_results, 1):
                context = item.get("关键词上下文", "").strip()
                if not context:
                    continue
                context_lines.append(f"{idx}. {context}")
            if len(context_lines) > 2:
                response.content[0]["text"] += "\n" + "\n".join(context_lines)
        return response

    async def fetch_url_page_text(self, url: str, symbol: str | None = None) -> ToolResponse:
        """返回url对应网页的文本结果，如果不为空则保存到本地。
        Args:
            url (str):
                网页地址。
            symbol (str | None):
                新闻对应股票代码或名称。如果无法判断，可以不提供。
        """
        bytes = fetch_page_html(url)
        page_text, img_urls = extract_text_and_images(bytes, url)
        page_text = page_text or ""
        if page_text:
            # 保存网页提取的文本为单独的文件
            entity = get_entity_info(long_term=self.long_term, text=symbol or page_text)
            domain = urlparse(url).netloc
            if domain.startswith("www."):
                domain = domain[4:]
            cite_id = make_cite_id(
                "web_page",
                str(entity["code"]) if entity else "page",
                url_part(url),
                hash_parts=(url,),
                max_part_len=56,
            )

            published_date = None
            try:
                published_date = find_date(
                    bytes,
                    url=url,
                    original_date=True,
                    extensive_search=True,
                    deferred_url_extractor=True,
                )
            except Exception:
                published_date = None

            desc = ""
            time = None
            if published_date:
                published_date = fmt_yyyymmdd(published_date)
                desc = desc + f"网页发布时间：{published_date} "
                time = {"point": published_date}
            if entity:
                desc = desc + f"发布关于{entity['name']}（{entity['code']}）的内容:"

            desc = desc + page_text[:50]

            self.short_term.save_material(
                cite_id=cite_id,
                content=page_text,
                description=desc,
                source=f"web search（来源：{domain}）",
                entity=entity,
                time=time
            )

            text_block: TextBlock = {
                "type": "text",
                "text": (
                    f"[fetch_url_page_text] url:{url}对应的网页文本结果获取如下：\n"
                    f"Material 已写入 cite_id='{cite_id}' TXT 格式）\n"
                ),
            }
            return ToolResponse(content=[text_block], metadata={"cite_id": cite_id})
        else:
            text = f"[fetch_url_page_text] url:{url}对应的网页文本为空。"
            return ToolResponse(
                content=[
                    TextBlock(
                    type="text",
                    text=text,
                ),
            ],
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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://so.eastmoney.com/"
    }

    try:
        r: requests.Response = requests.get(url, params=params, headers=headers)
        data_text = r.text
        data_json = json.loads(
            re.search(r'^\w+\((.*)\)$', data_text).group(1)
        )
        temp_df = pd.DataFrame(data_json["result"]["cmsArticleWebOld"])
        assert len(temp_df) > 0, "当前时间范围内不存在相关新闻，请调整时间区间或关键词后重试。"
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
    except Exception as e:
        return pd.DataFrame()


