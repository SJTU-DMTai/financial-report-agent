# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Callable, Any, Dict, Union

import io
import time
import akshare as ak
import pandas as pd
import re
from urllib.parse import urlparse, parse_qs
import requests
import fitz
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit
from ..memory.short_term import ShortTermMemoryStore, MaterialType
import json

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


class MaterialTools:
    def __init__(self, short_term: Optional[ShortTermMemoryStore] = None) -> None:
        self.short_term = short_term

    def read_material(
            self,
            ref_id: str,
            start_index: int | None = None,
            end_index: int | None = None,
            query_key: str | None = None,
    ) -> ToolResponse:
        """
        统一读取material。支持读取全文、通过参数、按行/条目截取其中部分。
        - 表格（MaterialType.TABLE）：按行号切片，可选按列筛选。
        - 文本（MaterialType.TEXT）：按行号切片。
        - JSON（MaterialType.JSON）：按列表索引切片，可通过 query_key 获取特定条目内容。

        如果start_index， end_index，query_key都为空表示读取全文。
        
        Args:
            ref_id (str): Material 的唯一标识 ID。
            start_index (int | None): 
                - 对于表格：起始行号（包含）。
                - 对于文本/Markdown：起始行号（包含）。
                - 对于 JSON list：起始条目索引（包含）。
                - 如果为空表示从第0行开始。
            end_index (int | None): 
                - 对于表格：结束行号（不包含）。
                - 对于文本/Markdown：结束行号（不包含）。
                - 对于 JSON list：结束条目索引（不包含）。
                - 如果为空表示到最后一条结束。
            query_key (str | None):
                - 对于 JSON list：可选，用于对每个条目提取该字段。
                - 对于表格：可选，用于筛选特定列（如 "Date,Close"）。
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
            # 2. 策略分发 (Strategy Dispatch)
            if meta.m_type == MaterialType.TABLE:
                return self._read_table_impl(ref_id, start_index, end_index, query_key)
            elif meta.m_type == MaterialType.TEXT:
                return self._read_text_impl(ref_id, start_index, end_index)
            elif meta.m_type == MaterialType.JSON:
                return self._read_json_impl(ref_id, start_index, end_index, query_key)
            else:  
                return ToolResponse(
                    content=[TextBlock(type="text", text=f"[read_material]不支持的文件类型: {meta.m_type}")],
                    metadata={"ref_id": ref_id}
                )
        except Exception as e:
            return ToolResponse(
            content=[TextBlock(type="text", text=f"[read_material]读取失败: {str(e)}")],
            metadata={"ref_id": ref_id}
            )


    # ========== 内部函数（不注册为 tool） ==========

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
        
        sliced_df = df.iloc[start:end]
        
        # 转换为 Markdown 或 CSV 字符串给 LLM
        preview_str = sliced_df.to_markdown(index=False, disable_numparse=True)
        
        text = (f"[read_material] ID: {ref_id}\n"
                f"完整 material 共 {total_rows} 行。已读取范围: 行 [{start}, {end})。\n"
                f"内容:\n{preview_str}")
                
        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"ref_id": ref_id, "type": "table", "rows": len(sliced_df)}
        )

    def _read_text_impl(self, ref_id, start_line, end_line):
        # 适用于 .txt, .md
        content = self.short_term.load_material(ref_id) # 返回 str
        lines = content.split('\n')
        total_lines = len(lines)

        start = start_line if start_line is not None else 0
        end = end_line if end_line is not None else total_lines

        # 截取
        sliced_lines = lines[start:end]
        preview_str = "\n".join(sliced_lines)

        text = (f"[read_material] ID: {ref_id}\n"
                f"完整 material 共 {total_lines} 行。已读取范围: 行 [{start}, {end})。\n"
                f"内容:\n{preview_str}")
        
        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"ref_id": ref_id, "type": "text", "lines": len(sliced_lines)}
        )

    def _read_json_impl(
        self,
        ref_id: str,
        start_index: int | None,
        end_index: int | None,
        key_path: str | None,
    ) -> ToolResponse:
        data = self.short_term.load_material(ref_id)  # 返回 dict 或 list，目前只有搜索结果为json list

        # if isinstance(data, list):
        n = len(data)
        # 处理切片：start_index/end_index 控制“第几条”
        start = start_index if start_index is not None else 0
        end = end_index if end_index is not None else n

        # 边界保护
        start = max(0, min(start, n))
        end = max(start, min(end, n))

        sliced = data[start:end]

        
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

        # 防止过长
        # if len(json_str) > 4000:
        #     json_str = json_str[:4000] + "\n... (content truncated)"

        text = (
            f"[read_material] ID: {ref_id}\n"
            f"完整 material 共 {n} 条。已读取 JSON 列表范围 [{start}, {end})\n"
            f"内容:\n{json_str}"
        )

        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"ref_id": ref_id, "type": "json_list"}
        )
    def _save_df_to_material(
            self,
            df: pd.DataFrame,
            ref_id: str
    ) -> int:
        """DataFrame 存入 short-term material（CSV），返回行数。"""
        if self.short_term is not None:
            self.short_term.save_material(ref_id=ref_id, 
                                          content=df, 
                                          source="AKshare API")
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

        df = ak.stock_zh_a_spot_em()
        if symbol is not None:
            # 文档中 "代码" 列为股票代码
            df = df[df["代码"] == symbol]

            ref_id = f"{symbol or 'all'}_realtime_price_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
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
            period: str = "daily",
            start_date: str = "20100101",
            end_date: str = "20991231",
            adjust: str = "",       
    ) -> ToolResponse:
        """获取指定 A 股股票的历史行情（日/周/月），并保存表格结果到Material当中，返回Material标识ref_id。
        拉取指定股票在给定时间区间和周期上的历史行情数据（开盘价、收盘价、成交量、涨跌幅等），
        支持不复权、前复权和后复权数据，并将结果保存。
        适用场景：生成个股 K 线、收益率曲线、回测信号等历史行情分析；作为生成研报中“股价表现”“历史走势”等章节的基础数据。

        Args:
            symbol (str):
                沪深京 A 股股票代码（不带市场标识），例如 "000001"。
            period (str):
                支持的取值：
                - "daily": 日频数据；
                - "weekly": 周频数据；
                - "monthly": 月频数据。
            start_date (str):
                历史行情起始日期，格式为 "YYYYMMDD"，例如 "20100101"。
            end_date (str):
                历史行情结束日期，格式为 "YYYYMMDD"，例如 "20251231"。
            adjust (str):
                复权方式，支持的取值：
                - "": 不复权（默认）；
                - "qfq": 前复权；
                - "hfq": 后复权。

        """
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )


        ref_id = f"{symbol}_history_price_{start_date}_{end_date}_{int(time.time())}"

        self._save_df_to_material(df, ref_id)

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
    ) -> ToolResponse:
        """获取指定个股的新闻资讯数据，并保存表格结果到Material当中，返回Material标识ref_id。
        相关的最新新闻资讯（默认为当日最近约 100 条），包括新闻标题、内容摘要、发布时间、来源和链接等，
        适用场景：为个股研报生成“新闻动态”“舆情分析”等部分提供原始素材；需要快速获取近期与某股票相关的新闻列表。

        Args:
            symbol (str):
                个股新闻检索关键词，通常为股票代码，例如 "603777"；

        """
        df = ak.stock_news_em(symbol=symbol)

        ref_id = f"{symbol}_news_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_stock_news_material] 个股新闻（symbol={symbol}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )

    async def fetch_disclosure_material(
            self,
            symbol: str,
            market: str = "沪深京",
            keyword: str = "",
            category: str = "",
            start_date: str = "20000101",
            end_date: str = "20991231",
    ) -> ToolResponse:
        """获取指定股票的信息披露公告，并保存表格结果到Material当中，返回Material标识ref_id。
        抓取指定 symbol 在给定时间区间内的各类信息披露公告，
        可按市场、公告类别和关键词进行过滤，并将结果保存为表格。
        适用场景：生成研报中的“公司公告梳理”“信息披露情况”章节；快速定位某段时间内的年报、季报、重大事项、股权变动等公告列表。

        Args:
            symbol (str):
                股票代码，例如 "000001"。
            market (str):
                市场类型，支持的取值包括：
                - "沪深京"（默认）、"港股"、"三板"、"基金"、"债券"、"监管"、"预披露"。
            keyword (str):
                公告搜索关键词，例如 "股权激励"、"增发"；为空字符串时不做关键词过滤。
            category (str):
                公告类别，支持的取值包括（示例）：
                - "年报"、"半年报"、"一季报"、"三季报"、"业绩预告"、"权益分派"、
                "董事会"、"监事会"、"股东大会"、"日常经营"、"公司治理"、"中介报告"、
                "首发"、"增发"、"股权激励"、"配股"、"解禁"、"公司债"、"可转债"、
                "其他融资"、"股权变动"、"补充更正"、"澄清致歉"、"风险提示"、
                "特别处理和退市"、"退市整理期"。
                - 为空字符串时：不按类别过滤，返回全部信息披露公告。
            start_date (str):
                公告起始日期，格式为 "YYYYMMDD"，例如 "20230618"。
            end_date (str):
                公告结束日期，格式为 "YYYYMMDD"，例如 "20231219"。

        """


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
                f"[fetch_disclosure_material] 信息披露公告搜索结果为空，"
                f"建议修改或放宽搜索条件（symbol={symbol}, market={market}, keyword={keyword}, "
                f"category={category or '全部'}, {start_date}~{end_date}）"
            )
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=text,
                    ),
                ],
            )

        
        df = df.head(10) # 避免获取的公告数量过多取前10条，后续可以改成按照某些条件排序取前10条

        # 2. 遍历 df 行，构造 PDF URL 并抽文本
        texts: list[str] = []
        for _, row in df.iterrows():
            link = row.get("公告链接")
            announce_date = row.get("公告时间")
            pdf_url = _build_pdf_url(link, announce_date)
            text = _fetch_pdf_text(pdf_url, referer=link)
            
            if len(text) > 50000:
                text = text[:50000]
                text += "\n...[内容过长，已截断]"

            texts.append(text)

        # 3. 新增「公告」列，删除「公告链接」列
        df["公告"] = texts
        if "公告链接" in df.columns:
            df = df.drop(columns=["公告链接"])
        ref_id = f"{symbol}_disclosure_{category or 'all'}_{int(time.time())}"



        self._save_df_to_material(df, ref_id)
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
        ref_id = f"{symbol}_balance_{safe_indicator}_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
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
        ref_id = f"{symbol}_profit_{safe_indicator}_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
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
        ref_id = f"{symbol}_cashflow_{safe_indicator}_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
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

        ref_id = f"{symbol}_top10_free_{date}_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
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
        ref_id = f"{symbol}_top10_{date}_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_top10_shareholders_material] 十大股东（symbol={symbol}, date={date}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol, "date": date},
        )

    async def fetch_main_shareholders_material(
            self,
            stock: str,
    ) -> ToolResponse:
        """获取指定股票的主要股东信息，并保存表格结果到Material当中，返回Material标识ref_id。
        抓取所有历史披露的主要股东信息，
        包括股东名称、持股数量、持股比例、股本性质、截至日期、公告日期等。
        适用场景：分析公司历史上的主要股东变化；辅助研报中“股权结构与股东情况”章节的撰写。

        Args:
            stock (str):
                股票代码，例如 "600004"。
        """
        df = ak.stock_main_stock_holder(stock=stock)
        ref_id = f"{stock}_main_holders_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_main_shareholders_material] 主要股东（stock={stock}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"stock": stock},
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

        ref_id = f"{symbol}_shareholder_count_detail_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
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


        ref_id = f"{symbol}_shareholder_change_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
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

        ref_id = f"{symbol}_business_description_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_business_description_material] 主营介绍（symbol={symbol}）"
        return _build_tool_response_from_df(
            df,
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

        ref_id = f"{symbol}_business_composition_{int(time.time())}"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_business_composition_material] 主营构成（symbol={symbol}）"
        return _build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )

    # ===================== 通用读取函数 =====================

    # def read_table_material(
    #         self,
    #         ref_id: str,
    #         max_rows: int | None = None,  # 默认显示全部
    # ) -> ToolResponse:
    #     """读取任意表格 Material，并返回预览信息。

    #     Args:
    #         ref_id (str):
    #             Material 标识，用于定位需要读取的表格。
    #         max_rows (int | None):
    #             用于控制预览行数：
    #             - 为 None（默认）：预览全部数据；
    #             - 为正整数：仅预览前 max_rows 行。


    #     """
    #     df = self.short_term.load_material(ref_id=ref_id)

    #     if df is None:
    #         text = f"[read_table_material] 未找到 ref_id='{ref_id}' 对应的 Material。"
    #         return ToolResponse(
    #             content=[TextBlock(type="text", text=text)],
    #             metadata={"ref_id": ref_id, "found": False},
    #         )

    #     preview_str, total_rows, used_rows, _ = _preview_df(df, max_rows)
    #     text = (
    #         f"[read_table_material] 成功读取 ref_id='{ref_id}' 对应的表格，"
    #         f"共 {total_rows} 条记录。以下为前 {used_rows} 行预览：\n"
    #         f"{preview_str}"
    #     )
    #     return ToolResponse(
    #         content=[TextBlock(type="text", text=text)],
    #         metadata={
    #             "ref_id": ref_id,
    #             "row_count": total_rows,
    #             "preview_rows": used_rows,
    #             "found": True,
    #         },
    #     )

