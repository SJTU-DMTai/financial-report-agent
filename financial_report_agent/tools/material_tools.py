from __future__ import annotations

from typing import Optional, Callable, Any, Dict, Union

import io

import akshare as ak
import pandas as pd

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

from ..memory.short_term import ShortTermMemoryStore

class MaterialTools:

    def __init__(self, short_term: Optional[ShortTermMemoryStore] = None) -> None:
        self.short_term = short_term

    # ========== 内部函数（不注册为 tool） ==========

    def _save_df_to_material(
        self,
        df: pd.DataFrame,
        ref_id: str
    ) -> int:
        """DataFrame 存入 short-term material（CSV），返回行数。"""
        if self.short_term is not None:
            csv_text = df.to_csv(index=False)
            self.short_term.save_material(ref_id=ref_id, content=csv_text, ext="csv")
        return len(df)


    def _preview_df(self, df: pd.DataFrame, max_rows: int | None = None) -> tuple[str, int, int, list[str]]:
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
        self,
        df: pd.DataFrame,
        ref_id: str,
        header: str,
        preview_rows: int = 10,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> ToolResponse:
        """统一构造 ToolResponse，包含预览文本和基础 metadata。"""
        preview_str, total_rows, used_rows, columns_names = self._preview_df(df, preview_rows)  # 预览的时候返回前10行
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

    # def _read_df_from_material(
    #     self,
    #     ref_id: str,
    # ) -> Optional[pd.DataFrame]:
    #     """从 short-term material 中读取 CSV 并还原为 DataFrame。"""
    #     csv_text = self.short_term.load_material(ref_id=ref_id, ext="csv")
    #     if not csv_text:
    #         return None
    #     return pd.read_csv(io.StringIO(csv_text))


    def add_exchange_prefix(self, symbol: str, type: str) -> str:
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



    # ===================== 股价数据 =====================


    async def fetch_realtime_price_material(
        self,
        symbol: str | None = None,
        ref_id: str | None = None
    ) -> ToolResponse:
        """获取沪深京 A 股实时行情数据，并将结果保存为表格。
        适用场景：需要查询某只 A 股当前价格、涨跌幅、成交量等实时指标；需要一次性拉取全市场实时行情，作为选股或打分模型的输入。

        Args:
            symbol (Optional[str]):
                沪深京 A 股股票代码（不带市场标识），例如 "000001"。
                - 为 None 时：保留全部 A 股的实时行情数据；
                - 不为 None 时：仅保留 DataFrame 中 "代码" 列等于该值的记录。
            ref_id (Optional[str]):
                用于在 short_term 中保存本次结果的 Material 标识。
                - 为 None 时：默认使用 f"{symbol or 'all'}_realtime_spot"。

        """

        df = ak.stock_zh_a_spot_em()
        if symbol is not None:
            # 文档中 "代码" 列为股票代码
            df = df[df["代码"] == symbol]

        if ref_id is None:
            ref_id = f"{symbol or 'all'}_realtime_spot"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_realtime_price_material] 股价实时行情（symbol={symbol or 'ALL'}）"
        return self._build_tool_response_from_df(
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
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定 A 股股票的历史行情（日/周/月），并写入表格。
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
            ref_id (Optional[str]):
                Material 标识，用于在 short_term 中区分不同请求。
                - 为 None 时：默认生成 f"{symbol}_history_{period}_{start_date}_{end_date}_{adj}"，
                其中 adj 为 adjust 或 "none"。

        """
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        if ref_id is None:
            adj = adjust or "none"
            ref_id = f"{symbol}_history_{period}_{start_date}_{end_date}_{adj}"

        self._save_df_to_material(df, ref_id)

        header = (
            f"[fetch_history_price_material] {symbol} {period} 股价历史行情 "
            f"{start_date}~{end_date} adjust='{adjust or '无'}'"
        )
        return self._build_tool_response_from_df(
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
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定个股的新闻资讯数据，并写入表格.
        相关的最新新闻资讯（默认为当日最近约 100 条），包括新闻标题、内容摘要、发布时间、来源和链接等，
        适用场景：为个股研报生成“新闻动态”“舆情分析”等部分提供原始素材；需要快速获取近期与某股票相关的新闻列表。

        Args:
            symbol (str):
                个股新闻检索关键词，通常为股票代码，例如 "603777"；
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{symbol}_news_em"。

        """
        df = ak.stock_news_em(symbol=symbol)

        if ref_id is None:
            ref_id = f"{symbol}_news_em"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_stock_news_material] 个股新闻（symbol={symbol}）"
        return self._build_tool_response_from_df(
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
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票的信息披露公告，并写入表格。
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
            ref_id (Optional[str]):
                Material 标识，默认值为
                f"{symbol}_disclosure_{category or 'all'}_{start_date}_{end_date}"。

        """

        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=symbol,
            market=market,
            keyword=keyword,
            category=category,
            start_date=start_date,
            end_date=end_date,
        )

        if ref_id is None:
            ref_id = f"{symbol}_disclosure_{category or 'all'}_{start_date}_{end_date}"

        self._save_df_to_material(df, ref_id)
        header = (
            f"[fetch_disclosure_material] 信息披露公告（symbol={symbol}, market={market}, "
            f"category={category or '全部'}, {start_date}~{end_date}）"
        )
        return self._build_tool_response_from_df(
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
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票的资产负债表数据，并写入表格。
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
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{symbol}_balance_{indicator}"。
        """

        df = ak.stock_financial_debt_ths(symbol=symbol, indicator=indicator)

        if ref_id is None:
            safe_indicator = indicator.replace(" ", "")
            ref_id = f"{symbol}_balance_{safe_indicator}"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_balance_sheet_material] 资产负债表（symbol={symbol}, indicator={indicator}）"
        return self._build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol, "indicator": indicator},
        )


    async def fetch_profit_table_material(
        self,
        symbol: str,
        indicator: str = "按报告期",
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票的利润表数据，并写入表格。
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
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{symbol}_profit_{indicator}"。
        """

        df = ak.stock_financial_benefit_ths(symbol=symbol, indicator=indicator)

        if ref_id is None:
            safe_indicator = indicator.replace(" ", "")
            ref_id = f"{symbol}_profit_{safe_indicator}"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_profit_table_material] 利润表（symbol={symbol}, indicator={indicator}）"
        return self._build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol, "indicator": indicator},
        )


    async def fetch_cashflow_table_material(
        self,
        symbol: str,
        indicator: str = "按报告期",
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票的现金流量表数据，并写入表格。
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
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{symbol}_cashflow_{indicator}"。
        """
        df = ak.stock_financial_cash_ths(symbol=symbol, indicator=indicator)

        if ref_id is None:
            safe_indicator = indicator.replace(" ", "")
            ref_id = f"{symbol}_cashflow_{safe_indicator}"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_cashflow_table_material] 现金流量表（symbol={symbol}, indicator={indicator}）"
        return self._build_tool_response_from_df(
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
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票在某一报告期的十大流通股东，并写入表格。

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
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{symbol}_top10_free_{date}"。
        """

        df = ak.stock_gdfx_free_top_10_em(symbol=self.add_exchange_prefix(symbol,"lower"), date=date)

        if ref_id is None:
            ref_id = f"{symbol}_top10_free_{date}"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_top10_float_shareholders_material] 十大流通股东（symbol={symbol}, date={date}）"
        return self._build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol, "date": date},
        )


    async def fetch_top10_shareholders_material(
        self,
        symbol: str,
        date: str,
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票在某一报告期的十大股东（总股本口径），并写入表格。
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
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{symbol}_top10_{date}"。
        """
        df = ak.stock_gdfx_top_10_em(symbol=self.add_exchange_prefix(symbol,"lower"), date=date)

        if ref_id is None:
            ref_id = f"{symbol}_top10_{date}"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_top10_shareholders_material] 十大股东（symbol={symbol}, date={date}）"
        return self._build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol, "date": date},
        )


    async def fetch_main_shareholders_material(
        self,
        stock: str,
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票的主要股东信息，并写入表格。
        抓取所有历史披露的主要股东信息，
        包括股东名称、持股数量、持股比例、股本性质、截至日期、公告日期等。
        适用场景：分析公司历史上的主要股东变化；辅助研报中“股权结构与股东情况”章节的撰写。

        Args:
            stock (str):
                股票代码，例如 "600004"。
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{stock}_main_holders"。
        """
        df = ak.stock_main_stock_holder(stock=stock)

        if ref_id is None:
            ref_id = f"{stock}_main_holders"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_main_shareholders_material] 主要股东（stock={stock}）"
        return self._build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"stock": stock},
        )


    async def fetch_shareholder_count_detail_material(
        self,
        symbol: str,
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票的股东户数详情，并写入表格。
        获取指定 symbol 的全部历史数据，包括股东户数统计截止日、区间涨跌幅、股东户数本次/上次/增减及其比例、户均持股市值与数量、
        总市值、总股本及股本变动原因等。
        适用场景：分析股东户数与股价表现的关系；评估筹码集中度变化和市场参与者结构。

        Args:
            symbol (str):
                股票代码，例如 "000001"。
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{symbol}_shareholder_count_detail"。
        """
        df = ak.stock_zh_a_gdhs_detail_em(symbol=symbol)

        if ref_id is None:
            ref_id = f"{symbol}_shareholder_count_detail"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_shareholder_count_detail_material] 股东户数详情（symbol={symbol}）"
        return self._build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )


    async def fetch_shareholder_change_material(
        self,
        symbol: str,
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票的股东持股变动统计信息，并写入表格。
        抓取所有披露的股东持股变动记录，包括公告日期、变动股东、变动数量、交易均价、剩余股份总数、变动期间和变动途径等。
        适用场景： 跟踪重要股东和机构的减持 / 增持行为；分析股价波动背后的股东行为因素。

        Args:
            symbol (str):
                股票代码，例如 "688981"。
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{symbol}_shareholder_change"。
        """

        df = ak.stock_shareholder_change_ths(symbol=symbol)

        if ref_id is None:
            ref_id = f"{symbol}_shareholder_change"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_shareholder_change_material] 股东持股变动（symbol={symbol}）"
        return self._build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )


    # ===================== 业务范围 =====================


    async def fetch_business_description_material(
        self,
        symbol: str,
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票的主营业务介绍，并写入表格。
        抓取公司主营业务、产品类型、产品名称及经营范围等字段，
        适用场景：生成研报中“公司简介”“主营业务与商业模式”章节的基础描述；快速了解公司业务结构和核心产品。

        Args:
            symbol (str):
                股票代码，例如 "000066"。
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{symbol}_business_description_ths"。

        """
        df = ak.stock_zyjs_ths(symbol=symbol)

        if ref_id is None:
            ref_id = f"{symbol}_business_description_ths"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_business_description_material] 主营介绍（symbol={symbol}）"
        return self._build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )


    async def fetch_business_composition_material(
        self,
        symbol: str,
        ref_id: str | None = None,
    ) -> ToolResponse:
        """获取指定股票的主营构成数据，并写入表格。
        抓取按产品、地区等维度划分的主营收入、成本、利润、及对应比例和毛利率等历史数据。
        适用场景：分析公司按产品/地区划分的收入与利润结构；研报中“业务结构分析”“毛利率拆解”等章节的数据来源。

        Args:
            symbol (str):
                股票代码，例如 "000063"。
            ref_id (Optional[str]):
                Material 标识，默认值为 f"{symbol}_business_composition_em"。
        """

        df = ak.stock_zygc_em(symbol=self.add_exchange_prefix(symbol,"upper"))

        if ref_id is None:
            ref_id = f"{symbol}_business_composition_em"

        self._save_df_to_material(df, ref_id)
        header = f"[fetch_business_composition_material] 主营构成（symbol={symbol}）"
        return self._build_tool_response_from_df(
            df,
            ref_id=ref_id,
            header=header,
            extra_meta={"symbol": symbol},
        )


    # ===================== 通用读取函数 =====================


    def read_table_material(
        self,
        ref_id: str,
        max_rows: int | None = None,  # 默认显示全部
    ) -> ToolResponse:
        """读取任意表格 Material，并返回预览信息。

        Args:
            ref_id (str):
                Material 标识，用于定位需要读取的表格。
            max_rows (int | None):
                用于控制预览行数：
                - 为 None（默认）：预览全部数据；
                - 为正整数：仅预览前 max_rows 行。


        """
        df = self.short_term.load_material(ref_id=ref_id, ext="csv")
        
        if df is None:
            text = f"[read_table_material] 未找到 ref_id='{ref_id}' 对应的 Material。"
            return ToolResponse(
                content=[TextBlock(type="text", text=text)],
                metadata={"ref_id": ref_id, "found": False},
            )

        preview_str, total_rows, used_rows, _ = self._preview_df(df, max_rows)
        text = (
            f"[read_table_material] 成功读取 ref_id='{ref_id}' 对应的表格，"
            f"共 {total_rows} 条记录。以下为前 {used_rows} 行预览：\n"
            f"{preview_str}"
        )
        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={
                "ref_id": ref_id,
                "row_count": total_rows,
                "preview_rows": used_rows,
                "found": True,
            },
        )

