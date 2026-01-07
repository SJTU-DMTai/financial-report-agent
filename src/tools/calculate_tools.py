# -*- coding: utf-8 -*-
# from __future__ import annotations
import time
from pathlib import Path
import re
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import json
import math
import pandas as pd
from agentscope.message import TextBlock, ImageBlock, Base64Source
from agentscope.tool import ToolResponse
from agentscope.tool._coding._python import execute_python_code
from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import LongTermMemoryStore
from ..utils.get_entity_info import get_entity_info
class CalculateTools:
    """金融研报场景的计算工具集合。"""

    def __init__(self, short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore) -> None:
        self.short_term = short_term
        self.long_term = long_term

    # --------- 通用内部工具函数 ---------
    def _get_required(self, params: Dict[str, Any], key: str) -> float:
        if key not in params:
            raise KeyError(f"缺少必需参数: {key}")
        try:
            return float(params[key])
        except Exception as e:
            raise ValueError(f"参数 {key} 无法转换为浮点数: {e}")

    def _normalize_rate(self, r: Any) -> float:
        """将利率/增长率规范为小数形式。若 >= 1 或者 <= -1 则按百分比处理，如 10 -> 0.10"""
        val = float(r)
        if val >= 1 or val <= -1:
            return val / 100.0
        return val
    

    def _save_calc_result(
        self,
        tool_name: str,          # 如 "calculate_valuation_metric"
        sub_type: str | None,           # 如 metric_type / ratio_type / forecast_type 等
        result: Any,
        result_type: str | None,
        params: Dict[str, Any] | None = None,
        code: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        把本次计算的输入输出写入 material，并返回 ref_id。
        """

        ref_id = f"{tool_name}_result_{int(time.time())}"

        source = "计算工具"
        if code:
            final_description = (description or "").strip() or "自定义 Python 数据分析/计算结果"
        else:
            tool_label_map = {
                "calculate_valuation_metric": "估值指标计算",
                "calculate_financial_ratio": "财务比率计算",
                "calculate_cashflow_metric": "现金流指标计算",
                "calculate_timeseries_transform": "时间序列变换",
                "calculate_forecast_metric": "预测辅助计算",
                "calculate_math_metric": "数学计算",
            }

            subtype_label_map = {
                "calculate_valuation_metric": {
                    "pe": "市盈率（PE）",
                    "peg": "PEG",
                    "pb": "市净率（PB）",
                    "ev": "企业价值（EV）",
                    "ev_ebitda": "EV/EBITDA",
                    "ev_ebit": "EV/EBIT",
                    "dcf": "现金流折现（DCF）",
                    "terminal_value": "终值（Terminal Value）",
                    "discount_factor": "折现因子（Discount Factor）",
                    "wacc": "加权平均资本成本（WACC）",
                    "cost_equity": "股权成本（CAPM）",
                },
                "calculate_financial_ratio": {
                    "roe": "净资产收益率（ROE）",
                    "roic": "投入资本回报率（ROIC）",
                    "gross_margin": "毛利率（Gross Margin）",
                    "op_margin": "营业利润率（Operating Margin）",
                    "net_margin": "净利率（Net Margin）",
                    "yoy": "同比增速（YoY）",
                    "qoq": "环比增速（QoQ）",
                    "cagr": "复合年增长率（CAGR）",
                    "de_ratio": "资产负债率/杠杆（D/E）",
                    "interest_coverage": "利息保障倍数（Interest Coverage）",
                },
                "calculate_cashflow_metric": {
                    "fcf": "自由现金流（FCF）",
                    "nopat": "税后营业利润（NOPAT）",
                    "fcff": "企业自由现金流（FCFF）",
                    "ttm": "TTM 合计（最近12个月）",
                },
                "calculate_timeseries_transform": {
                    "align_quarterly_to_annual": "季度数据聚合为年度数据",
                    "rolling_average": "滑动平均（Rolling Average）",
                    "annualize_quarterly_value": "单季值年化",
                },
                "calculate_forecast_metric": {
                    "project_revenue": "收入预测序列",
                    "project_margin": "利润率预测序列",
                    "discount_series": "现金流折现序列",
                },
                "calculate_math_metric": {
                    "npv": "净现值（NPV）",
                    "linear_regression": "线性回归拟合（Linear Regression）",
                },
            }

            tool_label = tool_label_map.get(tool_name, tool_name)
            sub_label = ""
            if sub_type:
                sub_label = subtype_label_map.get(tool_name, {}).get(sub_type, str(sub_type))

            if sub_label:
                base_description = f"调用工具进行{tool_label}：{sub_label}"
            else:
                base_description = f"调用工具进行{tool_label}"

            extra = (description or "").strip()
            final_description = f"{base_description} {extra}" if extra else base_description
        content = [
            {
                "description": final_description,
                "tool": tool_name,
                "sub_type": sub_type,
                "parameters": params or {},
                "code": code or "",
                "result_type": result_type,
                "result": result,
            }
        ]
        entity = get_entity_info(self.long_term, final_description);

        try:
            self.short_term.save_material(
                ref_id=ref_id,
                content=content,
                description=final_description,
                entity=entity,
                source=source,
            )
        except Exception:
            return ""

        return ref_id

    # --------- 1. 估值类工具 ---------
    async def calculate_valuation_metric(
        self,
        metric_type: Literal[
            "pe",
            "peg",
            "pb",
            "ev",
            "ev_ebitda",
            "ev_ebit",
            "dcf",
            "terminal_value",
            "discount_factor",
            "wacc",
            "cost_equity",
        ],
        params: Dict[str, Any],
        description: str | None = None,
    ) -> ToolResponse:
        """
        预定义金融计算工具，计算常见估值类指标，并保存计算结果到Material当中，返回Material标识ref_id。
        调用模式：
            {
              "metric_type": "<见下方枚举之一>",
              "params": { ... 对应该类型的参数 ... }
              "description": "<可选，对本次计算的补充说明文字，建议简要说明计算对象（如股票名称）、时间范围以及所计算的指标或变换>"
            }

        通用约定：
        - 所有金额/数值字段（price, eps, ebitda 等）使用数值型（float 或 int）。
        - 所有“利率/增长率”字段（growth_rate, discount_rate, tax_rate, rf 等）支持两种输入：
            - 若传入大于 1 的数字，例如 10，视为百分比 10%（内部转换为 0.10）；
            - 若在 0~1 之间，例如 0.1，视为小数 0.10。
        - params 中的字段名必须与下方示例完全一致，否则会报错。

        支持的 metric_type 及其 params 结构：

        1) "pe"：市盈率 = price / eps
           params 示例：
           {
             "price": 12.5,   # 股价
             "eps": 1.25      # 每股收益
           }

        2) "peg"：PEG = pe / growth_rate
           params 示例：
           {
             "pe": 20.0,          # 已知市盈率
             "growth_rate": 15.0  # 利润增速，15 或 0.15 都表示 15%
           }

        3) "pb"：市净率 = price / bvps
           params 示例：
           {
             "price": 8.0,    # 股价
             "bvps": 4.0      # 每股净资产
           }

        4) "ev"：企业价值 EV = market_cap + net_debt + minority_interest
           params 示例：
           {
             "market_cap": 1000.0,         # 股权市值
             "net_debt": 200.0,            # 净负债
             "minority_interest": 50.0     # 少数股东权益
           }

        5) "ev_ebitda"：EV/EBITDA 倍数
           params 示例：
           {
             "ev": 1250.0,
             "ebitda": 250.0
           }

        6) "ev_ebit"：EV/EBIT 倍数
           params 示例：
           {
             "ev": 1250.0,
             "ebit": 200.0
           }

        7) "terminal_value"：终值（永续增长模型）
           公式：CF_{T+1} = last_cf * (1 + g)
                 Terminal = CF_{T+1} / (r - g)
           要求：r > g
           params 示例：
           {
             "last_cf": 100.0,   # 最后一个显性期的自由现金流 CF_T
             "g": 2.0,           # 永续增长率，2 或 0.02 表示 2%
             "r": 8.0            # 折现率，8 或 0.08 表示 8%
           }

        8) "discount_factor"：折现因子
           公式：discount_factor = 1 / (1 + r)^t
           params 示例：
           {
             "rate": 8.0,    # 折现率，8 或 0.08 表示 8%
             "t": 3          # 期数，从 1 开始
           }

        9) "dcf"：现金流折现 DCF
           公式：DCF = Σ_{t=1..N} CF_t / (1 + r)^t
                     (+ 可选终值 TV / (1 + r)^N)
           params 示例：
           {
             "cash_flows": [100, 110, 120],   # 第 1~N 期自由现金流
             "discount_rate": 8.0,            # 折现率
             "terminal_value": 1500.0         # 终值 TV，可选参数；不需要可不传
           }

        10) "wacc"：加权平均资本成本 WACC
            公式：WACC = we * Re + wd * Rd * (1 - tax_rate)
            约束：weight_equity + weight_debt ≈ 1.0
            params 示例：
            {
              "cost_equity": 12.0,      # 股权成本 Re
              "cost_debt": 5.0,         # 债务成本 Rd
              "tax_rate": 25.0,         # 所得税率
              "weight_equity": 0.6,     # 权益权重 we
              "weight_debt": 0.4        # 债务权重 wd
            }

        11) "cost_equity"：股权成本（CAPM）
            公式：Re = rf + beta * ERP
            params 示例：
            {
              "rf": 3.0,                  # 无风险利率
              "beta": 1.2,
              "equity_risk_premium": 5.0  # 股权风险溢价 ERP
            }
                
            """
        try:
            result: Any

            if metric_type == "pe":
                price = self._get_required(params, "price")
                eps = self._get_required(params, "eps")
                if eps == 0:
                    raise ValueError("eps 不能为 0")
                result = price / eps

            elif metric_type == "peg":
                pe = self._get_required(params, "pe")
                g = self._normalize_rate(self._get_required(params, "growth_rate"))
                if g == 0:
                    raise ValueError("growth_rate 不能为 0")
                result = pe / g

            elif metric_type == "pb":
                price = self._get_required(params, "price")
                bvps = self._get_required(params, "bvps")
                if bvps == 0:
                    raise ValueError("bvps 不能为 0")
                result = price / bvps

            elif metric_type == "ev":
                market_cap = self._get_required(params, "market_cap")
                net_debt = self._get_required(params, "net_debt")
                minority_interest = self._get_required(params, "minority_interest")
                result = market_cap + net_debt + minority_interest

            elif metric_type == "ev_ebitda":
                ev = self._get_required(params, "ev")
                ebitda = self._get_required(params, "ebitda")
                if ebitda == 0:
                    raise ValueError("ebitda 不能为 0")
                result = ev / ebitda

            elif metric_type == "ev_ebit":
                ev = self._get_required(params, "ev")
                ebit = self._get_required(params, "ebit")
                if ebit == 0:
                    raise ValueError("ebit 不能为 0")
                result = ev / ebit

            elif metric_type == "terminal_value":
                last_cf = self._get_required(params, "last_cf")
                g = self._normalize_rate(self._get_required(params, "g"))
                r = self._normalize_rate(self._get_required(params, "r"))
                if r <= g:
                    raise ValueError("要求 r > g 才能计算终值(Gordon 模型)")
                cf_next = last_cf * (1 + g)
                result = cf_next / (r - g)

            elif metric_type == "discount_factor":
                r = self._normalize_rate(self._get_required(params, "rate"))
                t = self._get_required(params, "t")
                if t < 0:
                    raise ValueError("t 不能为负")
                result = 1.0 / ((1.0 + r) ** t)

            elif metric_type == "dcf":
                cash_flows = params.get("cash_flows")
                if not isinstance(cash_flows, list) or len(cash_flows) == 0:
                    raise ValueError("cash_flows 必须是非空列表")
                discount_rate = self._normalize_rate(self._get_required(params, "discount_rate"))
                tv = params.get("terminal_value", None)
                dcf_value = 0.0
                for i, cf in enumerate(cash_flows, start=1):
                    dcf_value += float(cf) / ((1 + discount_rate) ** i)
                if tv is not None:
                    dcf_value += float(tv) / ((1 + discount_rate) ** len(cash_flows))
                result = dcf_value

            elif metric_type == "wacc":
                cost_equity = self._normalize_rate(self._get_required(params, "cost_equity"))
                cost_debt = self._normalize_rate(self._get_required(params, "cost_debt"))
                tax_rate = self._normalize_rate(self._get_required(params, "tax_rate"))
                we = self._get_required(params, "weight_equity")
                wd = self._get_required(params, "weight_debt")
                if abs(we + wd - 1.0) > 1e-6:
                    raise ValueError("weight_equity + weight_debt 应接近 1")
                result = we * cost_equity + wd * cost_debt * (1 - tax_rate)

            elif metric_type == "cost_equity":
                rf = self._normalize_rate(self._get_required(params, "rf"))
                beta = self._get_required(params, "beta")
                erp = self._normalize_rate(self._get_required(params, "equity_risk_premium"))
                result = rf + beta * erp

            else:
                raise ValueError(f"未知的 metric_type: {metric_type}")

            ref_id = self._save_calc_result(
                tool_name="calculate_valuation_metric",
                sub_type=metric_type,
                params=params,
                result=result,
                result_type="float",
                description=description,
            )

            text_block: TextBlock = {
                "type": "text",
                "text": (
                    "[calculate_valuation_metric] 估值指标计算完成。\n"
                    f"metric_type: {metric_type}\n"
                    f"输入参数: {json.dumps(params, ensure_ascii=False)}\n"
                    f"结果: {result}\n"
                    f"Material 已写入 ref_id='{ref_id}'（JSON 格式）\n"
                ),
            }
            return ToolResponse(content=[text_block], metadata={"ref_id": ref_id})

        except Exception as e:
            error_block: TextBlock = {
                "type": "text",
                "text": f"[calculate_valuation_metric] 计算失败: {e}",
            }
            return ToolResponse(content=[error_block])

    # --------- 2. 财务比率类工具 ---------
    async def calculate_financial_ratio(
        self,
        ratio_type: Literal[
            "roe",
            "roic",
            "gross_margin",
            "op_margin",
            "net_margin",
            "yoy",
            "qoq",
            "cagr",
            "de_ratio",
            "interest_coverage",
        ],
        params: Dict[str, Any],
        description: str | None = None,
    ) -> ToolResponse:
        """
        预定义金融计算工具，用于计算常见财务比率，并保存计算结果到Material当中，返回Material标识ref_id。

        调用模式：
            {
              "ratio_type": "<见下方枚举之一>",
              "params": { ... 对应该类型的参数 ... }
              "description": "<可选，对本次计算的补充说明文字，建议简要说明计算对象（如股票名称）、时间范围以及所计算的指标或变换>"
            }

        通用约定：
        - 金额/规模字段（net_income, revenue 等）使用数值型。
        - 返回结果通常为小数形式的比率，例如：
            - 0.25 表示 25%。
        - params 中字段名必须与下方示例一致。

        支持的 ratio_type 及其 params 结构：

        1) "roe"：净资产收益率 ROE
           公式：ROE = net_income / equity
           params 示例：
           {
             "net_income": 100.0,  # 净利润
             "equity": 400.0       # 期初或平均股东权益，根据你的口径
           }

        2) "roic"：投入资本回报率 ROIC
           公式：ROIC = nopat / invested_capital
           params 示例：
           {
             "nopat": 120.0,             # 税后营业利润
             "invested_capital": 800.0   # 投入资本
           }

        3) "gross_margin"：毛利率
           公式：毛利率 = gross_profit / revenue
           params 示例：
           {
             "gross_profit": 300.0,
             "revenue": 1000.0
           }

        4) "op_margin"：营业利润率
           公式：营业利润率 = operating_income / revenue
           params 示例：
           {
             "operating_income": 150.0,
             "revenue": 1000.0
           }

        5) "net_margin"：净利率
           公式：净利率 = net_income / revenue
           params 示例：
           {
             "net_income": 120.0,
             "revenue": 1000.0
           }

        6) "yoy"：同比增速
           公式：YoY = current / previous - 1
           params 示例：
           {
             "current": 1100.0,   # 本期值
             "previous": 1000.0   # 上年同期值
           }

        7) "qoq"：环比增速
           公式：QoQ = current / previous - 1
           params 示例：
           {
             "current": 260.0,   # 本季值
             "previous": 250.0   # 上季值
           }

        8) "cagr"：复合年增长率 CAGR
           公式：CAGR = (end / start)^(1/years) - 1
           params 示例：
           {
             "start": 100.0,   # 起始值
             "end": 150.0,     # 终值
             "years": 3        # 年数（可以是整数或浮点，通常为整数）
           }

        9) "de_ratio"：杠杆率 D/E
           公式：D/E = total_debt / equity
           params 示例：
           {
             "total_debt": 500.0,   # 总有息负债
             "equity": 400.0        # 股东权益
           }

        10) "interest_coverage"：利息保障倍数
            公式：Interest Coverage = ebit / interest_expense
            params 示例：
            {
              "ebit": 200.0,             # 息税前利润
              "interest_expense": 50.0   # 利息费用
            }

        """
        try:
            if ratio_type == "roe":
                ni = self._get_required(params, "net_income")
                equity = self._get_required(params, "equity")
                if equity == 0:
                    raise ValueError("equity 不能为 0")
                result = ni / equity

            elif ratio_type == "roic":
                nopat = self._get_required(params, "nopat")
                invested_capital = self._get_required(params, "invested_capital")
                if invested_capital == 0:
                    raise ValueError("invested_capital 不能为 0")
                result = nopat / invested_capital

            elif ratio_type == "gross_margin":
                gp = self._get_required(params, "gross_profit")
                rev = self._get_required(params, "revenue")
                if rev == 0:
                    raise ValueError("revenue 不能为 0")
                result = gp / rev

            elif ratio_type == "op_margin":
                op = self._get_required(params, "operating_income")
                rev = self._get_required(params, "revenue")
                if rev == 0:
                    raise ValueError("revenue 不能为 0")
                result = op / rev

            elif ratio_type == "net_margin":
                ni = self._get_required(params, "net_income")
                rev = self._get_required(params, "revenue")
                if rev == 0:
                    raise ValueError("revenue 不能为 0")
                result = ni / rev

            elif ratio_type in ("yoy", "qoq"):
                current = self._get_required(params, "current")
                previous = self._get_required(params, "previous")
                if previous == 0:
                    raise ValueError("previous 不能为 0")
                result = current / previous - 1.0

            elif ratio_type == "cagr":
                start = self._get_required(params, "start")
                end = self._get_required(params, "end")
                years = self._get_required(params, "years")
                if start <= 0 or end <= 0 or years <= 0:
                    raise ValueError("CAGR 要求 start > 0, end > 0 且 years > 0")
                result = (end / start) ** (1.0 / years) - 1.0

            elif ratio_type == "de_ratio":
                debt = self._get_required(params, "total_debt")
                equity = self._get_required(params, "equity")
                if equity == 0:
                    raise ValueError("equity 不能为 0")
                result = debt / equity

            elif ratio_type == "interest_coverage":
                ebit = self._get_required(params, "ebit")
                interest = self._get_required(params, "interest_expense")
                if interest == 0:
                    raise ValueError("interest_expense 不能为 0")
                result = ebit / interest

            else:
                raise ValueError(f"未知的 ratio_type: {ratio_type}")

            ref_id = self._save_calc_result(
                tool_name="calculate_financial_ratio",
                sub_type=ratio_type,
                params=params,
                result=result,
                result_type="float",
                description=description,
            )
            
            text_block: TextBlock = {
                "type": "text",
                "text": (
                    "[calculate_financial_ratio] 财务比率计算完成。\n"
                    f"ratio_type: {ratio_type}\n"
                    f"输入参数: {json.dumps(params, ensure_ascii=False)}\n"
                    f"结果: {result}\n"
                    f"Material 已写入 ref_id='{ref_id}'（JSON 格式）\n"
                ),
            }
            return ToolResponse(content=[text_block], metadata={"ref_id": ref_id})


        except Exception as e:
            error_block: TextBlock = {
                "type": "text",
                "text": f"[calculate_financial_ratio] 计算失败: {e}",
            }
            return ToolResponse(content=[error_block])

    # --------- 3. 现金流类工具 ---------
    async def calculate_cashflow_metric(
        self,
        metric_type: Literal["fcf", "nopat", "fcff", "ttm"],
        params: Dict[str, Any],
        description: str | None = None,
    ) -> ToolResponse:
        """
        预定义金融计算工具，用于计算自由现金流等与现金流相关的指标，并保存计算结果到Material当中，返回Material标识ref_id。

        调用模式：
            {
              "metric_type": "<见下方枚举之一>",
              "params": { ... 对应该类型的参数 ... }
              "description": "<可选，对本次计算的补充说明文字，建议简要说明计算对象（如股票名称）、时间范围以及所计算的指标或变换>"
            }

        通用约定：
        - 所有金额字段使用数值型。
        - 利率字段（tax_rate 等）遵从统一规则：
            - 若 > 1，例如 25，视为 25%；
            - 若在 0~1 之间，例如 0.25，视为 25%。

        支持的 metric_type 及其 params 结构：

        1) "fcf"：自由现金流 FCF
           公式：FCF = operating_cf - capex
           params 示例：
           {
             "operating_cf": 300.0,   # 经营活动产生的现金流量净额
             "capex": 120.0           # 资本性支出
           }

        2) "nopat"：税后营业利润 NOPAT
           公式：NOPAT = ebit * (1 - tax_rate)
           params 示例：
           {
             "ebit": 200.0,      # 息税前利润
             "tax_rate": 25.0    # 所得税率，25 或 0.25 都表示 25%
           }

        3) "fcff"：企业自由现金流 FCFF
           公式：FCFF = nopat + depreciation - capex - change_in_working_capital
           params 示例：
           {
             "nopat": 150.0,                     # 税后营业利润
             "depreciation": 50.0,               # 折旧摊销
             "capex": 120.0,                     # 资本性支出
             "change_in_working_capital": 20.0   # 营运资本变化（增加为正，减少为负）
           }

        4) "ttm"：TTM 合计（最近 12 个月合计，如 TTM EBITDA）
           说明：
           - 将 values 中最后 window 期的值求和。
           - 常用于从季度数据构造 TTM 指标（例如 window=4）。

           params 示例：
           {
             "values": [100.0, 110.0, 120.0, 130.0, 140.0],  # 时间顺序排列
             "window": 4                                      # 可选，默认 4
           }
           返回结果：
           - 对于上述示例，取最后 4 个值：110+120+130+140。

        """
        try:
            if metric_type == "fcf":
                ocf = self._get_required(params, "operating_cf")
                capex = self._get_required(params, "capex")
                result = ocf - capex

            elif metric_type == "nopat":
                ebit = self._get_required(params, "ebit")
                tax_rate = self._normalize_rate(self._get_required(params, "tax_rate"))
                result = ebit * (1 - tax_rate)

            elif metric_type == "fcff":
                nopat = self._get_required(params, "nopat")
                dep = self._get_required(params, "depreciation")
                capex = self._get_required(params, "capex")
                delta_wc = self._get_required(params, "change_in_working_capital")
                result = nopat + dep - capex - delta_wc

            elif metric_type == "ttm":
                values = params.get("values")
                if not isinstance(values, list) or len(values) == 0:
                    raise ValueError("values 必须是非空列表")
                window = int(params.get("window", 4))
                if window <= 0:
                    raise ValueError("window 必须为正整数")
                if len(values) < window:
                    raise ValueError("values 长度必须至少为 window")
                result = sum(float(v) for v in values[-window:])

            else:
                raise ValueError(f"未知的 metric_type: {metric_type}")

            ref_id = self._save_calc_result(
                tool_name="calculate_cashflow_metric",
                sub_type=metric_type,
                params=params,
                result=result,
                result_type="float",
                description=description,
            )

            text_block: TextBlock = {
                "type": "text",
                "text": (
                    "[calculate_cashflow_metric] 现金流指标计算完成。\n"
                    f"metric_type: {metric_type}\n"
                    f"输入参数: {json.dumps(params, ensure_ascii=False)}\n"
                    f"结果: {result}\n"
                    f"Material 已写入 ref_id='{ref_id}'（JSON 格式）\n"
                ),
            }

            return ToolResponse(content=[text_block], metadata={"ref_id": ref_id})

        except Exception as e:
            error_block: TextBlock = {
                "type": "text",
                "text": f"[calculate_cashflow_metric] 计算失败: {e}",
            }
            return ToolResponse(content=[error_block])

    # --------- 4. 时间序列变换工具 ---------
    async def calculate_timeseries_transform(
        self,
        transform_type: Literal[
            "align_quarterly_to_annual",
            "rolling_average",
            "annualize_quarterly_value",
        ],
        params: Dict[str, Any],
        description: str | None = None,
    ) -> ToolResponse:
        """
        预定义金融计算工具，用于对时间序列进行常见的变换与对齐处理，并保存计算结果到Material当中，返回Material标识ref_id。

        调用模式：
            {
              "transform_type": "<见下方枚举之一>",
              "params": { ... 对应该类型的参数 ... }
              "description": "<可选，对本次计算的补充说明文字，建议简要说明计算对象（如股票名称）、时间范围以及所计算的指标或变换>"
            }

        通用约定：
        - 所有时间序列均按时间顺序排列（从旧到新）。
        - 返回值可能是列表或单个数值，具体见每种类型说明。

        支持的 transform_type 及其 params 结构：

        1) "align_quarterly_to_annual"：季度数据聚合为年度数据
           逻辑：
           - 每 4 个连续季度聚合为 1 个年度值（简单求和）。
           - 要求 quarters 的长度必须是 4 的整数倍。

           params 示例：
           {
             "quarters": [10.0, 12.0, 11.0, 13.0,
                          14.0, 15.0, 16.0, 17.0]
           }
           返回结果：
           - [46.0, 62.0]   # 第一年 10+12+11+13，第二年 14+15+16+17

        2) "rolling_average"：滑动平均（简单移动平均）
           逻辑：
           - 对 values 进行窗口为 window 的简单平均。
           - 返回列表长度与 values 相同。
           - 在前 window-1 个位置，返回 None（因为窗口不完整）。

           params 示例：
           {
             "values": [1.0, 2.0, 3.0, 4.0, 5.0],
             "window": 3
           }
           返回结果：
           - [None, None, 2.0, 3.0, 4.0]
             解释：第 3 个点平均 (1+2+3)/3 = 2.0，以此类推。

        3) "annualize_quarterly_value"：将单季值年化
           逻辑：
           - annualized = quarterly_value * 4
           - 适用于年化季度 EPS、收入等。

           params 示例：
           {
             "quarterly_value": 2.5
           }
           返回结果：
           - 10.0

        """
        try:
            if transform_type == "align_quarterly_to_annual":
                quarters = params.get("quarters")
                if not isinstance(quarters, list) or len(quarters) == 0:
                    raise ValueError("quarters 必须是非空列表")
                if len(quarters) % 4 != 0:
                    raise ValueError("quarters 长度必须是 4 的整数倍")
                annual = []
                for i in range(0, len(quarters), 4):
                    annual.append(sum(float(v) for v in quarters[i:i+4]))
                result = annual

            elif transform_type == "rolling_average":
                values = params.get("values")
                if not isinstance(values, list) or len(values) == 0:
                    raise ValueError("values 必须是非空列表")
                window = int(params.get("window", 3))
                if window <= 0:
                    raise ValueError("window 必须为正整数")
                if window > len(values):
                    raise ValueError("window 不能大于 values 长度")
                ma: List[Union[float, None]] = []
                for i in range(len(values)):
                    if i + 1 < window:
                        ma.append(None)
                    else:
                        window_slice = values[i + 1 - window : i + 1]
                        ma.append(sum(float(v) for v in window_slice) / window)
                result = ma

            elif transform_type == "annualize_quarterly_value":
                qv = self._get_required(params, "quarterly_value")
                result = qv * 4.0

            else:
                raise ValueError(f"未知的 transform_type: {transform_type}")

            ref_id = self._save_calc_result(
                tool_name="calculate_timeseries_transform",
                sub_type=transform_type,
                params=params,
                result=result,
                result_type="float",
                description=description,
            )

            text_block: TextBlock = {
                "type": "text",
                "text": (
                    "[calculate_timeseries_transform] 时间序列变换完成。\n"
                    f"transform_type: {transform_type}\n"
                    f"输入参数: {json.dumps(params, ensure_ascii=False)}\n"
                    f"结果: {json.dumps(result, ensure_ascii=False)}\n"
                    f"Material 已写入 ref_id='{ref_id}'（JSON 格式）\n"
                ),
            }
            return ToolResponse(content=[text_block], metadata={"ref_id": ref_id})            
            
        except Exception as e:
            error_block: TextBlock = {
                "type": "text",
                "text": f"[calculate_timeseries_transform] 计算失败: {e}",
            }
            return ToolResponse(content=[error_block])

    # --------- 5. 预测辅助工具 ---------
    async def calculate_forecast_metric(
        self,
        forecast_type: Literal["project_revenue", "project_margin", "discount_series"],
        params: Dict[str, Any],
        description: str | None = None,
    ) -> ToolResponse:
        """
        预定义金融计算工具，预测场景下的常用辅助计算工具，并保存计算结果到Material当中，返回Material标识ref_id。

        调用模式：
            {
              "forecast_type": "<见下方枚举之一>",
              "params": { ... 对应该类型的参数 ... }
              "description": "<可选，对本次计算的补充说明文字，建议简要说明计算对象（如股票名称）、时间范围以及所计算的指标或变换>"
            }

        通用约定：
        - 金额类、比率类字段均为数值型。
        - 利率/折现率类字段遵从统一规则：
            - 若 > 1，例如 10，视为 10%；
            - 若在 0~1 之间，例如 0.1，视为 10%。

        支持的 forecast_type 及其 params 结构：

        1) "project_revenue"：按固定增速预测未来收入
           逻辑：
           - 从 base 开始，每年按 growth_rate 递增。
           - 返回长度为 years 的列表，表示未来每年的收入。

           params 示例：
           {
             "base": 1000.0,         # 基期收入（当前或最近一年）
             "growth_rate": 10.0,    # 年增长率，10 或 0.10 都表示 10%
             "years": 3              # 预测年数（正整数）
           }
           返回结果示例：
           - [1100.0, 1210.0, 1331.0]

        2) "project_margin"：按固定变化率预测利润率
           逻辑：
           - base_margin 为基期利润率（小数，如 0.35 表示 35%）。
           - 每年利润率 = base_margin + change_per_year * year_index。
           - year_index 从 1 开始计数。
           - 返回长度为 years 的列表。

           params 示例：
           {
             "base_margin": 0.35,        # 基期利润率（小数）
             "change_per_year": 0.02,    # 每年变化量，例如 0.02 表示每年+2pct
             "years": 3
           }
           返回结果示例：
           - [0.37, 0.39, 0.41]

        3) "discount_series"：将一系列未来现金流折现到当前
           逻辑：
           - 对 values 中的每一期现金流，按给定折现率 discount_rate 折现。
           - 假设第 1 个值对应 t=1，第 2 个值对应 t=2，以此类推。
           - 单期折现公式：CF_t / (1 + r)^t

           params 示例：
           {
             "values": [100.0, 110.0, 120.0],  # 未来第 1~3 年的现金流
             "discount_rate": 8.0              # 折现率，8 或 0.08 都表示 8%
           }
           返回结果示例：
           - [92.5926..., 94.4881..., 95.2380...]  # 每期折现后的数值

        """
        try:
            if forecast_type == "project_revenue":
                base = self._get_required(params, "base")
                growth_rate = self._normalize_rate(self._get_required(params, "growth_rate"))
                years = int(params.get("years", 1))
                if years <= 0:
                    raise ValueError("years 必须为正整数")
                projections: List[float] = []
                current = base
                for _ in range(years):
                    current = current * (1 + growth_rate)
                    projections.append(current)
                result = projections

            elif forecast_type == "project_margin":
                base_margin = self._get_required(params, "base_margin")
                change_per_year = float(params.get("change_per_year", 0.0))
                years = int(params.get("years", 1))
                if years <= 0:
                    raise ValueError("years 必须为正整数")
                margins: List[float] = []
                for year in range(1, years + 1):
                    margins.append(base_margin + change_per_year * year)
                result = margins

            elif forecast_type == "discount_series":
                values = params.get("values")
                if not isinstance(values, list) or len(values) == 0:
                    raise ValueError("values 必须是非空列表")
                discount_rate = self._normalize_rate(self._get_required(params, "discount_rate"))
                discounted: List[float] = []
                for i, v in enumerate(values, start=1):
                    discounted.append(float(v) / ((1 + discount_rate) ** i))
                result = discounted

            else:
                raise ValueError(f"未知的 forecast_type: {forecast_type}")

            ref_id = self._save_calc_result(
                tool_name="calculate_forecast_metric",
                sub_type=forecast_type,
                params=params,
                result=result,
                result_type="float",
                description=description,
            )

            text_block: TextBlock = {
                "type": "text",
                "text": (
                    "[calculate_forecast_metric] 预测相关计算完成。\n"
                    f"forecast_type: {forecast_type}\n"
                    f"输入参数: {json.dumps(params, ensure_ascii=False)}\n"
                    f"结果: {json.dumps(result, ensure_ascii=False)}\n"
                    f"Material 已写入 ref_id='{ref_id}'（JSON 格式）\n"
                ),
            }
            return ToolResponse(content=[text_block], metadata={"ref_id": ref_id})

        except Exception as e:
            error_block: TextBlock = {
                "type": "text",
                "text": f"[calculate_forecast_metric] 计算失败: {e}",
            }
            return ToolResponse(content=[error_block])

    # --------- 6. 通用数学类工具 ---------
    async def calculate_math_metric(
        self,
        metric_type: Literal["npv", "linear_regression"],
        params: Dict[str, Any],
        description: str | None = None,
    ) -> ToolResponse:
        """
        预定义的通用数学类工具，目前支持：
        - "npv"：净现值计算
        - "linear_regression"：一元线性回归拟合（y = a * x + b）
        使用工具进行计算并保存计算结果到Material当中，返回Material标识ref_id。
        
        调用模式：
            {
              "metric_type": "<见下方枚举之一>",
              "params": { ... 对应该类型的参数 ... }
              "description": "<可选，对本次计算的补充说明文字，建议简要说明计算对象（如股票名称）、时间范围以及所计算的指标或变换>"
            }

        通用约定：
        - 所有现金流、数值数据使用数值型（float 或 int）。
        - 折现率 discount_rate 遵从统一规则：
            - 若 > 1（例如 10），视为 10%；
            - 若在 0~1 之间（例如 0.1），视为 10%。

        支持的 metric_type 及其 params 结构：

        1) metric_type = "npv"

        功能：
            计算一串未来现金流的净现值（不包含 t=0 的初始投资）。
        公式：
            NPV = Σ_{t=1..N} CF_t / (1 + r)^t

        说明：
        - 若你还需要包含初始投资 I（发生在 t=0），可以自行处理：
              总 NPV = -I + 本函数返回的 NPV。
        - 本函数只折现 params["cash_flows"] 中每一期的数值。

        params 结构示例：
            {
              "cash_flows": [100.0, 110.0, 120.0],  # 第 1~N 期的现金流
              "discount_rate": 8.0                  # 折现率，8 或 0.08 均表示 8%
            }

        返回结果：
        - result 为数值，表示净现值。

        2) metric_type = "linear_regression"

        功能：
            对一组二维数据点 (x_i, y_i) 做一元线性回归拟合：
                y ≈ slope * x + intercept

        数据要求：
        - x 和 y 必须为长度相等的列表。
        - 样本数量 n >= 2。
        - 所有 x 不应全部相同（否则无法拟合直线）。

        基本公式（带截距的普通最小二乘 OLS）：
            记：
                x_mean = Σ x_i / n
                y_mean = Σ y_i / n
                Sxx = Σ (x_i - x_mean)^2
                Sxy = Σ (x_i - x_mean) * (y_i - y_mean)
                Syy = Σ (y_i - y_mean)^2

            则：
                slope = Sxy / Sxx
                intercept = y_mean - slope * x_mean
                r = Sxy / sqrt(Sxx * Syy)        （皮尔逊相关系数）
                r_squared = r^2                  （决定系数）

        params 结构示例：
            基本用法（带截距）：
            {
              "x": [1.0, 2.0, 3.0, 4.0],
              "y": [2.0, 2.5, 3.5, 4.0]
            }

            可选参数：
            - "fit_intercept": bool，是否拟合截距，默认 True。
              - 若为 False，则按“过原点回归”拟合：
                    slope = Σ(x_i * y_i) / Σ(x_i^2)
                    intercept = 0.0
              - r 与 r_squared 仍按上面的定义，用 (x_i, y_i) 的相关系数计算。

            示例（显式指定 fit_intercept）：
            {
              "x": [1.0, 2.0, 3.0, 4.0],
              "y": [2.0, 2.5, 3.5, 4.0],
              "fit_intercept": true
            }

        返回结果：
        - result 为一个字典（在文本中以 JSON 字符串形式展示），包含：
            {
              "slope": float,        # 斜率 a
              "intercept": float,    # 截距 b
              "r": float | null,     # 相关系数，若无法定义则为 null
              "r_squared": float | null,  # 决定系数，r 的平方
              "n": int,              # 样本点数量
              "x_mean": float,       # x 的均值（fit_intercept=True 时有意义）
              "y_mean": float        # y 的均值
            }
          说明：
          - 若样本退化（例如 Sxx = 0 或 Syy = 0），r 和 r_squared 将返回 null。

        """
        try:
            if metric_type == "npv":
                cash_flows = params.get("cash_flows")
                if not isinstance(cash_flows, list) or len(cash_flows) == 0:
                    raise ValueError("cash_flows 必须是非空列表")
                discount_rate = self._normalize_rate(float(params["discount_rate"]))
                npv_value = 0.0
                for i, cf in enumerate(cash_flows, start=1):
                    npv_value += float(cf) / ((1 + discount_rate) ** i)
                result: Any = npv_value

            elif metric_type == "linear_regression":
                x = params.get("x")
                y = params.get("y")
                if not isinstance(x, list) or not isinstance(y, list):
                    raise ValueError("x 和 y 必须是列表")
                if len(x) == 0 or len(y) == 0:
                    raise ValueError("x 和 y 不可为空")
                if len(x) != len(y):
                    raise ValueError("x 和 y 的长度必须相同")

                n = len(x)
                if n < 2:
                    raise ValueError("样本数量 n 必须至少为 2")

                x_vals = [float(v) for v in x]
                y_vals = [float(v) for v in y]

                fit_intercept = bool(params.get("fit_intercept", True))

                if fit_intercept:
                    x_mean = sum(x_vals) / n
                    y_mean = sum(y_vals) / n

                    sxx = sum((xi - x_mean) ** 2 for xi in x_vals)
                    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x_vals, y_vals))
                    syy = sum((yi - y_mean) ** 2 for yi in y_vals)

                    if sxx == 0:
                        raise ValueError("Sxx = 0，所有 x 值相同，无法拟合线性回归")

                    slope = sxy / sxx
                    intercept = y_mean - slope * x_mean
                else:
                    # 过原点回归：y ≈ slope * x，intercept = 0
                    sum_xx = sum(xi * xi for xi in x_vals)
                    if sum_xx == 0:
                        raise ValueError("Σ x_i^2 = 0，所有 x 值为 0，无法拟合线性回归")
                    sum_xy = sum(xi * yi for xi, yi in zip(x_vals, y_vals))
                    slope = sum_xy / sum_xx
                    intercept = 0.0
                    x_mean = sum(x_vals) / n
                    y_mean = sum(y_vals) / n
                    # 为了计算 r 与 r_squared，仍按带截距的定义计算 Sxx、Sxy、Syy
                    sxx = sum((xi - x_mean) ** 2 for xi in x_vals)
                    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x_vals, y_vals))
                    syy = sum((yi - y_mean) ** 2 for yi in y_vals)

                # 计算相关系数 r 和 r_squared，如退化则返回 None
                if sxx > 0 and syy > 0:
                    r_value = sxy / math.sqrt(sxx * syy)
                    r_squared = r_value ** 2
                else:
                    r_value = None
                    r_squared = None

                result = {
                    "slope": slope,
                    "intercept": intercept,
                    "r": r_value,
                    "r_squared": r_squared,
                    "n": n,
                    "x_mean": x_mean,
                    "y_mean": y_mean,
                }

            else:
                raise ValueError(f"未知的 metric_type: {metric_type}")

            ref_id = self._save_calc_result(
                tool_name="calculate_math_metric",
                sub_type=metric_type,
                params=params,
                result=result,
                result_type="float",
                description=description,
            )

            text_block: TextBlock = {
                "type": "text",
                "text": (
                    "[calculate_math_metric] 数学指标计算完成。\n"
                    f"metric_type: {metric_type}\n"
                    f"输入参数: {json.dumps(params, ensure_ascii=False)}\n"
                    "结果: "
                    f"{json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else result}\n"
                    f"Material 已写入 ref_id='{ref_id}'（JSON 格式）\n"
                ),
            }
            
            return ToolResponse(content=[text_block], metadata={"ref_id": ref_id})            
            
            

        except Exception as e:
            error_block: TextBlock = {
                "type": "text",
                "text": f"[calculate_math_metric] 计算失败: {e}",
            }
            return ToolResponse(content=[error_block])
        


    async def calculate_or_analysis_by_python_code(
        self,
        code: str,
        material_map: dict[str, str] | None = None,
        description: str | None = None,
    ) -> ToolResponse:
        """
        执行自行编写的 Python 数据分析 / 计算代码，并返回“打印出来的结果”，保存计算结果到Material当中，返回Material标识ref_id。
        你可以使用自己传入的数据，也可以通过 ref_id 引用之前保存的 Material 中的数值数据，并将其作为变量在代码中使用。

        1. 你需要编写一段 Python 代码片段，实现数据分析或数值计算逻辑。
           示例（计算某组收益率的均值和标准差）：

               import numpy as np

               returns = [0.01, 0.02, -0.005, 0.03]
               mean_ret = float(np.mean(returns))
               std_ret = float(np.std(returns, ddof=1))

               # 将要返回给上层的结果放入变量 result 中
               result = {
                   "mean_return": mean_ret,
                   "std_return": std_ret,
               }

           注意：只写核心逻辑，不需要写 if __name__ == "__main__"。

        2. 已预先导入的库（在子进程环境中）：
               import math
               import statistics
               import datetime
               import numpy as np
               import pandas as pd
           你可以直接使用这些库，无需再次导入；重复导入也不会报错。
           你也可以自行导入其他需要的库，例如sklearn、scipy等。

        3. 结果约定（关键）：
           - 你必须在代码中，将最终要返回的结果赋值给变量 `result`。
           - 本工具会调用：
                 print(result)
           - 对于 pandas.DataFrame / Series，result 会以 DataFrame/Series 自带的
             字符串表格形式输出（多行文本），外层会原样返回。

        Args:
            code (str):
                Python 代码片段，仅包含分析/计算逻辑。
            material_map (dict[str, str] | None):
                可选，本次计算中需要访问的material映射，可以通过设置此参数将material中的数值数据注入到code中的变量进行访问。
                material_map的构成如下：
                    - key: 你在代码中使用的变量名
                    - value: 已有 Material 的 ref_id
                若为 None，则不注入任何material变量。
                注意，可以被注入到变量的material必须包含数值数据，比如TABLE会完整加载为dataframe，calculate_*_result会提取其中result字段的数据。
            description (str | None):
                可选，对本次计算的补充说明文字，建议简要说明计算对象（如股票名称）、时间范围以及所计算的指标或变换。

        Returns:
            ToolResponse:
                content 中包含:
                - TextBlock:
                    - 若成功：
                        - 描述本次计算的含义
                        - 输出 result 的字符串表示（可能是多行，例如 DataFrame 表格）
                    - 若失败：
                        - 错误信息，以及 stdout / stderr 便于调试

                metadata 中包含:
                - "raw_stdout": 执行子进程的标准输出（包含 result 打印）
                - "raw_stderr": 执行子进程的标准错误
        """


        material_map = material_map or {}
        material_injection_lines: list[str] = []
        for var_name, ref_id in material_map.items():
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", var_name):
                text = (
                    "[calculate_or_analysis_by_python_code] Material 注入失败。\n"
                    f"变量名 '{var_name}' 不是合法的 Python 标识符，请仅使用字母、数字和下划线，"
                    "且不能以数字开头。\n"
                    f"触发位置: material_map['{var_name}'] = '{ref_id}'\n"
                )
                error_block: TextBlock = {
                    "type": "text",
                    "text": text,
                }
                return ToolResponse(
                    content=[error_block],
                    metadata={
                        "material_map": material_map,
                    },
                )

            try:
                mat_obj = self.short_term.load_material_numerical(ref_id)
            except Exception as e:
                text = (
                    "[calculate_or_analysis_by_python_code] Material 数值加载失败。\n"
                    f"变量名: {var_name}\n"
                    f"ref_id: {ref_id}\n"
                    f"异常信息: {e}\n\n"
                    "请检查该 Material 是否存在、类型是否为 TABLE 或 JSON，"
                    "以及 JSON 中是否包含可供提取的 result 字段。"
                )
                error_block: TextBlock = {
                    "type": "text",
                    "text": text,
                }
                return ToolResponse(
                    content=[error_block],
                    metadata={
                        "material_map": material_map,
                    },
                )

            # DataFrame：以 CSV 形式注入，在子进程中用 StringIO + read_csv 重建
            if isinstance(mat_obj, pd.DataFrame):
                csv_str = mat_obj.to_csv(index=False)
                csv_literal = repr(csv_str)  # 安全的 Python 字面量
                material_injection_lines.append(
                    f"{var_name} = pd.read_csv(StringIO({csv_literal}))"
                )
            else:
                # 其他对象：直接用 repr 注入
                material_injection_lines.append(
                    f"{var_name} = {repr(mat_obj)}"
                )

        material_injection_code = "\n".join(material_injection_lines)
        # 子进程执行包装代码：只做基础导入 + 执行用户代码 + 打印 result
        python_wrapper = f"""
import math
import statistics
import datetime
import numpy as np
import pandas as pd
from io import StringIO
import json
# ==== Material 数值注入开始 ====
{material_injection_code}
# ==== Material 数值注入结束 ====

# ==== 用户代码开始 ====
{code}
# ==== 用户代码结束 ====

# 将变量 result 的字符串表示打印出来，并用标签包裹，方便外层正则提取
try:
    _r = result
except NameError:
    print("<result_error>" +
          "变量 'result' 未定义，请在代码中将最终计算结果赋值给 result。" +
          "</result_error>")
else:

    print("<result_repr>")
    try:
        print(_r)
    except Exception as _e:
        print("打印 result 时发生错误:", _e)
    print("</result_repr>")

    try:
        if isinstance(_r, pd.DataFrame):
            # DataFrame 转为 records，方便后面还原
            payload = _r.to_dict(orient="records")
            r_type = "DataFrame"
        elif isinstance(_r, pd.Series):
            payload = _r.to_dict()
            r_type = "Series"
        else:
            # 其他类型要求是 JSON 可序列化的（标量 / list / dict 等）
            payload = _r
            r_type = type(_r).__name__

        result_json = json.dumps(
            {{"result_type": r_type, "result": payload}},
            ensure_ascii=False
        )
    except Exception as _e:
        # 实在编码失败，退化成纯文本
        result_json = json.dumps(
            {{
                "result_type": "text_repr",
                "result": repr(_r),
                "error": f"encode_error: {{_e}}"
            }},
            ensure_ascii=False
        )

    print("<result_json>")
    print(result_json)
    print("</result_json>")

"""

        # 调用已有的执行工具，在子进程中运行代码
        exec_resp = await execute_python_code(python_wrapper, timeout=60)

        # 从 execute_python_code 的返回中提取标准输出 / 错误
        text_blocks = [b for b in exec_resp.content if b.get("type") == "text"]
        if not text_blocks:
            error_block: TextBlock = {
                "type": "text",
                "text": (
                    "[calculate_or_analysis_by_python_code] 执行失败："
                    "内置 execute_python_code 未返回任何文本输出。"
                ),
            }
            return ToolResponse(content=[error_block])

        raw_text = text_blocks[0]["text"]

        # 解析 <returncode>/<stdout>/<stderr>
        m_ret = re.search(r"<returncode>(.*?)</returncode>", raw_text, re.S)
        m_out = re.search(r"<stdout>(.*?)</stdout>", raw_text, re.S)
        m_err = re.search(r"<stderr>(.*?)</stderr>", raw_text, re.S)

        try:
            returncode = int(m_ret.group(1).strip()) if m_ret else -1
        except Exception:
            returncode = -1

        stdout = m_out.group(1) if m_out else ""
        stderr = m_err.group(1) if m_err else ""

        # 优先看包装代码是否输出 <result_error>
        m_err_tag = re.search(r"<result_error>(.*?)</result_error>", stdout, re.S)
        m_repr = re.search(r"<result_repr>(.*?)</result_repr>", stdout, re.S)
        m_json = re.search(r"<result_json>(.*?)</result_json>", stdout, re.S)

        if returncode != 0:
            # 子进程本身执行失败
            text = (
                "[calculate_or_analysis_by_python_code] 代码执行失败。\n"
                f"returncode = {returncode}\n"
                "stdout:\n"
                f"{stdout}\n\n"
                "stderr:\n"
                f"{stderr}\n\n"
                "请检查你生成的 Python 代码是否有语法错误或运行时错误，"
                "并确保最终结果赋值给变量 result。"
            )
            error_block: TextBlock = {
                "type": "text",
                "text": text,
            }
            return ToolResponse(
                content=[error_block],
                metadata={
                    "raw_stdout": stdout,
                    "raw_stderr": stderr,
                },
            )

        if m_err_tag:
            # result 未定义或打印 result 时出错
            tag_error = m_err_tag.group(1).strip()
            text = (
                "[calculate_or_analysis_by_python_code] 结果处理失败。\n"
                f"错误信息: {tag_error}\n\n"
                "stdout:\n"
                f"{stdout}\n\n"
                "stderr:\n"
                f"{stderr}\n\n"
                "请在代码中确保定义变量 result（例如 result = ...），"
                "并避免在打印 result 时抛出异常。"
            )
            error_block: TextBlock = {
                "type": "text",
                "text": text,
            }
            return ToolResponse(
                content=[error_block],
                metadata={
                    "raw_stdout": stdout,
                    "raw_stderr": stderr,
                },
            )

        if not m_repr:
            # 没拿到 result_repr，说明用户代码没有正确走到 result 打印逻辑
            text = (
                "[calculate_or_analysis_by_python_code] 未找到结果标记 <result_repr>。\n"
                "请确认你在代码中正确地将最终结果赋值给变量 result，"
                "并且没有覆盖或删除包装逻辑打印的标签。\n\n"
                "stdout:\n"
                f"{stdout}\n\n"
                "stderr:\n"
                f"{stderr}\n"
            )
            error_block: TextBlock = {
                "type": "text",
                "text": text,
            }
            return ToolResponse(
                content=[error_block],
                metadata={
                    "raw_stdout": stdout,
                    "raw_stderr": stderr,
                },
            )

        # 提取 result 的打印文本（中间部分可能是多行）
        result_text = m_repr.group(1).strip("\n")


        structured_result = result_text
        result_type = "text_repr"
        if m_json:
            try:
                payload = json.loads(m_json.group(1).strip())
                result_type = payload.get("result_type", "text_repr")
                structured_result = payload.get("result", result_text)
            except Exception:
                structured_result = result_text
                result_type = "text_repr"

        if description is None:
            description = "自定义 Python 数据分析/计算结果"

        ref_id = self._save_calc_result(
                tool_name="calculate_or_analysis_by_python_code",
                sub_type=None,
                result=structured_result,
                result_type=result_type,
                code=code,
                description=description,
            )

        text_block: TextBlock = {
            "type": "text",
            "text": (
                "[calculate_or_analysis_by_python_code] 代码执行完成。\n"
                f"描述: {description}\n"
                "结果（以下为你在代码中 print(result) 输出的内容）：\n"
                f"{result_text}\n"
                f"Material 已写入 ref_id='{ref_id}'（JSON 格式）\n"
            ),
        }
        
        return ToolResponse(content=[text_block], metadata={"ref_id": ref_id})            