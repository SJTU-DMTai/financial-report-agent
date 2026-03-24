# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from pathlib import Path

from src.memory.short_term import ShortTermMemoryStore
from src.tools.graphic_tools import GraphicTools

CURRENT_FILE = Path(__file__).resolve()  # tests/test_graphictools.py
PROJECT_ROOT = CURRENT_FILE.parent.parent

async def run_charts(short_term: ShortTermMemoryStore) -> None:
    gt = GraphicTools(short_term=short_term)

    print(f"[test_graphic_tools] manuscript_dir = {short_term.manuscript_dir}")

    # 1) line
    resp = await gt.generate_chart_by_template(
        chart_type="line",
        title="折线图：收入与利润",
        x_label="年份",
        y_label="金额(亿元)",
        data={
            "x": ["2022", "2023", "2024", "2025"],
            "series": [
                {"name": "收入", "values": [100, 120, 150, 180]},
                {"name": "利润", "values": [10, 12, 18, 22]},
            ],
        },
        figsize=[8.0, 4.5],
        style=None,
    )
    print(resp.content[0]["text"])

    # 2) bar
    resp = await gt.generate_chart_by_template(
        chart_type="bar",
        title="柱状图：各季度销量",
        x_label="季度",
        y_label="销量(万台)",
        data={
            "x": ["Q1", "Q2", "Q3", "Q4"],
            "series": [{"name": "销量", "values": [25, 28, 35, 40]}],
        },
        figsize=[8.0, 4.5],
    )
    print(resp.content[0]["text"])

    # 3) stacked_bar
    resp = await gt.generate_chart_by_template(
        chart_type="stacked_bar",
        title="堆积柱状图：收入结构",
        x_label="年份",
        y_label="金额(亿元)",
        data={
            "x": ["2023", "2024", "2025"],
            "series": [
                {"name": "国内", "values": [80, 95, 110]},
                {"name": "海外", "values": [20, 30, 45]},
            ],
        },
        figsize=[8.0, 4.5],
    )
    print(resp.content[0]["text"])

    # 4) bar_line（柱线组合）
    resp = await gt.generate_chart_by_template(
        chart_type="bar_line",
        title="柱线组合：成交量与收盘价",
        x_label="交易日",
        y_label="成交量",
        data={
            "x": ["D1", "D2", "D3", "D4", "D5"],
            "bar_series": [{"name": "成交量", "values": [120, 180, 150, 220, 200]}],
            "line_series": [{"name": "收盘价", "values": [10.2, 10.6, 10.4, 11.0, 10.8]}],
        },
        figsize=[8.5, 4.8],
    )
    print(resp.content[0]["text"])

    # 5) pie
    resp = await gt.generate_chart_by_template(
        chart_type="pie",
        title="饼图：市场分布",
        data={"labels": ["A股", "港股", "美股"], "values": [55, 25, 20]},
        figsize=[8.0, 4.5],
    )
    print(resp.content[0]["text"])

    # 6) scatter（带 group）
    resp = await gt.generate_chart_by_template(
        chart_type="scatter",
        title="散点图：样本分布（含分组）",
        x_label="特征X",
        y_label="特征Y",
        data={
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1.2, 1.9, 3.1, 3.8, 5.2, 5.9],
            "group": ["A", "A", "A", "B", "B", "B"],
        },
        figsize=[8.0, 4.5],
    )
    print(resp.content[0]["text"])

    # 7) regression
    resp = await gt.generate_chart_by_template(
        chart_type="regression",
        title="回归图：拟合效果",
        x_label="X",
        y_label="Y",
        data={
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1.1, 2.0, 2.8, 4.2, 5.1, 5.9],
            "point_label": "样本点",
        },
        figsize=[8.0, 4.5],
    )
    print(resp.content[0]["text"])

    # 8) python_code
    code = r"""
import numpy as np
x = np.arange(0, 10)
y = np.sin(x)
plt.figure(figsize=(8, 4.5))
plt.plot(x, y, marker="o", linewidth=2)
plt.title("python_code：sin(x) 曲线")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True, linestyle="--", alpha=0.3)
"""
    resp = await gt.generate_chart_by_python_code(code=code, caption="sin(x)")
    print(resp.content[0]["text"])

    print("[test_graphic_tools] done. 请到 manuscript_dir 查看生成的 .png 文件。")

    # 9) line：多系列 + x 标签很多且很长（触发：xticks 稀疏 + 旋转 + legend 右移 + 标题换行）
    resp = await gt.generate_chart_by_template(
        chart_type="line",
        title="折线图：多条指标对比（用于测试长标题自动换行与图例过多时右侧展示）",
        x_label="月份",
        y_label="指标值",
        data={
            "x": [f"2025-{m:02d}-月度" for m in range(1, 13)],
            "series": [
                {"name": "收入", "values": [100, 102, 105, 108, 110, 112, 115, 118, 121, 125, 130, 136]},
                {"name": "利润", "values": [10, 11, 10, 12, 13, 13, 14, 15, 16, 17, 18, 20]},
                {"name": "毛利率", "values": [22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28]},
                {"name": "费用率", "values": [12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16]},
                {"name": "经营现金流", "values": [8, 7, 9, 6, 10, 12, 11, 13, 12, 14, 16, 18]},
            ],
        },
        figsize=[10.0, 4.8],
    )
    print(resp.content[0]["text"])

    # 10) bar：多系列（并列柱）+ x 很多（触发：xticks 稀疏/旋转 + legend 右移）
    resp = await gt.generate_chart_by_template(
        chart_type="bar",
        title="柱状图：多地区多品类销量（用于测试并列柱与密集类别轴）",
        x_label="品类",
        y_label="销量(万件)",
        data={
            "x": [f"品类{i}" for i in range(1, 17)],
            "series": [
                {"name": "华东", "values": [12, 14, 11, 16, 18, 15, 13, 12, 19, 21, 17, 16, 18, 20, 22, 23]},
                {"name": "华南", "values": [10, 12, 9, 13, 15, 14, 11, 10, 16, 18, 15, 14, 16, 17, 19, 20]},
                {"name": "华北", "values": [9, 11, 8, 12, 14, 13, 10, 9, 15, 16, 13, 12, 14, 16, 17, 18]},
                {"name": "西部", "values": [7, 8, 6, 9, 10, 9, 8, 7, 11, 12, 10, 9, 10, 11, 12, 13]},
            ],
        },
        figsize=[11.0, 4.8],
    )
    print(resp.content[0]["text"])

    # 11) stacked_bar：堆积层数更多（5 层）+ legend 右移
    resp = await gt.generate_chart_by_template(
        chart_type="stacked_bar",
        title="堆积柱状图：收入结构（更多分项，用于测试堆积层颜色与图例布局）",
        x_label="年份",
        y_label="金额(亿元)",
        data={
            "x": ["2021", "2022", "2023", "2024", "2025"],
            "series": [
                {"name": "国内-线上", "values": [25, 30, 35, 42, 50]},
                {"name": "国内-线下", "values": [30, 34, 40, 46, 52]},
                {"name": "海外-直销", "values": [12, 14, 16, 18, 22]},
                {"name": "海外-渠道", "values": [10, 12, 14, 16, 18]},
                {"name": "其他", "values": [3, 4, 5, 6, 7]},
            ],
        },
        figsize=[10.5, 4.8],
    )
    print(resp.content[0]["text"])

    # 12) bar_line：多柱（2 组）+ 多线（3 条）+ x 很多（触发：xticks 稀疏/旋转 + 合并 legend）
    resp = await gt.generate_chart_by_template(
        chart_type="bar_line",
        title="柱线组合：成交量/换手率（柱）与价格/均线（线）——用于测试双轴多序列与图例合并",
        x_label="交易日",
        y_label="柱(成交量/换手率)",
        data={
            "x": [f"D{i}" for i in range(1, 16)],
            "bar_series": [
                {"name": "成交量", "values": [120, 180, 150, 220, 200, 260, 240, 210, 230, 280, 300, 270, 250, 260, 310]},
                {"name": "换手率(%)", "values": [2.1, 2.8, 2.4, 3.2, 3.0, 3.6, 3.3, 3.1, 3.0, 3.8, 4.0, 3.7, 3.4, 3.5, 4.2]},
            ],
            "line_series": [
                {"name": "收盘价", "values": [10.2, 10.6, 10.4, 11.0, 10.8, 11.3, 11.1, 10.9, 11.0, 11.5, 11.7, 11.4, 11.2, 11.3, 11.8]},
                {"name": "MA(5)", "values": [10.2, 10.4, 10.4, 10.55, 10.6, 10.82, 10.92, 11.02, 11.02, 11.16, 11.26, 11.3, 11.36, 11.42, 11.48]},
                {"name": "MA(10)", "values": [10.2, 10.4, 10.4, 10.55, 10.6, 10.72, 10.77, 10.83, 10.89, 10.98, 11.07, 11.13, 11.17, 11.22, 11.27]},
            ],
        },
        figsize=[11.5, 5.2],
    )
    print(resp.content[0]["text"])

    # 13) pie：多分类（10 类），测试扇区颜色与 legend 右侧展示
    resp = await gt.generate_chart_by_template(
        chart_type="pie",
        title="饼图：收入构成（10 分类，用于测试颜色区分与右侧图例）",
        data={
            "labels": [f"业务{i}" for i in range(1, 11)],
            "values": [18, 14, 12, 10, 9, 8, 7, 7, 8, 7],
        },
        figsize=[10.0, 5.0],
    )
    print(resp.content[0]["text"])

    # 14) scatter：3~4 组，且 group 标签顺序被打乱（测试：dict.fromkeys 去重保持出现顺序）
    resp = await gt.generate_chart_by_template(
        chart_type="scatter",
        title="散点图：样本分布（多组且出现顺序混杂，用于测试分组去重顺序与图例）",
        x_label="特征X",
        y_label="特征Y",
        data={
            "x": [1, 1.5, 2.2, 2.8, 3.1, 3.7, 4.0, 4.5, 5.2, 5.8, 6.1, 6.6],
            "y": [1.1, 1.4, 2.0, 2.6, 3.2, 3.4, 4.1, 4.4, 5.0, 5.6, 6.2, 6.5],
            "group": ["B", "A", "C", "B", "A", "D", "C", "D", "B", "A", "D", "C"],
        },
        figsize=[9.5, 4.8],
    )
    print(resp.content[0]["text"])

    # 15) regression：噪声更大、点更多（测试：polyfit 稳定性 + 排序后画回归线）
    resp = await gt.generate_chart_by_template(
        chart_type="regression",
        title="回归图：带噪声样本（用于测试回归线与点样式）",
        x_label="广告投入(万元)",
        y_label="销售额(万元)",
        data={
            "x": [5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
            "y": [48, 60, 66, 79, 83, 95, 104, 110, 120, 131, 135, 150, 158],
            "point_label": "样本点（含噪声）",
        },
        figsize=[9.5, 4.8],
    )
    print(resp.content[0]["text"])

    # 16) python_code：混合柱 + 线（测试：用户代码里不显式 color 时的整体协调）
    code = r"""
import numpy as np
x = np.arange(1, 13)
bar1 = np.array([12, 14, 13, 16, 18, 17, 15, 14, 19, 21, 20, 22])
bar2 = np.array([9, 10, 8, 11, 13, 12, 10, 9, 14, 15, 14, 16])
line = np.array([10.2, 10.5, 10.4, 10.8, 11.1, 11.0, 10.9, 10.7, 11.2, 11.4, 11.3, 11.6])

plt.figure(figsize=(11, 5))
idx = np.arange(len(x))
w = 0.35
plt.bar(idx - w/2, bar1, width=w, label="成交量A")
plt.bar(idx + w/2, bar2, width=w, label="成交量B")
plt.plot(idx, line, marker="o", linewidth=2.2, label="价格")
plt.xticks(idx, [f"M{m}" for m in x], rotation=45, ha="right")
plt.title("python_code：柱+线混合（测试默认配色轮换在多类型元素上的表现）")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
"""
    resp = await gt.generate_chart_by_python_code(code=code, caption="bar+line")
    print(resp.content[0]["text"])


def run_workflow() -> None:
    short_term_dir = PROJECT_ROOT / "data" / "memory" / "short_term" / "history_short_term" / "20260122_203405_835679"
    short_term = ShortTermMemoryStore(
        base_dir=short_term_dir,
        do_post_init=False,
    )

    asyncio.run(run_charts(short_term))


if __name__ == "__main__":
    run_workflow()
    # 运行方式：
    # python -m tests.test_graphictools
