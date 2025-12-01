# from __future__ import annotations
import time
from pathlib import Path
import re
from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse

import base64
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm


from agentscope.message import TextBlock, ImageBlock, Base64Source
from agentscope.tool import ToolResponse
from agentscope.tool._coding._python import execute_python_code
from ..memory.short_term import ShortTermMemoryStore

import config


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

cfg = config.Config()
matplotlib.use("Agg")


FONT_PATH = cfg.get_font_path("chinese")

if FONT_PATH:
    try:
        # 把字体文件注册进 Matplotlib
        fm.fontManager.addfont(FONT_PATH)
        font_prop = fm.FontProperties(fname=FONT_PATH)

        # 设置为全局默认字体
        matplotlib.rcParams["font.family"] = font_prop.get_name()
        matplotlib.rcParams["font.sans-serif"] = [font_prop.get_name()]
        matplotlib.rcParams["axes.unicode_minus"] = False

        print(f"[graphic_tools] 已加载中文字体: {FONT_PATH} ({font_prop.get_name()})")
    except Exception as e:
        print(f"[graphic_tools] 加载中文字体失败: {FONT_PATH} -> {e}")
        matplotlib.rcParams["axes.unicode_minus"] = False
else:
    print("[graphic_tools] 未在 config.yaml 中找到 font.chinese 配置")
    matplotlib.rcParams["axes.unicode_minus"] = False


class GraphicTools:

    def __init__(self, short_term: ShortTermMemoryStore) -> None:
        self.short_term = short_term
        self.tools = [
            self.generate_chart_by_python_code,
            self.generate_chart_by_template,
        ]

    def _save_chart(self, img_bytes: bytes, chart_id: str) -> str:
        """保存 PNG 图片到本地 charts/ 目录，返回文件路径"""

        file_path = self.short_term.manuscript_dir / f"{chart_id}.png"
        with open(file_path, "wb") as f:
            f.write(img_bytes)
        return file_path


    def _apply_common_style(
        self,
        ax,
        title: Optional[str],
        x_label: Optional[str],
        y_label: Optional[str],
    ) -> None:
        """统一设置标题、轴标签、网格等基础样式。"""
        if title:
            ax.set_title(title)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        ax.grid(True, linestyle="--", alpha=0.3)


    def _fig_to_base64(self,fig) -> str:
        """将 matplotlib Figure 转为 base64 字符串。"""
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_bytes = buf.getvalue()
        return base64.b64encode(img_bytes).decode("ascii")


    def _validate_series_xy(self,data: Dict[str, Any]) -> tuple[List[Any], List[Dict[str, Any]]]:
        """
        校验并提取适用于 line / bar / stacked_bar 的数据格式:
        {
            "x": [...],
            "series": [
                {"name": "收入", "values": [...]},
                {"name": "利润", "values": [...]},
            ]
        }
        """
        x = data.get("x")
        series = data.get("series")

        if not isinstance(x, list) or not isinstance(series, list) or len(series) == 0:
            raise ValueError(
                "数据格式应为: {'x': [...], 'series': [{'name': str, 'values': [...]}, ...]}"
            )

        n = len(x)
        for s in series:
            values = s.get("values")
            if not isinstance(values, list) or len(values) != n:
                raise ValueError("series 中每个 'values' 必须为列表，且长度与 x 相同")

        return x, series


    def _plot_line(self, ax, data: Dict[str, Any]) -> None:
        x, series = self._validate_series_xy(data)
        for s in series:
            y = s["values"]
            label = s.get("name")
            ax.plot(x, y, marker="o", label=label)

        if any(s.get("name") for s in series):
            ax.legend()


    def _plot_bar(self,ax, data: Dict[str, Any]) -> None:
        x, series = self._validate_series_xy(data)
        n_series = len(series)
        index = np.arange(len(x))
        width = 0.8 / max(n_series, 1)

        for i, s in enumerate(series):
            values = s["values"]
            label = s.get("name")
            offset = (i - (n_series - 1) / 2) * width
            ax.bar(index + offset, values, width=width, label=label)

        ax.set_xticks(index)
        ax.set_xticklabels(x)

        if any(s.get("name") for s in series):
            ax.legend()


    def _plot_stacked_bar(self,ax, data: Dict[str, Any]) -> None:
        x, series = self._validate_series_xy(data)
        index = np.arange(len(x))
        bottom = np.zeros(len(x))

        for s in series:
            values = np.array(s["values"], dtype=float)
            label = s.get("name")
            ax.bar(index, values, bottom=bottom, label=label)
            bottom += values

        ax.set_xticks(index)
        ax.set_xticklabels(x)

        if any(s.get("name") for s in series):
            ax.legend()


    def _plot_bar_line(self, ax, data: Dict[str, Any]) -> None:
        """
        柱线组合图数据格式:
        {
            "x": [...],
            "bar_series": [
                {"name": "成交量", "values": [...]},
                ...
            ],
            "line_series": [
                {"name": "收盘价", "values": [...]},
                ...
            ]
        }
        """
        x = data.get("x")
        bar_series = data.get("bar_series")
        line_series = data.get("line_series")

        if not isinstance(x, list) or not isinstance(bar_series, list) or not isinstance(
            line_series, list
        ):
            raise ValueError(
                "bar_line 图数据格式应为: "
                "{'x': [...], 'bar_series': [...], 'line_series': [...]} "
                "且三者均为列表"
            )

        n = len(x)
        index = np.arange(n)

        for s in bar_series:
            values = s.get("values")
            if not isinstance(values, list) or len(values) != n:
                raise ValueError("bar_series 中每个 'values' 长度必须与 x 一致")

        for s in line_series:
            values = s.get("values")
            if not isinstance(values, list) or len(values) != n:
                raise ValueError("line_series 中每个 'values' 长度必须与 x 一致")

        # 柱状图
        n_bar = len(bar_series)
        width = 0.8 / max(n_bar, 1)
        bar_cmap = plt.get_cmap("Set2")
        line_cmap = plt.get_cmap("Set1")

        bar_colors = bar_cmap(np.linspace(0, 1, len(bar_series)))
        line_colors = line_cmap(np.linspace(0, 1, len(line_series)))

        for i, (s, c) in enumerate(zip(bar_series, bar_colors)):
            values = s["values"]
            label = s.get("name")
            offset = (i - (n_bar - 1) / 2) * width
            ax.bar(index + offset, values, width=width, label=label, color=c)

        ax.set_xticks(index)
        ax.set_xticklabels(x)

        # 线图放在第二个 y 轴上（金融研报中常见：左轴是量，右轴是价）
        ax2 = ax.twinx()
        for s, c in zip(line_series, line_colors):
            values = s["values"]
            label = s.get("name")
            ax2.plot(index, values, marker="o", linestyle="-", label=label, color=c)

    
        # 合并图例
        handles, labels = [], []
        for axis in (ax, ax2):
            h, lab = axis.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(lab)
        if labels:
            ax.legend(handles, labels, loc="best")


    def _plot_pie(self,ax, data: Dict[str, Any]) -> None:
        """
        饼状图数据格式:
        {
            "labels": ["A股", "美股", ...],
            "values": [10, 20, ...]
        }
        """
        labels = data.get("labels")
        values = data.get("values")

        if not isinstance(labels, list) or not isinstance(values, list):
            raise ValueError("pie 图数据格式应为: {'labels': [...], 'values': [...]}")

        if len(labels) != len(values):
            raise ValueError("pie 图中 labels 与 values 长度必须一致")

        ax.pie(
            values,
            labels=labels,
            autopct="%.1f%%",
            startangle=90,
            counterclock=False,
        )
        ax.axis("equal")  # 饼图为正圆


    def _plot_scatter(self, ax, data: Dict[str, Any]) -> None:
        """
        散点图数据格式:
        {
            "x": [...],
            "y": [...],
            "group": [... 可选, 与 x 同长度, 分类标签]
        }
        """
        x = data.get("x")
        y = data.get("y")
        group = data.get("group")

        if not isinstance(x, list) or not isinstance(y, list):
            raise ValueError("scatter 图数据格式应为: {'x': [...], 'y': [...]}")

        if len(x) != len(y):
            raise ValueError("scatter 图中 x 与 y 长度必须一致")

        if isinstance(group, list) and len(group) == len(x):
            # 按 group 分类绘制
            unique_groups = list(dict.fromkeys(group))  # 保持顺序 dedup
            for g in unique_groups:
                xs = [xi for xi, gi in zip(x, group) if gi == g]
                ys = [yi for yi, gi in zip(y, group) if gi == g]
                ax.scatter(xs, ys, label=str(g), alpha=0.8)
            ax.legend()
        else:
            ax.scatter(x, y, alpha=0.8)


    def _plot_regression(self, ax, data: Dict[str, Any]) -> None:
        """
        回归线图数据格式:
        {
            "x": [...],
            "y": [...],
            "point_label": "样本点说明（可选）"
        }
        """
        x = data.get("x")
        y = data.get("y")
        point_label = data.get("point_label", "样本点")

        if not isinstance(x, list) or not isinstance(y, list):
            raise ValueError("regression 图数据格式应为: {'x': [...], 'y': [...]}")

        if len(x) != len(y):
            raise ValueError("regression 图中 x 与 y 长度必须一致")

        if len(x) < 2:
            raise ValueError("回归线至少需要两个样本点")

        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)

        # 散点
        ax.scatter(x_arr, y_arr, alpha=0.8, label=point_label)

        # 一阶线性回归
        k, b = np.polyfit(x_arr, y_arr, 1)
        order = np.argsort(x_arr)
        x_sorted = x_arr[order]
        y_pred = k * x_sorted + b

        ax.plot(x_sorted, y_pred, linestyle="--", linewidth=2, label="线性回归")
        ax.legend()


    async def generate_chart_by_template(
        self,
        chart_type: Literal[
            "line",
            "bar",
            "bar_line",
            "pie",
            "stacked_bar",
            "scatter",
            "regression",
        ],
        data: Dict[str, Any],
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        style: str | None = None,
        figsize: list[float] | None = None,
    ) -> ToolResponse:
        """使用预定义模版自动生成金融研报中常见图表，并返回图表chart_id用于引用。

        Args:
            chart_type:
                图表类型，可选:
                - "line": 折线图
                - "bar": 柱状图
                - "bar_line": 柱线组合图
                - "pie": 饼状图
                - "stacked_bar": 堆积柱状图
                - "scatter": 散点图
                - "regression": 带回归线的散点图
            data (dict):
                图表数据，必须是可 JSON 序列化的结构。不同图表类型对应的数据格式为:
                1) 对于 "line" / "bar" / "stacked_bar":
                    {
                        "x": [x1, x2, ...],
                        "series": [
                            {"name": "收入", "values": [y1, y2, ...]},
                            {"name": "利润", "values": [y1, y2, ...]},
                        ]
                    }

                2) 对于 "bar_line":
                    {
                        "x": [...],
                        "bar_series": [
                            {"name": "成交量", "values": [...]},
                            ...
                        ],
                        "line_series": [
                            {"name": "收盘价", "values": [...]},
                            ...
                        ]
                    }

                3) 对于 "pie":
                    {
                        "labels": ["A股", "美股", ...],
                        "values": [10, 20, ...]
                    }

                4) 对于 "scatter":
                    {
                        "x": [...],
                        "y": [...],
                        "group": [... 可选, 与 x 同长度, 分类标签]
                    }

                5) 对于 "regression":
                    {
                        "x": [...],
                        "y": [...],
                        "point_label": "样本点说明(可选)"
                    }

            title (str | None):
                图表标题。
            x_label (str | None):
                X 轴名称。
            y_label (str | None):
                Y 轴名称。
            style (str | None):
                matplotlib 样式字符串，例如 "ggplot"、"seaborn-v0_8"。
                若为空则使用默认样式。
            figsize (List[float] | None):
                图尺寸 [宽, 高]，单位英寸，例如 [8, 4]。为空则使用默认尺寸。

        Returns:
            ToolResponse:
                content 中包含:
                - TextBlock: 简要描述生成的图表类型和参数
        """
        # matplotlib 一些通用设置
        plt.rcParams["axes.unicode_minus"] = False  # 避免负号显示为方块

        # 可选样式
        if style:
            try:
                plt.style.use(style)
            except Exception:
                # 样式错误时忽略，保持默认
                pass

        if not figsize or len(figsize) != 2:
            figsize = [8.0, 4.5]

        try:
            fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))

            # 根据 chart_type 调用不同模板
            if chart_type == "line":
                self._plot_line(ax, data)
            elif chart_type == "bar":
                self._plot_bar(ax, data)
            elif chart_type == "bar_line":
                self._plot_bar_line(ax, data)
            elif chart_type == "pie":
                self._plot_pie(ax, data)
            elif chart_type == "stacked_bar":
                self._plot_stacked_bar(ax, data)
            elif chart_type == "scatter":
                self._plot_scatter(ax, data)
            elif chart_type == "regression":
                self._plot_regression(ax, data)
            else:
                raise ValueError(
                    f"未知的 chart_type: {chart_type}，"
                    "应为 'line' | 'bar' | 'bar_line' | 'pie' | "
                    "'stacked_bar' | 'scatter' | 'regression'"
                )

            # 对于非饼图，统一设置样式
            if chart_type != "pie":
                self._apply_common_style(ax, title, x_label, y_label)
            else:
                # 饼图通常标题单独居中
                if title:
                    ax.set_title(title)

            img_b64 = self._fig_to_base64(fig)
            plt.close(fig)

            chart_id = f"chart_{int(time.time() * 1000)}"
            # 保存到本地
            file_path = self._save_chart(base64.b64decode(img_b64), chart_id)

            text_block: TextBlock = {
                "type": "text",
                "text": (
                    f"[generate_chart_by_template]图表已生成（ID: {chart_id}，类型: {chart_type}）。\n"
                    f"请在 Manuscript 的 Markdown 中按如下方式引用该图表（chart_id 作为占位）：\n\n"
                    f"![请在这里填写图的说明文字，如果没有则为空](chart:{chart_id})\n\n"
                ),
            }

            return ToolResponse(
                content=[text_block],
            )

        except Exception as e:
            # 出错时返回错误信息（纯文本）
            error_block: TextBlock = {
                "type": "text",
                "text": f"[generate_chart_by_template] 生成图表失败: {e}",
            }
            return ToolResponse(content=[error_block])

    async def generate_chart_by_python_code(
        self,
        code: str,
        caption: str | None = None,
    ) -> ToolResponse:
        """执行自行编写的绘图代码，并返回图表chart_id用于引用。
            1. 你需要编写绘图逻辑的 Python 代码片段，例如:
                plt.figure()
                plt.plot(dates, prices, marker="o")
                plt.title("XX 股票价格走势")
            无需写 if __name__ == "__main__" 等结构。
            2. 已自动导入 matplotlib.pyplot 为 plt，并导入 seaborn 为 sns。
            3. 将所有绘制逻辑限定在单个图表中（当前 Figure），避免生成多张图。

        Args:
            code (str):
                绘图 Python 代码片段。
            caption (str | None):
                推荐的图注文字，若为空则使用默认提示。

        Returns:
            ToolResponse:
                content 中包含:
                - TextBlock: 简要描述生成的图表

        """
        # 在子进程中使用非交互式后端
        python_wrapper = f"""
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 绘图代码开始 ====
{code}
# ==== 绘图代码结束 ====

# 若用户没有主动创建图，则使用当前图
buf = io.BytesIO()
plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
buf.seek(0)
img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
print("<img_b64>" + img_b64 + "</img_b64>")
"""

        # 调用内置执行 Python 代码工具（在临时文件 + 子进程中执行）
        exec_resp = await execute_python_code(python_wrapper, timeout=60)

        # 解析内置工具返回的 <returncode>/<stdout>/<stderr> 格式
        # execute_python_code 会返回 ToolResponse(content=[TextBlock(...)]),
        # 其中 text 形如: "<returncode>0</returncode><stdout>...</stdout><stderr>...</stderr>"
        text_blocks = [b for b in exec_resp.content if b["type"] == "text"]
        if not text_blocks:
            error_block: TextBlock = {
                "type": "text",
                "text": (
                    "[generate_chart_by_python_code] 执行失败："
                    "内置 execute_python_code 未返回任何文本输出。"
                ),
            }
            return ToolResponse(content=[error_block])

        raw_text = text_blocks[0]["text"]
        m_ret = re.search(r"<returncode>(.*?)</returncode>", raw_text, re.S)
        m_out = re.search(r"<stdout>(.*?)</stdout>", raw_text, re.S)
        m_err = re.search(r"<stderr>(.*?)</stderr>", raw_text, re.S)

        try:
            returncode = int(m_ret.group(1).strip()) if m_ret else -1
        except Exception:
            returncode = -1

        stdout = m_out.group(1) if m_out else ""
        stderr = m_err.group(1) if m_err else ""

        # 从 stdout 中提取 <img_b64>...</img_b64>
        m_img = re.search(r"<img_b64>(.*?)</img_b64>", stdout, re.S)

        if returncode != 0 or not m_img:
            # 执行错误或没有成功输出图片
            text = (
                "[generate_chart_by_python_code] 图表生成失败。\n"
                f"returncode = {returncode}\n"
                "stdout:\n"
                f"{stdout}\n\n"
                "stderr:\n"
                f"{stderr}\n\n"
                "请检查你生成的 Python 绘图代码是否报错，"
                "以及是否成功绘制了图像。"
            )
            error_block: TextBlock = {
                "type": "text",
                "text": text,
            }
            return ToolResponse(content=[error_block])

        img_b64 = m_img.group(1).strip()
        img_bytes = base64.b64decode(img_b64)

        # 生成 chart_id 并保存到本地
        chart_id = f"chart_{int(time.time() * 1000)}"
        file_path = self._save_chart(img_bytes, chart_id)


        text_block: TextBlock = {
            "type": "text",
            "text": (
                f"[generate_chart_by_python_code] 图表已生成（ID: {chart_id}）。\n"
                f"请在 Manuscript 的 Markdown 中按如下方式引用该图表（chart_id 作为占位）：\n\n"
                f"![请在这里填写图的说明文字，如果没有则为空](chart:{chart_id})\n\n"
            ),
        }

        return ToolResponse(
            content=[text_block],
            metadata={
                "chart_id": chart_id,
                "file_path": file_path,
            },
        )