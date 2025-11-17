from __future__ import annotations

from pathlib import Path
import pdfkit
import re
from typing import List, Tuple
from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse

from memory.short_term import ShortTermMemoryStore
from .material_tools import *



class WriterTools:

    def __init__(self, short_term: ShortTermMemoryStore):
        self.short_term = short_term

    # ---- 内部函数 ----
    def _parse_outline_sections(self, outline: str) -> List[Tuple[str, str, str]]:
        """把 outline.md 划分为若干 section。

        返回: List[(section_id, title, body_markdown)]
        简化策略：
        - 以一级标题 `#` 作为章节分割点
        - section_id 形如 `sec_01_行业分析`，保证字典序 == 章节顺序
        """
        lines = outline.splitlines()
        sections: List[Tuple[str, str, str]] = []

        current_title = None
        current_body_lines: List[str] = []
        index = 0  # 用于编号

        def flush():
            nonlocal current_title, current_body_lines, index
            if current_title is None:
                return
            title = current_title.strip("# ").strip()
            # 简单 slug 化做 section_id
            slug = re.sub(r"\s+", "_", title)
            slug = re.sub(r"[^\w\-一-龥]", "", slug)  # 保留中文和常见字符
            index += 1
            prefix = f"{index:02d}"
            section_id = f"sec_{prefix}_{slug}"
            body = "\n".join(current_body_lines).strip()
            sections.append((section_id, title, body))
            current_title = None
            current_body_lines = []

        for line in lines:
            if line.startswith("# "):  # 一级标题
                flush()
                current_title = line
            else:
                if current_title is None:
                    # 出现在第一个 # 之前的内容可以直接忽略或归入引言
                    continue
                current_body_lines.append(line)

        flush()
        return sections


    # ---- Manuscript Tool ----
    def draft_manuscript_from_outline(
        self,
    ) -> ToolResponse:
        """根据现有的 outline.md 生成按章节拆分的 HTML 草稿骨架。
        调用此工具时，将根据大纲内容，自动创建对应章节的初始 HTML 草稿，并返回生成的章节 ID 列表。

        """
        outline = self.short_term.load_outline()
        if not outline.strip():
            return ToolResponse(
                content=[TextBlock(type="text", text="[draft_manuscript_from_outline] outline 为空")],
                metadata={"sections": []},
            )

        sections = self._parse_outline_sections(outline)

        section_ids = []
        for section_id, title, body_md in sections:
            # 这里先用最简单的 HTML stub，后续你可以换成真正的 markdown→HTML 渲染
            body_html = (
                f"<h1>{title}</h1>"
                "<hr/>"
                "<p>（请根据大纲要点在此撰写正文，可调用 Searcher 工具补充材料。）</p>"
                "<pre>"
                f"{body_md}"
                "</pre>"
            )
            self.short_term.save_manuscript_section(section_id, body_html)
            section_ids.append(section_id)

        text = "[draft_manuscript_from_outline] 已生成以下章节草稿:\n" + "\n".join(section_ids)
        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"sections": section_ids},
        )


    def read_manuscript_section(
        self,
        section_id: str,
    ) -> ToolResponse:
        """读取指定章节的 HTML 草稿。
        可调用此工具获取某个章节点的完整 HTML 内容，用于阅读。如果该章节不存在，将返回说明信息。

        Args:
            section_id (str):
                要读取的章节唯一标识符。
        """
        html = self.short_term.load_manuscript_section(section_id)
        if not html:
            html = f"[read_manuscript_section] section {section_id} 不存在或为空。"
        return ToolResponse(
            content=[TextBlock(type="text", text=html)],
            metadata={"section_id": section_id},
        )


    def replace_manuscript_section(
        self,
        section_id: str,
        new_html: str,
    ) -> ToolResponse:
        """
        以新的 HTML 内容替换指定章节。调用此工具可直接覆盖某个章节的草稿内容。
        调用时需提供章节 ID 以及更新后的完整 HTML 字符串。

            Args:
                section_id (str):
                    要更新的章节唯一标识符。
                new_html (str):
                    用于替换原章节的完整 HTML 内容。
            """
        self.short_term.save_manuscript_section(section_id, new_html)
        return ToolResponse(
            content=[TextBlock(type="text", text=f"[replace_manuscript_section] 已更新 {section_id}")],
            metadata={"section_id": section_id},
        )

    # ---- Graphic Tool（占位） ----
    def generate_chart(self, description: str, output_dir: str = "data/output/charts") -> ToolResponse:
        """根据描述生成图表并返回文件路径（占位实现）。"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        fake_path = path / "chart_placeholder.png"
        fake_path.write_bytes(b"")  # 占位空文件
        text = f"[generate_chart] 已生成图表: {fake_path}"
        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"chart_path": str(fake_path)},
        )

    # ---- PDF Converter ----
    def html_to_pdf(
        self,
        output_filename: str = "report.pdf",
        output_dir: str = "data/output/reports",
    ) -> ToolResponse:
        """将所有 Manuscript 章节按顺序合并并导出为 PDF 文件。

        此工具会自动收集当前已生成的全部章节（例如 sec_01_xxx.html 等），
        按章节编号的字典序排序后依次拼接为一个完整的 HTML 文档，并最终导出 PDF。
        仅需提供输出文件名及输出目录（如有需要），工具将负责：
        - 自动读取所有已有章节
        - 按章节序号排序
        - 拼接内容
        - 生成 PDF 文件并返回文件路径

        Args:
            output_filename (str):
                生成的 PDF 文件名。默认为 "report.pdf"。
            output_dir (str):
                PDF 输出目录。若不存在会自动创建。

        """
        assert self.short_term is not None

        # 1. 收集所有 section 文件并按文件名排序
        sec_files = sorted(self.short_term.manuscript_dir.glob("sec_*.html"))
        if not sec_files:
            text = "[html_to_pdf] 未找到任何章节文件 (sec_*.html)，无法生成 PDF。"
            return ToolResponse(
                content=[TextBlock(type="text", text=text)],
                metadata={"pdf_path": None},
            )

        # 2. 按顺序拼接 body 片段
        body_parts: list[str] = []
        section_ids: list[str] = []
        for path in sec_files:
            html_fragment = path.read_text(encoding="utf-8")
            if not html_fragment:
                continue
            body_parts.append(html_fragment)
            section_ids.append(path.stem)  # 比如 sec_01_行业分析

        if not body_parts:
            text = "[html_to_pdf] 所有章节为空，无法生成 PDF。"
            return ToolResponse(
                content=[TextBlock(type="text", text=text)],
                metadata={"pdf_path": None},
            )

        full_html = "<html><body>\n" + "\n<hr/>\n".join(body_parts) + "\n</body></html>"

        # 3. 调用 pdfkit 生成 PDF
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / output_filename

        # if wkhtmltopdf_path:
        #     config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        #     pdfkit.from_string(full_html, str(pdf_path), configuration=config)
        # else:
        #     pdfkit.from_string(full_html, str(pdf_path))
        pdfkit.from_string(full_html, str(pdf_path))
        text = f"[html_to_pdf] 已输出 PDF: {pdf_path}"
        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={
                "pdf_path": str(pdf_path),
                "sections": section_ids,
            },
        )


    # ---- 高级工具：调用 Searcher agent ----
    def searcher_tool(self, searcher: ReActAgent) -> Callable[[str], ToolResponse]:
        """把 Searcher agent 封装成 Writer 可见的工具函数。"""

        async def search_with_searcher(query: str) -> ToolResponse:
            """使用指定的 Searcher 工具 基于 query 执行一次检索并返回总结结果。

            Args:
                query (str): 检索需求的自然语言描述。

            """
            msg = Msg(
                name="Planner",
                content=query,
                role="user",
            )
            res = await searcher(msg)
            # 直接把 Searcher 的 Msg.content 作为工具返回内容，
            # 这样 Planner 在 ReAct 流程中就能看到原始检索总结
            return ToolResponse(
                content=res.content,
                metadata={"from_agent": searcher.name},
            )

        # closure 的 __name__ 会是 'search_with_searcher'，作为工具名即可。
        return search_with_searcher


# ---- Toolkit Builder ----
def build_writer_toolkit(
    short_term: ShortTermMemoryStore,
    searcher: ReActAgent,
) -> Toolkit:
    toolkit = Toolkit()
    writer_tools = WriterTools(short_term=short_term)
    toolkit.register_tool_function(writer_tools.draft_manuscript_from_outline)

    toolkit.register_tool_function(writer_tools.read_manuscript_section)

    toolkit.register_tool_function(writer_tools.replace_manuscript_section)
    # toolkit.register_tool_function(generate_chart)

    toolkit.register_tool_function(writer_tools.html_to_pdf)

    toolkit.register_tool_function(writer_tools.searcher_tool(searcher))

    # -------- Material Tools --------

    # ========================================
    # 通用读取工具 Reader
    # ========================================

    tools = MaterialTools(short_term=short_term)
    toolkit.register_tool_function(
        tools.read_table_material
    )
    toolkit.tools

    return toolkit
