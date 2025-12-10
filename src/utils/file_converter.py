# -*- coding: utf-8 -*-

from src.memory.short_term import ShortTermMemoryStore
import pdfkit
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import (
    TableItem,
    PictureItem,
    TextItem,
    SectionHeaderItem,
    ListItem,
    DocItem
)
import markdown
import re
import html
from typing import Optional, Union
import time


def add_citation(
    ref_id: str,
    detail: str,
    short_term: ShortTermMemoryStore,
    citation_index_map: dict,
    citations: list,
) -> Optional[int]:
    """
    根据 (ref_id, detail) 获取或创建引用编号。
    如果在 registry 里找不到 ref_id，则返回 None（表示不做替换）。
    """
    # 没有 get_material_meta 就直接放弃
    if not hasattr(short_term, "get_material_meta"):
        return None

    meta = short_term.get_material_meta(ref_id)
    # ref_id 在 registry 中不存在：不生成引用，调用方应保持原文不变
    if meta is None:
        return None

    key = (ref_id, detail)
    if key in citation_index_map:
        return citation_index_map[key]

    idx = len(citations) + 1
    citation_index_map[key] = idx
    citations.append(
        {
            "index": idx,
            "ref_id": ref_id,
            "detail": detail,
            "meta": meta,
        }
    )
    return idx


def _inject_refs(
    md_text: str,
    short_term: ShortTermMemoryStore,
    citation_index_map: dict,
    citations: list,
) -> str:
    """
    扫描 md_text 中的ref_id标记，
    支持以下形式，具有一定容错能力：
      - [ref_id:xxx|yyy]
      - ref_id:xxx|yyy
      - ref_id:xxx
      - [ref_id:xxx|yyy；ref_id:zzz]
      - [ref_id:xxx|yyy；zzz]

    其中 yyy 为可选位置描述可缺省。

    """
# ---------- 第一轮：处理显式写出的 ref_id:xxx 或 ref_id:xxx|yyy ----------
    # 不要求方括号存在，detail(yyy) 可缺省
    pattern_main = re.compile(
        r"ref_id[:=]([0-9A-Za-z_\u4e00-\u9fff\-]+)(?:\|([^；\]\s]+))?"
    )

    text = md_text
    result_parts = []
    last_end = 0

    for m in pattern_main.finditer(text):
        # 先把上一个匹配之后的原文拼上
        result_parts.append(text[last_end:m.start()])

        ref_id = m.group(1).strip()
        detail = (m.group(2) or "").strip()

        idx = add_citation(ref_id, detail, short_term, citation_index_map, citations)

        if idx is None:
            # 没找到 meta：不要替换，原样保留
            result_parts.append(text[m.start():m.end()])
        else:
            # 用 (1)、(2)… 的形式替换
            result_parts.append(f'(<a href="#ref-{idx}" class="ref">{idx}</a>)')

        last_end = m.end()

    # 拼接剩余部分
    result_parts.append(text[last_end:])
    text_after_main = "".join(result_parts)

    # ---------- 第二轮：处理 “(1)；zzz” 这种漏写 ref_id 的情况 ----------
    # 典型来源：
    #   原文：[ref_id:xxx|yyy；zzz]
    #   第一轮后：[(1)；zzz]
    # 这里尝试把 zzz 当作 ref_id 去解析，如果 registry 里没有，就保持原样。
    pattern_follow = re.compile(
        r'(\(<a href="#ref-(\d+)" class="ref">\2</a>\))\s*([；;])\s*([0-9A-Za-z_\u4e00-\u9fff\-]+)'
    )

    text = text_after_main
    result_parts = []
    last_end = 0

    for m in pattern_follow.finditer(text):
        result_parts.append(text[last_end:m.start()])

        anchor = m.group(1)          # 已有的 (1)
        sep = m.group(3)             # 分号：；或 ;
        ref_id2 = m.group(4).strip() # 疑似漏写 ref_id: 的部分

        idx2 = add_citation(ref_id2, "", short_term, citation_index_map, citations)

        if idx2 is None:
            # 第二个“疑似 ref_id”在 registry 中找不到：保持原样，不改
            result_parts.append(text[m.start():m.end()])
        else:
            anchor2 = f'(<a href="#ref-{idx2}" class="ref">{idx2}</a>)'
            result_parts.append(f"{anchor}{sep}{anchor2}")

        last_end = m.end()

    result_parts.append(text[last_end:])
    return "".join(result_parts)

def _replace_chart_placeholders(md_text: str, manuscript_dir: Path) -> str:
    """
    将 Markdown 中形如
        ![说明文字](chart:chart_id)
    的图片占位符替换为真实图片路径，例如
        ![说明文字](/abs/path/to/manuscript_dir/chart_id.png)
    """
    pattern = re.compile(
        r'!\[(?P<alt>.*?)\]\(chart:(?P<chart_id>[a-zA-Z0-9_\-]+)\)'
    )

    def repl(match: re.Match) -> str:
        alt = match.group("alt")
        chart_id = match.group("chart_id")
        img_path = manuscript_dir / f"{chart_id}.png"
        # 使用 POSIX 路径，避免反斜杠问题
        return f'![{alt}]({img_path.as_posix()})'

    return pattern.sub(repl, md_text)


def _normalize_tables(md_text: str) -> str:
    """
    1. 保证表格块前有一个空行；
    2. 删除表格内部行之间的“空行”，让表格行连续。
    """
    lines = md_text.splitlines()

    def is_table_line(line: str) -> bool:
        # 简单判断：以 | 开头并且包含至少两个 |
        return bool(re.match(r'^\s*\|.*\|\s*$', line))

    new_lines: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 2) 删除“表格行之间的空行”
        if stripped == "":
            if (
                i > 0
                and i + 1 < len(lines)
                and is_table_line(lines[i - 1])
                and is_table_line(lines[i + 1])
            ):
                # 这是“前后都是表格行”的空行，跳过
                continue

        # 1) 如果这是表格行，且前一行是非空非表格行，则补一个空行
        if is_table_line(line):
            if new_lines and new_lines[-1].strip() != "" and not is_table_line(new_lines[-1]):
                new_lines.append("")  # 补一行空行

        new_lines.append(line)

    return "\n".join(new_lines)


def _remove_word_count_tags(md_text: str) -> str:
    """
    删除形如（字数：712）的字数标记。
    兼容全角/半角括号：
        （字数：712）
        (字数：712)
    """
    pattern = re.compile(r"[（(]字数：\s*\d+[)）]")
    return pattern.sub("", md_text)


def md_to_pdf(
        short_term : ShortTermMemoryStore,
        output_filename: str = "report.pdf",
        output_dir: str = "data/output/reports",
    ):
    """将所有 Manuscript 章节按顺序合并并导出为 PDF 文件。
    """
    # 1. 收集所有 section 文件并按文件名排序
    sec_files = sorted(short_term.manuscript_dir.glob("sec_*.md"))
    if not sec_files:
        return "[md_to_pdf] 未找到任何章节文件 (sec_*.md)，无法生成 PDF。"

    # 2. 按顺序把每个 Markdown 章节渲染为 HTML，并拼接 body 片段
    body_parts: list[str] = []
    section_ids: list[str] = []

    # 全局引用信息：key=(ref_id, detail) -> index
    citation_index_map: dict[tuple[str, str], int] = {}
    citations: list[dict] = []

    for path in sec_files:
        md_text = path.read_text(encoding="utf-8").strip()
        if not md_text:
            continue
        
        md_text = _remove_word_count_tags(md_text)
        md_text = _inject_refs(md_text, short_term, citation_index_map, citations)
        md_text = _replace_chart_placeholders(md_text, short_term.manuscript_dir)
        md_text = _normalize_tables(md_text)
        # print(md_text+"\n\n")
        # 将 Markdown 转为 HTML 片段
        html_fragment = markdown.markdown(
            md_text,
            extensions=[
                "extra",        # 支持表格、脚注等常用扩展
                "tables",
                "fenced_code",
            ],
            output_format="html",
        )
        # print(html_fragment+"\n\n")
        body_parts.append(html_fragment)
        section_ids.append(path.stem)  # 比如 sec_01_行业分析

    if not body_parts:
        return "[md_to_pdf] 所有章节为空，无法生成 PDF。"

    # 章节之间用分隔线（也可以加上分页样式）
    body_html = '\n<hr style="page-break-after: always; border: none;" />\n'.join(body_parts)
    
    
    # 3. 生成附录区域（改为附录 -> 数据来源附录 + 预留部分）
    appendix_html = ""
    if citations:
        appendix_lines: list[str] = []
        appendix_lines.append('<hr style="page-break-before: always; border: none;" />')

        # 附录大标题
        appendix_lines.append('<h1>附录</h1>')

        # ========== 第一部分：数据来源附录 ==========
        appendix_lines.append('<h2>第一部分：数据来源附录</h2>')
        appendix_lines.append('<ol>')

        # 按 index 排序，保证顺序一致
        for c in sorted(citations, key=lambda x: x["index"]):
            idx = c["index"]
            ref_id = c["ref_id"]
            detail = c["detail"]
            meta = c["meta"]

            esc_ref_id = html.escape(ref_id)
            esc_detail = html.escape(detail) if detail else ""
            if meta is not None:
                esc_filename = html.escape(meta.filename)
                esc_m_type = html.escape(meta.m_type.value)
                esc_desc = html.escape(meta.description) if meta.description else ""
                esc_source = html.escape(meta.source) if meta.source else ""
            else:
                esc_filename = ""
                esc_m_type = ""
                esc_desc = ""
                esc_source = ""

            li_parts: list[str] = []
            li_parts.append(f'<p><strong>{esc_ref_id}</strong>')
            if esc_detail:
                li_parts.append(f'（引用字段/行：{esc_detail}）')
            li_parts.append('</p>')

            if meta is not None:
                li_parts.append("<ul>")
                li_parts.append(f"<li>文件名：{esc_filename}</li>")
                li_parts.append(f"<li>类型：{esc_m_type}</li>")
                if esc_desc:
                    li_parts.append(f"<li>描述：{esc_desc}</li>")
                if esc_source:
                    li_parts.append(f"<li>来源：{esc_source}</li>")
                li_parts.append("</ul>")
            else:
                li_parts.append("<p><em>警告：未在 registry 中找到该 ref_id 对应的元数据。</em></p>")

            li_html = f'<li id="ref-{idx}">' + "".join(li_parts) + "</li>"
            appendix_lines.append(li_html)

        appendix_lines.append("</ol>")

        # ========== 第二部分：预留（暂时为空） ==========

        # appendix_lines.append('<h2>第二部分：附加资料（预留）</h2>')
        # appendix_lines.append('<p>（本部分暂未添加内容。）</p>')

        appendix_html = "\n".join(appendix_lines)

    if appendix_html:
        body_html = body_html + "\n\n" + appendix_html
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: "Noto Sans CJK SC", "Microsoft YaHei", "SimSun", "Songti SC", sans-serif;
            font-size: 19px; 
            line-height: 1.7;
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-weight: bold;
        }}
        code, pre {{
            font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 12px auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        table, th, td {{
            border: 1px solid #ccc;
        }}
        th, td {{
            padding: 4px 8px;
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>"""

    if not output_filename.lower().endswith(".pdf"):
        output_filename = output_filename + ".pdf"

    # 3. 调用 pdfkit 生成 PDF
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / output_filename

    html_path = out_dir / f"source_{time.time()}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    options = {
        "encoding": "UTF-8",
        "enable-internal-links": None,
        "enable-local-file-access": None,
    }
    pdfkit.from_string(full_html, str(pdf_path), options=options)

    text = f"[md_to_pdf] 已输出 PDF: {pdf_path}"
    return text


def pdf_to_markdown(
    short_term : ShortTermMemoryStore,
):
    """
    将 demonstration report PDF 转换为结构化 Markdown，提取图片和表格并以特定格式嵌入。
    """
    # 1. 路径与目录准备
    output_dir = short_term.demonstration_dir

    pdf_files = list(short_term.demonstration_dir.glob("*.pdf")) + list(short_term.demonstration_dir.glob("*.PDF"))

    if not pdf_files:
        raise FileNotFoundError("demonstration_dir 中没有任何 PDF 文件")
    latest_pdf = max(pdf_files, key=lambda f: f.stat().st_mtime)
    
    images_out_path = output_dir / "images"
    
    # 创建目录
    images_out_path.mkdir(parents=True, exist_ok=True)

    # 2. 配置 Docling (开启图片生成)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = True  # 必须开启才能裁剪图片

    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"开始解析: {latest_pdf} ...")
    result = converter.convert(latest_pdf)
    doc = result.document

    # 3. 构建 Markdown 内容
    md_lines = []
    image_counter = 0
    table_counter = 0

    # Docling 的 iterate_items() 会按照阅读顺序遍历所有元素
    for item, level in doc.iterate_items():
        
        # --- 处理 标题 (Headers) ---
        if isinstance(item, SectionHeaderItem):
            # level 通常从 0 或 1 开始，加 1 个 # 保证至少是 H1
            prefix = "#" * (level + 1)
            md_lines.append(f"\n{prefix} {item.text}\n")

        # --- 处理 普通文本 (Text) ---
        elif isinstance(item, TextItem) and not isinstance(item, (TableItem, PictureItem, SectionHeaderItem)):
            # 过滤掉空的或无意义的文本
            if item.text.strip():
                md_lines.append(f"{item.text}\n")

        # --- 处理 列表 (List) ---
        elif isinstance(item, ListItem):
             md_lines.append(f"* {item.text}\n")

        # --- 处理 表格 (Tables) ---
        elif isinstance(item, TableItem):
            table_counter += 1
            # 获取标题，如果没有则使用默认名称
            # caption = item.captions if item.captions else f"Unknown Table {table_counter}"
            
            # # 导出为 Pandas DataFrame
            # df = item.export_to_dataframe(doc)
            
            # # 转换为 Markdown 格式的表格字符串 (LLM 读这个最容易)
            # # index=False 去掉 Pandas 的索引列，通常研报表格不需要索引
            # md_table_str = df.to_markdown(index=False)

            # # 构造特定标记块
            # block = (
            #     f"\n::: TABLE [Title: {caption}] :::\n"
            #     f"{md_table_str}\n"
            #     f"::: END TABLE :::\n"
            # )
            # md_lines.append(block)
            # print(f"   -> 提取表格: {caption}")

        # --- 处理 图片 (Images) ---
        elif isinstance(item, PictureItem):
            image_counter += 1
            # caption = f"Unknown Figure {image_counter}" # 设置默认值
            # if hasattr(item, "captions") and item.captions:
            #     caption_texts = []
            #     for c in item.captions:
            #         if hasattr(c, "text"):
            #             caption_texts.append(c.text)
            #         else:
            #             caption_texts.append(str(c))
                        
            #     if caption_texts:
            #         caption = " ".join(caption_texts)
            
            # # 获取并保存图片
            # image_obj = item.get_image(doc)
            # if image_obj:
                # if image_obj.width < 50 or image_obj.height < 50:
                #         continue
            #     # 图片命名: page_X_fig_Y.png
            #     page_no = item.prov[0].page_no
            #     img_filename = f"p{page_no}_fig_{image_counter}.png"
            #     img_save_path = images_out_path / img_filename
                
            #     image_obj.save(img_save_path)
                
            #     # 构造特定标记块 (存相对路径，方便 Markdown 预览)
            #     rel_path = f"images/{img_filename}"
            #     block = (
            #         f"\n::: IMAGE [Title: {caption}] :::\n"
            #         f"![{caption}]({rel_path})\n"
            #         f"::: END IMAGE :::\n"
            #     )
            #     md_lines.append(block)
            #     print(f"   -> 提取图片: {img_filename}")

    # 4. 写入最终的 Markdown 文件
    final_md_content = "\n".join(md_lines)
    short_term.save_demonstration(final_md_content)


    print(f"\n转换完成！")
    print(f"图片目录: {images_out_path}")
    print(f"提取统计: 表格 {table_counter} 个, 图片 {image_counter} 张")
