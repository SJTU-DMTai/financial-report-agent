# -*- coding: utf-8 -*-

# ========== 在导入任何模块之前设置环境变量 ==========
import os
import sys

# 禁用所有进度条显示
os.environ["TQDM_DISABLE"] = "1"  # 全局禁用 tqdm 进度条
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # 禁用 huggingface 进度条
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # 减少 transformers 输出

# Windows 特定编码修复
if sys.platform == 'win32':
    try:
        import io
        # 修复标准输出的编码
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
        
        # 设置控制台代码页为 UTF-8
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
    except:
        pass
    
from src.memory.working import Section, Segment
from src.memory.short_term import ShortTermMemoryStore
import pdfkit
from pdfkit.configuration import Configuration
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import markdown
import re
import html
from typing import Optional, Union, List, Dict, Tuple
import time
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

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



def _build_appendix_md(citations: List[Dict]) -> str:
    """
    根据 citations 构造“附录”的 Markdown 文本。

    要求与 add_citation / _inject_refs 完全对齐：
    - _inject_refs 在正文中插入 (<a href="#ref-{idx}" class="ref">{idx}</a>)
    - 这里必须生成带 id="ref-{idx}" 的锚点，供 href 跳转

    结构示例（Markdown）：
    # 附录
    ## 第一部分：数据来源附录

    1. <a id="ref-1"></a>**REF_ID**（引用字段/行：detail）

        - 文件名：`xxx`
        - 类型：`yyy`
        - 描述：`zzz`
        - 来源：`...`
    """
    if not citations:
        return ""

    lines: List[str] = []

    # 附录大标题
    lines.append("# 附录")
    lines.append("")

    # 第一部分标题
    lines.append("## 第一部分：数据来源附录")
    lines.append("")
    lines.append("以下为文中引用到的数据及材料来源：")
    lines.append("")

    # 按 index 排序，保证顺序稳定
    for c in sorted(citations, key=lambda x: x["index"]):
        idx = c["index"]
        ref_id = c["ref_id"]
        detail = c["detail"]
        meta = c["meta"]

        # 做一下 HTML 转义，避免特殊字符干扰
        esc_ref_id = html.escape(str(ref_id)) if ref_id is not None else ""
        esc_detail = html.escape(str(detail)) if detail else ""

        # 有序列表项首行：
        # 1. <a id="ref-1"></a>**REF_ID**（引用字段/行：detail）
        # 注意：<a id="ref-1"></a> 是真正的锚点，和正文里的 href="#ref-1" 完全对应
        first_line = f'1. <a id="ref-{idx}"></a>**{esc_ref_id}**'
        if esc_detail:
            first_line += f'（引用字段/行：{esc_detail}）'
        lines.append(first_line)
        lines.append("")
        # 元信息：使用缩进的无序列表
        if meta is not None:
            esc_filename = html.escape(str(meta.filename)) if getattr(meta, "filename", None) else ""
            esc_m_type = html.escape(str(meta.m_type.value)) if getattr(meta, "m_type", None) else ""

            if esc_filename:
                lines.append(f"    - 文件名：`{esc_filename}`")
            if esc_m_type:
                lines.append(f"    - 类型：`{esc_m_type}`")

            if getattr(meta, "description", None):
                esc_desc = html.escape(str(meta.description))
                lines.append(f"    - 描述：`{esc_desc}`")

            if getattr(meta, "source", None):
                esc_source = html.escape(str(meta.source))
                lines.append(f"    - 来源：`{esc_source}`")
        else:
            lines.append("    - *警告：未在 registry 中找到该 ref_id 对应的元数据*")

        # 每条引用之间空一行，增强可读性
        lines.append("")

    return "\n".join(lines)

def md_to_pdf(
        md_text: str,
        short_term : ShortTermMemoryStore,
        output_dir: str = "data/output/reports",
    ):
    """将所有 Manuscript 章节按顺序合并并导出为 PDF 文件。
    """
    # 全局引用信息：key=(ref_id, detail) -> index
    citation_index_map: dict[tuple[str, str], int] = {}
    citations: list[dict] = []
    cleaned_sections_md: list[str] = []

    for line in md_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            title = s.lstrip("#").strip()
            if title not in ("摘要", "研报摘要"):
                main_title = title
            break
        else:
            # 第一行不是标题，直接放弃自动抽取
            break

    md_text = _remove_word_count_tags(md_text)
    md_text = _inject_refs(md_text, short_term, citation_index_map, citations)
    md_text = _replace_chart_placeholders(md_text, short_term.manuscript_dir)
    md_text = _normalize_tables(md_text)
    cleaned_sections_md.append(md_text)

    if not cleaned_sections_md:
        return "[md_to_pdf] 所有章节为空，无法生成 PDF。"

    appendix_md = _build_appendix_md(citations)
    if appendix_md:
        cleaned_sections_md.append(appendix_md)

    # ==== 1) 把所有章节的 Markdown 合成一个字符串，用特殊标记做分隔 ====
    PAGE_BREAK_MARK = "[[PAGE_BREAK_TOKEN_123]]"
    combined_md = f"\n\n{PAGE_BREAK_MARK}\n\n".join(cleaned_sections_md)

    # ==== 2) 用带 toc 扩展的 Markdown 一次性转换 ====
    md = markdown.Markdown(
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "toc",
        ],
        output_format="html",
    )
    body_html_full = md.convert(combined_md)
    toc_inner_html = md.toc

    # ==== 3) 按标记把 body_html_full 切回每一章 ====
    split_token = f"<p>{PAGE_BREAK_MARK}</p>"
    html_sections = body_html_full.split(split_token)

    # ==== 4) 构造“目录页”的 HTML，把 markdown 生成的目录包一层，方便样式 & 分页 ====
    toc_page_html = f"""
<div class="toc-page">
  <h1 class="toc-title">目录</h1>
  {toc_inner_html}
</div>
""".strip()

    # 假定第一个章节就是摘要（sec_01），把目录页插在它后面
    if html_sections:
        html_sections.insert(1, toc_page_html)
    else:
        html_sections = [toc_page_html]

    # ==== 5) 把章节+目录按页分隔线连接回去 ====
    body_html = '\n<hr style="page-break-after: always; border: none;" />\n'.join(html_sections)
    
    # # 3. 生成附录区域（改为附录 -> 数据来源附录 + 预留部分）
    # appendix_html = ""
    # if citations:
    #     appendix_lines: list[str] = []
    #     appendix_lines.append('<hr style="page-break-before: always; border: none;" />')

    #     # 附录大标题
    #     appendix_lines.append('<h1>附录</h1>')

    #     # ========== 第一部分：数据来源附录 ==========
    #     appendix_lines.append('<h2>第一部分：数据来源附录</h2>')
    #     appendix_lines.append('<ol>')

    #     # 按 index 排序，保证顺序一致
    #     for c in sorted(citations, key=lambda x: x["index"]):
    #         idx = c["index"]
    #         ref_id = c["ref_id"]
    #         detail = c["detail"]
    #         meta = c["meta"]

    #         esc_ref_id = html.escape(ref_id)
    #         esc_detail = html.escape(detail) if detail else ""
    #         if meta is not None:
    #             esc_filename = html.escape(meta.filename)
    #             esc_m_type = html.escape(meta.m_type.value)
    #             esc_desc = html.escape(meta.description) if meta.description else ""
    #             esc_source = html.escape(meta.source) if meta.source else ""
    #         else:
    #             esc_filename = ""
    #             esc_m_type = ""
    #             esc_desc = ""
    #             esc_source = ""

    #         li_parts: list[str] = []
    #         li_parts.append(f'<p><strong>{esc_ref_id}</strong>')
    #         if esc_detail:
    #             li_parts.append(f'（引用字段/行：{esc_detail}）')
    #         li_parts.append('</p>')

    #         if meta is not None:
    #             li_parts.append("<ul>")
    #             li_parts.append(f"<li>文件名：{esc_filename}</li>")
    #             li_parts.append(f"<li>类型：{esc_m_type}</li>")
    #             if esc_desc:
    #                 li_parts.append(f"<li>描述：{esc_desc}</li>")
    #             if esc_source:
    #                 li_parts.append(f"<li>来源：{esc_source}</li>")
    #             li_parts.append("</ul>")
    #         else:
    #             li_parts.append("<p><em>警告：未在 registry 中找到该 ref_id 对应的元数据。</em></p>")

    #         li_html = f'<li id="ref-{idx}">' + "".join(li_parts) + "</li>"
    #         appendix_lines.append(li_html)

    #     appendix_lines.append("</ol>")

    #     # ========== 第二部分：预留（暂时为空） ==========

    #     # appendix_lines.append('<h2>第二部分：附加资料（预留）</h2>')
    #     # appendix_lines.append('<p>（本部分暂未添加内容。）</p>')

    #     appendix_html = "\n".join(appendix_lines)

    # if appendix_html:
    #     body_html = body_html + "\n\n" + appendix_html
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
        /* ==== 总标题：文档中第一个 h1 的特殊样式 ==== */
        body > h1:first-of-type {{
            font-size: 38px;
            font-weight: 1000;
            text-align: center;
            margin-top: 40px;
            margin-bottom: 30px;
            letter-spacing: 0.05em;
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

        .toc {{
            margin-top: 12px;
        }}
        .toc-title {{
            text-align: center;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            margin: 4px 0;
        }}
        .toc a {{
            text-decoration: none;
        }}

        .toc > ul > li > a {{
            font-weight: 700;
            color: #000;
        }}

        /* 二级目录 */
        .toc > ul > li > ul > li > a {{
            padding-left: 1.5em;
            font-weight: 500;
            color: #555;
        }}

        /* 三级目录 */
        .toc > ul > li > ul > li > ul > li > a {{
            padding-left: 3em;
            font-size: 0.9em;
            color: #777;
        }}

    </style>
</head>
<body>
{body_html}
</body>
</html>"""

    if not main_title:
        now = datetime.now()
        main_title = f"金融研报_{now.strftime('%Y%m%d_%H%M%S')}"

    # 生成文件名，替换掉不安全字符
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", main_title)
    safe_name = safe_name.strip()
    safe_name = safe_name + ".pdf"

    # 3. 调用 pdfkit 生成 PDF
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / safe_name

    html_path = out_dir / f"source_{time.time()}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    options = {
        "encoding": "UTF-8",
        "enable-internal-links": None,
        "enable-local-file-access": None,
        "footer-center": "[page] / [topage]",
        "footer-font-size": "10",
        "footer-spacing": "5",
    }
    
    cfg = config.Config()
    WKHTMLTOPDF_PATH = cfg.get_wkhtmltopdf_path()
    pdfkit_config = Configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
    pdfkit.from_string(full_html, str(pdf_path), options=options, configuration=pdfkit_config)

    text = f"[md_to_pdf] 已输出 PDF: {pdf_path}"
    return text


def detect_section(line: str) -> Tuple[int, str]:
    """
    Detect sections using multiple patterns.
    """
    line2 = re.sub(r"<span.+</span>", "", line).strip()
    # Pattern 1: Standard (1., 1.1, 1.1.1)
    pattern1 = r'#+\s+([0-9一二三四五六七八九十IVX]+(?:\.\d+)*)[、.\s章节]?\s*(.+)'
    match = re.match(pattern1, line2)
    if match:
        section_num = match.group(1)
        title = match.group(2).strip()
        level = min(section_num.count('.') + 2, 6)
        title = f"{section_num} {title}"
        if section_num.count('.') == 0:
            title += "."
        return level, title

    pattern2 = r'#+\s+<span id=.+></span>\s*(.+)$'
    match = re.match(pattern2, line)
    if match and re.search(r"[图表]\s?\d", line) is None:
        title = match.group(1).strip()
        level = line.split(" ")[0].count("#")
        return level, title
    return 0, ""


def pdf_to_markdown(
    pdf_path: Union[str, Path], output_path: Union[str, Path]
):
    """
    将 demonstration report PDF 转换为结构化 Markdown，提取图片和表格并以特定格式嵌入。
    """
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))
    text, metadata, images = text_from_rendered(rendered)

    processed_lines = []

    found_h1 = False
    found_section_1 = False
    summary_content = []
    skip_toc = False
    title = None

    for line in text.split('\n\n'):
        line = line.strip()
        if len(line) == 0:
            continue
        if title is None:
            if "[Table\\_Title]" not in text:
                if match := re.match(r"^#+ (.+)", line):
                    title = match.group(1) if detect_section(line)[0] == 0 else pdf_path.name
                    processed_lines.append("# " + title)
            else:
                if match := re.match(r"^#{1,3}\s+\[Table\\_Title] (.+)", line):
                    title = match.group(1)
                    processed_lines.append("# " + title)
        else:
            if match := re.match(r"^#{2,4}\s+\[Table\\_Summary] (.+)", line):
                processed_lines.append("## " + match.group(1) + " （摘要）")
            else:
                # Check if this is a TOC section (目录)
                if '目录' in line and line.startswith('#'):
                    skip_toc = True
                    continue

                # Skip TOC content (until we find a numbered section or new header)
                if skip_toc:
                    if line.startswith('#'):
                        level, title = detect_section(line)
                        if level:
                            skip_toc = False
                    else:
                        continue

                # Detect if line starts with markdown header
                if re.search(r"^#+ ", line):
                    level, title = detect_section(line)
                    if level:
                        # Use level based on section number
                        processed_lines.append('#' * min(level, 6) + ' ' + title)
                    else:
                        processed_lines.append('**' + line.strip("#").strip() + '**')
                else:
                    processed_lines.append(line)
    final_text = '\n\n'.join(processed_lines)
    if isinstance(output_path, Path):
        output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_text, encoding="utf-8")
    return final_text, images

def markdown_to_sections(markdown: Union[str, Path, List[str]]) -> Section:
    """
    将 markdown 文件递归解析为嵌套的 Section 结构，支持任意深度
    非标题内容作为 Element 的 reference 存储
    """
    if isinstance(markdown, Path):
        markdown = markdown.read_text(encoding="utf-8").split("\n\n")
    elif isinstance(markdown, str):
        markdown = markdown.split("\n\n")
    title = None
    for i, line in enumerate(markdown):
        if line.startswith("# "):
            title = line[2:]
            break
    if title is None: i = -1
    root = Section(section_id=0, title=title, segments=[],
                   subsections=[], level=1)
    _parse_lines_as_section(markdown[i+1:], root)
    if '摘要' in root.subsections[0].title:
        root.title += "\n".join([e.reference for e in root.segments if e.reference])
        root.segments = []
    return root


def _parse_lines_as_section(lines: List[str], parent: Section) -> int:
    """
    递归解析行列表，构建 Section 树

    Args:
        lines: 剩余要解析的行
        parent: 当前父 Section
        min_level: 当前层级（# 的数量）

    Returns:
        已处理的行数
    """
    i = 0
    section_count = 0
    current_content = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 检测标题
        if stripped.startswith('#'):
            level = len(stripped) - len(stripped.lstrip('#'))

            # 如果级别低于当前级别，保存当前内容并返回
            if level <= parent.level:
                if current_content:
                    elem = Segment(reference='\n\n'.join(current_content).strip())
                    parent.segments.append(elem)
                return i

            # 如果级别大于当前级别，创建新 Section
            else:
                # 保存当前积累的内容到父级
                if current_content:
                    elem = Segment(reference='\n\n'.join(current_content).strip())
                    parent.segments.append(elem)
                    current_content = []

                title = stripped.lstrip('#').strip()
                new_section = Section(
                    section_id=section_count + 1,
                    level=level,
                    title=title,
                    segments=[],
                    subsections=[]
                )
                parent.subsections.append(new_section)
                section_count += 1
                i += 1

                # 递归处理子级别
                consumed = _parse_lines_as_section(lines[i:], new_section)
                i += consumed
        else:
            # 非标题行，积累到当前内容
            current_content.append(line)
            i += 1

    # 处理末尾的内容
    if current_content:
        elem = Segment(reference='\n\n'.join(current_content).strip())
        parent.segments.append(elem)

    return i


def section_to_markdown(section: Section, level: int = 1) -> str:
    """
    将 Section 树递归转换为 Markdown 文本

    Args:
        section: 要转换的 Section 对象
        level: 当前标题级别（# 的数量）

    Returns:
        生成的 Markdown 文本
    """
    lines = [f"{'#' * level} {section.title}\n"]
    # 添加当前 Section 的所有 segments 内容
    for elem in section.segments:
        if elem.content:
            lines.append(elem.content + '\n')

    # 递归处理所有子 Section
    for subsection in section.subsections:
        sub_md = section_to_markdown(subsection, level + 1)
        if sub_md:
            lines.append(sub_md + '\n')

    return "\n".join(lines).strip()