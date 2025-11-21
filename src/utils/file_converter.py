
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

def html_to_pdf(
        short_term : ShortTermMemoryStore,
        output_filename: str = "report.pdf",
        output_dir: str = "data/output/reports",
    ):
        """将所有 Manuscript 章节按顺序合并并导出为 PDF 文件。
        """

        # 1. 收集所有 section 文件并按文件名排序
        sec_files = sorted(short_term.manuscript_dir.glob("sec_*.html"))
        if not sec_files:
            text = "[html_to_pdf] 未找到任何章节文件 (sec_*.html)，无法生成 PDF。"

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


        body_html = "\n<hr/>\n".join(body_parts)
        full_html = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: "Noto Sans CJK SC", "Microsoft YaHei", "SimSun", "Songti SC", sans-serif;
                }}
            </style>
        </head>
        <body>
        {body_html}
        </body>
        </html>"""
        # 3. 调用 pdfkit 生成 PDF
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / output_filename

        # if wkhtmltopdf_path:
        #     config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        #     pdfkit.from_string(full_html, str(pdf_path), configuration=config)
        # else:
        #     pdfkit.from_string(full_html, str(pdf_path))

        options = {
            "encoding": "UTF-8",
        }
        pdfkit.from_string(full_html, str(pdf_path), options=options)
        text = f"[html_to_pdf] 已输出 PDF: {pdf_path}"

        return text



def pdf_to_markdown(
    short_term : ShortTermMemoryStore,
):
    """
    将 PDF 转换为结构化 Markdown，提取图片和表格并以特定格式嵌入。
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
