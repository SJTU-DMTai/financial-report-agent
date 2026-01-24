# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import List, Dict, Any, Tuple
from ddgs import DDGS
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from agentscope.agent import ReActAgent
from agentscope.message import Msg
from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import LongTermMemoryStore
from .material_tools import *
import re
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import time as time_module
from trafilatura import extract
from htmldate import find_date
from urllib.parse import urlparse
from ..utils.call_agent_with_retry import call_agent_with_retry
from ..utils.get_entity_info import get_entity_info
from ..utils.retrieve_in_memory import retrieve_in_memory
class SearchTools:

    def __init__(self, short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore) -> None:
        self.short_term = short_term
        self.long_term = long_term

    # ================= 辅助函数 =================

    @staticmethod
    def _fetch_page_html(url: str, timeout: int = 10) -> bytes:
        """用 requests 获取网页 HTML 内容"""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
            return b""
        return resp.content   # 注意：这里返回的是 bytes

    @staticmethod
    def _extract_text_and_images(html: bytes, base_url: str) -> Tuple[str, List[str]]:
        """文本使用 trafilatura 提取主内容；图片使用 BeautifulSoup 提取。"""
        if not html:
            return "", []

        try:
            text = extract(
                html,
                url=base_url,
                output_format="txt",    # 返回纯文本
                include_comments=False,
                include_tables=True     # 如需保留表格内容
            ) or ""
        except Exception:
            text = ""

        img_urls: List[str] = []
        try:
            soup = BeautifulSoup(html, "html.parser")
            for img in soup.find_all("img"):
                src = img.get("src") or ""
                if not src:
                    continue
                full_url = urljoin(base_url, src)
                img_urls.append(full_url)
        except Exception:
            pass


        return text, img_urls


    def _calculate_batch_relevance(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """
        批量计算相关性得分 (Batch Processing TF-IDF)。
        原理：将 Query 和所有文档放在同一个语料库中计算，这样能利用 IDF 正确降低通用词权重。
        """
        if not candidates:
            return []

        stop_words = [
            # 虚词
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "个", "为", "之", "与", "及", "等", "或", "但是", "对于", "我们", "他们", "相关", "进行", "可以",
            # 防止年份匹配导致的误判
            "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2030"
        ]
        # 1. 准备语料库：索引0为查询，后续为各个文档
        # 组合 Title + Description + Text(前5000字) 以获得最佳匹配效果
        corpus = [query]
        for item in candidates:
            # 权重优化：标题重复3次以增加标题匹配的权重
            doc_content = (item['title'] + " ") * 3 + item['page_description'] + " " + item['page_text'][:5000]
            corpus.append(doc_content)

        # 2. 向量化 (Fit Transform once)
        vectorizer = TfidfVectorizer(
            tokenizer=jieba.cut,
            max_features=50000,
            stop_words=stop_words, 
            token_pattern=r"(?u)\b\w+\b" # 覆盖默认正则以支持中文
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # 极端情况：语料库为空或分词后无有效词
            return [0.0] * len(candidates)

        # 3. 计算相似度
        # query_vec 是矩阵第0行, doc_vecs 是第1行到最后
        query_vec = tfidf_matrix[0:1]
        doc_vecs = tfidf_matrix[1:]

        # 计算余弦相似度，结果是一个 shape 为 (1, n_docs) 的矩阵
        similarities = cosine_similarity(query_vec, doc_vecs).flatten()

        return [float(score) for score in similarities]
    

    #searcher agent使用
    async def search_engine(self, query: str, max_results: int = 10) -> ToolResponse:
        """进行 Web 搜索并返回搜索结果预览，并保存每一条搜索结果到Material当中，返回每一条Material标识ref_id。
        - 调用搜索引擎，根据给定关键词返回若干条过滤后的搜索结果，适合获取大致信息或者是新闻等。
        - 如果需要完整、可核查的原文内容，或者是结构化数据请调用其他工具。
        Args:
            query (str):
                搜索内容。
            max_results (int):
                返回的最大结果数量。
        """
        try:
            max_results = int(max_results) # 防止传入字符串如"10"导致搜索失败
        except (TypeError, ValueError):
            max_results = 10

        candidates: List[Dict[str, Any]] = []
        item_ref_ids: List[str] = []
        try:
            ddgs = DDGS()
            # 1) 调用 DuckDuckGo 搜索接口
            raw_results = ddgs.text(
                query=query,
                backend="auto",
                region="cn-zh",
                max_results=max_results*2,
            )

            for r in raw_results:
                title = r.get("title", "") or "无标题"
                link = r.get("href", "") or ""
                desc = r.get("body", "") or ""

                title = re.sub(r"\s+", "", title).replace("　", "")
                desc = re.sub(r"\s+", "", desc).replace("　", "")

                if not link.startswith("http"):
                    # 非正常链接直接跳过
                    continue

                # 2) 抓取网页 HTML 并抽取文本 + 图片
                try:
                    html_bytes = self._fetch_page_html(link)
                    if not html_bytes:
                        continue
                    page_text, img_urls = self._extract_text_and_images(html_bytes, link)
                    published_date = None
                    try:
                        published_date = find_date(
                            html_bytes,
                            url=link,
                            original_date=True,
                            extensive_search=True,
                            deferred_url_extractor=True,   # 降低从 URL 猜日期的优先级，减少误判
                        )
                    except Exception:
                        published_date = None

                except Exception:
                    # 单条失败不影响整体
                    continue

                if not page_text.strip():
                     # 没有有效文本也跳过
                    continue

                # 取前 300 字作为备用摘要
                snippet = page_text.replace("\n", " ")
                snippet = re.sub(r"\s{2,}", " ", snippet)
                snippet = snippet[:300] + ("..." if len(snippet) > 300 else "")

                # 暂存候选项，暂不计算分数
                candidates.append({
                    "title": title,
                    "link": link,
                    "page_description": desc or snippet, # 优先用搜索结果摘要，没有则用正文摘要
                    "page_text": page_text,
                    "published_date": published_date,
                    # "images": img_urls
                })



            # 如果一个都没通过过滤，就退回到“未找到”
            if not candidates:
                text = f"[search_engine] 对查询「{query}」未找到足够相关的结果。"
            else:
                scores = self._calculate_batch_relevance(query, candidates)
                # 将分数回填给 candidates
                for i, score in enumerate(scores):
                    candidates[i]['relevance'] = score

                # 按相关性排序（高到低）
                candidates.sort(key=lambda x: x.get("relevance", 0), reverse=True)

                if len(candidates) > max_results:
                    candidates = candidates[:max_results]

                pre_ref_id = f"search_engine_{int(time_module.time())}"
                
                for i, item in enumerate(candidates):
                    item_ref_id = pre_ref_id + f"{i:03d}"
                    published_date = item.get("published_date")
                    time = {"point": published_date} if published_date else None
                    
                    # entity = get_entity_info(long_term=self.long_term, text=query)
                    
                    entity = get_entity_info(long_term=self.long_term, text=candidates[i]["page_description"]+candidates[i]["page_text"])
                    
                    desc = ""
                    if published_date:
                        desc = desc+ f"网页发布时间：{published_date} "
                    
                    if entity:
                        desc = desc+f"发布关于{entity['name']}（{entity['code']}）的内容:"
                    # else:
                    #     desc = desc+f"发布关于{query}的内容:"
                    desc = desc + candidates[i]["title"] + " "
                    desc = desc + candidates[i]["page_description"]+ " "
                    link = candidates[i].get("link", "")
                    domain = urlparse(link).netloc
                    if domain.startswith("www."):
                        domain = domain[4:]

                    self.short_term.save_material(
                        ref_id=item_ref_id,
                        content=[candidates[i]],
                        time=time,
                        entity=entity,
                        description=desc,
                        # source=f"Search Engine 搜索「{query}」的结果"
                        source=f"Search Engine 搜索结果（来源：{domain}）"
                    )
                    item_ref_ids.append(item_ref_id)


                # 以下仅仅为调试使用 （便于看到单次搜索内容）
                # index_payload = {
                #     "query": query,
                #     "max_results": max_results,
                #     "result_count": len(candidates),
                #     "items": [
                #         {
                #             "index": i,
                #             "ref_id": item_ref_ids[i],
                #             "title": candidates[i].get("title", ""),
                #             "link": candidates[i].get("link", ""),
                #             "description": candidates[i].get("page_description", ""),
                #             "page_text": candidates[i].get("page_text"),
                #             "relevance": candidates[i].get("relevance", 0.0),
                #         }
                #         for i in range(len(candidates))
                #     ],
                # }
                # self.short_term.save_material(
                #     ref_id=pre_ref_id,
                #     content=index_payload,
                #     description=f"Search Engine 搜索「{query}」的结果",
                #     source="Search Engine",
                # )
                # 以上仅仅为调试使用 （便于看到单次搜索内容）


                lines: List[str] = [f"[search_engine] 搜索：{query}", 
                                    "以下为搜索结果预览（每条结果已单独写入 Material）：",
                                    ]
                for i, item in enumerate(candidates, start=0):
                    title = item["title"]
                    link = item["link"]
                    desc = item["page_description"]
                    # images = item.get("images") or []
                    # relevance = item.get("relevance", 0.0)

                    lines.append(f"第{i}条. {title}")
                    lines.append(f"   链接: {link}")
                    lines.append(f"   Material 已写入 ref_id='{item_ref_ids[i]}'（JSON 格式）")
                    lines.append(f"   搜索摘要: {desc}")

                    page_text_i = item.get("page_text", "")
                    snippet = page_text_i.replace("\n", " ")
                    snippet = re.sub(r"\s{2,}", " ", snippet)
                    snippet = snippet[:1000] + ("......[内容过长，已截断，如需要完整阅读请对此条结果单独使用read_material工具]" if len(snippet) > 1000 else "")
                    lines.append(f"   页面正文摘录: {snippet}")

                    # if images:
                    #     # 只显示前 2 个图片链接，避免过长
                    #     lines.append("   图片链接示例:")
                    #     for img_url in images[:2]:
                    #         lines.append(f"      - {img_url}")
                    # lines.append(f"   相关性得分: {relevance:.2f}")
                    lines.append("")  # 空行分隔
                    lines.append("")

                text = "\n".join(lines)

        except Exception as e:
            text = f"[search_engine] 搜索出错：{e}"

        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=text,
                ),
            ],
        )

    def searcher_tool(self, searcher: ReActAgent) -> Callable[[str], ToolResponse]:
        """把 Searcher agent 封装成 agent 可见的工具函数。"""
        async def search_with_searcher(query: str) -> ToolResponse:
            """使用指定的 Searcher 工具基于 query 执行一次检索并返回总结结果。同时将获取的结果保存为Material。
            Args:
                query (str): 检索需求的自然语言描述。
            """
            retrieve_fn = get_retrieve_fn(self.short_term, self.long_term)
            final_prompt = await retrieve_fn(query)
            final_prompt = final_prompt.content
            msg = Msg(
                name="user",
                content=final_prompt,
                role="user",
            )
            await searcher.memory.clear()
            res = await call_agent_with_retry(searcher,msg)
            
            return ToolResponse(
                content=res.content,
                metadata={"from_agent": searcher.name},
            )

        return search_with_searcher

    @staticmethod
    async def fetch_url_page_text(url: str) -> ToolResponse:
        """返回url对应网页的文本结果。
        Args:
            url (str):
                网页地址。
        """
        bytes = SearchTools._fetch_page_html(url)
        page_text, img_urls = SearchTools._extract_text_and_images(bytes, url)

        page_text = page_text or ""

        return ToolResponse(
            content=[
                TextBlock(type="text", text=page_text),
            ],
        )

def get_retrieve_fn(short_term, long_term) -> Callable[str]:
    async def retrieve_local_material(query: str) -> ToolResponse:
        """
        在已保存的本地材料中按关键词搜索和query相关的材料，返回部分预览内容。
        Args:
            query (str):
                搜索内容。
        """
        candidates = retrieve_in_memory(
            short_term=short_term,
            long_term=long_term,
            query=query,
        )
        if candidates:
            lines = [
                f"你需要检索的信息：{query}",
                "",
                "用户本地保存的候选材料（materials）及其部分预览如下:",
                "",
            ]

            for i, meta in enumerate(candidates, 1):
                ref_id = meta.get("ref_id", "")
                desc = meta.get("description", "")
                src = meta.get("source", "")
                m_type = meta.get("m_type", "")
                lines.append(f"第{i}条材料：Material 的唯一标识 ref_id={ref_id}")
                if desc:
                    lines.append(f"    简短描述: {desc}")
                if src:
                    lines.append(f"    来源: {src}")

                try:
                    content = short_term.load_material(ref_id) if short_term is not None else None
                except Exception:
                    content = None

                # (A) 搜索引擎：search_engine_*
                if isinstance(ref_id, str) and ref_id.startswith("search_engine_"):
                    page_text_preview = ""
                    if isinstance(content, list) and content:
                        first = content[0] if isinstance(content[0], dict) else None
                        if isinstance(first, dict):
                            page_text = first.get("page_text") or ""
                            page_text_preview = page_text[:100]
                    lines.append("    部分内容预览：")
                    lines.append(f"   {page_text_preview}")


                # (B) 计算结果：calculate_*
                elif isinstance(ref_id, str) and ref_id.startswith("calculate_"):
                    params = None
                    result = None
                    if isinstance(content, list) and content:
                        first = content[0] if isinstance(content[0], dict) else None
                        if isinstance(first, dict):
                            params = first.get("parameters", None)
                            result = first.get("result", None)
                    if params:
                        lines.append("    计算参数:")
                        lines.append(f"    {params}")
                    lines.append("    计算结果:")
                    lines.append(f"    {result}")

                # (C) 表格：非前缀类时，用 m_type==table 给出前几行
                elif m_type == "table":
                    preview = ""
                    if isinstance(content, pd.DataFrame) and not content.empty:
                        df_preview = content.head(3).copy()
                        MAX_CELL_CHARS = 200
                        SUFFIX = "…[内容过长，已截断]"
                        for col in df_preview.columns:
                            df_preview[col] = df_preview[col].apply(
                                lambda v: (
                                    v[:MAX_CELL_CHARS] + SUFFIX
                                    if isinstance(v, str) and len(v) > MAX_CELL_CHARS
                                    else v
                                )
                            )

                        preview = df_preview.to_csv(index=False)

                    lines.append("    前3行预览:")
                    lines.append(preview)

                lines.append("")  # 空行分隔
            lines.append("如果以上无合适材料，请重新调用工具获取数据。")
            res = "\n".join(lines)
        else:
            res = f"对于{query}，本地尚未保存相关材料。请调用合适的工具获取数据。"
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=res,
                ),
            ],
        )

    return retrieve_local_material
