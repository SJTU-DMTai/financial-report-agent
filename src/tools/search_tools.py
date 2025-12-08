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
from ..memory.long_term import ToolUseExperienceStore
from .material_tools import *
import re
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import time
from trafilatura import extract
from ..utils.call_agent_with_retry import call_agent_with_retry


class SearchTools:

    def __init__(self, short_term: ShortTermMemoryStore) -> None:
        self.short_term = short_term


    # ================= 辅助函数 =================

    def _fetch_page_html(self, url: str, timeout: int = 10) -> bytes:
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


    def _extract_text_and_images(self, html: bytes, base_url: str) -> Tuple[str, List[str]]:
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
            doc_content = (item['title'] + " ") * 3 + item['description'] + " " + item['page_text'][:5000]
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
        """进行 Web 搜索并返回搜索结果预览，并保存搜索结果到Material当中，返回Material标识ref_id。

        - 调用 DuckDuckGo 的搜索引擎接口，根据给定关键词返回若干条过滤后的搜索结果，适合获取大致信息或者是新闻等。
        - 如果需要完整、可核查的原文内容，或者是结构化数据请调用其他工具。
        Args:
            query (str):
                搜索内容。
            max_results (int):
                返回的最大结果数量。
        """
        try:
            ddgs = DDGS()
            # 1) 调用 DuckDuckGo 搜索接口
            raw_results = ddgs.text(
                query=query,
                backend="auto",
                region="cn-zh",
                max_results=max_results*2,
            )
            ref_id = None
            candidates: List[Dict[str, Any]] = []

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
                    "description": desc or snippet, # 优先用搜索结果摘要，没有则用正文摘要
                    "page_text": page_text,
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

                new_candidates = []
                for i, item in enumerate(candidates):
                    new_item = {"index": i}  # 放最前
                    new_item.update(item)    # 其余字段按原顺序追加
                    new_candidates.append(new_item)

                candidates = new_candidates

                ref_id = f"search_engine_{int(time.time())}"
                self.short_term.save_material(
                    ref_id=ref_id,
                    content=candidates,
                    description=f"Search Engine 搜索「{query}」的结果",
                    source="Search Engine"
                )

                lines: List[str] = [f"[search_engine] 搜索：{query}", 
                                    f"Material 已写入 ref_id='{ref_id}'（JSON 格式）",
                                    "以下为搜索结果预览",
                                    ""]
                for i, item in enumerate(candidates, start=0):
                    title = item["title"]
                    desc = item.get("description", "无摘要")
                    link = item["link"]
                    # images = item.get("images") or []
                    # relevance = item.get("relevance", 0.0)

                    lines.append(f"第{i}条. {title}")
                    lines.append(f"   链接: {link}")
                    lines.append(f"   搜索摘要: {desc}")
                    if snippet:
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
            metadata={"ref_id": ref_id, "result_count": len(candidates)}
        )

    # planner和writer调用
    def searcher_tool(self, searcher: ReActAgent) -> Callable[[str], ToolResponse]:
        """把 Searcher agent 封装成 agent 可见的工具函数。"""
        async def search_with_searcher(query: str) -> ToolResponse:
            """使用指定的 Searcher 工具 基于 query 执行一次检索并返回总结结果。同时将获取的结果保存为Material。

            Args:
                query (str): 检索需求的自然语言描述。

            """
            msg = Msg(
                name="user",
                content=query,
                role="user",
            )
            # res = await searcher(msg)
            res = await call_agent_with_retry(searcher,msg)

            await searcher.memory.clear()
            
            return ToolResponse(
                content=res.content,
                metadata={"from_agent": searcher.name},
            )

        return search_with_searcher
