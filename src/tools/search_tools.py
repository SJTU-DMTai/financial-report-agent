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
class SearchTools:

    def __init__(self, short_term: ShortTermMemoryStore) -> None:
        self.short_term = short_term


    # ================= 辅助函数 =================

    def _fetch_page_html(self, url: str, timeout: int = 8) -> bytes:
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


    def _score_relevance(self, query: str, title: str, text: str) -> float:
        """
        使用 TF-IDF + 余弦相似度 计算 query 与网页内容的相关性
        返回值 0.0~1.0 越高越相关
        """
        # 为了提高效率，只取前 N 字
        text = text[:5000]

        # 把 query 与 文本 组合成语料
        corpus = [query, title + " " + text]

        # 使用 tf-idf 向量化
        vectorizer = TfidfVectorizer(
            tokenizer=jieba.cut,
            max_features=20000,     # 控制维度
            stop_words=None,
        )
        try:
            tfidf = vectorizer.fit_transform(corpus)
        except ValueError:
            return 0.0

        # 计算余弦相似度
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])

        return float(sim[0][0])

    #searcher agent使用
    async def search_engine(self, query: str, max_results: int = 10) -> ToolResponse:
        """进行 Web 搜索并返回搜索结果预览，并保存搜索结果到Material当中，返回Material标识ref_id。

        - 调用 DuckDuckGo 的搜索引擎接口，根据给定关键词返回若干条过滤后的搜索结果，适合获取大致信息或者是新闻等。
        - 如果需要完整、可核查的原文内容，或者是结构化数据请调用其他工具。
        - 查询语句中的空格被视为逻辑“或”（OR）操作，因此每个关键词会分别参与匹配，请避免加入过于宽泛或无关的词语，导致无法搜索到需要的结果。
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
                max_results=max_results,
            )
            ref_id = None
            filtered: List[Dict[str, Any]] = []

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

                # 3) 计算相关性得分
                relevance = self._score_relevance(query, title, desc or snippet)

                # 得分 >= 0.2
                if relevance < 0.2:
                    continue

                filtered.append(
                    {
                        "title": title or "无标题",
                        "link": link,
                        "description": desc or snippet,  # 优先用搜索摘要，备用真实摘要
                        "page_text": page_text,
                        # "images": img_urls,
                        "relevance": relevance,
                    }
                )

            # 如果一个都没通过过滤，就退回到“未找到”
            if not filtered:
                text = f"[search_engine] 对查询「{query}」未找到足够相关的结果。"
            else:
                # 按相关性排序（高到低）
                filtered.sort(key=lambda x: x.get("relevance", 0), reverse=True)

                ref_id = f"search_engine_{int(time.time())}"
                self.short_term.save_material(
                    ref_id=ref_id,
                    content=filtered,
                    description=f"Search Engine 搜索「{query}」的结果",
                )

                lines: List[str] = [f"[search_engine] 搜索：{query}", 
                                    f"Material 已写入 ref_id='{ref_id}'（JSON 格式）",
                                    "以下为搜索结果预览",
                                    ""]
                for i, item in enumerate(filtered, start=0):
                    title = item["title"]
                    desc = item.get("description", "无摘要")
                    link = item["link"]
                    snippet = item.get("page_snippet", "")
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
            metadata={"ref_id": ref_id, "result_count": len(filtered)}
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
            res = await searcher(msg)
            return ToolResponse(
                content=res.content,
                metadata={"from_agent": searcher.name},
            )

        return search_with_searcher
