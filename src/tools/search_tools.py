# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import time as time_module
from typing import Any, Callable, Dict, List
from urllib.parse import urlparse

from ddgs import DDGS
import jieba
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from agentscope.agent import ReActAgent
from agentscope.message import Msg
from ..memory.short_term import ShortTermMemoryStore
from ..memory.long_term import LongTermMemoryStore
from .material_tools import bind_query_tool, extract_keyword_context_snippets, get_retrieve_fn
from ..utils.call_with_retry import call_agent_with_retry
from ..utils.get_entity_info import get_entity_info


async def _run_searcher_tool(
    searcher: ReActAgent,
    short_term: ShortTermMemoryStore,
    long_term: LongTermMemoryStore,
    query: str,
) -> ToolResponse:
    retrieve_fn = get_retrieve_fn(short_term, long_term)
    final_prompt = await retrieve_fn(query)
    final_prompt = final_prompt.content
    msg = Msg(
        name="user",
        content=final_prompt,
        role="user",
    )
    await searcher.memory.clear()
    res = await call_agent_with_retry(searcher, msg)
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=res.get_text_content(),
            ),
        ],
        metadata={"from_agent": searcher.name},
    )


class SearchTools:

    def __init__(self, short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore) -> None:
        self.short_term = short_term
        self.long_term = long_term


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
    

    def _fetch_bocha_candidates(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        bocha_api_key = os.getenv("BOCHA_API_KEY", "")
        bocha_url = "https://api.bochaai.com/v1/web-search"
        headers = {
            "Authorization": f"Bearer {bocha_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": max_results * 2,
        }

        response = requests.post(bocha_url, headers=headers, data=json.dumps(payload), timeout=15)
        response.raise_for_status()
        res_data = response.json()

        if "data" in res_data and "webPages" in res_data["data"]:
            raw_results = res_data["data"]["webPages"]["value"]
        else:
            raw_results = res_data.get("data", [])

        candidates: List[Dict[str, Any]] = []
        for item in raw_results:
            title = item.get("name", "") or "无标题"
            link = item.get("url", "") or ""
            summary = item.get("summary", "") or item.get("snippet", "") or ""
            snippet = item.get("snippet", "") or ""
            published_date = item.get("datePublished", "")
            if not link.startswith("http"):
                continue

            candidates.append({
                "title": re.sub(r"\s+", "", title).replace("　", ""),
                "link": link,
                "page_description": snippet,
                "page_text": summary,
                "published_date": published_date,
                "search_provider": "Bocha",
            })
        return candidates

    def _fetch_duckduckgo_candidates(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        raw_results = DDGS().text(
            query=query,
            backend="auto",
            region="cn-zh",
            max_results=max_results * 2,
        )

        candidates: List[Dict[str, Any]] = []
        for item in raw_results:
            title = item.get("title", "") or "无标题"
            link = item.get("href", "") or ""
            desc = item.get("body", "") or ""
            if not link.startswith("http"):
                continue

            clean_desc = re.sub(r"\s+", " ", desc).replace("　", "").strip()
            candidates.append({
                "title": re.sub(r"\s+", "", title).replace("　", ""),
                "link": link,
                "page_description": clean_desc,
                "page_text": clean_desc,
                "published_date": "",
                "search_provider": "DuckDuckGo",
            })
        return candidates

    async def search_engine(
        self, 
        query: str,
        max_results: int = 10
    ) -> ToolResponse:
        """进行 Web 搜索并返回搜索结果预览，并保存每一条搜索结果到Material当中，返回每一条Material标识cite_id，以及命中关键词的片段。
        - 调用搜索引擎，根据给定关键词返回若干条搜索结果，适合获取大致信息或者是新闻等。
        - 如果需要完整、可核查的原文内容，或者是结构化数据请调用其他工具。
        
        Args:
            query (str):
                搜索关键词，例如 "德明利 营业收入 2024"、"德明利 AI存储 产能"。
            max_results (int):
                返回的最大结果数量。
        """
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 10

        query = (query or "").strip()

        provider = "Bocha"
        bocha_error = ""
        duckduckgo_error = ""
        try:
            candidates = self._fetch_bocha_candidates(query=query, max_results=max_results)
        except Exception as e:
            candidates = []
            bocha_error = str(e)

        if not candidates:
            provider = "DuckDuckGo"
            try:
                candidates = self._fetch_duckduckgo_candidates(query=query, max_results=max_results)
            except Exception as e:
                candidates = []
                duckduckgo_error = str(e)

        if not candidates:
            details = ""
            if bocha_error or duckduckgo_error:
                details = f" Bocha失败: {bocha_error or '无结果'}; DuckDuckGo失败: {duckduckgo_error or '无结果'}"
            text = f"[search_engine] 对查询「{query}」未找到足够相关的结果。{details}"
            return ToolResponse(content=[TextBlock(type="text", text=text)])

        item_cite_ids: List[str] = []
        try:
            scores = self._calculate_batch_relevance(query, candidates)
            for i, score in enumerate(scores):
                candidates[i]["relevance"] = score

            candidates.sort(key=lambda x: x.get("relevance", 0), reverse=True)
            if len(candidates) > max_results:
                candidates = candidates[:max_results]

            pre_cite_id = f"search_engine_{int(time_module.time())}"
            for i, item in enumerate(candidates):
                item_cite_id = pre_cite_id + f"{i:03d}"
                published_date = item.get("published_date")
                time = {"point": published_date} if published_date else None

                entity = get_entity_info(
                    long_term=self.long_term,
                    text=item["page_description"] + item["page_text"],
                )

                desc_text = ""
                if published_date:
                    desc_text += f"网页发布时间：{published_date} "
                if entity:
                    desc_text += f"发布关于{entity['name']}（{entity['code']}）的内容:"
                desc_text += item["title"] + " " + item["page_description"] + " "

                link = item.get("link", "")
                domain = urlparse(link).netloc
                if domain.startswith("www."):
                    domain = domain[4:]

                self.short_term.save_material(
                    cite_id=item_cite_id,
                    content=[item],
                    time=time,
                    entity=entity,
                    description=desc_text,
                    source=f"Search Engine 搜索结果（{provider}，来源：{domain}）",
                )
                item_cite_ids.append(item_cite_id)

            valid_keywords = [k.strip() for k in re.split(r"\s+", query) if k.strip()]
            if not valid_keywords and query:
                valid_keywords = [query]

            lines: List[str] = [
                f"[search_engine] 搜索：{query}",
                f"搜索来源：{provider}",
                "以下为搜索结果预览（每条结果已单独写入 Material）：",
            ]
            for i, item in enumerate(candidates, start=0):
                title = item["title"]
                link = item["link"]
                desc_text = item["page_description"]

                lines.append(f"第{i}条. {title}")
                lines.append(f"   链接: {link}")
                lines.append(f"   Material 已写入 cite_id='{item_cite_ids[i]}'（JSON 格式）")
                lines.append(f"   搜索摘要: {desc_text}")

                page_text_i = item.get("page_text", "")
                matched_contexts = extract_keyword_context_snippets(
                    text=page_text_i,
                    keywords=valid_keywords,
                    context_chars=100,
                    min_keyword_len=2,
                    ignore_case=True,
                    merge_gap_chars=20,
                    highlight=True,
                )

                if matched_contexts:
                    lines.append("   页面正文关键词命中上下文:")
                    for idx, ctx in enumerate(matched_contexts):
                        kws_str = ", ".join(ctx["keywords"])
                        lines.append(f"      片段{idx+1} [{kws_str}]: {ctx['snippet']}")
                else:
                    page_text_clean = re.sub(r"\s+", " ", page_text_i).strip()
                    snippet = page_text_clean[:500] + ("......[后续内容已截断]" if len(page_text_clean) > 500 else "")
                    lines.append(f"   未在正文中精确命中关键词，页面正文开头摘录: {snippet}")

                lines.append("")
                lines.append("")

            lines.append("   [如果片段信息不足，请单独使用 read_material 工具读取上述 cite_id 获取全文，或者 fetch_url_page_text 工具获取根据链接网页全文]")
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
        return bind_query_tool(
            _run_searcher_tool,
            "search_with_searcher",
            """使用指定的 Searcher 工具基于 query 执行一次检索并返回总结结果。同时将获取的结果保存为Material。
            Args:
                query (str): 检索需求的自然语言描述。
            """,
            searcher,
            self.short_term,
            self.long_term,
        )
