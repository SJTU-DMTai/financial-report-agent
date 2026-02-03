# -*- coding: utf-8 -*-
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from trafilatura import extract
from typing import Optional, Callable, Any, Dict, Union, List, Tuple

def fetch_page_html(url: str, timeout: int = 10) -> bytes:
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

def extract_text_and_images(html: bytes, base_url: str) -> Tuple[str, List[str]]:
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