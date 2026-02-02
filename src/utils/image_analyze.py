# -*- coding: utf-8 -*-
import base64
from io import BytesIO
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import re
IMG_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
VLM_TAG_RE = re.compile(r"<!--VLM:([^>]+?)-->")

def extract_image_refs_all(md_text: str) -> list[str]:
    """保留重复（用于计数）。"""
    refs = []
    for p in IMG_REF_RE.findall(md_text or ""):
        refs.append(Path(p.strip().strip('"').strip("'")).name)
    return refs

def extract_image_refs_unique(md_text: str) -> list[str]:
    """去重但保持顺序（用于逐图处理）。"""
    refs_all = extract_image_refs_all(md_text)
    seen = set()
    out = []
    for r in refs_all:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out

def pil_to_base64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def image_complexity_score(img: Image.Image) -> tuple[float, float, int]:
    """
    返回 (entropy, edge_density, unique_colors_approx)
    """
    small = img.convert("RGB").resize((128, 128))
    arr = np.asarray(small).astype(np.float32)

    small64 = img.convert("RGB").resize((64, 64))
    arr64 = np.asarray(small64).reshape(-1, 3)
    unique_colors = len({(int(r), int(g), int(b)) for r, g, b in arr64})

    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float32)
    p = hist / (hist.sum() + 1e-9)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    gx = np.abs(np.diff(gray.astype(np.float32), axis=1))
    gy = np.abs(np.diff(gray.astype(np.float32), axis=0))
    edge_density = float((gx.mean() + gy.mean()) / 255.0)

    return entropy, edge_density, unique_colors

def should_skip_image(img: Image.Image, ref_count: int) -> tuple[bool, str]:
    """
    过滤掉示例PDF里面一些装饰性的、与研报内容无关的图片
    过滤规则：
    (1) 重复引用：直接跳过
    (2) 小图：跳过（头像/图标/徽标概率高）
    (3) 横幅/竖条装饰：跳过
    (4) 低复杂度：跳过
    (5) 颜色极少 + 边缘少：跳过（图标/徽标）
    """
    if ref_count > 1:
        return True, f"duplicated_ref(count={ref_count})"

    w, h = img.size
    area = w * h
    ar = w / (h + 1e-9)

    if area < 50_000 or max(w, h) < 300 or min (w, h) < 100:
        return True, f"too_small({w}x{h})"

    if (ar > 5.0 and h < 180) or (ar < 0.2 and w < 180):
        return True, f"banner_like(ar={ar:.2f},{w}x{h})"

    entropy, edge_density, uniq = image_complexity_score(img)

    if entropy < 3.0 and edge_density < 0.04:
        return True, f"low_complexity(entropy={entropy:.2f},edge={edge_density:.3f})"

    if uniq < 40 and edge_density < 0.06:
        return True, f"icon_like(colors={uniq},edge={edge_density:.3f})"

    return False, "keep"

def normalize_vlm_output(text: str) -> str:
    """
    把 VLM 输出里的 “分类：xxx” 去掉“分类：”，转换为 “【xxx】”
    其它内容保持原样。
    """
    lines = [l.rstrip() for l in (text or "").splitlines()]
    out = []
    for l in lines:
        m = re.match(r"^\s*分类[:：]\s*(.+?)\s*$", l)
        if m:
            out.append(f"【{m.group(1).strip()}】")
        else:
            out.append(l)

    # 去掉首尾空行
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out).strip()

def coerce_to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = []
        for item in x:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(x)

async def analyze_one_image_vlm(vlm_model, img: Image.Image, prompt: str) -> str:
    """
    单次多模态 LLM 调用。
    注意：这里使用 OpenAI 兼容的 image_url data URI schema。
    """
    b64 = pil_to_base64_png(img)
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ],
        },
    ]
    res = await vlm_model(messages=messages, temperature=0.0)
    raw = getattr(res, "content", res)
    return normalize_vlm_output(coerce_to_text(raw))

def format_injected_block(vlm_text: str) -> str:
    """
    注入到 markdown 原文中的块：用 blockquote，避免影响标题/结构解析。
    """
    if not vlm_text.strip():
        return ""
    q = ["> 【图片内容解析】"]
    for ln in vlm_text.splitlines():
        ln = ln.strip()
        if ln:
            q.append(f"> {ln}")
    return "\n" + "\n".join(q) + "\n"

def load_images_from_md_dir(demo_md_path: Path) -> dict[str, Image.Image]:
    """当 md 已存在时，从 md 同目录加载被引用的图片文件。"""
    md_text = demo_md_path.read_text(encoding="utf-8", errors="ignore")
    refs = extract_image_refs_unique(md_text)
    images = {}
    base_dir = demo_md_path.parent
    for name in refs:
        p = base_dir / name
        if p.exists():
            try:
                images[name] = Image.open(p).convert("RGB")
            except Exception:
                pass
    return images

async def inject_vlm_into_demo_markdown(
    demo_md_path: Path,
    images: dict[str, Image.Image],
    vlm_model,
    image_prompt: str,
) -> None:
    """
    将图片解析结果注入到 demonstration 的 markdown 原文中：
    - 只处理在 markdown 中出现过的图片
    - 对每个图片引用行：若下一条非空行是 > 【图片内容解析】，则认为之前已注入，跳过该图
    - 重复出现（引用次数>1）的图片直接跳过
    - 过滤头像/图标/装饰图
    """
    if isinstance(demo_md_path, str):
        demo_md_path = Path(demo_md_path)
    md_text = demo_md_path.read_text(encoding="utf-8", errors="ignore")
    lines = md_text.splitlines()

    # 全局图片引用计数（用于“重复引用跳过”）
    refs_all = extract_image_refs_all(md_text)
    counter = Counter(refs_all)

    changed = False
    out_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        out_lines.append(line)

        m = IMG_REF_RE.search(line)
        if not m:
            i += 1
            continue

        img_name = Path(m.group(1)).name

        # (1) 若已注入，跳过解析
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines) and lines[j].strip().startswith("> 【图片内容解析】"):
            i += 1
            continue

        # (2) 重复引用直接跳过
        if counter.get(img_name, 0) > 1:
            i += 1
            continue

        img = images.get(img_name)
        if img is None:
            i += 1
            continue

        # (3) 过滤头像/图标/装饰
        skip, _reason = should_skip_image(img, counter.get(img_name, 0))
        if skip:
            i += 1
            continue

        # (4) 单次多模态调用并注入
        vlm_text = await analyze_one_image_vlm(vlm_model, img, image_prompt +
                                               f"\n\n注：该图片来自于名为“{demo_md_path.stem}”的研报，如果是无关插图，请直接输出“无关插图”并结束")
        if vlm_text:
            if "无关插图" in vlm_text:
                i += 1
                continue
            out_lines.append(format_injected_block(vlm_text).rstrip("\n"))
            changed = True

        i += 1

    if changed:
        demo_md_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")