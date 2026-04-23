import re
import math
import traceback
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import asyncio

from rapidfuzz import fuzz
from semhash import SemHash
from filelock import FileLock

from src.pipelines.planning import process_pdf_to_outline
from src.memory.working import Section
from src.utils.call_with_retry import call_chatbot_with_retry
from src.utils.local_file import DEMO_DIR
from src.utils.instance import llm_reasoning, llm_instruct, formatter, cfg, create_emb_model
from src.utils.retrieve_in_memory import _tokenize_for_bm25, _bm25_scores

# 最大并发数限制
MAX_CONCURRENT = 8

# ===================== 股票代码 -> 公司名称 查询 =====================

_code_name_df: Optional[pd.DataFrame] = None


def _load_code_name_df() -> pd.DataFrame:
    """延迟加载 a_share_code_name.csv，返回 DataFrame。"""
    global _code_name_df
    if _code_name_df is None:
        csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "memory" / "long_term" / "a_share_code_name.csv"
        if csv_path.exists():
            _code_name_df = pd.read_csv(csv_path, dtype=str)
        else:
            _code_name_df = pd.DataFrame(columns=["code", "name"])
    return _code_name_df


def get_entity_name_by_code(stock_code: str) -> str:
    """根据股票代码查找公司名称，找不到则返回空字符串。"""
    code = str(stock_code).zfill(6)
    df = _load_code_name_df()
    hit = df[df["code"] == code]
    return "" if hit.empty else hit.iloc[0]["name"]


def extract_text_from_content(content) -> str:
    """无论content是列表还是字典，都能安全地提取'text'值。"""
    if isinstance(content, list) and content:
        first_item = content[0]
        if isinstance(first_item, dict): return first_item.get('text', '').strip()
    elif isinstance(content, dict): return content.get('text', '').strip()
    return ""

async def get_content_from_response(response_msg) -> str:
    """统一处理流式和非流式的、各种格式的大模型响应。"""
    if hasattr(response_msg, 'content') and response_msg.content:
        return extract_text_from_content(response_msg.content)
    full_content = ""
    try:
        async for chunk in response_msg:
            if hasattr(chunk, 'content') and chunk.content:
                full_content += extract_text_from_content(chunk.content)
    except TypeError: pass
    return full_content.strip()

def get_all_evidences_from_section(section: Section) -> List[Tuple[str, str]]:
    """递归地从Section对象中收集所有论据。

    evidences 是 OrderedDict[论据, 具体事实]，返回 (论据, 具体事实) 元组列表。

    Returns:
        List of (key, value) tuples.
    """
    evidences = []
    if section.segments:
        for segment in section.segments:
            if segment.evidences:
                if isinstance(segment.evidences, dict):
                    for k, v in segment.evidences.items():
                        if k and k.strip():
                            evidences.append((k.strip(), v.strip() if v else ""))
                else:
                    evidences += segment.evidences
    if section.subsections:
        for subsection in section.subsections:
            if '摘要' not in subsection.title:
                evidences.extend(get_all_evidences_from_section(subsection))
    return evidences

async def extract_unique_evidences_from_pdf(pdf_path: Path, save_dir: Path, only_evidence: bool = False, stock_entity_name: str = "") -> List[Tuple[str, str]]:
    pdf_stem = pdf_path.stem  # 去掉.pdf后缀
    evidence_filename = f"{pdf_stem}_evidences.json"
    evidence_path = save_dir / cfg.llm_name / evidence_filename
    # 尝试直接读取
    if evidence_path.exists():
        print(f"    - 检测到已有的evidences，加载: {evidence_filename}")
        evidences = json.loads(evidence_path.read_text(encoding="utf-8"))
        return evidences

    # 从文件名中提取研报写作日期
    report_date = _parse_report_date_from_path(pdf_path)
    if report_date:
        print(f"  - 研报写作日期: {report_date.strftime('%Y-%m-%d')}", flush=True)

    print(f"\n-> 开始提取并清洗文件: {pdf_path.name}", flush=True)
    manuscript = await process_pdf_to_outline(pdf_path, save_dir, llm_reasoning, llm_instruct, formatter, only_evidence)
    print(f"  - 从 {pdf_path.name} 的结构中提取论据...")
    return await extract_unique_evidences(manuscript, evidence_path, stock_entity_name=stock_entity_name, report_date=report_date)


async def extract_unique_evidences(manuscript: Section, evidence_path: Path, stock_entity_name: str = "", report_date: Optional[datetime] = None) -> List[Tuple[str, str]]:
    if evidence_path.exists():
        return json.loads(evidence_path.read_text(encoding="utf-8"))
    evidences = get_all_evidences_from_section(manuscript)
    print(f"  - 对 {len(evidences)} 条原始论据进行语义去重...", flush=True)

    evidences = await drop_duplicate_evidences_by_similarity(evidences, stock_entity_name=stock_entity_name, report_date=report_date)
    # 保存到long_term_dir
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidences, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"    -> Evidence已保存到: {evidence_path}", flush=True)
    return evidences

def _bm25_similarity_matrix(
    texts_a: List[str],
    texts_b: List[str],
) -> np.ndarray:
    """以 texts_a 中的每条文本作为 query，对 texts_b 计算 BM25 分数矩阵 (M, N)。

    分数会被 min-max 归一化到 [0, 1] 区间以便与 cosine 相似度混合。
    """
    tokens_a = [_tokenize_for_bm25(t) for t in texts_a]
    tokens_b = [_tokenize_for_bm25(t) for t in texts_b]

    M, N = len(texts_a), len(texts_b)
    matrix = np.zeros((M, N), dtype=np.float32)

    for i, q_tokens in enumerate(tokens_a):
        scores = _bm25_scores(q_tokens, tokens_b)
        matrix[i] = scores

    # min-max 归一化（整体）
    mn, mx = matrix.min(), matrix.max()
    if mx - mn > 1e-8:
        matrix = (matrix - mn) / (mx - mn)
    else:
        matrix[:] = 0.0
    return matrix

# ===================== 时间解析工具函数 =====================

def _parse_report_date_from_path(pdf_path: Path) -> Optional[datetime]:
    """从 PDF 文件名中提取研报写作日期。

    文件名格式: {stock_code}_{YYYY-MM-DD}_xxx.pdf

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        datetime 对象，解析失败返回 None。
    """
    parts = pdf_path.stem.split("_")
    if len(parts) >= 2:
        try:
            return datetime.strptime(parts[1], "%Y-%m-%d")
        except ValueError:
            pass
    return None


# 相对时间词 -> (年偏移, 季度/半年/月标识) 的映射规则
_RELATIVE_TIME_PATTERNS: List[Tuple[str, callable]] = []


def _resolve_relative_time(text: str, report_date: datetime) -> str:
    """将论据文本中的相对时间词转换为具体的时间标签。

    支持的相对时间词包括：
    - 季度类：本季度、上一季度、上季度、下一季度、前N个季度
    - 年度类：本年度、上年度、去年、前年、今年、明年
    - 半年类：上半年、下半年
    - 月份类：本月、上月、上个月
    - 模糊类：近N年、近N个季度、未来N年

    Args:
        text: 论据文本。
        report_date: 研报写作日期。

    Returns:
        替换后的文本，相对时间词被替换为具体时间（如 "2025年Q2"）。
    """
    year = report_date.year
    month = report_date.month
    quarter = (month - 1) // 3 + 1  # 1-4

    def _quarter_str(y, q):
        """生成季度字符串，处理跨年。"""
        while q > 4:
            y += 1
            q -= 4
        while q < 1:
            y -= 1
            q += 4
        return f"{y}年Q{q}"

    def _half_year_str(y, half):
        return f"{y}年{'上' if half == 1 else '下'}半年"

    result = text

    # === 季度类 ===
    # "前N个季度" / "近N个季度"
    m = re.search(r'[前近](\d+)个?季度', result)
    if m:
        n = int(m.group(1))
        start_q_label = _quarter_str(year, quarter - n)
        end_q_label = _quarter_str(year, quarter - 1)
        result = result[:m.start()] + f"{start_q_label}至{end_q_label}" + result[m.end():]

    # "未来N个季度"
    m = re.search(r'未来(\d+)个?季度', result)
    if m:
        n = int(m.group(1))
        start_q_label = _quarter_str(year, quarter + 1)
        end_q_label = _quarter_str(year, quarter + n)
        result = result[:m.start()] + f"{start_q_label}至{end_q_label}" + result[m.end():]

    # "下一季度" / "下季度"
    result = re.sub(r'下一?季度', _quarter_str(year, quarter + 1), result)
    # "上一季度" / "上季度"
    result = re.sub(r'上一?季度', _quarter_str(year, quarter - 1), result)
    # "本季度" / "当季度"
    result = re.sub(r'[本当]季度', _quarter_str(year, quarter), result)

    # === 年度类 ===
    # "近N年" / "过去N年"
    m = re.search(r'[近过去](\d+)年', result)
    if m:
        n = int(m.group(1))
        result = result[:m.start()] + f"{year - n}年至{year}年" + result[m.end():]

    # "未来N年"
    m = re.search(r'未来(\d+)年', result)
    if m:
        n = int(m.group(1))
        result = result[:m.start()] + f"{year + 1}年至{year + n}年" + result[m.end():]

    result = re.sub(r'前年', f"{year - 2}年", result)
    result = re.sub(r'去年|上年度|上一年度|上一年', f"{year - 1}年", result)
    result = re.sub(r'今年|本年度|本年|当年度', f"{year}年", result)
    result = re.sub(r'明年|下一年度|下一年', f"{year + 1}年", result)

    # === 半年类 ===
    half = 1 if month <= 6 else 2
    result = re.sub(r'上半年', _half_year_str(year, 1), result)
    result = re.sub(r'下半年', _half_year_str(year, 2), result)

    # === 月份类 ===
    def _month_str(y, m_val):
        while m_val > 12:
            y += 1
            m_val -= 12
        while m_val < 1:
            y -= 1
            m_val += 12
        return f"{y}年{m_val}月"

    result = re.sub(r'上个?月', _month_str(year, month - 1), result)
    result = re.sub(r'[本当]月', _month_str(year, month), result)
    result = re.sub(r'下个?月', _month_str(year, month + 1), result)

    return result


def _extract_time_tag(text: str) -> str:
    """从论据文本中提取时间标签，用于分组。

    提取规则（按优先级）：
    1. 具体年份+季度：如 "2025年Q2" -> "2025-Q2"
    2. 具体年份+半年：如 "2025年上半年" -> "2025-H1"
    3. 具体年份+月份：如 "2025年3月" -> "2025-03"
    4. 年份范围：如 "2022年至2025年" -> "2022~2025"
    5. 单独年份：如 "2025年" -> "2025"
    6. 无时间信息 -> "NO_TIME"

    如果存在多个时间标记，取第一个作为主时间标签。

    Args:
        text: 论据文本（已经过相对时间转换）。

    Returns:
        时间标签字符串。
    """
    # 年份范围
    m = re.search(r'(\d{4})\s*年?\s*[至到~-]\s*(\d{4})\s*年?', text)
    if m:
        return f"{m.group(1)}~{m.group(2)}"

    # 年份+季度
    m = re.search(r'(\d{4})\s*年\s*(?:第\s*)?([一二三四1-4])\s*季度', text)
    if m:
        q_map = {'一': '1', '二': '2', '三': '3', '四': '4'}
        q = q_map.get(m.group(2), m.group(2))
        return f"{m.group(1)}-Q{q}"

    # 年份+Q格式
    m = re.search(r'(\d{4})\s*年\s*Q([1-4])', text)
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"

    # 年份+半年
    m = re.search(r'(\d{4})\s*年\s*(上|下)\s*半年', text)
    if m:
        h = '1' if m.group(2) == '上' else '2'
        return f"{m.group(1)}-H{h}"

    # 年份+月份
    m = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月', text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"

    # 单独年份
    m = re.search(r'(\d{4})\s*年', text)
    if m:
        return f"{m.group(1)}"

    return "NO_TIME"


def _group_evidences_by_time(
    evidences: List[Tuple[str, str]],
    report_date: Optional[datetime] = None,
) -> Dict[str, List[Tuple[int, Tuple[str, str]]]]:
    """按时间标签对论据进行分组。

    先将相对时间转为绝对时间，再提取时间标签进行分组。

    Args:
        evidences: List of (evidence_text, value) tuples.
        report_date: 研报写作日期，用于解析相对时间。

    Returns:
        Dict mapping time_tag -> List of (original_index, (evidence_text, value)).
    """
    groups: Dict[str, List[Tuple[int, Tuple[str, str]]]] = defaultdict(list)
    for idx, (text, value) in enumerate(evidences):
        resolved_text = text
        if report_date:
            resolved_text = _resolve_relative_time(text, report_date)
        tag = _extract_time_tag(resolved_text)
        groups[tag].append((idx, (text, value)))
    return groups


# ===================== rapidfuzz 辅助函数 =====================

def _rapidfuzz_dedup_evidences(
    evidences: List[Tuple[str, str]],
    fuzzy_threshold: float = 85.0,
) -> List[Tuple[str, str]]:
    """使用 rapidfuzz 的 token_set_ratio 对论据进行模糊去重。

    通过 Union-Find 将 token_set_ratio >= fuzzy_threshold 的论据归为一组，
    每组只保留第一条（出现最早的）。

    注意：本函数不做时间分组，调用方应先按时间分组后再调用。

    Args:
        evidences: List of (evidence_text, value) tuples.
        fuzzy_threshold: token_set_ratio 阈值（0-100），越高越保守。

    Returns:
        去重后的 (evidence_text, value) 列表。
    """
    if len(evidences) <= 1:
        return evidences

    N = len(evidences)
    texts = [e for e, _ in evidences]

    # Union-Find
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    for i in range(N):
        for j in range(i + 1, N):
            score = fuzz.token_set_ratio(texts[i], texts[j])
            if score >= fuzzy_threshold:
                union(i, j)

    # 每组保留 root（最小 ID）对应的论据
    kept_indices = set()
    for i in range(N):
        kept_indices.add(find(i))

    unique_evidences = [evidences[i] for i in sorted(kept_indices)]
    return unique_evidences


# ===================== 新版本：基于 rapidfuzz + semhash 的去重 =====================

async def drop_duplicate_evidences_by_similarity(
    evidences: List[Tuple[str, str]],
    semhash_threshold: float = 0.90,
    stock_entity_name: str = "",
    report_date: Optional[datetime] = None,
) -> List[Tuple[str, str]]:
    """基于 rapidfuzz 模糊去重 + semhash 语义去重进行两阶段去重（带时间分组约束）。

    去重策略：
    1. 先按时间标签进行一次分组。
    2. 在每个时间组内，依次执行：
       a. rapidfuzz 的 token_set_ratio 模糊去重
       b. semhash 语义哈希补充去重（阈值 >= semhash_threshold）

    对于判定为重复的一组证据，保留第一条（出现最早的）。

    Args:
        evidences: List of (evidence_text, value) tuples.
        semhash_threshold: semhash 语义相似度阈值（0-1），越高越保守。
        stock_entity_name: 公司名称，非空时会将 evidence 中的公司名替换为"公司"以提升去重效果。
        report_date: 研报写作日期，用于解析相对时间并按时间分组去重。

    Returns:
        去重后的 (evidence_text, value) 列表。
    """
    if not evidences:
        return evidences

    # 先做简单的文本清洗去重（与原版一致）
    _pairs: List[Tuple[str, str]] = []
    seen_texts: List[str] = []
    for text, value in evidences:
        cleaned = text.replace("查询", "").replace("确认", "").replace("计算", "").replace("获取", "")
        if stock_entity_name:
            cleaned = cleaned.replace(stock_entity_name, "")
        cleaned = (cleaned.replace("本公司", "").replace("本股票", "").replace("目标公司", "").replace("目标股票", "")
                   .replace("研报公司", "").replace("研报股票", "").replace("该公司", "").replace("该股票", "")
                   .replace("公司的", "").replace("股票的", "").replace("公司", ""))
        if cleaned not in seen_texts:
            seen_texts.append(cleaned)
            _pairs.append((cleaned, value))
    evidences = _pairs

    if len(evidences) <= 1:
        return evidences

    N_before = len(evidences)

    # ===== 一次性按时间标签分组 =====
    time_groups = _group_evidences_by_time(evidences, report_date)
    if report_date:
        print(f"    - 时间分组结果：{len(time_groups)} 个时间组 -> {', '.join(f'{k}({len(v)}条)' for k, v in sorted(time_groups.items()))}", flush=True)

    print(f"    - 对 {N_before} 条论据按时间组进行 rapidfuzz + semhash 两阶段去重...", flush=True)

    # ===== 在每个时间组内依次执行 rapidfuzz 去重 + semhash 去重 =====
    final_evidences: List[Tuple[str, str]] = []

    for tag, group_items in sorted(time_groups.items()):
        group_evs = [ev for _, ev in group_items]
        n_group_before = len(group_evs)

        if n_group_before <= 1:
            final_evidences.extend(group_evs)
            continue

        # --- 阶段1：rapidfuzz 模糊去重 ---
        group_evs = _rapidfuzz_dedup_evidences(group_evs)
        n_after_fuzz = len(group_evs)

        # --- 阶段2：semhash 语义去重 ---
        if len(group_evs) > 1:
            records = [{'text': group_evs[i][0], 'idx': i} for i in range(len(group_evs))]
            try:
                sh = SemHash.from_records(records, columns=['text'])
                result = sh.self_deduplicate(threshold=semhash_threshold)
                kept_indices = {rec['idx'] for rec in result.selected}
                group_evs = [group_evs[i] for i in sorted(kept_indices)]
            except Exception as e:
                print(f"      时间组 [{tag}]: semhash 出错 ({e})，保留 rapidfuzz 去重结果", flush=True)

        n_after_sem = len(group_evs)
        if n_group_before != n_after_sem:
            print(f"      时间组 [{tag}]: {n_group_before} -> {n_after_fuzz}(fuzz) -> {n_after_sem}(semhash)", flush=True)

        final_evidences.extend(group_evs)

    print(f"    - 两阶段去重总结：{N_before} -> {len(final_evidences)} 条", flush=True)
    return final_evidences


# ===================== 新版本：基于相似度的匹配 =====================

async def find_best_matches_by_similarity(
    source_evidences: List[Tuple[str, str]],
    ref_evidences: List[Tuple[str, str]],
    min_threshold: float = 0.75,
) -> List[Tuple[str, str, str, str]]:
    """基于 embedding cosine 相似度为参考论据匹配源论据。

    使用 embedding model 对 source 和 ref 的论据文本分别编码，
    计算 cosine 相似度矩阵，然后按分数降序贪心配对。

    Args:
        source_evidences: List of (evidence_text, value) from the source.
        ref_evidences: List of (evidence_text, value) from the reference.
        min_threshold: 最低 cosine 相似度阈值，低于此值视为无匹配。

    Returns:
        List of (source_evidence, ref_evidence, source_value, ref_value) tuples.
    """
    if not source_evidences or not ref_evidences:
        return []

    emb_model = create_emb_model()

    source_texts = [e for e, _ in source_evidences]
    ref_texts = [e for e, _ in ref_evidences]

    print(f"    - 正在通过 embedding model 计算 {len(ref_texts)} x {len(source_texts)} 的相似度矩阵...", flush=True)

    # 调用 embedding model 编码
    source_resp = await emb_model(source_texts)
    ref_resp = await emb_model(ref_texts)

    source_embs = np.array(source_resp.embeddings, dtype=np.float32)  # (S, D)
    ref_embs = np.array(ref_resp.embeddings, dtype=np.float32)        # (R, D)

    # L2 归一化后点积 = cosine similarity
    source_norms = np.linalg.norm(source_embs, axis=1, keepdims=True)
    ref_norms = np.linalg.norm(ref_embs, axis=1, keepdims=True)
    source_embs = source_embs / np.maximum(source_norms, 1e-8)
    ref_embs = ref_embs / np.maximum(ref_norms, 1e-8)

    # sim_matrix: (R, S)
    sim_matrix = ref_embs @ source_embs.T

    # 贪心匹配：按最高分数优先配对，避免一个 source 被多个 ref 重复匹配
    candidates = []
    for r_idx in range(len(ref_texts)):
        for s_idx in range(len(source_texts)):
            score = float(sim_matrix[r_idx, s_idx])
            if score >= min_threshold:
                candidates.append((score, r_idx, s_idx))
    candidates.sort(key=lambda x: x[0], reverse=True)

    evidence_pairs: List[Tuple[str, str, str, str]] = []
    matched_ref = set()
    matched_src = set()
    for score, r_idx, s_idx in candidates:
        if r_idx in matched_ref or s_idx in matched_src:
            continue
        s_text, s_value = source_evidences[s_idx]
        r_text, r_value = ref_evidences[r_idx]
        evidence_pairs.append((s_text, r_text, s_value, r_value))
        matched_ref.add(r_idx)
        matched_src.add(s_idx)
        print(f"      匹配 (score={score:.3f}): {s_text[:60]}  <->  {r_text[:60]}，[{s_value}] vs [{r_value}] ")

    print(f"    -> 配对完成，成功构建了 {len(evidence_pairs)} 对可供判断的论据（阈值={min_threshold}）。")
    return evidence_pairs


def _rapidfuzz_match(
    source_evidences: List[Tuple[str, str]],
    ref_evidences: List[Tuple[str, str]],
    match_threshold: float = 60.0,
    source_report_date: Optional[datetime] = None,
    ref_report_date: Optional[datetime] = None,
) -> List[Tuple[str, str, str, str]]:
    """使用 rapidfuzz 的 token_set_ratio 进行两个论据列表之间的模糊匹配（带时间约束）。

    先将两边论据中的相对时间转为具体时间，提取时间标签。
    匹配时优先在相同时间标签的论据之间进行，NO_TIME 组的论据可以与任意组匹配。
    对每对 (source, ref) 计算 token_set_ratio 分数，按分数降序贪心配对。

    Args:
        source_evidences: List of (evidence_text, value) from the source.
        ref_evidences: List of (evidence_text, value) from the reference.
        match_threshold: token_set_ratio 最低匹配阈值（0-100）。
        source_report_date: 源研报写作日期。
        ref_report_date: 参考研报写作日期。

    Returns:
        List of (source_evidence, ref_evidence, source_value, ref_value) tuples.
    """
    if not source_evidences or not ref_evidences:
        return []

    # 为每条论据计算时间标签（先转绝对时间再提取标签）
    source_tags = []
    for text, _ in source_evidences:
        resolved = _resolve_relative_time(text, source_report_date) if source_report_date else text
        source_tags.append(_extract_time_tag(resolved))

    ref_tags = []
    for text, _ in ref_evidences:
        resolved = _resolve_relative_time(text, ref_report_date) if ref_report_date else text
        ref_tags.append(_extract_time_tag(resolved))

    source_texts = [e for e, _ in source_evidences]
    ref_texts = [e for e, _ in ref_evidences]

    # 计算所有 (ref, source) 对的 token_set_ratio 分数，加时间约束
    candidates = []
    for r_idx, r_text in enumerate(ref_texts):
        for s_idx, s_text in enumerate(source_texts):
            r_tag = ref_tags[r_idx]
            s_tag = source_tags[s_idx]
            # 时间约束：只有同一时间标签或其中一方为 NO_TIME 时才允许匹配
            if r_tag != s_tag and r_tag != "NO_TIME" and s_tag != "NO_TIME":
                continue
            score = fuzz.token_set_ratio(r_text, s_text)
            if score >= match_threshold:
                candidates.append((score, r_idx, s_idx))

    # 按分数降序排列，贪心匹配
    candidates.sort(key=lambda x: x[0], reverse=True)

    evidence_pairs: List[Tuple[str, str, str, str]] = []
    matched_src = set()
    matched_ref = set()

    for score, r_idx, s_idx in candidates:
        if s_idx in matched_src or r_idx in matched_ref:
            continue
        s_text, s_value = source_evidences[s_idx]
        r_text, r_value = ref_evidences[r_idx]
        evidence_pairs.append((s_text, r_text, s_value, r_value))
        matched_src.add(s_idx)
        matched_ref.add(r_idx)
        tag_info = f"[{source_tags[s_idx]}↔{ref_tags[r_idx]}]" if source_report_date or ref_report_date else ""
        print(f"      rapidfuzz匹配 (score={score:.1f}) {tag_info}: [{s_value}] {s_text[:60]}  <->  [{r_value}] {r_text[:60]}")

    return evidence_pairs


async def find_best_matches(
    source_evidences: List[Tuple[str, str]],
    ref_evidences: List[Tuple[str, str]],
    source_report_date: Optional[datetime] = None,
    ref_report_date: Optional[datetime] = None,
) -> List[Tuple[str, str, str, str]]:
    """使用 rapidfuzz 的 token_set_ratio 进行模糊匹配（带时间约束），为参考论据列表中的每一项在源论据列表中找到最佳匹配。

    匹配前先将相对时间转为具体时间，只在相同时间标签的论据之间进行匹配。
    如果 rapidfuzz 未产生有效匹配，则回退到 BM25 匹配。

    Args:
        source_evidences: List of (evidence_text, value) from the source (report).
        ref_evidences: List of (evidence_text, value) from the reference.
        source_report_date: 源研报写作日期，用于解析相对时间。
        ref_report_date: 参考研报写作日期，用于解析相对时间。

    Returns:
        List of (source_evidence, ref_evidence, source_value, ref_value) tuples.
    """
    if not source_evidences or not ref_evidences:
        return []

    date_info = ""
    if source_report_date:
        date_info += f" source_date={source_report_date.strftime('%Y-%m-%d')}"
    if ref_report_date:
        date_info += f" ref_date={ref_report_date.strftime('%Y-%m-%d')}"
    print(f"    - 使用 rapidfuzz 进行模糊匹配（时间约束）: {len(source_evidences)} x {len(ref_evidences)}{date_info} ...", flush=True)

    try:
        evidence_pairs = _rapidfuzz_match(
            source_evidences, ref_evidences,
            source_report_date=source_report_date,
            ref_report_date=ref_report_date,
        )
    except Exception as e:
        print(f"    - rapidfuzz 匹配过程出错: {e}，回退到 BM25 匹配", flush=True)
        traceback.print_exc()
        evidence_pairs = []

    if not evidence_pairs:
        # 回退方案：使用 BM25 进行匹配
        print(f"    - rapidfuzz 未产生有效匹配，回退到 BM25 匹配...", flush=True)
        evidence_pairs = await _fallback_bm25_match(source_evidences, ref_evidences)

    print(f"    -> 配对完成，成功构建了 {len(evidence_pairs)} 对可供判断的论据。")
    return evidence_pairs


async def _fallback_bm25_match(
    source_evidences: List[Tuple[str, str]],
    ref_evidences: List[Tuple[str, str]],
    min_threshold: float = 0.5,
) -> List[Tuple[str, str, str, str]]:
    """BM25 回退匹配方案，当 dedupe 匹配失败时使用。

    Args:
        source_evidences: List of (evidence_text, value) from the source.
        ref_evidences: List of (evidence_text, value) from the reference.
        min_threshold: 最低 BM25 归一化相似度阈值。

    Returns:
        List of (source_evidence, ref_evidence, source_value, ref_value) tuples.
    """
    source_texts = [e for e, _ in source_evidences]
    ref_texts = [e for e, _ in ref_evidences]

    bm25_sim = _bm25_similarity_matrix(ref_texts, source_texts)  # (len(ref), len(source))

    # 贪心匹配
    candidates = []
    for r_idx in range(len(ref_texts)):
        for s_idx in range(len(source_texts)):
            score = bm25_sim[r_idx, s_idx]
            if score >= min_threshold:
                candidates.append((score, r_idx, s_idx))
    candidates.sort(key=lambda x: x[0], reverse=True)

    evidence_pairs: List[Tuple[str, str, str, str]] = []
    matched_ref = set()
    matched_src = set()
    for score, r_idx, s_idx in candidates:
        if r_idx in matched_ref or s_idx in matched_src:
            continue
        s_text, s_value = source_evidences[s_idx]
        r_text, r_value = ref_evidences[r_idx]
        evidence_pairs.append((s_text, r_text, s_value, r_value))
        matched_ref.add(r_idx)
        matched_src.add(s_idx)

    return evidence_pairs


async def find_locations_in_outline(outline_content_str: str, evidences_to_find: List[str]) -> Dict[str, str]:
    """在研报大纲中定位论据的位置。"""
    print(f"  - 正在大纲中定位 {len(evidences_to_find)} 条共通论据的位置...")
    if not evidences_to_find:
        return {}

    numbered_evidences = "\n".join(f"EV_{i + 1}: {e}" for i, e in enumerate(evidences_to_find))

    # 为了防止Prompt过长，对outline_content进行简化，只保留关键信息
    outline_json = json.loads(outline_content_str)
    simplified_outline = []

    def simplify_section(section, prefix):
        current_loc = f"{prefix}s{section.get('section_id')}"
        simplified_outline.append(f"章节点: {current_loc}, 标题: \"{section.get('title')}\"")
        if section.get('segments'):
            for i, seg in enumerate(section['segments']):
                seg_loc = f"{current_loc}_p{seg.get('segment_id', i + 1)}"
                if seg.get('evidences'):
                    simplified_outline.append(
                        f"  - 段落点: {seg_loc}, 论据: {json.dumps(seg.get('evidences'), ensure_ascii=False)}")
        if section.get('subsections'):
            for sub in section['subsections']:
                simplify_section(sub, f"{current_loc}_")

    simplify_section(outline_json, "")
    simplified_outline_text = "\n".join(simplified_outline)

    prompt = """你是一名精准的文本定位专家。你的任务是在给定的研报大纲结构中，为一系列指定的论据找到它们的确切位置路径。

## 背景与定义
1.  **研报大纲**: 以下是一份简化版的研报结构，每一行代表一个章节或段落。
2.  **位置路径格式**: 路径由 `s{id}` (章节) 和 `p{id}` (段落) 组成，层层相连。
    - `s{id}`: 代表 `section_id`。
    - `p{id}`: 代表 `segment_id`。
    - **示例**: `s1_s3_p2` 表示在 `section_id=1` 的章节下的 `section_id=3` 的子章节中的 `segment_id=2` 的段落。

## 任务指令
1.  仔细阅读下面提供的 "研报大纲结构"。
2.  对于 "待查找论据列表" 中的每一条论据（以 EV_ID 开头），在大纲的 `evidences` 数组中找到与之**完全匹配或语义上非常相似**的条目。
3.  记录下该条目所属的 "段落点" (例如 `s1_s3_p2`)。
4.  你的答案必须包裹在<ANSWER>和</ANSWER>内。其中每一行是论据的ID和找到的位置路径字符串，逗号隔开。每一条论据ID都必须有对应的路径，不能有遗漏。

### 输出示例
<ANSWER>
EV_1,s1_s2_p1
EV_2,s3_p1
EV_3,s2_s3_p2
</ANSWER>
"""

    def _parse(response: str) -> Dict[str, str]:
        locations_map = {}
        answer_match = re.search(r"<ANSWER>(.+)</?ANSWER>", response, re.DOTALL)
        assert answer_match is not None, "输出格式错误：未找到<ANSWER>和</ANSWER>标签包裹的内容。"
        answer_content = answer_match.group(1).strip().replace('-', '_')
        for line in answer_content.splitlines():
            parts = line.strip().split(",", 1)
            if len(parts) < 2:
                print(f"输出格式错误：每行应包含论据ID和位置路径，用逗号分隔。错误行: {line}")
            ev_id, location = parts
            assert ev_id.strip().startswith("EV_"), f"输出格式错误：论据ID没有用EV_开头。错误行: {line}"
            locations_map[ev_id.strip()] = location.strip()
        return locations_map

    locations_map_by_id = await call_chatbot_with_retry(
        llm_instruct, formatter,
        prompt, f"**研报大纲结构:**\n---\n{simplified_outline_text}\n---\n"
                f"**待查找论据列表:**\n---\n{numbered_evidences}\n---\n",
        hook=_parse, handle_hook_exceptions=(AssertionError,), max_retries=5,
    )
    locations_map_by_text = {}
    for i, evidence in enumerate(evidences_to_find):
        ev_id = f"EV_{i + 1}"
        location = locations_map_by_id[ev_id]
        locations_map_by_text[evidence] = location
    print("    -> 定位完成。")
    return locations_map_by_text

def find_report_pairs(pdf_directory: Path, earliest_date: str = "2025-01-01", latest_date: str = "2025-12-31") -> List[Tuple[str, Path, Path]]:
    """
    配对时间相近的两个研报。

    Args:
        pdf_directory: PDF文件所在目录
        earliest_date: 最早日期，格式为 'YYYY-MM-DD'
        latest_date: 最晚日期，格式为 'YYYY-MM-DD'

    Returns:
        研报对列表，每项为 (stock_code, old_report_path, new_report_path)
    """
    print(f"--- 步骤 1: 正在扫描文件夹并配对研报 (时间范围: {earliest_date} 到 {latest_date}) ---")

    from datetime import datetime
    earliest_dt = datetime.strptime(earliest_date, "%Y-%m-%d")
    latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")

    # 按股票代码和报告日期存储所有研报
    stock_reports = defaultdict(list)

    for pdf_file in pdf_directory.glob("*.pdf"):
        parts = pdf_file.name.split("_")
        if len(parts) < 2: continue
        stock_code = parts[0]
        date_str = parts[1]

        if not stock_code.isdigit() or len(stock_code) < 6: continue

        # 解析日期
        report_date = datetime.strptime(date_str, "%Y-%m-%d")

        # 检查是否在指定时间范围内
        if not (earliest_dt <= report_date <= latest_dt): continue

        stock_reports[stock_code].append((report_date, pdf_file))

    pairs = []
    for stock_code, reports_list in stock_reports.items():
        if len(reports_list) < 2: continue

        # 按日期排序
        reports_list.sort(key=lambda x: x[0])

        # 配对相邻的两个报告
        for i in range(len(reports_list) - 1):
            old_date, old_path = reports_list[i]
            new_date, new_path = reports_list[i + 1]
            if old_date == new_date: continue
            pairs.append((stock_code, old_path, new_path))
            print(f"  - 成功配对: {stock_code} -> {old_path.name} vs {new_path.name}")

    print(f"配对完成！共找到 {len(pairs)} 对符合条件的研报。")
    return pairs

def save_result_with_lock(result_data: Dict, output_json_path: Path) -> bool:
    """
    使用文件锁机制安全地保存单个结果到JSON文件。

    Args:
        result_data: 要保存的结果数据字典
        output_json_path: 输出JSON文件的路径

    Returns:
        bool: 保存成功返回 True，失败返回 False
    """
    if result_data is None:
        return False

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_json_path.with_suffix('.lock')

    # 创建文件锁，超时时间为 60 秒
    with FileLock(str(lock_path), timeout=60) as lock:
        try:
            # 读取现有的 JSON 数据
            results_cache = {}
            if output_json_path.exists():
                # try:
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                # except (json.JSONDecodeError, KeyError) as e:
                #     print(f"  ⚠️  警告: 读取现有结果文件时出错: {e}，将创建新文件。")
                results_cache = {(res['stock_code'], res['old_report'], res['new_report']): res for res in existing_results}

            # 更新或插入新结果
            results_cache[(result_data['stock_code'], result_data['old_report'], result_data['new_report'])] = result_data

            # 将更新后的结果写回文件
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(list(results_cache.values()), f, ensure_ascii=False, indent=4)

            print(f"  ✓ 结果已安全保存到 {output_json_path.name}")
            return True
        except Exception as e:
            print(f"  ✗ 保存结果时出错: {e}")
            traceback.print_exc()
        finally:
            lock.release(force=True)
    return False


async def process_single_stock_pair(stock_code: str, old_path: Path, new_path: Path,
                                    long_term_dir: Path, output_json_path: Path,
                                    results_cache: Dict, semaphore: asyncio.Semaphore) -> Optional[Dict]:
    """
    处理单个股票对的函数，可并发执行。

    Args:
        stock_code: 股票代码
        old_path: 旧报告路径
        new_path: 新报告路径
        long_term_dir: 长期记忆目录
        output_json_path: 输出JSON文件路径
        results_cache: 结果缓存字典
        semaphore: 用于限制并发数的信号量

    Returns:
        该股票的处理结果数据字典
    """

    async with semaphore:
        try:
            print(f"\n======= 正在处理股票: {stock_code} =======", flush=True)

            stock_entity_name = get_entity_name_by_code(stock_code)
            if stock_entity_name:
                print(f"  - 公司名称: {stock_entity_name}", flush=True)

            evidences_old = await extract_unique_evidences_from_pdf(old_path, long_term_dir / "demonstration", stock_entity_name=stock_entity_name)
            evidences_new = await extract_unique_evidences_from_pdf(new_path, long_term_dir / "demonstration", stock_entity_name=stock_entity_name)
            # ===== 并发提取两份报告的论据 =====
            # evidences_old, evidences_new = await asyncio.gather(
            #     extract_unique_evidences_from_pdf(old_path, long_term_dir / "demonstration"),
            #     extract_unique_evidences_from_pdf(new_path, long_term_dir / "demonstration")
            # )
            # 从文件名中提取研报写作日期，用于匹配时的时间约束
            old_report_date = _parse_report_date_from_path(old_path)
            new_report_date = _parse_report_date_from_path(new_path)
            common_evidences_texts = await find_best_matches(
                evidences_old, evidences_new,
                source_report_date=old_report_date,
                ref_report_date=new_report_date,
            )

            assert len(common_evidences_texts) > 0, "!!! 未找到任何共通论据"
            common_evidences_with_locs = [
                {"text": (old_text, new_text)}
                for old_text, new_text in common_evidences_texts
            ]

            result_data = {
                "stock_code": stock_code, "old_report": old_path.name, "new_report": new_path.name,
                "old_evidence_count": len(evidences_old), "new_evidence_count": len(evidences_new),
                "common_evidence_count": len(common_evidences_with_locs),
                # "common_evidences": common_evidences_with_locs
            }

            print(f"--- 股票 {stock_code} 处理完成 ---", flush=True)

            # 立即使用文件锁保存结果
            save_result_with_lock(result_data, output_json_path)

            return result_data

        except Exception as e:
            traceback.print_exc()
            print(f"!!! 处理股票 {stock_code} 时发生错误: {e}，跳过该股票。", flush=True)
            return None

def _process_single_stock_pair(stock_code: str, old_path: Path, new_path: Path,
                                    long_term_dir: Path, output_json_path: Path,
                                    results_cache: Dict, semaphore) -> Optional[Dict]:
    return asyncio.run(process_single_stock_pair(stock_code, old_path, new_path, long_term_dir, output_json_path, results_cache, semaphore))

async def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    long_term_dir = PROJECT_ROOT / "data" / "memory" / "long_term"

    # 配对研报，指定时间范围
    report_pairs = find_report_pairs(DEMO_DIR, earliest_date="2025-01-01", latest_date="2025-12-31")
    if not report_pairs: print("未找到任何可处理的研报对，程序退出。"); return

    existing_results = []
    output_json_path = PROJECT_ROOT / "output" / "comparison_results.json"
    output_txt_path = PROJECT_ROOT  / "output" / "comparison_results.txt"
    if output_json_path.exists():
        with open(output_json_path, 'r', encoding='utf-8') as f:
            try: existing_results = json.load(f)
            except json.JSONDecodeError: print(f"警告: {output_json_path} 文件为空或已损坏，将创建新文件。")
    results_cache = {(res['stock_code'], res['old_report'], res['new_report']): res for res in existing_results}
    print(f"检测到 {len(results_cache)} 个已保存的结果。")

    print(f"\n--- 步骤 2: 开始批量处理研报对（并发模式，最大并发数: {MAX_CONCURRENT}）... ---")

    # 创建信号量，用于限制并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # results = []
    # for stock_code, old_path, new_path in report_pairs:
    #     if (stock_code, old_path.name, new_path.name) not in results_cache:
    #         res = await process_single_stock_pair(stock_code, old_path, new_path, long_term_dir, output_json_path, results_cache, semaphore)
    #         results.append(res)

    # 创建并发任务列表
    tasks = [
        process_single_stock_pair(stock_code, old_path, new_path, long_term_dir, output_json_path, results_cache, semaphore)
        for stock_code, old_path, new_path in report_pairs if (stock_code, old_path.name, new_path.name) not in results_cache
    ]
    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # # 使用线程池并发执行所有任务
    # results = []
    # with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
    #     # 提交所有任务
    #     futures = {
    #         executor.submit(
    #             _process_single_stock_pair,
    #             stock_code, old_path, new_path,
    #             long_term_dir, output_json_path, results_cache, semaphore
    #         ): stock_code
    #         for stock_code, old_path, new_path in report_pairs if (stock_code, old_path.name, new_path.name) not in results_cache
    #     }
    #     # 收集结果
    #     for future in as_completed(futures):
    #         stock_code = futures[future]
    #         try:
    #             result = future.result()
    #             results.append(result)
    #         except Exception as e:
    #             print(f"任务执行中发生异常 (股票 {stock_code}): {e}")
    #             traceback.print_exc()
    #             results.append(None)

    # 处理结果统计
    successful_count = 0
    failed_count = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"任务执行中发生异常: {result}")
            failed_count += 1
            continue
        if result is not None and isinstance(result, dict):
            successful_count += 1
            results_cache[(result['stock_code'], result['old_report'], result['new_report'])] = result
        else:
            failed_count += 1

    print(f"\n✓ 成功处理: {successful_count} 个任务")
    print(f"✗ 失败处理: {failed_count} 个任务")
    print(f"(所有结果已在处理过程中实时保存)")

    print("\n--- 步骤 3: 所有股票处理完毕，正在生成最终排序报告... ---")
    if not results_cache: print("未找到任何结果，无法生成排序报告。"); return
    
    final_results = list(results_cache.values())
    final_results.sort(key=lambda x: x["common_evidence_count"], reverse=True)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("研报共通论据数量排序 \n====================================\n\n")
        for i, res in enumerate(final_results):
            f.write(f"第 {i+1} 名: 股票代码 {res['stock_code']}\n")
            f.write(f"  - 共通论据数量: {res['common_evidence_count']}\n")
            f.write(f"  - (old论据数: {res['old_evidence_count']}, new论据数: {res['new_evidence_count']})\n")
            f.write(f"  - old 报告: {res['old_report']}\n")
            f.write(f"  - new 报告: {res['new_report']}\n\n")

    print(f"排序报告已生成: {output_txt_path}"); print("--- 全部任务完成！ ---")


if __name__ == "__main__":
    asyncio.run(main())