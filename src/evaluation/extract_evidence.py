import re
import traceback
from pathlib import Path

import json
import difflib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from filelock import FileLock
try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None
try:
    from semhash import SemHash
except ImportError:
    SemHash = None

from src.memory.working import Evidence, Section, evidence_pairs
from src.utils.local_file import DEMO_DIR
from src.utils.retrieve_in_memory import _bm25_scores, _tokenize_for_bm25
import config

# 最大并发数限制
MAX_CONCURRENT = 8
CONFIG = config.Config()


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

EvidenceTuple = Tuple[str, str]
EvidenceMatch = Tuple[str, str, str, str]


_code_name_df: Optional[pd.DataFrame] = None


def _load_code_name_df() -> pd.DataFrame:
    global _code_name_df
    if _code_name_df is None:
        csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "memory" / "long_term" / "a_share_code_name.csv"
        if csv_path.exists():
            _code_name_df = pd.read_csv(csv_path, dtype=str)
        else:
            _code_name_df = pd.DataFrame(columns=["code", "name"])
    return _code_name_df


def get_entity_name_by_code(stock_code: str) -> str:
    code = str(stock_code).zfill(6)
    df = _load_code_name_df()
    hit = df[df["code"] == code]
    return "" if hit.empty else str(hit.iloc[0]["name"])


def _normalize_evidence_tuple(item) -> EvidenceTuple:
    if isinstance(item, Evidence):
        return item.text.strip(), (item.fact or "").strip()
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("evidence") or item.get("key") or "").strip()
        fact = str(item.get("fact") or "").strip()
        if not text and len(item) == 1:
            key, val = next(iter(item.items()))
            return str(key).strip(), str(val).strip() if val is not None else ""
        return text, fact
    if isinstance(item, (list, tuple)) and item:
        text = str(item[0]).strip()
        fact = str(item[1]).strip() if len(item) > 1 and item[1] is not None else ""
        return text, fact
    return str(item).strip(), ""


def _normalize_evidence_tuples(items) -> List[EvidenceTuple]:
    normalized = []
    for item in items or []:
        text, fact = _normalize_evidence_tuple(item)
        if text:
            normalized.append((text, fact))
    return normalized


def get_all_evidences_from_section(section: Section) -> List[EvidenceTuple]:
    """递归地从 Section 对象中收集 (论据描述, 具体事实)。"""
    evidences: List[EvidenceTuple] = []
    if section.segments:
        for segment in section.segments:
            evidences.extend(evidence_pairs(segment.evidences))
    if section.subsections:
        for subsection in section.subsections:
            if "摘要" not in (subsection.title or ""):
                evidences.extend(get_all_evidences_from_section(subsection))
    return evidences


async def extract_unique_evidences_from_pdf(
    pdf_path: Path,
    save_dir: Path,
    only_evidence: bool = False,
    stock_entity_name: str = "",
) -> List[EvidenceTuple]:
    pdf_stem = pdf_path.stem
    evidence_filename = f"{pdf_stem}_evidences.json"
    evidence_path = save_dir / CONFIG.llm_name / evidence_filename
    if evidence_path.exists():
        print(f"    - 检测到已有的evidences，加载: {evidence_filename}")
        return _normalize_evidence_tuples(json.loads(evidence_path.read_text(encoding="utf-8")))

    report_date = _parse_report_date_from_path(pdf_path)
    if report_date:
        print(f"  - 研报写作日期: {report_date.strftime('%Y-%m-%d')}", flush=True)

    print(f"\n-> 开始提取并清洗文件: {pdf_path.name}", flush=True)
    from src.pipelines.planning import process_pdf_to_outline
    from src.utils.instance import formatter, llm_instruct, llm_reasoning

    manuscript = await process_pdf_to_outline(pdf_path, save_dir, llm_reasoning, llm_instruct, formatter, only_evidence)
    print(f"  - 从 {pdf_path.name} 的结构中提取论据...")
    return await extract_unique_evidences(
        manuscript,
        evidence_path,
        stock_entity_name=stock_entity_name,
        report_date=report_date,
    )


async def extract_unique_evidences(
    manuscript: Section,
    evidence_path: Path,
    stock_entity_name: str = "",
    report_date: Optional[datetime] = None,
) -> List[EvidenceTuple]:
    if evidence_path.exists():
        return _normalize_evidence_tuples(json.loads(evidence_path.read_text(encoding="utf-8")))

    evidences = get_all_evidences_from_section(manuscript)
    print(f"  - 对 {len(evidences)} 条原始论据进行语义去重...", flush=True)
    evidences = await drop_duplicate_evidences_by_similarity(
        evidences,
        stock_entity_name=stock_entity_name,
        report_date=report_date,
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidences, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"    -> Evidence已保存到: {evidence_path}", flush=True)
    return evidences


def _parse_report_date_from_path(pdf_path: Path) -> Optional[datetime]:
    parts = pdf_path.stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(parts[1], "%Y-%m-%d")
    except ValueError:
        return None


def _quarter_label(year: int, quarter: int) -> str:
    while quarter > 4:
        year += 1
        quarter -= 4
    while quarter < 1:
        year -= 1
        quarter += 4
    return f"{year}年Q{quarter}"


def _replace_relative_quarter(text: str, report_date: datetime) -> str:
    year = report_date.year
    quarter = (report_date.month - 1) // 3 + 1
    result = text
    result = re.sub(r"下一?季度", _quarter_label(year, quarter + 1), result)
    result = re.sub(r"上一?季度", _quarter_label(year, quarter - 1), result)
    return re.sub(r"[本当]季度", _quarter_label(year, quarter), result)


def _replace_relative_year(text: str, report_date: datetime) -> str:
    year = report_date.year
    result = text
    result = re.sub(r"前年", f"{year - 2}年", result)
    result = re.sub(r"去年|上年|上一年|上年度", f"{year - 1}年", result)
    result = re.sub(r"今年|本年|本年度", f"{year}年", result)
    return re.sub(r"明年|下一年", f"{year + 1}年", result)


def _resolve_relative_time(text: str, report_date: Optional[datetime]) -> str:
    if report_date is None:
        return text
    return _replace_relative_year(_replace_relative_quarter(text, report_date), report_date)


def _extract_time_tag(text: str) -> str:
    quarter_match = re.search(r"(20\d{2})\s*年?\s*[Qq]\s*([1-4])", text)
    if quarter_match:
        return f"{quarter_match.group(1)}Q{quarter_match.group(2)}"
    half_match = re.search(r"(20\d{2})\s*年?\s*([上下])半年", text)
    if half_match:
        return f"{half_match.group(1)}H{'1' if half_match.group(2) == '上' else '2'}"
    year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", text)
    if year_match:
        return year_match.group(1)
    return "NO_TIME"


def _clean_evidence_text(text: str, stock_entity_name: str = "") -> str:
    cleaned = (text or "").strip()
    for token in ("查询", "确认", "计算", "获取"):
        cleaned = cleaned.replace(token, "")
    if stock_entity_name:
        cleaned = cleaned.replace(stock_entity_name, "")
    for token in ("本公司", "本股票", "目标公司", "目标股票", "研报公司", "研报股票", "该公司", "该股票", "公司的", "股票的"):
        cleaned = cleaned.replace(token, "")
    return cleaned.replace("公司", "").strip()


def _similarity_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if fuzz is not None:
        return float(fuzz.token_set_ratio(left, right)) / 100.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _group_evidences_by_time(
    evidences: List[EvidenceTuple],
    report_date: Optional[datetime],
) -> Dict[str, List[EvidenceTuple]]:
    groups: Dict[str, List[EvidenceTuple]] = defaultdict(list)
    for evidence in evidences:
        resolved = _resolve_relative_time(evidence[0], report_date)
        groups[_extract_time_tag(resolved)].append(evidence)
    return groups


def _semhash_filter_evidences(evidences: List[EvidenceTuple], semhash_threshold: float) -> List[EvidenceTuple]:
    if SemHash is None or len(evidences) <= 1:
        return evidences
    try:
        records = [{"text": evidence[0], "idx": idx} for idx, evidence in enumerate(evidences)]
        result = SemHash.from_records(records, columns=["text"]).self_deduplicate(threshold=semhash_threshold)
        kept_indices = {record["idx"] for record in result.selected}
        return [evidences[idx] for idx in sorted(kept_indices)]
    except Exception as exc:
        print(f"      semhash 去重失败，保留 rapidfuzz 结果: {exc}", flush=True)
        return evidences


def _dedupe_evidence_group(evidences: List[EvidenceTuple], fuzzy_threshold: float) -> List[EvidenceTuple]:
    kept: List[EvidenceTuple] = []
    for evidence in evidences:
        if any(_similarity_score(evidence[0], existing[0]) >= fuzzy_threshold for existing in kept):
            continue
        kept.append(evidence)
    return kept


async def drop_duplicate_evidences_by_similarity(
    evidences: List[EvidenceTuple],
    fuzzy_threshold: float = 0.85,
    semhash_threshold: float = 0.90,
    stock_entity_name: str = "",
    report_date: Optional[datetime] = None,
) -> List[EvidenceTuple]:
    normalized = []
    seen = set()
    for text, value in _normalize_evidence_tuples(evidences):
        cleaned = _clean_evidence_text(text, stock_entity_name)
        key = (cleaned, value)
        if cleaned and key not in seen:
            seen.add(key)
            normalized.append((cleaned, value))

    if len(normalized) <= 1:
        return normalized

    groups = _group_evidences_by_time(normalized, report_date)
    final_evidences: List[EvidenceTuple] = []
    for tag, group in sorted(groups.items()):
        deduped = _dedupe_evidence_group(group, fuzzy_threshold)
        deduped = _semhash_filter_evidences(deduped, semhash_threshold)
        if len(group) != len(deduped):
            print(f"      时间组 [{tag}]: {len(group)} -> {len(deduped)}", flush=True)
        final_evidences.extend(deduped)
    print(f"    - 两阶段去重总结：{len(normalized)} -> {len(final_evidences)} 条", flush=True)
    return final_evidences


async def drop_duplicate_evidences(evidences: List[str]) -> List[str]:
    pairs = [(evidence, "") for evidence in evidences]
    deduped = await drop_duplicate_evidences_by_similarity(pairs)
    return [text for text, _ in deduped]


def _bm25_similarity_matrix(texts_a: List[str], texts_b: List[str]) -> np.ndarray:
    tokens_a = [_tokenize_for_bm25(t) for t in texts_a]
    tokens_b = [_tokenize_for_bm25(t) for t in texts_b]
    matrix = np.zeros((len(texts_a), len(texts_b)), dtype=np.float32)
    for idx, query_tokens in enumerate(tokens_a):
        matrix[idx] = _bm25_scores(query_tokens, tokens_b)
    minimum = float(matrix.min()) if matrix.size else 0.0
    maximum = float(matrix.max()) if matrix.size else 0.0
    if maximum - minimum > 1e-8:
        matrix = (matrix - minimum) / (maximum - minimum)
    return matrix


def _rapidfuzz_match(
    source_evidences: List[EvidenceTuple],
    ref_evidences: List[EvidenceTuple],
    match_threshold: float = 0.60,
    source_report_date: Optional[datetime] = None,
    ref_report_date: Optional[datetime] = None,
) -> List[EvidenceMatch]:
    candidates = []
    source_tags = [_extract_time_tag(_resolve_relative_time(text, source_report_date)) for text, _ in source_evidences]
    ref_tags = [_extract_time_tag(_resolve_relative_time(text, ref_report_date)) for text, _ in ref_evidences]
    for ref_idx, ref_evidence in enumerate(ref_evidences):
        for source_idx, source_evidence in enumerate(source_evidences):
            if ref_tags[ref_idx] != source_tags[source_idx] and "NO_TIME" not in {ref_tags[ref_idx], source_tags[source_idx]}:
                continue
            score = _similarity_score(ref_evidence[0], source_evidence[0])
            if score >= match_threshold:
                candidates.append((score, ref_idx, source_idx))
    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_ref = set()
    matched_source = set()
    matches: List[EvidenceMatch] = []
    for score, ref_idx, source_idx in candidates:
        if ref_idx in matched_ref or source_idx in matched_source:
            continue
        source_text, source_value = source_evidences[source_idx]
        ref_text, ref_value = ref_evidences[ref_idx]
        matches.append((source_text, ref_text, source_value, ref_value))
        matched_ref.add(ref_idx)
        matched_source.add(source_idx)
    return matches


async def _fallback_bm25_match(
    source_evidences: List[EvidenceTuple],
    ref_evidences: List[EvidenceTuple],
    min_threshold: float = 0.5,
) -> List[EvidenceMatch]:
    source_texts = [text for text, _ in source_evidences]
    ref_texts = [text for text, _ in ref_evidences]
    bm25_sim = _bm25_similarity_matrix(ref_texts, source_texts)
    candidates = []
    for ref_idx in range(len(ref_texts)):
        for source_idx in range(len(source_texts)):
            score = float(bm25_sim[ref_idx, source_idx])
            if score >= min_threshold:
                candidates.append((score, ref_idx, source_idx))
    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_ref = set()
    matched_source = set()
    matches: List[EvidenceMatch] = []
    for score, ref_idx, source_idx in candidates:
        if ref_idx in matched_ref or source_idx in matched_source:
            continue
        source_text, source_value = source_evidences[source_idx]
        ref_text, ref_value = ref_evidences[ref_idx]
        matches.append((source_text, ref_text, source_value, ref_value))
        matched_ref.add(ref_idx)
        matched_source.add(source_idx)
    return matches


def _embeddings_to_array(response) -> np.ndarray:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if embeddings is None:
        raise ValueError("embedding response does not contain embeddings")
    return np.array(embeddings, dtype=np.float32)


async def _embedding_match(
    source_evidences: List[EvidenceTuple],
    ref_evidences: List[EvidenceTuple],
    min_threshold: float,
) -> List[EvidenceMatch]:
    from src.utils.instance import create_emb_model

    emb_model = create_emb_model()
    source_texts = [text for text, _ in source_evidences]
    ref_texts = [text for text, _ in ref_evidences]
    print(
        f"    - 正在通过 embedding model 计算 {len(ref_texts)} x {len(source_texts)} 的相似度矩阵...",
        flush=True,
    )

    source_response = await emb_model(source_texts)
    ref_response = await emb_model(ref_texts)
    source_embeddings = _embeddings_to_array(source_response)
    ref_embeddings = _embeddings_to_array(ref_response)

    source_norms = np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    ref_norms = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    source_embeddings = source_embeddings / np.maximum(source_norms, 1e-8)
    ref_embeddings = ref_embeddings / np.maximum(ref_norms, 1e-8)

    sim_matrix = ref_embeddings @ source_embeddings.T
    candidates = []
    for ref_idx in range(len(ref_texts)):
        for source_idx in range(len(source_texts)):
            score = float(sim_matrix[ref_idx, source_idx])
            if score >= min_threshold:
                candidates.append((score, ref_idx, source_idx))
    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_ref = set()
    matched_source = set()
    matches: List[EvidenceMatch] = []
    for score, ref_idx, source_idx in candidates:
        if ref_idx in matched_ref or source_idx in matched_source:
            continue
        source_text, source_value = source_evidences[source_idx]
        ref_text, ref_value = ref_evidences[ref_idx]
        matches.append((source_text, ref_text, source_value, ref_value))
        matched_ref.add(ref_idx)
        matched_source.add(source_idx)
        print(
            f"      embedding匹配 (score={score:.3f}): {source_text[:60]}  <->  {ref_text[:60]}",
            flush=True,
        )
    return matches


async def find_best_matches_by_similarity(
    source_evidences: List[EvidenceTuple],
    ref_evidences: List[EvidenceTuple],
    source_report_date: Optional[datetime] = None,
    ref_report_date: Optional[datetime] = None,
    embedding_threshold: float = 0.7,
) -> List[EvidenceMatch]:
    source_evidences = _normalize_evidence_tuples(source_evidences)
    ref_evidences = _normalize_evidence_tuples(ref_evidences)
    if not source_evidences or not ref_evidences:
        return []

    try:
        matches = await _embedding_match(source_evidences, ref_evidences, embedding_threshold)
    except Exception as exc:
        print(f"    - embedding 匹配失败: {exc}，回退到 rapidfuzz/BM25 匹配", flush=True)
        matches = []
    if matches:
        print(f"    -> 配对完成，成功构建了 {len(matches)} 对可供判断的论据（embedding阈值={embedding_threshold}）。")
        return matches

    matches = _rapidfuzz_match(
        source_evidences,
        ref_evidences,
        source_report_date=source_report_date,
        ref_report_date=ref_report_date,
    )
    if not matches:
        matches = await _fallback_bm25_match(source_evidences, ref_evidences)
    print(f"    -> 配对完成，成功构建了 {len(matches)} 对可供判断的论据。")
    return matches


async def find_best_matches(
    source_evidences: List[EvidenceTuple],
    ref_evidences: List[EvidenceTuple],
    source_report_date: Optional[datetime] = None,
    ref_report_date: Optional[datetime] = None,
) -> List[EvidenceMatch]:
    return await find_best_matches_by_similarity(
        source_evidences,
        ref_evidences,
        source_report_date=source_report_date,
        ref_report_date=ref_report_date,
    )


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

    from src.utils.call_with_retry import call_chatbot_with_retry
    from src.utils.instance import formatter, llm_instruct

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
            evidences_old = await extract_unique_evidences_from_pdf(
                old_path,
                long_term_dir / "demonstration",
                stock_entity_name=stock_entity_name,
            )
            evidences_new = await extract_unique_evidences_from_pdf(
                new_path,
                long_term_dir / "demonstration",
                stock_entity_name=stock_entity_name,
            )
            # ===== 并发提取两份报告的论据 =====
            # evidences_old, evidences_new = await asyncio.gather(
            #     extract_unique_evidences_from_pdf(old_path, long_term_dir / "demonstration"),
            #     extract_unique_evidences_from_pdf(new_path, long_term_dir / "demonstration")
            # )
            common_evidences_texts = await find_best_matches(
                evidences_old,
                evidences_new,
                source_report_date=_parse_report_date_from_path(old_path),
                ref_report_date=_parse_report_date_from_path(new_path),
            )

            common_evidences_with_locs = []
            assert len(common_evidences_texts) > 0, "!!! 未找到任何共通论据"
            if common_evidences_texts:
                print(f"\n  --- 开始为 {len(common_evidences_texts)} 条共通论据定位 ---", flush=True)
                outline_old_path = long_term_dir / "demonstration" / CONFIG.llm_name / f"{old_path.stem}_outline.json"
                outline_new_path = long_term_dir / "demonstration" / CONFIG.llm_name / f"{new_path.stem}_outline.json"
                if not outline_new_path.exists():
                    outline_new_path = long_term_dir / "demonstration" / f"{new_path.stem}_outline.json"
                if not outline_old_path.exists():
                    outline_old_path = long_term_dir / "demonstration" / f"{old_path.stem}_outline.json"

                outline_old_content = outline_old_path.read_text(encoding='utf-8')
                outline_new_content = outline_new_path.read_text(encoding='utf-8')

                # ===== 并发定位两份报告中论据的位置 =====
                locations_old, locations_new = await asyncio.gather(
                    find_locations_in_outline(outline_old_content, [e[0] for e in common_evidences_texts]),
                    find_locations_in_outline(outline_new_content, [e[1] for e in common_evidences_texts])
                )
                for old_text, new_text, old_value, new_value in common_evidences_texts:
                    common_evidences_with_locs.append({
                        "text": (old_text, new_text),
                        "value": (old_value, new_value),
                        "location_old": locations_old.get(old_text, "NONE"),
                        "location_new": locations_new.get(new_text, "NONE")
                    })

            # 过滤掉任何包含 "NONE" 的共通论据
            initial_count = len(common_evidences_with_locs)

            filtered_common_evidences = [
                item for item in common_evidences_with_locs
                if "NONE" not in item.get("location_old", "NONE") and \
                   "NONE" not in item.get("location_new", "NONE")
            ]

            filtered_count = len(filtered_common_evidences)
            if initial_count > filtered_count:
                print(f"  - 过滤完成：移除了 {initial_count - filtered_count} 条无法在两份报告中同时定位的共通论据。", flush=True)

            result_data = {
                "stock_code": stock_code, "old_report": old_path.name, "new_report": new_path.name,
                "old_evidence_count": len(evidences_old), "new_evidence_count": len(evidences_new),
                "common_evidence_count": filtered_count,
                # "common_evidences": filtered_common_evidences
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
    output_txt_path = PROJECT_ROOT / "output" / "comparison_results.txt"
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
