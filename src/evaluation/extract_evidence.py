import re
from pathlib import Path

import json
import difflib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None
try:
    from semhash import SemHash
except ImportError:
    SemHash = None

from src.memory.working import Evidence, Section, evidence_pairs
from src.utils.retrieve_in_memory import _bm25_scores, _tokenize_for_bm25


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
