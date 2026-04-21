# -*- coding: utf-8 -*-
"""
Benchmark评估流程
- 读取benchmark.json配置
- 配对新研报与参考研报
- 计算structure、evidence、content三方面指标
- 缓存human_report的outline和evidence以加速后续评估
"""
import argparse
import sys
import os
import json
import asyncio
import re
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from src.memory.working import Section, Segment
from src.evaluation.extract_evidence import extract_unique_evidences
from src.evaluation.eval_structure import num_of_segment, structure_score
from src.evaluation.eval_evidence import evidence_coverage_and_accuracy
from src.evaluation.eval_content import get_content_score, ContentScore
from src.utils.instance import llm_reasoning, llm_instruct, formatter
from src.pipelines.planning import process_pdf_to_outline
from src.utils import local_file


@dataclass
class BenchmarkItem:
    """benchmark.json中的单个条目"""
    stock_code: str
    date: str
    reference: str
    human_report: str


@dataclass
class StructureMetrics:
    """结构指标"""
    total_segments: int
    avg_segments_per_section: float
    comprehensiveness: float      # 结构完整性评分
    logicality: float             # 逻辑性评分


@dataclass
class EvidenceMetrics:
    """论据指标"""
    coverage_ratio: float  # 覆盖率
    accuracy_ratio: float  # 准确率


@dataclass
class BenchmarkResult:
    """单个benchmark评估的完整结果"""
    stock_code: str
    date: str
    human_report_name: str
    new_report_name: str
    structure: StructureMetrics
    evidence: EvidenceMetrics
    content: ContentScore
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def _build_pair_cache_key(stock_code: str, date: str, human_report_name: str, new_report_name: str) -> str:
    return f"{stock_code}|{date}|{human_report_name}|{new_report_name}"


def _load_json_file(json_path: Path) -> Optional[Any]:
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


_CITE_REFERENCE_PATTERN = re.compile(r"\[\^cite_id[:=][^\]]+?\]")
_CHART_REFERENCE_PATTERN = re.compile(r"\[\^chart_id[:=][^\]]+?\]")
_CHART_IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\(chart:[a-zA-Z0-9_\-]+\)")
def _strip_references_for_scoring(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    cleaned = _CHART_IMAGE_PATTERN.sub("", text)
    cleaned = _CITE_REFERENCE_PATTERN.sub("", cleaned)
    cleaned = _CHART_REFERENCE_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+([，。；：！？,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _sanitize_segment_for_scoring(segment: Segment) -> None:
    segment.topic = _strip_references_for_scoring(segment.topic)
    segment.reference = _strip_references_for_scoring(segment.reference)
    segment.content = _strip_references_for_scoring(segment.content)


def _sanitize_section_for_scoring(section: Section) -> Section:
    sanitized = deepcopy(section)
    pending_sections = [sanitized]

    while pending_sections:
        current = pending_sections.pop()
        current.title = _strip_references_for_scoring(current.title)

        for segment in current.segments or []:
            _sanitize_segment_for_scoring(segment)

        for subsection in current.subsections or []:
            pending_sections.append(subsection)

    return sanitized


def _sanitize_evidence_list(evidences: List[str]) -> List[str]:
    cleaned_evidences: List[str] = []

    for evidence in evidences:
        cleaned = _strip_references_for_scoring(evidence)
        if cleaned and cleaned not in cleaned_evidences:
            cleaned_evidences.append(cleaned)

    return cleaned_evidences


def _content_score_to_dict(scores: Any) -> Optional[Dict[str, float]]:
    if hasattr(scores, "model_dump"):
        scores = scores.model_dump()
    elif hasattr(scores, "dict"):
        scores = scores.dict()

    if not isinstance(scores, dict):
        return None

    normalized_scores: Dict[str, float] = {}
    for dim in ("insightfulness", "readability", "relevance", "sufficiency"):
        value = scores.get(dim)
        if isinstance(value, (int, float)):
            normalized_scores[dim] = float(value)
    return normalized_scores if normalized_scores else None


def _average_content_scores(score_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    avg_scores = {
        "insightfulness": 0.0,
        "readability": 0.0,
        "relevance": 0.0,
        "sufficiency": 0.0,
    }
    if not score_dicts:
        return avg_scores

    for dim in avg_scores:
        values = [score[dim] for score in score_dicts if dim in score]
        avg_scores[dim] = sum(values) / len(values) if values else 0.0
    return avg_scores


def _collect_segment_tasks_for_content(section: Section) -> List[Tuple[str, str]]:
    segment_tasks: List[Tuple[str, str]] = []
    pending_sections: List[Tuple[Section, str]] = [(section, section.title or "全文")]

    while pending_sections:
        current_section, parent_topic = pending_sections.pop()
        current_topic = current_section.title or parent_topic or "全文"

        for segment in current_section.segments or []:
            segment_text = segment.content or segment.reference
            if not segment_text:
                continue
            segment_topic = segment.topic or current_topic
            segment_tasks.append((segment_text, segment_topic))

        for subsection in reversed(current_section.subsections or []):
            pending_sections.append((subsection, current_topic))

    return segment_tasks


def _deserialize_benchmark_result(data: Dict) -> BenchmarkResult:
    return BenchmarkResult(
        stock_code=data["stock_code"],
        date=data["date"],
        human_report_name=data["human_report_name"],
        new_report_name=data["new_report_name"],
        structure=StructureMetrics(**data["structure"]),
        evidence=EvidenceMetrics(**data["evidence"]),
        content=ContentScore(**data["content"]),
        timestamp=data.get("timestamp", datetime.now().isoformat()),
    )


async def _load_or_extract_evidences(section: Section, evidence_path: Path) -> List[str]:
    cached_evidences = _load_json_file(evidence_path)
    if cached_evidences is not None:
        if isinstance(cached_evidences, list):
            print(f"      - 复用evidence缓存: {evidence_path.name}")
            return cached_evidences
    return await extract_unique_evidences(section, evidence_path)


def _serialize_benchmark_result(result: BenchmarkResult) -> Dict:
    """将嵌套的Pydantic ContentScore转换为可JSON序列化的字典。"""
    data = asdict(result)
    if hasattr(result.content, "model_dump"):
        data["content"] = result.content.model_dump()
    else:
        data["content"] = result.content.dict()
    return data


def _build_summary(
    benchmark_items: List[BenchmarkItem],
    results: List[BenchmarkResult],
    successful: int,
    failed: int,
    none_count: int,
    failure_reasons: Dict[str, str],
) -> Dict:
    return {
        "total": len(benchmark_items),
        "successful": successful,
        "failed": failed,
        "NONE": none_count,
        "results": [_serialize_benchmark_result(r) for r in results],
        "failure_reasons": failure_reasons,
        "timestamp": datetime.now().isoformat(),
    }


def _write_summary(output_path: Path, summary: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _get_content_average(content: Dict) -> float:
    values = [
        content.get("insightfulness", 0.0),
        content.get("readability", 0.0),
        content.get("relevance", 0.0),
        content.get("sufficiency", 0.0),
    ]
    values = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else 0.0


def _build_new_report_patterns(stock_code: str, date: str) -> List[str]:
    """支持 YYYYMMDD 和 YYYY-MM-DD 两种新报告文件名。"""
    patterns: List[str] = []

    def add_pattern(date_str: str) -> None:
        pattern = f"{stock_code}_{date_str}*.md"
        if pattern not in patterns:
            patterns.append(pattern)

    add_pattern(date)

    compact_date = date.replace("-", "")
    if len(compact_date) == 8 and compact_date.isdigit():
        add_pattern(compact_date)
        add_pattern(f"{compact_date[:4]}-{compact_date[4:6]}-{compact_date[6:]}")

    return patterns


def _resolve_report_paths(
    item: BenchmarkItem,
    new_reports_dir: Path,
) -> Tuple[Optional[Path], Optional[Path], Optional[str], Optional[str]]:
    ref_path = local_file.DEMO_DIR / item.human_report
    if not ref_path.exists():
        return None, None, "human_report", item.human_report

    matching_files = list(
        new_reports_dir.glob(f"{item.stock_code}_{item.date}*.md")
    )

    if not matching_files:
        alt_patterns = _build_new_report_patterns(item.stock_code, item.date)[1:]
        seen_files = set()
        for pattern in alt_patterns:
            for path in new_reports_dir.glob(pattern):
                if path not in seen_files:
                    seen_files.add(path)
                    matching_files.append(path)

    if not matching_files:
        return ref_path, None, "new_report", f"{item.stock_code}_{item.date}*.md"

    if len(matching_files) > 1:
        print(f"  ! 找到 {len(matching_files)} 个匹配报告，使用第一个")

    return ref_path, matching_files[0], None, None


async def evaluate_structure(new_section: Section, human_section: Section) -> StructureMetrics:
    """评估结构指标（包括完整性和逻辑性）"""
    print(f"    - 正在评估structure指标...")
    total_segments, avg_segments_per_section = num_of_segment(new_section)

    # 基于human_report相对评估结构
    comprehensiveness, logicality = await structure_score(new_section, human_section)

    return StructureMetrics(
        total_segments=total_segments,
        avg_segments_per_section=avg_segments_per_section,
        comprehensiveness=comprehensiveness,
        logicality=logicality,
    )


async def evaluate_content(new_section: Section) -> ContentScore:
    """
    评估内容指标。
    - 对所有 segment 逐段评分
    - 各维度对全文取平均
    """
    print(f"    - 正在评估content指标...")
    segment_tasks = _collect_segment_tasks_for_content(new_section)
    if not segment_tasks:
        raise RuntimeError(
            "新报告缺少可用于content评估的segment内容，请检查评估输入或outline解析结果。"
        )

    print(f"      - 共需评估 {len(segment_tasks)} 个segment")
    segment_score_results = await asyncio.gather(
        *[
            get_content_score(llm_reasoning, formatter, content, topic)
            for content, topic in segment_tasks
        ],
        return_exceptions=True,
    )
    failed_count = 0
    score_dicts: List[Dict[str, float]] = []

    for scores in segment_score_results:
        if isinstance(scores, Exception):
            failed_count += 1
            continue
        normalized_scores = _content_score_to_dict(scores)
        if normalized_scores is None:
            failed_count += 1
            continue
        score_dicts.append(normalized_scores)

    if not score_dicts:
        raise RuntimeError("segment content评分失败，未获得任何有效的评分结果。")

    avg_scores = _average_content_scores(score_dicts)
    print(f"      - 成功评估 {len(score_dicts)}/{len(segment_tasks)} 个segment")
    if failed_count > 0:
        print(f"      - 评估失败 {failed_count} 个segment")

    return ContentScore(
        insightfulness=avg_scores["insightfulness"],
        readability=avg_scores["readability"],
        relevance=avg_scores["relevance"],
        sufficiency=avg_scores["sufficiency"],
    )


async def benchmark_single_pair(
    stock_code: str,
    date: str,
    new_report_path: Path,
    human_report_path: Path,
    long_term_dir: Path,
) -> Optional[BenchmarkResult]:
    """
    评估一对(new_report, human_report)

    Args:
        stock_code: 股票代码
        date: 报告日期
        new_report_path: 新研报PDF路径
        human_report_path: 参考研报PDF路径
        long_term_dir: 长期记忆目录

    Returns:
        BenchmarkResult 或 None (评估失败)
    """

    print(f"\n{'='*60}")
    print(f"正在评估: {stock_code} ({date})")
    print(f"{'='*60}")

    async def _do_evaluation():
        print(f"[1/4] 处理报告...")
        new_section = await process_pdf_to_outline(new_report_path, new_report_path.parent, llm_reasoning,
                                                   llm_instruct, formatter, only_evidence=True)
        human_section = await process_pdf_to_outline(human_report_path, long_term_dir / 'evidences', llm_reasoning,
                                                     llm_instruct, formatter, only_evidence=True)
        sanitized_new_section = _sanitize_section_for_scoring(new_section)
        sanitized_human_section = _sanitize_section_for_scoring(human_section)

        print(f"[2/4] 抽取论据...")
        new_evidence_path = new_report_path.parent / f"{new_report_path.stem}_evidences.json"
        human_evidence_path = long_term_dir / 'evidences' / f"{human_report_path.stem}_evidences.json"
        new_evidences = await _load_or_extract_evidences(new_section, new_evidence_path)
        human_evidences = await _load_or_extract_evidences(human_section, human_evidence_path)
        sanitized_new_evidences = _sanitize_evidence_list(new_evidences)
        sanitized_human_evidences = _sanitize_evidence_list(human_evidences)

        print(f"[3/4] 评估指标...")
        structure_metrics = await evaluate_structure(sanitized_new_section, sanitized_human_section)
        coverage_ratio, accuracy_ratio = await evidence_coverage_and_accuracy(
            sanitized_new_evidences,
            sanitized_human_evidences,
        )
        content_metrics = await evaluate_content(sanitized_new_section)

        # 组装结果
        print(f"[4/4] 汇总结果...")
        result = BenchmarkResult(
            stock_code=stock_code,
            date=date,
            human_report_name=human_report_path.name,
            new_report_name=new_report_path.name,
            structure=structure_metrics,
            evidence=EvidenceMetrics(
                coverage_ratio=coverage_ratio, accuracy_ratio=accuracy_ratio
            ),
            content=content_metrics,
        )

        print(f"✓ 评估完成")
        return result

    return await _do_evaluation()


async def run_benchmark(
    benchmark_json_path: Path,
    new_reports_dir: Path,
    long_term_dir: Path,
    output_path: Optional[Path] = None,
) -> Optional[Dict[str, any]]:
    """
    执行完整的benchmark评估流程

    Args:
        benchmark_json_path: benchmark.json路径
        new_reports_dir: 新研报所在目录
        long_term_dir: 长期记忆目录
        output_path: 输出结果JSON的路径 (默认与benchmark.json同目录)

    Returns:
        {
            "total": 总数,
            "successful": 成功数,
            "failed": 失败数,
            "results": [BenchmarkResult, ...],
            "timestamp": str
        }
    """

    # 验证输入
    if not benchmark_json_path.exists():
        print(f"✗ 错误: benchmark.json 不存在: {benchmark_json_path}")
        return None

    if not new_reports_dir.exists():
        print(f"✗ 错误: 新研报目录不存在: {new_reports_dir}")
        return None

    if not long_term_dir.exists():
        print(f"✗ 错误: 长期记忆目录不存在: {long_term_dir}")
        return None


    # 设置输出路径
    if output_path is None:
        output_path = benchmark_json_path.parent / "benchmark_results.json"

    # 读取benchmark.json
    print(f"\n读取benchmark配置: {benchmark_json_path}")
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    if not isinstance(benchmark_data, list):
        print(f"✗ 错误: benchmark.json应为数组，实际为 {type(benchmark_data)}")
        return None

    benchmark_items = [BenchmarkItem(**item) for item in benchmark_data]

    print(f"✓ 读取了 {len(benchmark_items)} 个benchmark条目\n")

    results = []
    successful = 0
    failed = 0
    NONE = 0
    failure_reasons = {}
    existing_result_cache: Dict[str, BenchmarkResult] = {}
    cached_summary = _load_json_file(output_path)
    if isinstance(cached_summary, dict) and isinstance(cached_summary.get("results"), list):
        for item in cached_summary["results"]:
            result = _deserialize_benchmark_result(item)
            cache_key = _build_pair_cache_key(
                result.stock_code,
                result.date,
                result.human_report_name,
                result.new_report_name,
            )
            existing_result_cache[cache_key] = result
    pending_pairs = []

    for idx, item in enumerate(benchmark_items, 1):
        ref_path, new_report_path, missing_type, missing_detail = _resolve_report_paths(item, new_reports_dir)

        if missing_type == "human_report":
            print(f"[{idx}/{len(benchmark_items)}] 处理 {item.stock_code} ({item.date})")
            print(f"  ✗ 参考报告不存在: {item.human_report}")
            NONE += 1
            failure_reasons[f"human_report_NONE_{item.stock_code}_{item.date}"] = item.human_report
            continue

        if missing_type == "new_report":
            print(f"[{idx}/{len(benchmark_items)}] 处理 {item.stock_code} ({item.date})")
            print(f"  ✗ 未找到新报告（搜索模式: {missing_detail}）")
            NONE += 1
            failure_reasons[f"new_report_NONE_{item.stock_code}_{item.date}"] = missing_detail
            continue

        cache_key = _build_pair_cache_key(
            item.stock_code,
            item.date,
            ref_path.name,
            new_report_path.name,
        )
        cached_result = existing_result_cache.get(cache_key)
        if cached_result is not None:
            print(f"[{idx}/{len(benchmark_items)}] 处理 {item.stock_code} ({item.date})")
            print(f"  ✓ 复用pair缓存: {new_report_path.name} vs {ref_path.name}")
            results.append(cached_result)
            successful += 1
            continue

        pending_pairs.append((idx, item, ref_path, new_report_path))

    _write_summary(output_path, _build_summary(benchmark_items, results, successful, failed, NONE, failure_reasons))

    for idx, item, ref_path, new_report_path in pending_pairs:
        print(f"[{idx}/{len(benchmark_items)}] 处理 {item.stock_code} ({item.date})")

        # 执行评估
        try:
            result = await benchmark_single_pair(
                stock_code=item.stock_code,
                date=item.date,
                new_report_path=new_report_path,
                human_report_path=ref_path,
                long_term_dir=long_term_dir,
            )

            if result:
                results.append(result)
                successful += 1
            else:
                failed += 1
                failure_reasons[f"evaluation_failed_{item.stock_code}_{item.date}"] = "评估过程中失败"
        except Exception as e:
            traceback.print_exc()
            failed += 1
            failure_reasons[f"evaluation_failed_{item.stock_code}_{item.date}"] = str(e)
        finally:
            _write_summary(output_path, _build_summary(benchmark_items, results, successful, failed, NONE, failure_reasons))


    # 保存结果
    print(f"\n{'='*60}")
    print(f"评估完成！")
    print(f"{'='*60}")
    print(f"总数:   {len(benchmark_items)}")
    print(f"成功:   {successful} ({successful/len(benchmark_items)*100:.1f}%)" if len(benchmark_items) > 0 else "成功:   0")
    print(f"失败:   {failed}")
    print(f"找不到: {NONE}")

    if failure_reasons:
        print(f"\n失败原因详情:")
        for reason, detail in failure_reasons.items():
            print(f"  - {reason}: {detail}")

    summary = _build_summary(benchmark_items, results, successful, failed, NONE, failure_reasons)
    _write_summary(output_path, summary)
    print(f"\n✓ 结果已保存到: {output_path}")

    return summary


# ==============================================================================
# 辅助函数：结果分析和可视化
# ==============================================================================


def print_benchmark_summary(results_json_path: Path) -> None:
    """从结果JSON打印汇总统计"""
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"Benchmark结果汇总")
    print(f"{'='*60}")
    print(f"总数: {data['total']}")
    print(f"成功: {data['successful']}")
    print(f"失败: {data['failed']}")
    print(f"成功率: {data['successful']/data['total']*100:.1f}%\n" if data["total"] > 0 else "成功率: 0.0%\n")

    if not data["results"]:
        print("无评估结果")
        return

    # 计算各指标的平均值
    results = data["results"]

    structure_total_segments = [r["structure"]["total_segments"] for r in results]
    structure_avg_segments = [r["structure"]["avg_segments_per_section"] for r in results]
    structure_comprehensiveness = [r["structure"]["comprehensiveness"] for r in results]
    structure_logicality = [r["structure"]["logicality"] for r in results]
    evidence_coverage = [r["evidence"]["coverage_ratio"] for r in results]
    evidence_accuracy = [r["evidence"]["accuracy_ratio"] for r in results]
    content_scores = [_get_content_average(r["content"]) for r in results]
    content_insightfulness = [r["content"]["insightfulness"] for r in results]
    content_readability = [r["content"]["readability"] for r in results]
    content_relevance = [r["content"]["relevance"] for r in results]
    content_sufficiency = [r["content"]["sufficiency"] for r in results]

    def safe_avg(values):
        return sum(values) / len(values) if values else 0

    print("Structure指标:")
    print(f"  - 平均总segment数: {safe_avg(structure_total_segments):.1f}")
    print(f"  - 平均每section的segment数: {safe_avg(structure_avg_segments):.2f}")
    print(f"  - 平均完整性: {safe_avg(structure_comprehensiveness):.2f}/10")
    print(f"  - 平均逻辑性: {safe_avg(structure_logicality):.2f}/10")

    print("\nEvidence指标:")
    print(f"  - 平均覆盖率: {safe_avg(evidence_coverage):.2%}")
    print(f"  - 平均准确率: {safe_avg(evidence_accuracy):.2%}")

    print("\nContent指标:")
    print(f"  - 平均总分: {safe_avg(content_scores):.2f}/10")
    print(f"  - 平均洞察力: {safe_avg(content_insightfulness):.2f}/10")
    print(f"  - 平均可读性: {safe_avg(content_readability):.2f}/10")
    print(f"  - 平均相关性: {safe_avg(content_relevance):.2f}/10")
    print(f"  - 平均充分性: {safe_avg(content_sufficiency):.2f}/10")

    print("\n详细结果:")
    for r in results:
        print(f"\n  {r['stock_code']} ({r['date']}):")
        print(f"    Structure: {r['structure']['total_segments']} segments, "
              f"{r['structure']['avg_segments_per_section']:.2f} avg/section, "
              f"comprehensiveness={r['structure']['comprehensiveness']:.2f}, "
              f"logicality={r['structure']['logicality']:.2f}")
        print(f"    Evidence: {r['evidence']['coverage_ratio']:.2%} coverage, "
              f"{r['evidence']['accuracy_ratio']:.2%} accuracy")
        print(f"    Content: {_get_content_average(r['content']):.2f}/10 overall, "
              f"insightfulness={r['content']['insightfulness']:.2f}, "
              f"readability={r['content']['readability']:.2f}, "
              f"relevance={r['content']['relevance']:.2f}, "
              f"sufficiency={r['content']['sufficiency']:.2f}")


# ==============================================================================
# 主程序入口
# ==============================================================================


async def main(model_name):
    """示例使用"""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # 配置路径
    benchmark_json = PROJECT_ROOT / "benchmark.json"
    new_reports = PROJECT_ROOT / "data" / "output" / "reports" / model_name
    long_term = PROJECT_ROOT / "data" / "memory" / "long_term"
    output = PROJECT_ROOT / "data" / "output" / f"{model_name}_benchmark_results.json"

    # 执行评估
    summary = await run_benchmark(
        benchmark_json_path=benchmark_json,
        new_reports_dir=new_reports,
        long_term_dir=long_term,
        output_path=output,
    )

    # 打印汇总
    if summary:
        print_benchmark_summary(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 benchmark 测试，支持并行执行")
    parser.add_argument(
        "--model_name",
        type=str,
        default='qwen3-32b',
    )
    args = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(args.model_name))
