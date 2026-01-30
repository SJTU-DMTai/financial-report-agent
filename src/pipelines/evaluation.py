# -*- coding: utf-8 -*-
"""
Benchmark评估流程
- 读取benchmark.json配置
- 配对新研报与参考研报
- 计算structure、evidence、content三方面指标
- 缓存reference的outline和evidence以加速后续评估
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from src.memory.working import Section
from src.evaluation.extract_evidence import get_all_evidences_from_section
from src.evaluation.eval_structure import num_of_segment, structure_score
from src.evaluation.eval_evidence import evidence_coverage_and_accuracy
from src.evaluation.eval_content import get_content_score
from src.utils.instance import llm_reasoning, llm_instruct, formatter
from src.pipelines.planning import process_pdf_to_outline

CONCURRENCY_LIMIT = int(os.getenv("N_THREAD", 32))



@dataclass
class BenchmarkItem:
    """benchmark.json中的单个条目"""
    stock_code: str
    date: str
    reference: str


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
class ContentMetrics:
    """内容指标（各维度）"""
    insight: float
    readability: float
    relevance: float
    sufficiency: float
    average_score: float


@dataclass
class BenchmarkResult:
    """单个benchmark评估的完整结果"""
    stock_code: str
    date: str
    reference_name: str
    new_report_name: str
    structure: StructureMetrics
    evidence: EvidenceMetrics
    content: ContentMetrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


async def get_reference_evidences(
    reference_path: Path, save_dir: Path
) -> List[str]:
    """
    从save_dir读取reference的evidence.json，如果不存在则提取并保存

    Evidence文件命名规则: {reference_pdf_filename_without_ext}_evidences.json
    例如: 000333_2025-10-30_均衡发展，稳中求进_evidences.json

    Args:
        reference_path: 参考PDF路径
        save_dir: 长期记忆目录

    Returns:
        论据列表
    """
    # 生成evidence文件名（基于reference pdf文件名）
    pdf_stem = reference_path.stem  # 去掉.pdf后缀
    evidence_filename = f"{pdf_stem}_evidences.json"
    evidence_path = save_dir / evidence_filename

    # 尝试直接读取
    if evidence_path.exists():
        print(f"    - 检测到已有的evidences，加载: {evidence_filename}")
        evidences = json.loads(evidence_path.read_text(encoding="utf-8"))
        return evidences

    # 如果不存在，需要先获取reference的Section，然后提取evidence
    print(f"    - Evidence不存在，正在提取: {evidence_filename}")

    # 处理PDF生成Section
    ref_section = await process_pdf_to_outline(reference_path, save_dir,
                                               llm_reasoning, llm_instruct, formatter)

    # 提取evidences
    evidences = get_all_evidences_from_section(ref_section)

    # 保存到long_term_dir
    evidence_path.write_text(
        json.dumps(evidences, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"    -> Evidence已保存到: {evidence_path}")
    return evidences


async def evaluate_structure(new_section: Section) -> StructureMetrics:
    """评估结构指标（包括完整性和逻辑性）"""
    print(f"    - 正在评估structure指标...")
    total_segments, avg_segments_per_section = num_of_segment(new_section)

    # 调用structure_score获取补充指标
    comprehensiveness, logicality = await structure_score(new_section)

    return StructureMetrics(
        total_segments=total_segments,
        avg_segments_per_section=avg_segments_per_section,
        comprehensiveness=comprehensiveness,
        logicality=logicality,
    )


async def evaluate_evidence(
    new_section: Section, ref_section: Section, ref_evidences: Optional[List[str]] = None
) -> EvidenceMetrics:
    """评估论据指标（使用提供的evidences或从ref_section提取）"""
    print(f"    - 正在评估evidence指标...")

    coverage_ratio, accuracy_ratio = await evidence_coverage_and_accuracy(
        new_section, ref_section, ref_evidences
    )
    return EvidenceMetrics(
        coverage_ratio=coverage_ratio, accuracy_ratio=accuracy_ratio
    )


async def evaluate_content(new_section: Section) -> ContentMetrics:
    """
    评估内容指标，对每个section应用score_content
    取各维度所有segment的平均值

    改进：
    - 异步并发收集所有segment的评分，最大并发数由CONCURRENCY_LIMIT控制
    - 更好的错误处理和日志记录
    - 更高效的平均值计算
    """
    print(f"    - 正在评估content指标...")

    # 收集所有segment的tasks
    segment_tasks = []

    def collect_segments(section: Section, parent_topic: str = ""):
        """递归收集所有segment及其topic"""
        topic = section.title or parent_topic or "unknown"

        if section.segments:
            for segment in section.segments:
                if segment.content:
                    segment_tasks.append((segment.content, topic))

        if section.subsections:
            for subsection in section.subsections:
                collect_segments(subsection, topic)

    collect_segments(new_section)

    # 初始化维度scores
    dimension_scores = {
        "comprehensiveness": [],
        "insight": [],
        "readability": [],
        "relevance": [],
        "sufficiency": [],
    }

    # 创建信号量控制并发数
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def score_with_limit(content: str, topic: str):
        """使用信号量包装 get_content_score 调用"""
        async with semaphore:
            return await get_content_score(llm_reasoning, formatter, content, topic)

    # 异步评估所有segment
    print(f"      - 共需评估 {len(segment_tasks)} 个segment...")

    # 使用 asyncio.gather 并发执行所有评分任务
    segment_scores = await asyncio.gather(
        *[score_with_limit(content, topic) for content, topic in segment_tasks],
        return_exceptions=True
    )

    # 处理结果
    evaluated_count = 0
    failed_count = 0

    for scores in segment_scores:
        if isinstance(scores, Exception):
            failed_count += 1
            continue

        for dim in dimension_scores.keys():
            if dim in scores:
                dimension_scores[dim].append(scores[dim])
        evaluated_count += 1

    # 计算平均值，使用列表推导式优化
    avg_scores = {
        dim: (sum(scores) / len(scores) if scores else 0.0)
        for dim, scores in dimension_scores.items()
    }

    overall_avg = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0

    print(f"      - 成功评估 {evaluated_count}/{len(segment_tasks)} 个segment")
    if failed_count > 0:
        print(f"      - 评估失败 {failed_count} 个segment")

    return ContentMetrics(
        insight=avg_scores.get("insight", 0.0),
        readability=avg_scores.get("readability", 0.0),
        relevance=avg_scores.get("relevance", 0.0),
        sufficiency=avg_scores.get("sufficiency", 0.0),
        average_score=overall_avg,
    )


async def benchmark_single_pair(
    stock_code: str,
    date: str,
    new_report_path: Path,
    reference_path: Path,
    long_term_dir: Path,
) -> Optional[BenchmarkResult]:
    """
    评估一对(new_report, reference)

    Args:
        stock_code: 股票代码
        date: 报告日期
        new_report_path: 新研报PDF路径
        reference_path: 参考研报PDF路径
        long_term_dir: 长期记忆目录

    Returns:
        BenchmarkResult 或 None (评估失败)
    """

    print(f"\n{'='*60}")
    print(f"正在评估: {stock_code} ({date})")
    print(f"{'='*60}")

    async def _do_evaluation():
        save_dir = long_term_dir / "demonstration"
        print(f"[1/4] 处理参考报告...")
        ref_section = await process_pdf_to_outline(reference_path, save_dir, llm_reasoning, llm_instruct, formatter)

        # 获取reference的evidences（从long_term_dir读取或新提取）
        ref_evidences = await get_reference_evidences(reference_path, save_dir)

        print(f"[2/4] 处理新报告...")
        print(f"  -> {new_report_path.name}")
        new_section = await process_pdf_to_outline(new_report_path, save_dir, llm_reasoning, llm_instruct, formatter)

        print(f"[3/4] 评估指标...")
        structure_metrics = await evaluate_structure(new_section)
        evidence_metrics = await evaluate_evidence(new_section, ref_section, ref_evidences)
        content_metrics = await evaluate_content(new_section)

        # 组装结果
        print(f"[4/4] 汇总结果...")
        result = BenchmarkResult(
            stock_code=stock_code,
            date=date,
            reference_name=reference_path.name,
            new_report_name=new_report_path.name,
            structure=structure_metrics,
            evidence=evidence_metrics,
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
    not_found = 0
    failure_reasons = {}

    for idx, item in enumerate(benchmark_items, 1):
        print(f"[{idx}/{len(benchmark_items)}] 处理 {item.stock_code} ({item.date})")

        # 查找reference报告
        ref_path = new_reports_dir.parent / item.reference
        if not ref_path.exists():
            print(f"  ✗ 参考报告不存在: {item.reference}")
            not_found += 1
            failure_reasons[f"reference_not_found_{item.stock_code}"] = item.reference
            continue

        # 查找新报告
        matching_files = list(
            new_reports_dir.glob(f"{item.stock_code}_{item.date}_*.pdf")
        )

        if not matching_files:
            print(f"  ✗ 未找到新报告（搜索模式: {item.stock_code}_{item.date}_*.pdf）")
            not_found += 1
            failure_reasons[f"new_report_not_found_{item.stock_code}"] = f"{item.stock_code}_{item.date}_*.pdf"
            continue

        if len(matching_files) > 1:
            print(f"  ! 找到 {len(matching_files)} 个匹配报告，使用第一个")

        new_report_path = matching_files[0]

        # 执行评估
        result = await benchmark_single_pair(
            stock_code=item.stock_code,
            date=item.date,
            new_report_path=new_report_path,
            reference_path=ref_path,
            long_term_dir=long_term_dir,
        )

        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1
            failure_reasons[f"evaluation_failed_{item.stock_code}"] = "评估过程中失败"


    # 保存结果
    print(f"\n{'='*60}")
    print(f"评估完成！")
    print(f"{'='*60}")
    print(f"总数:   {len(benchmark_items)}")
    print(f"成功:   {successful} ({successful/len(benchmark_items)*100:.1f}%)" if len(benchmark_items) > 0 else "成功:   0")
    print(f"失败:   {failed}")
    print(f"找不到: {not_found}")

    if failure_reasons:
        print(f"\n失败原因详情:")
        for reason, detail in failure_reasons.items():
            print(f"  - {reason}: {detail}")

    summary = {
        "total": len(benchmark_items),
        "successful": successful,
        "failed": failed,
        "not_found": not_found,
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
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
    print(f"成功率: {data['successful']/data['total']*100:.1f}%\n")

    if not data["results"]:
        print("无评估结果")
        return

    # 计算各指标的平均值
    results = data["results"]

    structure_total_segments = [r["structure"]["total_segments"] for r in results]
    structure_avg_segments = [r["structure"]["avg_segments_per_section"] for r in results]
    evidence_coverage = [r["evidence"]["coverage_ratio"] for r in results]
    evidence_accuracy = [r["evidence"]["accuracy_ratio"] for r in results]
    content_scores = [r["content"]["average_score"] for r in results]

    def safe_avg(values):
        return sum(values) / len(values) if values else 0

    print("Structure指标:")
    print(f"  - 平均总segment数: {safe_avg(structure_total_segments):.1f}")
    print(f"  - 平均每section的segment数: {safe_avg(structure_avg_segments):.2f}")

    print("\nEvidence指标:")
    print(f"  - 平均覆盖率: {safe_avg(evidence_coverage):.2%}")
    print(f"  - 平均准确率: {safe_avg(evidence_accuracy):.2%}")

    print("\nContent指标:")
    print(f"  - 平均总分: {safe_avg(content_scores):.2f}/10")

    print("\n详细结果:")
    for r in results:
        print(f"\n  {r['stock_code']} ({r['date']}):")
        print(f"    Structure: {r['structure']['total_segments']} segments, "
              f"{r['structure']['avg_segments_per_section']:.2f} avg/section")
        print(f"    Evidence: {r['evidence']['coverage_ratio']:.2%} coverage, "
              f"{r['evidence']['accuracy_ratio']:.2%} accuracy")
        print(f"    Content: {r['content']['average_score']:.2f}/10 overall, "
              f"insight={r['content']['insight']:.2f}, "
              f"readability={r['content']['readability']:.2f}")


# ==============================================================================
# 主程序入口
# ==============================================================================


async def main():
    """示例使用"""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # 配置路径
    benchmark_json = PROJECT_ROOT / "data" / "benchmark.json"
    new_reports = PROJECT_ROOT / "data" / "output" / "reports"
    long_term = PROJECT_ROOT / "data" / "memory" / "long_term"
    output = PROJECT_ROOT / "data" / "benchmark_results.json"

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
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())


# ==============================================================================
# 使用说明
# ==============================================================================
#
# 【目录结构要求】
#
# financial-report-agent/
#   ├── data/
#   │   ├── benchmark.json              # 配置文件（需要创建）
#   │   ├── output/
#   │   │   └── reports/                # 新研报目录
#   │   │       ├── 000333_2025-10-30_xxx.pdf
#   │   │       ├── 002594_2025-06-24_xxx.pdf
#   │   │       └── ...
#   │   └── memory/
#   │       └── long_term/              # 长期记忆目录
#   └── {reference_reports_parent}/     # 参考报告所在目录（与new_reports同级）
#       ├── 000333_2025-10-30_均衡发展，稳中求进.pdf
#       ├── 002594_2024-06-24_首次覆盖报告：新能源车领军企业，剑指海外市场.pdf
#       └── ...
#
# 【benchmark.json 示例】
#
# [
#   {
#     "stock_code": "000333",
#     "date": "2025-10-30",
#     "reference": "000333_2025-10-30_均衡发展，稳中求进.pdf"
#   },
#   {
#     "stock_code": "002594",
#     "date": "2025-06-24",
#     "reference": "002594_2024-06-24_首次覆盖报告：新能源车领军企业，剑指海外市场.pdf"
#   }
# ]
#
# 【Python调用示例】
#
# import asyncio
# from pathlib import Path
# from src.evaluation.benchmark import run_benchmark, print_benchmark_summary
#
# async def main():
#     PROJECT_ROOT = Path(".")
#
#     summary = await run_benchmark(
#         benchmark_json_path=PROJECT_ROOT / "data" / "benchmark.json",
#         new_reports_dir=PROJECT_ROOT / "data" / "output" / "reports",
#         long_term_dir=PROJECT_ROOT / "data" / "memory" / "long_term",
#         output_path=PROJECT_ROOT / "data" / "benchmark_results.json",
#         cache_dir=PROJECT_ROOT / "data" / "benchmark_cache",
#     )
#
#     if summary:
#         print_benchmark_summary(PROJECT_ROOT / "data" / "benchmark_results.json")
#
# asyncio.run(main())
#
# 【输出结果说明】
#
# 运行完成后会生成两个输出：
#
# 1. benchmark_results.json (详细结果JSON)
#    {
#      "total": 2,
#      "successful": 2,
#      "failed": 0,
#      "results": [
#        {
#          "stock_code": "000333",
#          "date": "2025-10-30",
#          "reference_name": "000333_2025-10-30_均衡发展，稳中求进.pdf",
#          "new_report_name": "000333_2025-10-30_生成研报.pdf",
#          "structure": {
#            "total_segments": 45,
#            "avg_segments_per_section": 5.2
#          },
#          "evidence": {
#            "coverage_ratio": 0.85,      # 覆盖率
#            "accuracy_ratio": 0.92       # 准确率
#          },
#          "content": {
#            "comprehensiveness": 7.8,     # 信息覆盖度
#            "insight": 8.1,               # 分析深度
#            "readability": 8.5,           # 可读性
#            "relevance": 8.2,             # 相关性
#            "sufficiency": 7.9,           # 充分性
#            "average_score": 8.1          # 平均分
#          },
#          "timestamp": "2026-01-29T10:30:45.123456"
#        },
#        ...
#      ],
#      "timestamp": "2026-01-29T10:35:22.456789"
#    }
#
# 2. benchmark_cache/ (缓存目录，自动创建)
#    benchmark_cache/
#      ├── 000333_2025-10-30/
#      │   ├── outline.md         # 参考报告的outline markdown
#      │   ├── evidences.json     # 参考报告的unique evidences
#      │   └── metadata.json      # 缓存元数据
#      └── 002594_2025-06-24/
#          ├── outline.md
#          ├── evidences.json
#          └── metadata.json
#
# 【评估指标说明】
#
# Structure (结构) - 评估报告的组织结构
#   - total_segments: 新报告中所有segment的总数
#   - avg_segments_per_section: 每个section平均包含的segment数
#
# Evidence (论据) - 评估论据的覆盖和准确性
#   - coverage_ratio: 新报告覆盖的参考报告论据比例（0-1）
#   - accuracy_ratio: 新报告论据与参考报告一致的比例（0-1）
#
# Content (内容) - 评估生成内容的各维度质量（1-10分制）
#   - comprehensiveness: 信息覆盖的广度和深度
#   - insight: 分析见解的深度和价值
#   - readability: 结构清晰度、语言流畅度等
#   - relevance: 内容与主题的相关性
#   - sufficiency: 论据的充分性和支撑力
#   - average_score: 以上5个维度的平均分
#
# 【缓存优化】
#
# 系统会自动缓存参考报告的：
#   - outline markdown: 用于快速查阅报告结构
#   - unique evidences: 用于加速论据评估
#   - metadata: 缓存的创建时间、PDF大小等
#
# 如果同一个参考报告被多次评估，第二次及以后的评估会大幅加速。
# 删除cache目录可强制重新计算所有缓存。
#
# ==============================================================================
