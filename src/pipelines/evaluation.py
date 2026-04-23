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
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from src.memory.working import Section
from src.evaluation.extract_evidence import get_all_evidences_from_section, extract_unique_evidences_from_pdf, \
    extract_unique_evidences, get_entity_name_by_code, _parse_report_date_from_path
from src.evaluation.eval_structure import num_of_segment, structure_score
from src.evaluation.eval_evidence import evidence_coverage_and_accuracy
from src.evaluation.eval_content import get_content_score, ContentScore
from src.utils.instance import llm_reasoning, llm_instruct, formatter, cfg
from src.pipelines.planning import process_pdf_to_outline
from src.utils import local_file





class BenchmarkItem(BaseModel):
    """benchmark.json中的单个条目"""
    stock_code: str
    date: str
    reference: str
    human_report: str


class StructureMetrics(BaseModel):
    """结构指标"""
    total_segments: int
    avg_segments_per_section: float
    comprehensiveness: float      # 结构完整性评分
    logicality: float             # 逻辑性评分


class EvidenceMetrics(BaseModel):
    """论据指标"""
    coverage_ratio: float  # 覆盖率
    accuracy_ratio: float  # 准确率


class BenchmarkResult(BaseModel):
    """单个benchmark评估的完整结果"""
    stock_code: str
    date: str
    human_report_name: str
    new_report_name: str
    structure: StructureMetrics
    evidence: EvidenceMetrics
    content: ContentScore
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


def _serialize_benchmark_result(result: BenchmarkResult) -> Dict:
    """将嵌套的Pydantic ContentScore转换为可JSON序列化的字典。"""
    if isinstance(result, dict): return result
    return result.model_dump()


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


async def evaluate_content(new_section: Section) -> ContentScore:
    """
    评估内容指标，对每个section应用score_content
    取各维度所有segment的平均值

    改进：
    - 异步并发收集所有segment的评分
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
                # if segment.content:
                segment_tasks.append((segment.content, topic))

        if section.subsections:
            for subsection in section.subsections:
                collect_segments(subsection, topic)

    collect_segments(new_section)

    if not segment_tasks:
        return ContentScore(
            insightfulness=0.0,
            readability=0.0,
            relevance=0.0,
            sufficiency=0.0,
        )

    # 初始化维度scores
    dimension_scores = {
        "insightfulness": [],
        "readability": [],
        "relevance": [],
        "sufficiency": [],
    }

    # 异步评估所有segment
    print(f"      - 共需评估 {len(segment_tasks)} 个segment...")

    # 使用 asyncio.gather 并发执行所有评分任务
    segment_scores = await asyncio.gather(
        *[get_content_score(llm_reasoning, formatter, content, topic) for content, topic in segment_tasks],
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
            if dim in scores and isinstance(scores[dim], (int, float)):
                dimension_scores[dim].append(float(scores[dim]))
        evaluated_count += 1

    # 计算平均值，使用列表推导式优化
    avg_scores = {
        dim: (sum(scores) / len(scores) if scores else 0.0)
        for dim, scores in dimension_scores.items()
    }

    print(f"      - 成功评估 {evaluated_count}/{len(segment_tasks)} 个segment")
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

    print(f"[1/4] 处理报告...")
    new_section = await process_pdf_to_outline(new_report_path, new_report_path.parent, llm_reasoning,
                                               llm_instruct, formatter, only_evidence=True)
    human_section = await process_pdf_to_outline(human_report_path, long_term_dir / "demonstration", llm_reasoning,
                                                 llm_instruct, formatter, only_evidence=True)

    print(f"[2/4] 抽取论据...")
    stock_entity_name = get_entity_name_by_code(stock_code)
    # 从文件名中提取研报写作日期，用于时间约束去重
    new_report_date = _parse_report_date_from_path(new_report_path)
    human_report_date = _parse_report_date_from_path(human_report_path)
    new_evidences = await extract_unique_evidences(new_section,
                                                   new_report_path.parent / 'evidences' / f"{new_report_path.stem}_evidences.json",
                                                   stock_entity_name=stock_entity_name,
                                                   report_date=new_report_date)
    human_evidences = await extract_unique_evidences(human_section,
                                                     long_term_dir / 'evidences' / f"{human_report_path.stem}_evidences.json",
                                                     stock_entity_name=stock_entity_name,
                                                     report_date=human_report_date)

    print(f"[3/4] 评估指标...")
    structure_metrics = await evaluate_structure(new_section)
    coverage_ratio, accuracy_ratio = await evidence_coverage_and_accuracy(new_evidences, human_evidences)
    content_metrics = await evaluate_content(new_section)

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
    print(result)
    return result


def _load_existing_results(output_path: Path) -> Tuple[List[dict], set]:
    """加载已有的评估结果，返回结果列表和已完成的 (stock_code, date) 集合"""
    if not output_path.exists():
        return [], set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        existing_results = data.get("results", [])
        completed = {(r["stock_code"], r["date"]) for r in existing_results}
        print(f"✓ 读取已有结果: {len(completed)} 条，将跳过已完成的任务")
        return existing_results, completed
    except Exception as e:
        print(f"! 读取已有结果失败 ({e})，从头开始")
        return [], set()


def _save_incremental(output_path: Path, results: List[dict], total: int,
                      successful: int, failed: int, none_count: int) -> None:
    """将当前结果增量保存到文件"""
    summary = {
        "total": total,
        "successful": successful,
        "failed": failed,
        "NONE": none_count,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


async def run_benchmark(
    benchmark_json_path: Path,
    new_reports_dir: Path,
    long_term_dir: Path,
    output_path: Path,
) -> Optional[Dict[str, any]]:
    """
    执行完整的benchmark评估流程，支持断点继续。

    每次评估完一个任务后立即将结果追加保存到 output_path；
    启动时自动检测已保存的结果并跳过已完成的条目。

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

    # 读取benchmark.json
    print(f"\n读取benchmark配置: {benchmark_json_path}")
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    if not isinstance(benchmark_data, list):
        print(f"✗ 错误: benchmark.json应为数组，实际为 {type(benchmark_data)}")
        return None

    benchmark_items = [BenchmarkItem(**item) for item in benchmark_data]

    print(f"✓ 读取了 {len(benchmark_items)} 个benchmark条目")

    # 加载已有结果，支持断点继续
    existing_result_dicts, completed_keys = _load_existing_results(output_path)
    results: List[dict] = list(existing_result_dicts)
    successful = len([r for r in results if True])  # 已成功数即已保存结果数
    successful = len(results)
    failed = 0
    NONE = 0
    failure_reasons = {}

    _write_summary(output_path, _build_summary(benchmark_items, results, successful, failed, NONE, failure_reasons))

    for idx, item in enumerate(benchmark_items, 1):
        key = (item.stock_code, item.date)

        # 跳过已完成的条目
        if key in completed_keys:
            print(f"[{idx}/{len(benchmark_items)}] 跳过 {item.stock_code} ({item.date})（已有结果）")
            continue

        print(f"[{idx}/{len(benchmark_items)}] 处理 {item.stock_code} ({item.date})")

        # 查找human_report报告
        ref_path = local_file.DEMO_DIR / item.human_report
        if not ref_path.exists():
            ref_path = long_term_dir / "demonstration" / item.human_report.replace('.pdf', '.md')
        if not ref_path.exists():
            print(f"  ✗ 参考报告不存在: {ref_path}")
            NONE += 1
            failure_reasons[f"human_report_NONE_{item.stock_code}_{item.date}"] = item.human_report
            _write_summary(output_path, _build_summary(benchmark_items, results, successful, failed, NONE, failure_reasons))
            continue

        # 查找新报告
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
            print(f"  ✗ 未找到新报告（搜索模式: {item.stock_code}_{item.date}*.md）")
            NONE += 1
            failure_reasons[f"new_report_NONE_{item.stock_code}_{item.date}"] = f"{item.stock_code}_{item.date}*.md"
            _write_summary(output_path, _build_summary(benchmark_items, results, successful, failed, NONE, failure_reasons))
            continue

        if len(matching_files) > 1:
            print(f"  ! 找到 {len(matching_files)} 个匹配报告，使用第一个")

        new_report_path = matching_files[0]

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

    # 最终汇总输出
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

async def main(method_name, new_reports_dir):
    """示例使用"""
    # 配置路径
    benchmark_json = PROJECT_ROOT / "benchmark.json"
    # new_reports = PROJECT_ROOT / "output" / "reports" / model_name
    long_term = PROJECT_ROOT / "data" / "memory" / "long_term"
    output = PROJECT_ROOT / "output" / f"{method_name}_benchmark_results.json"

    # 执行评估
    summary = await run_benchmark(
        benchmark_json_path=benchmark_json,
        new_reports_dir=new_reports_dir,
        long_term_dir=long_term,
        output_path=output,
    )

    # 打印汇总
    if summary:
        print_benchmark_summary(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 benchmark 测试，支持并行执行")
    parser.add_argument(
        "--new_reports_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default='kalm-qwen3.5-397b',
    )
    args = parser.parse_args()
    if args.new_reports_path:
        args.new_reports_path = Path(args.new_reports_path)
    else:
        args.new_reports_path = PROJECT_ROOT / "output" / "reports" / args.method_name,

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(args.method_name, args.new_reports_path))