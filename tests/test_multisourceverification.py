# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from pathlib import Path

from src.memory.short_term import ShortTermMemoryStore
from src.memory.long_term import LongTermMemoryStore
from src.utils.multi_source_verification import (
    multi_source_verification,
    VerifyInput,
    VerifyTask,
)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent



REPORT_SNIPPET = """
2025年Q2，贵州茅台实现营业总收入910.94亿元，同比增长9.16%；净利润454.03亿元，同比增长8.89%。在白酒行业存量竞争背景下，公司保持接近10%的稳健增长，盈利能力维持高位：

- 毛利率：91.46%，同比基本持平
- 净利率：49.84%，同比提升0.3个百分点
- 产品结构：茅台酒755.9亿元（占比82.97%），系列酒137.6亿元，同比增长14.4%
""".strip()


def build_stores():
    short_term_dir = PROJECT_ROOT / "data" / "memory" / "short_term"
    long_term_dir = PROJECT_ROOT / "data" / "memory" / "long_term"

    short_term = ShortTermMemoryStore(
        base_dir=short_term_dir,
        do_post_init=False,
    )
    long_term = LongTermMemoryStore(base_dir=long_term_dir)
    return short_term, long_term


def pretty_print_results(title: str, results):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] claim: {r.claim}")
        print(f"    verdict: {r.verdict}  confidence: {r.confidence:.2f}")
        if r.slots:
            print(f"    slots: {r.slots}")
        if r.evidence:
            print(f"    evidence_ref_ids({len(r.evidence)}): {r.evidence}")
        if r.conflicts:
            print(f"    conflict_ref_ids({len(r.conflicts)}): {r.conflicts}")
        if r.rationale:
            print("    rationale:")
            for line in r.rationale.splitlines():
                print(f"      - {line}")


async def main():
    short_term, long_term = build_stores()

    # 1) FACTUAL：整段文本做多源验证（内部会抽取 claims）
    factual_input = VerifyInput(text=REPORT_SNIPPET)
    factual_results = await multi_source_verification(
        input=factual_input,
        task=VerifyTask.FACTUAL,
        short_term=short_term,
        long_term=long_term,
    )
    pretty_print_results("FACTUAL results (multi-source verify claims from snippet)", factual_results)

    # 2) CORROBORATE：排除既有来源 search_engine_1767926646001（以及同源材料）
    corroborate_text = "2025年Q2贵州茅台营业总收入910.94亿元，同比增长9.16%；净利润454.03亿元，同比增长8.89%。"
    corroborate_input = VerifyInput(
        text=corroborate_text,
        ref_id="search_engine_1767926646001",  # 你文中引用的来源
    )
    corroborate_results = await multi_source_verification(
        input=corroborate_input,
        task=VerifyTask.CORROBORATE,
        short_term=short_term,
        long_term=long_term,
    )
    pretty_print_results("CORROBORATE results (exclude base ref_id and same-source materials)", corroborate_results)

    BASE_REF_ID = "search_engine_1767929707002"

    # 1) 明显可被该材料支持的陈述（来自材料正文）
    supported_text = "1月1日，i茅台正式上架飞天53%vol 500ml贵州茅台酒，售价1499元。"
    inp1 = VerifyInput(text=supported_text, ref_id=BASE_REF_ID)
    res1 = await multi_source_verification(inp1, VerifyTask.GROUNDING, short_term, long_term)
    pretty_print_results("GROUNDING (SUPPORTED)",res1)

    # 2) 该材料不足以推出/可能缺失的陈述（用于观察 missing/unknown/refute）
    not_supported_text = "i茅台注册用户超8000万，月活超1700万。"
    inp2 = VerifyInput(text=not_supported_text, ref_id=BASE_REF_ID)
    res2 = await multi_source_verification(inp2, VerifyTask.GROUNDING, short_term, long_term)
    pretty_print_results("GROUNDING (NOT SUPPORTED)",res2)

if __name__ == "__main__":
    asyncio.run(main())
