import sys
import asyncio
from pathlib import Path

# 计算项目根目录：/financial-report-agents
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os

import pandas as pd
import pytest
from financial_report_agent.tools import material_tools as mt
from financial_report_agent.memory.short_term import ShortTermMemoryStore

short_term_memory_path = ROOT / "data" / "memory" / "short_term"
def _get_block_text(block) -> str:
    # 兼容 dict 和 TextBlock 两种情况
    if isinstance(block, dict):
        return block.get("text", str(block))
    if hasattr(block, "text"):
        return block.text
    return str(block)


def _assert_write_and_read(resp, short_term: ShortTermMemoryStore, max_rows: int = 5):
    assert "ref_id" in resp.metadata
    ref_id = resp.metadata["ref_id"]
    row_count = resp.metadata.get("row_count", None)

    read_resp = mt.read_table_material(
        ref_id=ref_id,
        short_term=short_term,
        max_rows=max_rows,
    )

    assert read_resp.metadata["found"] is True
    if row_count is not None:
        assert read_resp.metadata["row_count"] == row_count

    if read_resp.content:
        text = _get_block_text(read_resp.content[0])
        print(f"\n=== {ref_id} ===")
        print(text[:500])

@pytest.mark.asyncio
async def test_fetch_realtime_price_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_realtime_price_material(
        symbol="000001",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_history_price_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_history_price_material(
        symbol="000001",
        period="daily",
        start_date="20240101",
        end_date="20240131",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_stock_news_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_stock_news_material(
        symbol="603777",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_disclosure_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_disclosure_material(
        symbol="000001",
        market="沪深京",
        keyword="",
        category="",
        start_date="20230101",
        end_date="20231231",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_balance_sheet_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_balance_sheet_material(
        symbol="000063",
        indicator="按报告期",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_profit_table_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_profit_table_material(
        symbol="000063",
        indicator="按报告期",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_cashflow_table_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_cashflow_table_material(
        symbol="000063",
        indicator="按报告期",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_top10_float_shareholders_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    # 注意：date 需要是有披露数据的季度末日期，必要时你可以改成自己确定有数据的日期
    resp = await mt.fetch_top10_float_shareholders_material(
        symbol="sh688686",
        date="20240930",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_top10_shareholders_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_top10_shareholders_material(
        symbol="sh688686",
        date="20240930",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_main_shareholders_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_main_shareholders_material(
        stock="600004",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_shareholder_count_detail_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_shareholder_count_detail_material(
        symbol="000001",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_shareholder_change_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_shareholder_change_material(
        symbol="688981",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_business_description_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_business_description_material(
        symbol="000066",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)


@pytest.mark.asyncio
async def test_fetch_business_composition_material():
    short_term = ShortTermMemoryStore(short_term_memory_path)
    resp = await mt.fetch_business_composition_material(
        symbol="SH688041",
        short_term=short_term,
    )
    _assert_write_and_read(resp, short_term)