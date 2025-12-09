# -*- coding: utf-8 -*-
import sys
import asyncio
from pathlib import Path

# 计算项目根目录：/financial-report-agents
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os

import pandas as pd
import pytest
from src.tools import material_tools as mt
from src.memory.short_term import ShortTermMemoryStore

short_term_memory_path = ROOT / "data" / "memory" / "short_term"

test_tools = mt.MaterialTools(short_term=ShortTermMemoryStore(short_term_memory_path))

def _get_block_text(block) -> str:
    # 兼容 dict 和 TextBlock 两种情况
    if isinstance(block, dict):
        return block.get("text", str(block))
    if hasattr(block, "text"):
        return block.text
    return str(block)


def _assert_write_and_read(resp, max_rows: int = 5):
    assert "ref_id" in resp.metadata
    ref_id = resp.metadata["ref_id"]
    row_count = resp.metadata.get("row_count", None)

    read_resp = test_tools.read_table_material(
        ref_id=ref_id,
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
    
    resp = await test_tools.fetch_realtime_price_material(
        symbol="000001",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_history_price_material():
    
    resp = await test_tools.fetch_history_price_material(
        symbol="000001",
        period="daily",
        start_date="20240101",
        end_date="20240131",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_stock_news_material():
    
    resp = await test_tools.fetch_stock_news_material(
        symbol="603777",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_disclosure_material():
    
    resp = await test_tools.fetch_disclosure_material(
        symbol="000001",
        market="沪深京",
        keyword="",
        category="",
        start_date="20230101",
        end_date="20231231",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_balance_sheet_material():
    
    resp = await test_tools.fetch_balance_sheet_material(
        symbol="000063",
        indicator="按报告期",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_profit_table_material():
    
    resp = await test_tools.fetch_profit_table_material(
        symbol="000063",
        indicator="按报告期",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_cashflow_table_material():
    
    resp = await test_tools.fetch_cashflow_table_material(
        symbol="000063",
        indicator="按报告期",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_top10_float_shareholders_material():
    
    # 注意：date 需要是有披露数据的季度末日期，必要时你可以改成自己确定有数据的日期
    resp = await test_tools.fetch_top10_float_shareholders_material(
        symbol="688686",
        date="20240930",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_top10_shareholders_material():
    
    resp = await test_tools.fetch_top10_shareholders_material(
        symbol="688686",
        date="20240930",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_main_shareholders_material():
    
    resp = await test_tools.fetch_main_shareholders_material(
        stock="600004",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_shareholder_count_detail_material():
    
    resp = await test_tools.fetch_shareholder_count_detail_material(
        symbol="000001",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_shareholder_change_material():
    
    resp = await test_tools.fetch_shareholder_change_material(
        symbol="688981",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_business_description_material():
    
    resp = await test_tools.fetch_business_description_material(
        symbol="000066",
    )
    _assert_write_and_read(resp)


@pytest.mark.asyncio
async def test_fetch_business_composition_material():
    
    resp = await test_tools.fetch_business_composition_material(
        symbol="688041",
    )
    _assert_write_and_read(resp)