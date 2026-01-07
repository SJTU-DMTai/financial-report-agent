# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from src.memory.short_term import ShortTermMemoryStore
from src.utils.file_converter import md_to_pdf

import sys
CURRENT_FILE = Path(__file__).resolve()        # tests/test_pdf.py
PROJECT_ROOT = CURRENT_FILE.parent.parent

def run_workflow():

    short_term_dir = PROJECT_ROOT / "data" / "memory" / "short_term" / "history_short_term" / "20251202_111024_663686"
    
    short_term = ShortTermMemoryStore(
        base_dir=short_term_dir,
        do_post_init=False,
    )
    
    text = md_to_pdf(short_term=short_term)
    print(f"{text}")
    
if __name__ == "__main__":
    run_workflow() #python -m tests.test_pdf