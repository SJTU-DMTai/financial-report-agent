# -*- coding: utf-8 -*-
import collections
import os
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEMO_DIR = Path(os.getenv("DEMO_DIR", PROJECT_ROOT.parent / "pdf_filtered/"))
pdf_files = list(DEMO_DIR.glob("*.pdf")) + list(DEMO_DIR.glob("*.PDF"))
STOCK_REPORT_PATHS = collections.defaultdict(list)
for file in pdf_files:
    STOCK_REPORT_PATHS[file.name.split("_")[0]].append(file)
for k, v in STOCK_REPORT_PATHS.items():
    STOCK_REPORT_PATHS[k] = sorted(v, key=lambda x: x.name.split("_")[1])

