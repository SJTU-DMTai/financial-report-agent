# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "output" / "reports"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_import_stubs() -> None:
    if "pytest" not in sys.modules:
        sys.modules["pytest"] = types.ModuleType("pytest")
    if "src.utils.instance" in sys.modules:
        return

    import config

    stub = types.ModuleType("src.utils.instance")
    stub.cfg = config.Config()
    sys.modules["src.utils.instance"] = stub


_install_import_stubs()

from src.utils.file_converter import _strip_content_heading_number_prefixes


def _resolve_path(path_str: str | None) -> Path:
    if not path_str:
        return DEFAULT_REPORTS_DIR
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def clean_markdown_file(path: Path, dry_run: bool) -> bool:
    original_text = path.read_text(encoding="utf-8")
    cleaned_text = _strip_content_heading_number_prefixes(original_text)
    if original_text.endswith("\n") and cleaned_text and not cleaned_text.endswith("\n"):
        cleaned_text += "\n"
    if cleaned_text == original_text:
        return False
    if not dry_run:
        path.write_text(cleaned_text, encoding="utf-8")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="清洗已有报告 Markdown 中的标题编号。"
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="报告目录，默认 output/reports。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印会修改的文件，不写回。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports_dir = _resolve_path(args.reports_dir)
    if not reports_dir.exists():
        raise FileNotFoundError(f"报告目录不存在: {reports_dir}")

    changed_files = []
    for md_path in sorted(reports_dir.rglob("*.md")):
        if clean_markdown_file(md_path, args.dry_run):
            changed_files.append(md_path)
            print(f"{'would clean' if args.dry_run else 'cleaned'}: {md_path}")

    action = "would clean" if args.dry_run else "cleaned"
    print(f"{action} {len(changed_files)} markdown files under {reports_dir}")


if __name__ == "__main__":
    main()
