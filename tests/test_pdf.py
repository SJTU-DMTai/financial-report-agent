# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "reports"
DEFAULT_SHORT_TERM_ROOT = PROJECT_ROOT / "data" / "memory" / "short_term"


def _install_instance_stub() -> None:

    if "src.utils.instance" in sys.modules:
        return

    import config

    stub = types.ModuleType("src.utils.instance")
    stub.cfg = config.Config()
    sys.modules["src.utils.instance"] = stub


_install_instance_stub()

from src.memory.short_term import ShortTermMemoryStore
from src.memory.working import Section
from src.utils.file_converter import md_to_pdf, section_to_markdown


def _safe_print(message: str) -> None:
    os.write(1, f"{message}\n".encode("utf-8", errors="ignore"))


def _resolve_path(path_str: str | None, base_dir: Path) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _find_report_json(report_json_arg: str | None) -> Path:
    if report_json_arg:
        report_json_path = _resolve_path(report_json_arg, PROJECT_ROOT)
        if report_json_path is None or not report_json_path.exists():
            raise FileNotFoundError(f"报告 JSON 不存在: {report_json_arg}")
        return report_json_path

    candidates = sorted(DEFAULT_OUTPUT_DIR.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "未找到报告 JSON。请通过 --report-json 传入，例如 "
            "`data/output/reports/<llm_name>/603556_20260330.json`。"
        )
    return candidates[0]


def _infer_short_term_dir(short_term_arg: str | None, report_json_path: Path) -> Path:
    if short_term_arg:
        short_term_dir = _resolve_path(short_term_arg, PROJECT_ROOT)
        if short_term_dir is None or not short_term_dir.exists():
            raise FileNotFoundError(f"short_term 目录不存在: {short_term_arg}")
        return short_term_dir

    report_stem = report_json_path.stem
    candidate = DEFAULT_SHORT_TERM_ROOT / report_stem
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        "未能根据报告 JSON 自动推断 short_term 目录，请通过 --short-term-dir 显式传入。"
    )


def _infer_pdf_path(pdf_arg: str | None, report_json_path: Path) -> Path:
    if pdf_arg:
        pdf_path = _resolve_path(pdf_arg, PROJECT_ROOT)
        assert pdf_path is not None
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        return pdf_path

    return report_json_path.with_suffix(".pdf")


def _infer_md_path(md_arg: str | None, report_json_path: Path) -> Path | None:
    if md_arg is None:
        return None
    md_path = _resolve_path(md_arg, PROJECT_ROOT)
    assert md_path is not None
    md_path.parent.mkdir(parents=True, exist_ok=True)
    return md_path


def render_report_pdf(
    short_term_dir: Path,
    report_json_path: Path,
    pdf_path: Path,
    save_markdown_path: Path | None = None,
) -> str:
    short_term = ShortTermMemoryStore(
        base_dir=short_term_dir,
        do_post_init=False,
    )
    manuscript = Section.json(report_json_path.read_text(encoding="utf-8"))
    markdown_text = section_to_markdown(manuscript)

    if save_markdown_path is not None:
        save_markdown_path.write_text(markdown_text, encoding="utf-8")

    return md_to_pdf(markdown_text, short_term=short_term, pdf_path=pdf_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据现有 short_term 和报告 JSON 重新渲染 PDF。"
    )
    parser.add_argument(
        "--short-term-dir",
        help="short_term 目录路径，例如 data/memory/short_term/603556_20260330",
    )
    parser.add_argument(
        "--report-json",
        help="报告 JSON 路径，例如 data/output/reports/<llm_name>/603556_20260330.json",
    )
    parser.add_argument(
        "--pdf-path",
        help="输出 PDF 路径；默认与 report-json 同目录同名 .pdf",
    )
    parser.add_argument(
        "--save-markdown",
        nargs="?",
        const="__AUTO__",
        help="可选：同时导出 markdown。未带路径时默认保存为与 report-json 同名 .md",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report_json_path = _find_report_json(args.report_json)
    short_term_dir = _infer_short_term_dir(args.short_term_dir, report_json_path)
    pdf_path = _infer_pdf_path(args.pdf_path, report_json_path)

    md_path: Path | None = None
    if args.save_markdown is not None:
        md_path = report_json_path.with_suffix(".md") if args.save_markdown == "__AUTO__" else _infer_md_path(
            args.save_markdown, report_json_path
        )

    result = render_report_pdf(
        short_term_dir=short_term_dir,
        report_json_path=report_json_path,
        pdf_path=pdf_path,
        save_markdown_path=md_path,
    )

    _safe_print(f"short_term_dir: {short_term_dir}")
    _safe_print(f"report_json: {report_json_path}")
    if md_path is not None:
        _safe_print(f"markdown: {md_path}")
    _safe_print(result)


if __name__ == "__main__":
    main()
