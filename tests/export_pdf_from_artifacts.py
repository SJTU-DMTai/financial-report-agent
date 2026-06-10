# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import re
import sys
import types
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DEFAULT_LLM_NAME = "deepseek-v4-flash"
DEFAULT_RUN_IDS = ("603556_20260607", "002594_20260607")
INVALID_IMAGE_PATTERN = re.compile(
    r"<p>\s*<img\b[^>]*\bsrc=(?:\"(?:\.\.\.|)\"|'(?:\.\.\.|)')[^>]*>\s*</p>\s*",
    re.IGNORECASE,
)


def _safe_print(message: str) -> None:
    os.write(1, f"{message}\n".encode("utf-8", errors="ignore"))


def _install_instance_stub(llm_name: str) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import config

    cfg = config.Config(llm_name=llm_name)
    stub = sys.modules.get("src.utils.instance")
    if stub is None:
        stub = types.ModuleType("src.utils.instance")
        sys.modules["src.utils.instance"] = stub
    stub.cfg = cfg


def _set_short_term_cfg(llm_name: str) -> None:
    import config
    import src.memory.short_term as short_term_module

    short_term_module.cfg = config.Config(llm_name=llm_name)


def _resolve_path(path_text: str | None, base_dir: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _report_root(args: argparse.Namespace) -> Path:
    report_root = _resolve_path(args.report_root, PROJECT_ROOT)
    if report_root is not None:
        return report_root
    return PROJECT_ROOT / "output" / "reports" / args.llm_name


def _short_term_root(args: argparse.Namespace) -> Path:
    short_term_root = _resolve_path(args.short_term_root, PROJECT_ROOT)
    if short_term_root is not None:
        return short_term_root
    return PROJECT_ROOT / "data" / "memory" / "short_term"


def _default_pdf_path(report_root: Path, run_id: str, suffix: str) -> Path:
    return report_root / f"{run_id}{suffix}.pdf"


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _as_file_uri(path: Path | None) -> str:
    if path is None:
        return ""
    return path.resolve().as_uri()


def _existing_paths(report_root: Path, short_term_root: Path, run_id: str) -> dict[str, Path]:
    return {
        "artifact_dir": report_root / run_id,
        "md": report_root / f"{run_id}.md",
        "json": report_root / f"{run_id}.json",
        "short_term_dir": short_term_root / run_id,
    }


def _allowed_dirs(paths: dict[str, Path]) -> list[str]:
    candidates = [
        paths["artifact_dir"],
        paths["short_term_dir"] / "manuscript",
        paths["short_term_dir"] / "material",
    ]
    return [str(path.resolve()) for path in candidates if path.exists()]


def _pdfkit_configuration():
    import config
    from pdfkit.configuration import Configuration

    wkhtmltopdf_path = config.Config().get_wkhtmltopdf_path()
    if wkhtmltopdf_path:
        return Configuration(wkhtmltopdf=wkhtmltopdf_path)
    return Configuration()


def _pdf_options(paths: dict[str, Path]) -> dict:
    header_path = _latest_file(paths["artifact_dir"], "_header_*.html")
    footer_path = _latest_file(paths["artifact_dir"], "_footer_*.html")
    options = {
        "encoding": "UTF-8",
        "disable-smart-shrinking": None,
        "enable-local-file-access": None,
        "no-stop-slow-scripts": None,
        "load-error-handling": "ignore",
        "load-media-error-handling": "ignore",
        "allow": _allowed_dirs(paths),
        "margin-top": "18mm",
        "margin-bottom": "16mm",
        "margin-left": "14mm",
        "margin-right": "14mm",
        "page-size": "A4",
    }
    if header_path is not None:
        options["header-html"] = _as_file_uri(header_path)
        options["header-spacing"] = "6"
    if footer_path is not None:
        options["footer-html"] = _as_file_uri(footer_path)
        options["footer-spacing"] = "6"
    return options


def _sanitized_html_path(html_path: Path) -> Path:
    html_text = html_path.read_text(encoding="utf-8")
    sanitized_text = INVALID_IMAGE_PATTERN.sub("", html_text)
    if sanitized_text == html_text:
        return html_path
    sanitized_path = html_path.with_name(f"_sanitized_{html_path.name}")
    sanitized_path.write_text(sanitized_text, encoding="utf-8")
    return sanitized_path


def export_from_html(paths: dict[str, Path], pdf_path: Path) -> str:
    import pdfkit

    html_path = _latest_file(paths["artifact_dir"], "source_*.html")
    if html_path is None:
        raise FileNotFoundError(f"未找到 source_*.html: {paths['artifact_dir']}")
    source_html_path = _sanitized_html_path(html_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdfkit.from_file(
        str(source_html_path),
        str(pdf_path),
        options=_pdf_options(paths),
        configuration=_pdfkit_configuration(),
    )
    return f"html: {source_html_path}"


def _markdown_from_json(json_path: Path) -> str:
    from src.memory.working import Section
    from src.utils.file_converter import section_to_markdown

    manuscript = Section.from_json(json_path.read_text(encoding="utf-8"))
    return section_to_markdown(manuscript)


def _markdown_text(paths: dict[str, Path], save_markdown: bool) -> str:
    md_path = paths["md"]
    if md_path.exists():
        return md_path.read_text(encoding="utf-8")
    json_path = paths["json"]
    if not json_path.exists():
        raise FileNotFoundError(f"未找到报告 md/json: {md_path} / {json_path}")
    markdown_text = _markdown_from_json(json_path)
    if save_markdown:
        md_path.write_text(markdown_text, encoding="utf-8")
    return markdown_text


def export_from_markdown(paths: dict[str, Path], pdf_path: Path, save_markdown: bool) -> str:
    from src.memory.short_term import ShortTermMemoryStore
    from src.utils.file_converter import md_to_pdf

    short_term_dir = paths["short_term_dir"]
    if not short_term_dir.exists():
        raise FileNotFoundError(f"short_term 目录不存在: {short_term_dir}")
    markdown_text = _markdown_text(paths, save_markdown)
    short_term = ShortTermMemoryStore(base_dir=short_term_dir, do_post_init=False)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    md_to_pdf(markdown_text, short_term=short_term, pdf_path=pdf_path)
    return f"markdown/json: {paths['md'] if paths['md'].exists() else paths['json']}"


def export_one(args: argparse.Namespace, report_root: Path, short_term_root: Path, run_id: str) -> bool:
    paths = _existing_paths(report_root, short_term_root, run_id)
    pdf_path = _default_pdf_path(report_root, run_id, args.output_suffix)
    if args.dry_run:
        _safe_print(f"[DRY] {run_id}")
        _safe_print(f"      artifact_dir={paths['artifact_dir']}")
        _safe_print(f"      md={paths['md']}")
        _safe_print(f"      json={paths['json']}")
        _safe_print(f"      short_term_dir={paths['short_term_dir']}")
        _safe_print(f"      pdf={pdf_path}")
        return True

    try:
        source = ""
        if args.mode in ("auto", "html") and paths["artifact_dir"].exists():
            source = export_from_html(paths, pdf_path)
        elif args.mode in ("auto", "md", "json"):
            source = export_from_markdown(paths, pdf_path, args.save_markdown)
        else:
            raise FileNotFoundError(f"未找到可用导出中间文件: {run_id}")
        _safe_print(f"[OK] {run_id} -> {pdf_path}")
        _safe_print(f"     source={source}")
        return True
    except Exception as exc:
        can_fallback = args.mode == "auto" and (paths["md"].exists() or paths["json"].exists())
        if can_fallback:
            _safe_print(f"[WARN] {run_id} html 导出失败，尝试 markdown/json: {exc}")
            try:
                source = export_from_markdown(paths, pdf_path, args.save_markdown)
                _safe_print(f"[OK] {run_id} -> {pdf_path}")
                _safe_print(f"     source={source}")
                return True
            except Exception as fallback_exc:
                _safe_print(f"[FAIL] {run_id} markdown/json fallback 也失败: {fallback_exc}")
                return False
        _safe_print(f"[FAIL] {run_id}: {exc}")
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从已有报告中间文件补导出 PDF，不重新运行调研/写作 workflow。"
    )
    parser.add_argument(
        "run_ids",
        nargs="*",
        default=list(DEFAULT_RUN_IDS),
        help="报告 run id，例如 603556_20260607。",
    )
    parser.add_argument("--llm-name", default=DEFAULT_LLM_NAME)
    parser.add_argument("--report-root", help="报告输出目录，默认 output/reports/<llm-name>")
    parser.add_argument("--short-term-root", help="short_term 根目录，默认 data/memory/short_term")
    parser.add_argument("--mode", choices=("auto", "html", "md", "json"), default="auto")
    parser.add_argument(
        "--output-suffix",
        default="",
        help="输出 PDF 文件名后缀，例如 _recovered；默认覆盖/生成 <run_id>.pdf。",
    )
    parser.add_argument("--save-markdown", action="store_true", help="从 json 重建 markdown 时同时保存 .md")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _install_instance_stub(args.llm_name)
    _set_short_term_cfg(args.llm_name)
    report_root = _report_root(args)
    short_term_root = _short_term_root(args)
    results = [export_one(args, report_root, short_term_root, run_id) for run_id in args.run_ids]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
