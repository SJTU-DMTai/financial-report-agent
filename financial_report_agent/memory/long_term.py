from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass
class OutlineExperienceStore:
    """用简单的文件系统实现 Outline 的长期经验库。
    """

    base_dir: Path

    def ensure_dir(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_all(self) -> list[Path]:
        self.ensure_dir()
        return sorted(self.base_dir.glob("*.md"))

    def save_outline(self, task_id: str, outline_content: str, meta: dict[str, Any]) -> Path:
        self.ensure_dir()
        path = self.base_dir / f"{task_id}.md"
        path.write_text(outline_content, encoding="utf-8")
        meta_path = self.base_dir / f"{task_id}.meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_outline(self, task_id: str) -> str:
        path = self.base_dir / f"{task_id}.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")


@dataclass
class ToolUseExperienceStore:
    """记录工具调用经验 <tool_name, list of exp>。"""

    base_path: Path

    def ensure_dir(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)

    def append_experience(self, tool_name: str, exp: dict[str, Any]) -> None:
        self.ensure_dir()
        path = self.base_path / f"{tool_name}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(exp, ensure_ascii=False) + "\n")

    def load_experiences(self, tool_name: str) -> list[dict[str, Any]]:
        path = self.base_path / f"{tool_name}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
