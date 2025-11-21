from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import json
from io import StringIO
import shutil

@dataclass
class ShortTermMemoryStore:
    """负责当前任务的短期文件式记忆。
    """

    base_dir: Path

    @property
    def outline_path(self) -> Path:
        return self.base_dir / "outline.md"

    @property
    def material_dir(self) -> Path:
        return self.base_dir / "material"

    @property
    def manuscript_dir(self) -> Path:
        return self.base_dir / "manuscript"

    @property
    def demonstration_dir(self) -> Path:
        return self.base_dir / "demonstration"
    
    @property
    def demonstration_path(self) -> Path:
        return self.base_dir / "demonstration" / "demonstration.md"
    

    def __post_init__(self):
        # 转移之前的short_term memory，避免对当前任务造成干扰
        history_dir = self.base_dir / "history_short_term"
        history_dir.mkdir(parents=True, exist_ok=True)

        targets = [
            self.outline_path,
            self.material_dir,
            self.manuscript_dir,
        ]

        for target in targets:
            if target.exists():
                dest = history_dir / target.name

                # 目录移动
                if target.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(target, dest)
                    shutil.rmtree(target)

                # 文件移动
                else:
                    shutil.copy2(target, dest)
                    target.unlink()

        # 重建空目录结构，避免后续调用失败
        self.material_dir.mkdir(parents=True, exist_ok=True)
        self.manuscript_dir.mkdir(parents=True, exist_ok=True)


    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.material_dir.mkdir(parents=True, exist_ok=True)
        self.manuscript_dir.mkdir(parents=True, exist_ok=True)
        self.demonstration_dir.mkdir(parents=True, exist_ok=True)


    # ---- Outline ----
    def load_outline(self) -> str:
        if not self.outline_path.exists():
            return ""
        return self.outline_path.read_text(encoding="utf-8")

    def save_outline(self, content: str) -> None:
        self.ensure_dirs()
        self.outline_path.write_text(content, encoding="utf-8")

    # ---- Manuscript ----
    def save_manuscript_section(self, section_id: str, html: str) -> None:
        self.ensure_dirs()
        path = self.manuscript_dir / f"{section_id}.html"
        path.write_text(html, encoding="utf-8")

    def load_manuscript_section(self, section_id: str) -> str:
        path = self.manuscript_dir / f"{section_id}.html"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")
    
    # ---- demonstration ----
    def load_demonstration(self) -> str:
        if not self.demonstration_path.exists():
            return ""
        return self.demonstration_path.read_text(encoding="utf-8")

    def save_demonstration(self, content: str) -> None:
        self.ensure_dirs()
        self.demonstration_path.write_text(content, encoding="utf-8")
    
    # -----------------------------------------
    # Material 存储
    # -----------------------------------------

    def save_material(self, ref_id: str, content: str, ext: str = "md") -> None:
        """
        content：可以是 markdown / csv / json 文本。
        ext：决定文件后缀，支持 "md", "csv", "json"
        """
        self.ensure_dirs()
        path = self.material_dir / f"{ref_id}.{ext}"
        path.write_text(content, encoding="utf-8")

    def load_material(self, ref_id: str, ext: str = "md"):
        """
        如果 ext='csv' → 返回 pandas DataFrame
        如果 ext='json' → 返回 dict
        如果 ext='md' → 返回 str

        """
        path = self.material_dir / f"{ref_id}.{ext}"
        if not path.exists():
            return None

        text = path.read_text(encoding="utf-8")

        # --- 根据扩展名返回适当的数据类型 ---
        if ext == "csv":
            return pd.read_csv(StringIO(text))

        if ext == "json":
            return json.loads(text)

        # 默认是 markdown
        return text
