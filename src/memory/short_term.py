from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd
import json
from io import StringIO
import shutil
from typing import List, Tuple, Dict, Any, Union, Optional
from enum import Enum
import re
from datetime import datetime
class MaterialType(str, Enum):
    TABLE = "table"  # 对应 csv, excel
    TEXT = "text"    # 对应 md, txt
    JSON = "json"    # 对应 json 数据

@dataclass
class MaterialMeta:
    ref_id: str
    m_type: MaterialType
    filename: str
    description: str = ""
    source: str = ""  # 来源标记

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
    
    @property
    def registry_path(self) -> Path:
        return self.material_dir / "registry.json"

    def __post_init__(self):

        self._registry: Dict[str, MaterialMeta] = {}

        # 转移之前的short_term memory，避免对当前任务造成干扰
        history_root = self.base_dir / "history_short_term"
        history_root.mkdir(parents=True, exist_ok=True)

        batch_dir = history_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        batch_dir.mkdir(parents=True, exist_ok=True)

        targets = [
            self.outline_path,
            self.material_dir,
            self.manuscript_dir,
        ]

        for target in targets:
            if target.exists():
                dest = batch_dir / target.name
                if target.is_dir():
                    # 不要再删 dest 了，因为 batch_dir 是新建的，本来就不会存在
                    shutil.copytree(target, dest)
                    shutil.rmtree(target)
                else:
                    shutil.copy2(target, dest)
                    target.unlink()

        # 重建空目录结构，避免后续调用失败
        self.ensure_dirs()
        self._load_registry()


    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.material_dir.mkdir(parents=True, exist_ok=True)
        self.manuscript_dir.mkdir(parents=True, exist_ok=True)
        self.demonstration_dir.mkdir(parents=True, exist_ok=True)

    # ---- Registry 读写 ----

    def _load_registry(self):
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text(encoding="utf-8"))
                for key, val in data.items():
                    val['m_type'] = MaterialType(val['m_type']) # 恢复 Enum
                    self._registry[key] = MaterialMeta(**val)
            except:
                self._registry = {}

    def _save_registry(self):
        data = {k: {**asdict(v), 'm_type': v.m_type.value} for k, v in self._registry.items()}
        self.registry_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_material_meta(self, ref_id: str) -> Optional[MaterialMeta]:
        return self._registry.get(ref_id)

    # ---- Outline ----
    def load_outline(self) -> str:
        if not self.outline_path.exists():
            return ""
        return self.outline_path.read_text(encoding="utf-8")

    def save_outline(self, content: str) -> None:
        self.ensure_dirs()
        self.outline_path.write_text(content, encoding="utf-8")



    def _parse_outline_sections(self, outline: str) -> List[Tuple[str, str, str]]:
        """把 outline.md 划分为若干 section。
        返回: List[(section_id, title, body_markdown)]
        简化策略：
        - 以一级标题 `#` 作为章节分割点
        - section_id 形如 `sec_01_行业分析`，保证字典序 == 章节顺序
        """
        lines = outline.splitlines()
        sections: List[Tuple[str, str, str]] = []

        current_title = None
        current_body_lines: List[str] = []
        index = 0  # 用于编号

        def flush():
            nonlocal current_title, current_body_lines, index
            if current_title is None:
                return
            title = current_title.strip("# ").strip()
            # 简单 slug 化做 section_id
            slug = re.sub(r"\s+", "_", title)
            slug = re.sub(r"[^\w\-一-龥]", "", slug)  # 保留中文和常见字符
            index += 1
            prefix = f"{index:02d}"
            section_id = f"sec_{prefix}_{slug}"
            body = "\n".join(current_body_lines).strip()
            sections.append((section_id, title, body))
            current_title = None
            current_body_lines = []

        for line in lines:
            if line.startswith("# "):  # 一级标题
                flush()
                current_title = line
            else:
                if current_title is None:
                    # 出现在第一个 # 之前的内容可以直接忽略或归入引言
                    continue
                current_body_lines.append(line)

        flush()
        return sections


    # ---- Manuscript ----

    def draft_manuscript_from_outline(self):
        """根据现有的 outline.md 生成按章节拆分的多个 markdown 草稿骨架。
        根据大纲内容创建对应章节的初始 markdown 草稿，并返回生成的章节列表。
        """
        outline = self.load_outline()
        sections = self._parse_outline_sections(outline)

        for section_id, title, body_md in sections:
            body_markdown = (
                f"# {title}\n\n"
                "（请根据大纲要点在此撰写正文，可调用 Searcher 工具补充材料，调用generate chart工具绘图。）\n\n"
                f"{body_md}\n\n"
                )

            self.save_manuscript_section(section_id, body_markdown)
        return sections



    def save_manuscript_section(self, section_id: str, md: str) -> None:
        self.ensure_dirs()
        path = self.manuscript_dir / f"{section_id}.md"
        path.write_text(md, encoding="utf-8")

    def load_manuscript_section(self, section_id: str) -> str:
        path = self.manuscript_dir / f"{section_id}.md"
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

    # def save_material(self, ref_id: str, content: str, ext: str = "md") -> None:
    #     """
    #     content：可以是 markdown / csv / json 文本。
    #     ext：决定文件后缀，支持 "md", "txt", "csv", "json"
    #     """
    #     self.ensure_dirs()
    #     path = self.material_dir / f"{ref_id}.{ext}"
    #     path.write_text(content, encoding="utf-8")

    # def load_material(self, ref_id: str, ext: str = "md"):
    #     """
    #     如果 ext='csv' → 返回 pandas DataFrame
    #     如果 ext='json' → 返回 dict
    #     如果 ext='md'或'txt' → 返回 str

    #     """
    #     path = self.material_dir / f"{ref_id}.{ext}"
    #     if not path.exists():
    #         return None

    #     text = path.read_text(encoding="utf-8")

    #     # --- 根据扩展名返回适当的数据类型 ---
    #     if ext == "csv":
    #         return pd.read_csv(StringIO(text))

    #     if ext == "json":
    #         return json.loads(text)

    #     # 默认是 markdown
    #     return text


    def save_material(self, ref_id: str, content: Union[str, pd.DataFrame, dict, list], description: str = "", source: str = "", forced_ext: str = None) -> None:
        self.ensure_dirs()
        
        # 简化判断逻辑
        if isinstance(content, pd.DataFrame):
            ext, m_type = "csv", MaterialType.TABLE
            if content.index.name is None:
                content.index.name = "index"
            content.to_csv(self.material_dir / f"{ref_id}.csv", index=True)
        elif isinstance(content, (dict, list)):
            ext, m_type = "json", MaterialType.JSON
            (self.material_dir / f"{ref_id}.json").write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        else: # str
            ext = forced_ext or "txt"
            m_type = MaterialType.TEXT
            (self.material_dir / f"{ref_id}.{ext}").write_text(str(content), encoding="utf-8")

        self._registry[ref_id] = MaterialMeta(ref_id, m_type, f"{ref_id}.{ext}", description, source)
        self._save_registry()

    def load_material(self, ref_id: str) -> Union[pd.DataFrame, dict, str, None]:
        meta = self._registry.get(ref_id)
        if not meta: return None
        
        path = self.material_dir / meta.filename
        if not path.exists(): return None

        if meta.m_type == MaterialType.TABLE:
            return pd.read_csv(path, dtype=str)
        elif meta.m_type == MaterialType.JSON:
            return json.loads(path.read_text(encoding="utf-8"))
        else:
            return path.read_text(encoding="utf-8")