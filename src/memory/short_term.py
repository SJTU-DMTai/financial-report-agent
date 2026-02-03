# -*- coding: utf-8 -*-
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
    entity: Dict[str, str] = field(default_factory=lambda: {"name": "", "code": ""})
    time: Dict[str, str] = field(default_factory=dict)  # {}, {"point":...}, {"start":...,"end":...}
    description: str = ""
    source: str = ""  # 来源标记

@dataclass
class ShortTermMemoryStore:
    """负责当前任务的短期文件式记忆。
    """

    base_dir: Path
    do_post_init: bool = True

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
        if not self.do_post_init:
            self._load_registry()
            return
        # # 转移之前的short_term memory，避免对当前任务造成干扰
        # history_root = self.base_dir / "history_short_term"
        # history_root.mkdir(parents=True, exist_ok=True)
        #
        # batch_dir = history_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # batch_dir.mkdir(parents=True, exist_ok=True)
        #
        # targets = [
        #     self.outline_path,
        #     self.material_dir,
        #     self.manuscript_dir,
        # ]
        #
        # for target in targets:
        #     if target.exists():
        #         dest = batch_dir / target.name
        #         if target.is_dir():
        #             shutil.copytree(target, dest)
        #             shutil.rmtree(target)
        #         else:
        #             shutil.copy2(target, dest)
        #             target.unlink()

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



    def _parse_outline_sections(self, outline: str) -> Tuple[Optional[str], List[Tuple[str, str, str]]]:
        """把 outline.md 划分为若干 section。
        返回: (report_title, sections)

        report_title: 文档级研报标题
        sections: List[(section_id, title, body_markdown)]
        规则：
        - 第 1 行是文档级研报标题
        - 之后以一级标题 `#` 作为章节分割点
        - section_id 形如 `sec_01_行业分析`，保证字典序 == 章节顺序
        """
        lines = outline.splitlines()
        sections: List[Tuple[str, str, str]] = []

        if not lines:
            return None, sections
    
        report_title = lines[0]
        content_lines = lines[1:]
        current_title = None
        current_body_lines: List[str] = []
        index = 0

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

        for line in content_lines:
            if line.startswith("# "):  # 一级标题
                flush()
                current_title = line
            else:
                if current_title is None:
                    continue
                current_body_lines.append(line)

        flush()
        return report_title, sections


    # ---- Manuscript ----

    def draft_manuscript_from_outline(self):
        """根据现有的 outline.md 生成按章节拆分的多个 markdown 草稿骨架。
        根据大纲内容创建对应章节的初始 markdown 草稿，并返回生成的章节列表。
        """
        outline = self.load_outline()
        report_title, sections = self._parse_outline_sections(outline)

        for idx, (section_id, title, body_md) in enumerate(sections):
            if idx == 0 and report_title:
                body_markdown = (
                f"# {report_title}\n\n"
                f"# {title}\n\n"
                "（请根据大纲要点在此撰写正文，可调用 Searcher 工具补充材料，调用calculate工具进行计算，调用generate chart工具绘图。）\n\n"
                f"{body_md}\n\n"
                )
            else :
                body_markdown = (
                f"# {title}\n\n"
                "（请根据大纲要点在此撰写正文，可调用 Searcher 工具补充材料，调用calculate工具进行计算，调用generate chart工具绘图。）\n\n"
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

    def save_material(self, ref_id: str, 
        content: Union[str, pd.DataFrame, dict, list],
        description: str = "",
        source: str = "",
        entity: Optional[Dict[str, str]] = None,
        time: Optional[Dict[str, str]] = None,
        forced_ext: str = ""
    ) -> None:
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

        entity = entity if entity is not None else {"name": "", "code": ""}
        time = time if time is not None else {}
        
        _DESC_SEP_RE = re.compile(r"[，,。.;；:：/\\|()（）\[\]{}<>《》“”\"'!?！？\t\r\n]+")
        _DESC_WS_RE = re.compile(r"\s+")

        description = (description or "").strip()
        description = _DESC_SEP_RE.sub(" ", description)
        description = _DESC_WS_RE.sub(" ", description)
        description = description.lower()


        self._registry[ref_id] = MaterialMeta(
            ref_id=ref_id,
            m_type=m_type,
            filename=f"{ref_id}.{ext}",
            entity=entity,
            time=time,
            description=description,
            source=source,
        )
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
        
        
    def load_material_preview(
        self,
        ref_id:str,
        max_chars: int = 300,
        table_rows: int = 3
    ) :
        """
        从 Material 中提取预览字符串。
        """
        def _truncate(s: str) -> str:
            
            s = (s or "").replace("\r\n", "\n")
            if len(s) <= max_chars:
                return s
            return s[:max_chars] + "…[内容过长，已截断]"

        meta = self.get_material_meta(ref_id)
        content = self.load_material(ref_id)
        if meta is None or content is None:
            return ""


        # (A) 搜索引擎：search_engine_*
        if isinstance(ref_id, str) and ref_id.startswith("search_engine_"):
            page_text = ""
            if isinstance(content, list) and content and isinstance(content[0], dict):
                page_text = content[0].get("page_text") or ""
            return _truncate(page_text)

        # (B) 计算结果：calculate_*
        if isinstance(ref_id, str) and ref_id.startswith("calculate_"):
            params = None
            result = None
            if isinstance(content, list) and content and isinstance(content[0], dict):
                params = content[0].get("parameters", None)
                result = content[0].get("result", None)

            lines = []
            if params is not None:
                try:
                    params_str = json.dumps(params, ensure_ascii=False, indent=2) if isinstance(params, (dict, list)) else str(params)
                except Exception:
                    params_str = str(params)
                lines.append("计算参数:")
                lines.append(params_str)

            lines.append("计算结果:")
            if result is None:
                lines.append("")
            else:
                try:
                    result_str = json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, (dict, list)) else str(result)
                except Exception:
                    result_str = str(result)
                lines.append(result_str)

            return "\n".join(lines)

        # (C) 表格：m_type==table
        if meta.m_type == MaterialType.TABLE.value or meta.m_type == "table":
            if isinstance(content, pd.DataFrame) and not content.empty:
                df_preview = content.head(table_rows).copy()
                MAX_CELL_CHARS = 200
                SUFFIX = "…[内容过长，已截断]"
                for col in df_preview.columns:
                    df_preview[col] = df_preview[col].apply(
                        lambda v: (v[:MAX_CELL_CHARS] + SUFFIX) if isinstance(v, str) and len(v) > MAX_CELL_CHARS else v
                    )
                return df_preview.to_csv(index=False)
            return ""

        # (D) 文本：m_type==text

        if meta.m_type == MaterialType.TEXT.value or meta.m_type == "text":
            return _truncate(content if isinstance(content, str) else "")
        if isinstance(content, str):
            return _truncate(content)
        if isinstance(content, (dict, list)):
            try:
                return _truncate(json.dumps(content, ensure_ascii=False, indent=2))
            except Exception:
                return _truncate(str(content))

        return ""
    def load_material_numerical(self, ref_id: str):
        """
        从 Material 中提取适合数值计算的“数值部分”。
        返回：
            - pandas.DataFrame 或任意 Python 对象（通常是数值 / list / dict），可直接用于进一步计算。

        异常：
            ValueError: 无法从该 Material 中找到可用的数值数据。
        """
        meta = self.get_material_meta(ref_id)
        if meta is None:
            raise ValueError(f"Material ref_id='{ref_id}' 不存在。")

        obj = self.load_material(ref_id)
        if obj is None:
            raise ValueError(f"Material ref_id='{ref_id}' 对应内容不存在或已被删除。")

        # 表格类：直接返回 DataFrame，并尝试将能转成数值的列转成数值
        if meta.m_type == MaterialType.TABLE:
            if not isinstance(obj, pd.DataFrame):
                raise ValueError(
                    f"Material ref_id='{ref_id}' 类型为 TABLE，但 load_material 返回的不是 DataFrame。"
                )
            df = obj.copy()
            # 尝试将各列转为数值，无法转换的保持原样
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            return df
        if meta.m_type == MaterialType.JSON:
            data = obj

            def decode(record: dict):
                if "result" not in record:
                    raise ValueError(
                        f"Material ref_id='{ref_id}' 的记录中缺少 'result' 字段。"
                    )
                val = record["result"]
                r_type = record.get("result_type")

                if r_type is None:
                    return val

                if r_type == "DataFrame":
                    if isinstance(val, pd.DataFrame):
                        df = val
                    elif isinstance(val, (list, dict)):
                        df = pd.DataFrame(val)
                    else:
                        raise ValueError(
                            f"Material ref_id='{ref_id}' 标记为 DataFrame，但 result 类型为 {type(val).__name__}。"
                        )
                    df = df.copy()
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="ignore")
                    return df

                if r_type == "Series":
                    if isinstance(val, pd.Series):
                        ser = val
                    elif isinstance(val, (list, dict)):
                        ser = pd.Series(val)
                    else:
                        raise ValueError(
                            f"Material ref_id='{ref_id}' 标记为 Series，但 result 类型为 {type(val).__name__}。"
                        )
                    return ser

                return val

            if isinstance(data, dict):
                if "result" not in data:
                    raise ValueError(
                        f"Material ref_id='{ref_id}' 为 JSON dict，但不含 'result' 字段。"
                    )
                return decode(data)

            if isinstance(data, list):
                results = []
                for item in data:
                    if isinstance(item, dict) and "result" in item:
                        results.append(decode(item))
                if not results:
                    raise ValueError(
                        f"Material ref_id='{ref_id}' 为 JSON list，但没有任何元素包含 'result' 字段。"
                    )
                return results[0] if len(results) == 1 else results

            raise ValueError(
                f"Material ref_id='{ref_id}' 为 JSON，但顶层类型为 {type(data).__name__}。"
            )

        raise ValueError(
            f"Material ref_id='{ref_id}' 类型为 {meta.m_type.value}，不支持提取 numerical 数据。"
        )