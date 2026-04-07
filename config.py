# -*- coding: utf-8 -*-

import os
import yaml
from copy import deepcopy
from pathlib import Path


class Config:
    def __init__(self, path: str | None = None, llm_name: str = None, vlm_name: str = None):
        """
        加载顺序：
        1. 显式传入 path
        2. config.local.yaml
        3. config.yaml
        """
        if path is not None:
            config_path = Path(path)
        else:
            if (Path(__file__).resolve().parent / "config.local.yaml").exists():
                config_path = Path(__file__).resolve().parent / "config.local.yaml"
            elif (Path(__file__).resolve().parent / "config.yaml").exists():
                config_path = Path(__file__).resolve().parent / "config.yaml"
            else:
                raise FileNotFoundError(
                    "No config file found. Expected config.local.yaml or config.yaml"
                )

        with config_path.open("r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

        self._config_path = str(config_path)
        models = self.data["models"]
        self.llm_name = llm_name or os.getenv("LLM_NAME") or models["default"]
        self.vlm_name = vlm_name or os.getenv("VLM_NAME") or models.get("vlm_default") or models.get("default")

    def get_pdf_style(self) -> dict:
        style = self.data.get("pdf_style", {}) or {}
        if "base_color" not in style:
            style["base_color"] = "#498ECE"
        return style
         
    def get_model_cfg(self):
        models = self.data["models"]
        model_id = self.llm_name
        model_cfg = deepcopy(models[model_id])
        model_cfg.setdefault("reasoning_only", False)
        model_cfg.setdefault("non_reasoning_model_name", None)
        return model_cfg
    
    def get_vlm_cfg(self):
        models = self.data["models"]
        model_id = self.vlm_name
        if model_id is None:
            raise KeyError("未配置 models.vlm_default/models.default，且未设置 VLM_NAME")
        model_cfg = deepcopy(models[model_id])
        model_cfg.setdefault("reasoning_only", False)
        model_cfg.setdefault("non_reasoning_model_name", None)
        return model_cfg

    def get_max_verify_rounds(self) -> int:
        verify_config = self.data.get("verify_config", {}) or {}
        return int(verify_config.get("max_verify_rounds", 2))

    def is_multi_source_verification_enabled(self) -> bool:
        verify_config = self.data.get("verify_config", {}) or {}
        return bool(verify_config.get("enable_multi_source_verification", False))
    
    def get_wkhtmltopdf_path(self) -> str | None:
        wkhtmltopdf_cfg = self.data.get("wkhtmltopdf_path", {})
        return wkhtmltopdf_cfg.get("path")
    
    def get_planner_cfg(self) -> dict:
        return self.data.get("planner", {})