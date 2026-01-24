# -*- coding: utf-8 -*-

import os
import yaml
from pathlib import Path
class Config:
    def __init__(self, path: str | None = None):
        """
        加载顺序：
        1. 显式传入 path
        2. config.local.yaml
        3. config.yaml
        """
        if path is not None:
            config_path = Path(path)
        else:
            if Path("config.local.yaml").exists():
                config_path = Path("config.local.yaml")
            elif Path("config.yaml").exists():
                config_path = Path("config.yaml")
            else:
                raise FileNotFoundError(
                    "No config file found. Expected config.local.yaml or config.yaml"
                )

        with config_path.open("r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

        self._config_path = str(config_path)

    def get_pdf_style(self) -> dict:
        style = self.data.get("pdf_style", {}) or {}
        if "base_color" not in style:
            style["base_color"] = "#498ECE"
        return style
         
    def get_model_cfg(self):
        models = self.data["models"]
        model_id = os.getenv('LLM_NAME', models["default"])
        return models[model_id]

    def get_max_verify_rounds(self) -> int:
        verify_config = self.data["verify_config"]
        return verify_config["max_verify_rounds"]
    
    def get_wkhtmltopdf_path(self) -> str | None:
        wkhtmltopdf_cfg = self.data.get("wkhtmltopdf_path", {})
        return wkhtmltopdf_cfg.get("path")
    
    def get_planner_cfg(self) -> dict:
        return self.data.get("planner", {})