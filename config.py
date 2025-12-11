# -*- coding: utf-8 -*-
import yaml

class Config:
    def __init__(self, path: str = "config.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def get_model_cfg(self):
        models = self.data["models"]
        model_id = models["default"]
        return models[model_id]
    
    def get_font_path(self, key: str = "chinese") -> str | None:
            font_cfg = self.data.get("font", {})
            return font_cfg.get(key)
    
    def get_max_verify_rounds(self) -> int:
        verify_config = self.data["verify_config"]
        return verify_config["max_verify_rounds"]
    
    def get_wkhtmltopdf_path(self) -> str | None:
        wkhtmltopdf_cfg = self.data.get("wkhtmltopdf_path", {})
        return wkhtmltopdf_cfg.get("path")
    
    def get_planner_cfg(self) -> dict:
        return self.data.get("planner", {})