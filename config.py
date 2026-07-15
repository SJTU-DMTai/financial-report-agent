# -*- coding: utf-8 -*-

import os
import yaml
from copy import deepcopy
from pathlib import Path


class Config:
    def __init__(
        self,
        path: str | None = None,
        llm_name: str = None,
        vlm_name: str = None,
        outline_refine_name: str = None,
        embedding_name: str = None,
    ):
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
        self.outline_refine_name = (
            outline_refine_name
            or os.getenv("OUTLINE_REFINE_NAME")
            or models.get("outline_refine_default")
            or self.llm_name
        )
        self.embedding_name = (
            embedding_name
            or os.getenv("EMBEDDING_NAME")
            or models.get("embedding_default")
        )

    def get_pdf_style(self) -> dict:
        style = self.data.get("pdf_style", {}) or {}
        if "base_color" not in style:
            style["base_color"] = "#498ECE"
        return style

    def _get_model_cfg(self, model_id: str) -> dict:
        model_cfg = deepcopy(self.data["models"][model_id])
        model_cfg.setdefault("reasoning_only", False)
        model_cfg.setdefault("non_reasoning_model_name", None)
        return model_cfg

    def get_model_cfg(self):
        return self._get_model_cfg(self.llm_name)
    
    def get_vlm_cfg(self):
        model_id = self.vlm_name
        if model_id is None:
            raise KeyError("未配置 models.vlm_default/models.default，且未设置 VLM_NAME")
        return self._get_model_cfg(model_id)

    def get_outline_refine_model_cfg(self):
        model_id = self.outline_refine_name
        if model_id is None:
            raise KeyError("未配置 models.outline_refine_default/models.default，且未设置 OUTLINE_REFINE_NAME")
        model_cfg = self._get_model_cfg(model_id)
        model_cfg.setdefault("api_key_env", "OUTLINE_REFINE_API_KEY")
        return model_cfg

    def get_embedding_cfg(self):
        models = self.data["models"]
        model_id = self.embedding_name
        if model_id is None:
            raise KeyError("未配置 models.embedding_default，且未设置 EMBEDDING_NAME")
        model_cfg = deepcopy(models[model_id])
        model_cfg.setdefault("api_key_env", "EMB_API_KEY")
        return model_cfg

    def get_max_claims_per_segment(self) -> int:
        verify_config = self.data.get("verify_config", {}) or {}
        return int(verify_config.get("max_claims_per_segment", 15))

    def is_multi_source_verification_enabled(self) -> bool:
        verify_config = self.data.get("verify_config", {}) or {}
        return bool(verify_config.get("enable_multi_source_verification", False))

    def get_wkhtmltopdf_path(self) -> str | None:
        wkhtmltopdf_cfg = self.data.get("wkhtmltopdf_path", {})
        return wkhtmltopdf_cfg.get("path")
    
    def get_planner_cfg(self) -> dict:
        return self.data.get("planner", {})

    def get_tracking_board_cfg(self) -> dict:
        cfg = deepcopy(self.data.get("tracking_board", {}) or {})
        cfg.setdefault("max_quality_revise_attempts", 3)
        cfg.setdefault("max_verification_revise_attempts", 2)
        cfg.setdefault("max_replan_attempts", 1)
        cfg.setdefault("max_retry_attempts", 5)
        cfg.setdefault("evidence_concurrency", 4)
        cfg.setdefault("evidence_batch_size", 6)
        cfg.setdefault("segment_concurrency", 3)
        return cfg

    def get_citation_extraction_cfg(self, report_format: str | None = None) -> dict:
        evaluation_cfg = self.data.get("evaluation", {}) or {}
        citation_cfg = evaluation_cfg.get("citation_extraction", {}) or {}
        if report_format is None:
            return deepcopy(citation_cfg)
        return deepcopy(citation_cfg.get(report_format, {}) or {})

    def get_report_processing_llm_name(self) -> str:
        evaluation_cfg = self.data.get("evaluation", {}) or {}
        return str(evaluation_cfg.get("report_processing_llm_name") or self.llm_name)

    def get_report_processing_model_cfg(self) -> dict:
        model_cfg = self._get_model_cfg(self.get_report_processing_llm_name())
        model_cfg["api_key_env"] = "REPORT_PROCESSING_API_KEY"
        return model_cfg

