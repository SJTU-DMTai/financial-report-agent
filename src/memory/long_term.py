from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import akshare as ak

@dataclass
class LongTermMemoryStore:
    base_dir: Path

    def ensure_dir(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def code_name_path(self) -> Path:
        return self.base_dir / "a_share_code_name.csv"

    def refresh_code_name(self) -> pd.DataFrame:
        """手动更新：从 akshare 获取股票代码-股票名称，覆盖写入本地"""
        self.ensure_dir()
        df = ak.stock_info_a_code_name()
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["name"] = df["name"].astype(str)
        df.to_csv(self.code_name_path, index=False, encoding="utf-8-sig")
        return df

    def _load_code_name(self) -> Optional[pd.DataFrame]:
        if not self.code_name_path.exists():
            return None
        return pd.read_csv(self.code_name_path, dtype=str)

    def name_by_code(self, code: str) -> Optional[str]:
        """代码 -> 名称"""
        code = str(code).zfill(6)
        df = self._load_code_name()
        if df is None:
            df = self.refresh_code_name()
        hit = df[df["code"] == code]
        return None if hit.empty else hit.iloc[0]["name"]

    def codes_by_name(self, name: str, fuzzy: bool = True) -> List[Dict[str, str]]:
        """名称 -> 代码（可能多个），fuzzy: 是否支持模糊匹配"""
        df = self._load_code_name()
        if df is None:
            df = self.refresh_code_name()
        if fuzzy:
            hit = df[df["name"].str.contains(name, na=False)]
        else:
            hit = df[df["name"] == name]
        return hit[["code","name"]].to_dict(orient="records")
