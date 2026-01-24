# -*- coding: utf-8 -*-
from __future__ import annotations

import colorsys
from typing import Dict, Tuple, List


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    s = hex_color.lstrip("#")
    return (
        int(s[0:2], 16) / 255.0,
        int(s[2:4], 16) / 255.0,
        int(s[4:6], 16) / 255.0,
    )


def rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(
        int(round(_clamp(r) * 255)),
        int(round(_clamp(g) * 255)),
        int(round(_clamp(b) * 255)),
    )


def adjust_lightness(hex_color: str, l_mul: float) -> str:
    r, g, b = hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = _clamp(l * l_mul)
    return rgb01_to_hex(colorsys.hls_to_rgb(h, l, s))


def lighten(hex_color: str, t: float) -> str:
    r, g, b = hex_to_rgb01(hex_color)
    return rgb01_to_hex((
        r + (1 - r) * t,
        g + (1 - g) * t,
        b + (1 - b) * t,
    ))

_CATEGORICAL_PRIMARY: List[str] = [
    "#1B3C53",
    "#456882",
    "#77A6D8",
    "#86C2D5",
    "#76B7B2",
    "#B4EDD5",
    "#FFEAC0",
    "#D2C1B6",
    "#AFAFAC",
    "#A78F7F",
]

_CATEGORICAL_SECONDARY: List[str] = [
    "#ECC16A",
    "#FFF0C4",
    "#CC6846",
    "#AF3B20", 
    "#831007", 
    "#490B06", 
]



def _repeat_to_n(palette: List[str], n: int) -> List[str]:
    if n <= 0:
        return []
    return [palette[i % len(palette)] for i in range(n)]


def generate_palette(
    base_hex: str,
    *,
    series_n: int = 12,
    dark1_mul: float = 0.80,
    dark2_mul: float = 0.62,
    light1_t: float = 0.28,
    light2_t: float = 0.52,
) -> Dict[str, object]:
    """
    返回结构：
    {
      "theme": { "base","dark1","dark2","light1","light2" },
      "categorical_primary": [ ... ],
      "categorical_secondary": [ ... ],
    }
    """
    base = base_hex.upper()

    theme = {
        "base": base,
        "dark1": adjust_lightness(base, dark1_mul),
        "dark2": adjust_lightness(base, dark2_mul),
        "light1": lighten(base, light1_t),
        "light2": lighten(base, light2_t),
    }

    return {
        "theme": theme,
        "categorical_primary": _repeat_to_n(_CATEGORICAL_PRIMARY, series_n),
        "categorical_secondary": _repeat_to_n(_CATEGORICAL_SECONDARY, series_n),
    }
