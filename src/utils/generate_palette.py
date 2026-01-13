# -*- coding: utf-8 -*-
from __future__ import annotations
import colorsys
from typing import Dict, Tuple


# =====================
# 根据一个主要颜色，生成PDF配色、图表配色
# =====================

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


def rotate_hue(hex_color: str, degrees: float) -> str:
    r, g, b = hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + degrees / 360.0) % 1.0
    return rgb01_to_hex(colorsys.hls_to_rgb(h, l, s))


def generate_palette(
    base_hex: str,
    *,
    analogous_deg: float = 30.0,
    dark1_mul: float = 0.80,
    dark2_mul: float = 0.62,
    light1_t: float = 0.28,
    light2_t: float = 0.52,
) -> Dict[str, Dict[str, str]]:
    """
    返回结构：
    {
        "base": {
            "base": "...",
            "dark1": "...",
            "dark2": "...",
            "light1": "...",
            "light2": "..."
        },
        "analogous": {
            "base": "...",
            "dark1": "...",
            "dark2": "...",
            "light1": "...",
            "light2": "..."
        }
    }
    """

    base = base_hex.upper()
    ana = rotate_hue(base, analogous_deg)

    def scale(c: str) -> Dict[str, str]:
        return {
            "base": c,
            "dark1": adjust_lightness(c, dark1_mul),
            "dark2": adjust_lightness(c, dark2_mul),
            "light1": lighten(c, light1_t),
            "light2": lighten(c, light2_t),
        }

    return {
        "base": scale(base),
        "analogous": scale(ana),
    }

