"""Pitch geometry in metres.

World frame: origin at the near-left corner, +x along the touchline (towards the
right of a camera standing on the near touchline), +y across the pitch (away from
that camera), +z up. The camera is assumed to sit on the y < 0 side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchSpec:
    kind: str
    length: float
    width: float
    landmarks: dict  # name -> (x, y)

    def polylines(self) -> list[np.ndarray]:
        """Pitch markings as (N,2) world polylines (for overlays / synthetic data)."""
        return _football_lines(self) if self.kind == "football" else _futsal_lines(self)


def _arc(cx, cy, r, a0, a1, n=48):
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def _rect(x0, y0, x1, y1):
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], float)


def _football_lines(p: PitchSpec):
    L, W = p.length, p.width
    cy = W / 2
    lines = [_rect(0, 0, L, W), np.array([(L / 2, 0), (L / 2, W)], float), _arc(L / 2, cy, 9.15, 0, 360, 96)]
    a_half = math.degrees(math.acos(5.5 / 9.15))
    for xg, d in ((0.0, 1), (L, -1)):
        X = lambda dx: xg + d * dx  # noqa: E731
        lines.append(np.array([(X(0), cy - 20.16), (X(16.5), cy - 20.16), (X(16.5), cy + 20.16), (X(0), cy + 20.16)]))
        lines.append(np.array([(X(0), cy - 9.16), (X(5.5), cy - 9.16), (X(5.5), cy + 9.16), (X(0), cy + 9.16)]))
        lines.append(_arc(X(11), cy, 0.25, 0, 360, 16))
        c = 0 if d == 1 else 180
        lines.append(_arc(X(11), cy, 9.15, c - a_half, c + a_half))
    return lines


def _futsal_lines(p: PitchSpec):
    L, W = p.length, p.width
    cy = W / 2
    lines = [_rect(0, 0, L, W), np.array([(L / 2, 0), (L / 2, W)], float), _arc(L / 2, cy, 3.0, 0, 360, 64)]
    for xg, d in ((0.0, 1), (L, -1)):
        a1 = _arc(xg, cy - 1.5, 6.0, -90, 0) if d == 1 else _arc(xg, cy - 1.5, 6.0, 180, 270)
        a2 = _arc(xg, cy + 1.5, 6.0, 0, 90) if d == 1 else _arc(xg, cy + 1.5, 6.0, 90, 180)
        lines += [a1, a2, np.array([(xg + d * 6, cy - 1.5), (xg + d * 6, cy + 1.5)])]
        lines.append(_arc(xg + d * 6, cy, 0.15, 0, 360, 12))
        lines.append(_arc(xg + d * 10, cy, 0.15, 0, 360, 12))
    return lines


def football(length: float = 105.0, width: float = 68.0) -> PitchSpec:
    L, W = length, width
    cy = W / 2
    lm = {
        "corner_near_left": (0, 0), "corner_near_right": (L, 0),
        "corner_far_left": (0, W), "corner_far_right": (L, W),
        "halfway_near": (L / 2, 0), "halfway_far": (L / 2, W), "centre": (L / 2, cy),
    }
    for side, xg, d in (("left", 0.0, 1), ("right", L, -1)):
        lm[f"{side}_penalty_spot"] = (xg + d * 11, cy)
        lm[f"{side}_pa_goalline_near"] = (xg, cy - 20.16)
        lm[f"{side}_pa_goalline_far"] = (xg, cy + 20.16)
        lm[f"{side}_pa_corner_near"] = (xg + d * 16.5, cy - 20.16)
        lm[f"{side}_pa_corner_far"] = (xg + d * 16.5, cy + 20.16)
        lm[f"{side}_6yd_goalline_near"] = (xg, cy - 9.16)
        lm[f"{side}_6yd_goalline_far"] = (xg, cy + 9.16)
        lm[f"{side}_6yd_corner_near"] = (xg + d * 5.5, cy - 9.16)
        lm[f"{side}_6yd_corner_far"] = (xg + d * 5.5, cy + 9.16)
    return PitchSpec("football", L, W, lm)


def futsal(length: float = 40.0, width: float = 20.0) -> PitchSpec:
    L, W = length, width
    cy = W / 2
    lm = {
        "corner_near_left": (0, 0), "corner_near_right": (L, 0),
        "corner_far_left": (0, W), "corner_far_right": (L, W),
        "halfway_near": (L / 2, 0), "halfway_far": (L / 2, W), "centre": (L / 2, cy),
    }
    for side, xg, d in (("left", 0.0, 1), ("right", L, -1)):
        lm[f"{side}_penalty_spot"] = (xg + d * 6, cy)
        lm[f"{side}_second_penalty_spot"] = (xg + d * 10, cy)
        lm[f"{side}_post_near"] = (xg, cy - 1.5)
        lm[f"{side}_post_far"] = (xg, cy + 1.5)
        lm[f"{side}_pa_front_near"] = (xg + d * 6, cy - 1.5)
        lm[f"{side}_pa_front_far"] = (xg + d * 6, cy + 1.5)
    return PitchSpec("futsal", L, W, lm)


def get_pitch(kind: str = "football", length: float | None = None, width: float | None = None) -> PitchSpec:
    if kind == "football":
        return football(length or 105.0, width or 68.0)
    if kind == "futsal":
        return futsal(length or 40.0, width or 20.0)
    raise ValueError(f"unknown pitch kind {kind!r} (football | futsal)")
