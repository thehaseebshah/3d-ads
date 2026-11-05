"""Foreground (player / ball / referee) segmentation for a *fixed* camera.

A per-pixel median over frames sampled across the clip gives an empty-pitch plate.
Foreground = pixels whose colour differs from the plate. Distance is measured in Lab
with lightness down-weighted so cast shadows (mostly a lightness change) are not
mistaken for players.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_plate(video_path: str, n_frames: int = 30) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 1, 0), min(n_frames, max(total, 1))).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise IOError(f"could not read frames from {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


class ForegroundModel:
    def __init__(self, plate: np.ndarray, threshold: float = 12.0, l_weight: float = 0.35):
        self.plate = plate
        self.threshold = threshold
        self.l_weight = l_weight
        self._plate_lab = self._lab(plate)

    @staticmethod
    def _lab(img):
        return cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """Binary uint8 {0,255} foreground mask."""
        d = self._lab(frame) - self._plate_lab
        dist = np.sqrt((self.l_weight * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
        m = (dist > self.threshold).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        return cv2.dilate(m, k)
