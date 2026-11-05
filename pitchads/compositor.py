"""Composite ad layers into video frames with player occlusion and grass integration."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from .ads import AdSpec, Layer, footprint_world, render_layer
from .background import ForegroundModel, build_plate
from .camera import Camera


class Compositor:
    def __init__(self, camera: Camera, ads: list[AdSpec], plate: np.ndarray, threshold: float = 12.0):
        self.cam = camera
        self.fg = ForegroundModel(plate, threshold)
        self.items = []
        plate_lab = cv2.cvtColor(cv2.GaussianBlur(plate, (0, 0), 2), cv2.COLOR_BGR2LAB)
        plate_L = plate_lab[..., 0].astype(np.float32)
        hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        lines = ((hsv[..., 1] < 70) & (hsv[..., 2] > 165)).astype(np.float32)
        lines = cv2.GaussianBlur(cv2.dilate(lines, np.ones((3, 3), np.uint8)), (0, 0), 1.0)
        for ad in ads:
            layer = render_layer(ad, camera)
            if layer is None:
                continue
            sl = (slice(layer.y0, layer.y0 + layer.a.shape[0]), slice(layer.x0, layer.x0 + layer.a.shape[1]))
            if ad.type == "ground":
                layer = self._integrate_with_grass(layer, ad, plate_L[sl], lines[sl])
            fp = footprint_world(ad)
            self.items.append({
                "ad": ad, "layer": layer,
                "depth": float(camera.depth(np.append(fp.mean(axis=0)[:2], 0.0))[0]),
            })

    @staticmethod
    def _integrate_with_grass(layer: Layer, ad: AdSpec, L: np.ndarray, lines: np.ndarray) -> Layer:
        inside = layer.a > 0.5
        if inside.any():
            ratio = np.clip(L / max(float(L[inside].mean()), 1.0), 0.7, 1.3)
            shade = 1.0 + (ratio - 1.0) * ad.shade
            pm = layer.pm * shade[..., None]
        else:
            pm = layer.pm
        k = 1.0 - ad.keep_lines * lines
        return Layer(layer.x0, layer.y0, pm * (ad.opacity * k)[..., None], layer.a * ad.opacity * k)

    @staticmethod
    def _fade(ad: AdSpec, t: float) -> float:
        if t < ad.start or (ad.end is not None and t > ad.end):
            return 0.0
        f = min((t - ad.start) / ad.fade, 1.0) if ad.fade > 0 else 1.0
        if ad.end is not None and ad.fade > 0:
            f = min(f, (ad.end - t) / ad.fade)
        return float(np.clip(f, 0.0, 1.0))

    def render(self, frame: np.ndarray, t: float = 0.0, fg_mask: np.ndarray | None = None) -> np.ndarray:
        fg = self.fg.mask(frame) if fg_mask is None else fg_mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        comp_front = {}  # component id -> depth of its feet on the ground
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 12:
                continue
            gx, gy = self.cam.image_to_ground([[x + w / 2, y + h]])[0]
            comp_front[i] = float(self.cam.depth([[gx, gy, 0.0]])[0])
        out = frame.astype(np.float32)
        # far ads first so nearer ones overlap correctly
        for it in sorted(self.items, key=lambda i: -i["depth"]):
            ad, layer = it["ad"], it["layer"]
            f = self._fade(ad, t)
            if f <= 0:
                continue
            h, w = layer.a.shape
            sl = (slice(layer.y0, layer.y0 + h), slice(layer.x0, layer.x0 + w))
            if ad.type == "ground":
                occ = fg[sl] > 0                      # everything standing on the grass is in front of paint
            else:                                     # only things nearer to the camera than the object
                ids = [i for i, d in comp_front.items() if d < it["depth"]]
                occ = np.isin(labels[sl], ids) if ids else np.zeros((h, w), bool)
            soft = cv2.GaussianBlur(occ.astype(np.float32), (0, 0), 1.0)
            vis = (1.0 - soft) * f
            dst = out[sl]
            out[sl] = layer.pm * vis[..., None] + dst * (1.0 - (layer.a * vis)[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)


def render_video(video: str, camera: Camera, ads: list[AdSpec], out_path: str, bg_frames: int = 30,
                 threshold: float = 12.0, progress=None) -> None:
    plate = build_plate(video, bg_frames)
    comp = Compositor(camera, ads, plate, threshold)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != tuple(camera.size):
        raise ValueError(f"video is {w}x{h} but camera was calibrated for {tuple(camera.size)}")
    tmp = tempfile.mkdtemp()
    raw = os.path.join(tmp, "raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(comp.render(frame, i / fps))
        i += 1
        if progress and i % 25 == 0:
            progress(i)
    vw.release()
    cap.release()
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "18", out_path])
        if r.returncode == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            return
    shutil.move(raw, out_path)
