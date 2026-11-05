"""Synthetic fixed-camera match footage with known geometry, for demos and tests."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .camera import Camera
from .pitch import PitchSpec, get_pitch

PPM, MARGIN = 10, 6


def pitch_texture(spec: PitchSpec) -> np.ndarray:
    L, W = spec.length, spec.width
    w, h = int((L + 2 * MARGIN) * PPM), int((W + 2 * MARGIN) * PPM)
    rng = np.random.default_rng(1)
    img = np.zeros((h, w, 3), np.float32)
    stripe = int(round(L / 14 * PPM)) if spec.kind == "football" else int(round(L / 8 * PPM))
    for i, x in enumerate(range(0, w, stripe)):
        img[:, x:x + stripe] = (52, 128, 60) if i % 2 else (46, 114, 54)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
    img *= (1 + 0.06 * noise / noise.std())[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    to_px = lambda p: np.round(np.column_stack([(p[:, 0] + MARGIN) * PPM, (W + MARGIN - p[:, 1]) * PPM])).astype(np.int32)  # noqa: E731
    for pl in spec.polylines():
        cv2.polylines(img, [to_px(pl)], False, (235, 235, 235), 3, cv2.LINE_AA)
    return img


def _team_positions(spec: PitchSpec, t: float):
    """22 deterministic wandering players: (x, y, team)."""
    L, W = spec.length, spec.width
    form = [(0.06, .5), (.2, .15), (.2, .38), (.2, .62), (.2, .85), (.38, .2), (.38, .5), (.38, .8),
            (.55, .3), (.55, .7), (.62, .5)]
    out = []
    for team in (0, 1):
        for k, (fx, fy) in enumerate(form):
            fx = fx if team == 0 else 1 - fx
            ph = k * 1.7 + team * 0.9
            x = fx * L + 0.07 * L * np.sin(0.8 * t + ph) + 0.03 * L * np.sin(2.1 * t + 2 * ph)
            y = fy * W + 0.14 * W * np.sin(0.6 * t + 1.3 * ph) + 0.05 * W * np.cos(1.7 * t + ph)
            out.append((x, np.clip(y, 1.0, W - 1.0), team))
    return out


def make_camera(spec: PitchSpec, size=(1920, 540)) -> Camera:
    L, W = spec.length, spec.width
    scale = L / 105.0
    f = size[0] * 0.30  # ~ 120 degree horizontal FOV (ultra-wide panorama)
    return Camera.look_at((L / 2, -34.0 * scale, 18.0 * scale), (L / 2, W * 0.45, 0.0), f, size)


def generate(out_dir: str, kind: str = "football", seconds: float = 10.0, fps: int = 25, size=(1920, 540)):
    spec = get_pitch(kind)
    cam = make_camera(spec, size)
    os.makedirs(out_dir, exist_ok=True)
    W_, H_ = size
    tex = pitch_texture(spec)
    L, Wd = spec.length, spec.width
    T = np.array([[1 / PPM, 0, -MARGIN], [0, -1 / PPM, Wd + MARGIN], [0, 0, 1.0]])
    Hm = cam.ground_homography() @ T
    base = cv2.warpPerspective(tex, Hm, (W_, H_), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inside = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), Hm, (W_, H_), flags=cv2.INTER_NEAREST)
    yy = np.linspace(0, 1, H_)[:, None, None]
    stands = (np.array([70, 62, 58]) * (0.6 + 0.4 * yy) * np.ones((1, W_, 1))).astype(np.float32)
    stands += np.random.default_rng(3).normal(0, 6, stands.shape)
    base = np.where(inside[..., None] > 0, base, np.clip(stands, 0, 255).astype(np.uint8))
    base[:, :, :] = base  # explicit: static plate

    path = os.path.join(out_dir, "input.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_, H_))
    rng = np.random.default_rng(7)
    colors = {0: (40, 40, 220), 1: (230, 190, 40)}  # BGR: red / cyan-ish
    for fi in range(int(seconds * fps)):
        t = fi / fps
        frame = base.copy()
        ents = _team_positions(spec, t)
        # draw far -> near
        ents.sort(key=lambda e: -cam.depth([[e[0], e[1], 0]])[0])
        for x, y, team in ents:
            feet = cam.project([[x, y, 0.0]])[0]
            head = cam.project([[x, y, 1.75]])[0]
            d = cam.depth([[x, y, 0]])[0]
            r = max(cam.focal * 0.22 / d, 1.5)
            sh = cam.project([[x + 0.5, y + 0.3, 0.0]])[0]
            ov = frame.copy()
            cv2.ellipse(ov, tuple(map(int, sh)), (int(r * 1.8), int(r * 0.7)), 0, 0, 360, (15, 40, 15), -1, cv2.LINE_AA)
            frame = cv2.addWeighted(ov, 0.35, frame, 0.65, 0)
            cv2.line(frame, tuple(map(int, feet)), tuple(map(int, head)), colors[team], max(int(r * 2), 2), cv2.LINE_AA)
            cv2.circle(frame, tuple(map(int, head)), max(int(r * 0.9), 1), (150, 170, 220), -1, cv2.LINE_AA)
        bx, by = L / 2 + 25 * np.sin(0.5 * t), Wd / 2 + 15 * np.sin(0.9 * t + 1)
        bp = cam.project([[bx, by, 0.15]])[0]
        cv2.circle(frame, tuple(map(int, bp)), max(int(cam.focal * 0.11 / cam.depth([[bx, by, 0]])[0]), 2), (250, 250, 250), -1, cv2.LINE_AA)
        frame = np.clip(frame.astype(np.float32) + rng.normal(0, 1.5, frame.shape), 0, 255).astype(np.uint8)
        vw.write(frame)
    vw.release()

    # "annotator clicks": landmark pixels (1px jitter) -> what a human would supply
    names = list(spec.landmarks)
    lm = {}
    for n in names:
        p = cam.project([[*spec.landmarks[n], 0.0]])[0]
        if 5 < p[0] < W_ - 5 and 5 < p[1] < H_ - 5:
            lm[n] = [round(float(p[0] + rng.normal(0, 0.7)), 1), round(float(p[1] + rng.normal(0, 0.7)), 1)]
    with open(os.path.join(out_dir, "landmarks.json"), "w") as fh:
        json.dump({"pitch": kind, "image_size": [W_, H_], "landmarks": lm}, fh, indent=2)
    truth = os.path.join(out_dir, "truth_camera.json")
    cam.save(truth, spec)
    return path


def make_demo_ads(out_dir: str, spec: PitchSpec) -> str:
    """Write placeholder ad artwork + an ads.json sized for the pitch."""
    os.makedirs(os.path.join(out_dir, "ads"), exist_ok=True)

    def banner(name, text, c1, c2, size=(1200, 400)):
        w, h = size
        img = np.zeros((h, w, 3), np.uint8)
        for x in range(w):
            k = x / (w - 1)
            img[:, x] = [int(c1[i] * (1 - k) + c2[i] * k) for i in range(3)]
        cv2.rectangle(img, (6, 6), (w - 7, h - 7), (255, 255, 255), 8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 3.2, 8)
        s = min(1.0, (w - 80) / tw)
        cv2.putText(img, text, (int((w - tw * s) / 2), int((h + th * s) / 2)), cv2.FONT_HERSHEY_DUPLEX, 3.2 * s,
                    (255, 255, 255), max(int(8 * s), 2), cv2.LINE_AA)
        p = os.path.join(out_dir, "ads", name)
        cv2.imwrite(p, img)
        return os.path.join("ads", name)

    L, W = spec.length, spec.width
    s = L / 105.0
    ads = [
        {"id": "ground_a", "type": "ground", "image": banner("a.png", "ACME COLA", (30, 30, 200), (40, 120, 255)),
         "center": [L * 0.30, W * 0.50], "size": [18 * s, 6 * s]},
        {"id": "ground_b", "type": "ground", "image": banner("b.png", "NOVA BANK", (160, 80, 20), (230, 160, 60)),
         "center": [L * 0.72, W * 0.22], "size": [18 * s, 6 * s], "shade": 1.5},
        {"id": "box_c", "type": "box", "image": banner("c.png", "TURBO", (20, 140, 20), (60, 220, 120), (800, 400)),
         "front_image": banner("c_front.png", "TURBO", (20, 140, 20), (60, 220, 120), (800, 250)),
         "center": [L * 0.5, W * 0.72], "size": [7 * s, 3 * s], "height": 2.2, "side_color": [30, 120, 30]},
        {"id": "billboard_d", "type": "billboard", "image": banner("d.png", "SKYLINE", (120, 20, 120), (220, 90, 200), (900, 300)),
         "center": [L * 0.88, W * 0.55], "size": [8 * s, 0], "height": 2.6 * s, "start": 3.0, "end": 8.0},
    ]
    p = os.path.join(out_dir, "ads.json")
    with open(p, "w") as fh:
        json.dump({"ads": ads}, fh, indent=2)
    return p
