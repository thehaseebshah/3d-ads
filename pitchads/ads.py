"""Ad definitions and rasterisation into camera-space layers.

Ad types (all positions in pitch metres, `rotation_deg` is CCW about the vertical axis):
  ground     flat texture painted on the grass
  box        3D block: texture on top and on the front face (the side facing -y, towards
             the default near-touchline camera), solid colour on the other sides, ground shadow
  billboard  thin upright board with the texture on its front

Texture orientation: image left->right runs along +x, image top points to +y (far side),
so a logo reads correctly for a camera on the near touchline. Rotate 180 for the far side.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Camera

SS = 2  # supersampling factor for anti-aliased edges


@dataclass
class AdSpec:
    id: str
    type: str
    image: np.ndarray                       # BGRA uint8
    center: tuple                           # (x, y) metres
    size: tuple                             # (w along x, d along y) metres; billboard: (width, ignored)
    rotation_deg: float = 0.0
    height: float = 0.0                     # box / billboard height, metres
    opacity: float = 0.92                   # ground ads only
    front_image: np.ndarray | None = None   # BGRA, box front face (defaults to `image`)
    side_color: tuple = (40, 40, 40)        # BGR
    shade: float = 1.0                      # how strongly grass lighting modulates a ground ad
    keep_lines: float = 0.85                # 0..1, how much pitch markings show through a ground ad
    start: float = 0.0                      # seconds
    end: float | None = None
    fade: float = 0.5


@dataclass
class Layer:
    """Premultiplied-alpha crop of a full frame, positioned at (x0, y0)."""
    x0: int
    y0: int
    pm: np.ndarray   # HxWx3 float32 premultiplied BGR (0..255)
    a: np.ndarray    # HxW float32 (0..1)


def _bgra(img):
    if img is None:
        raise FileNotFoundError("ad image could not be read")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return img


def load_ads(path: str) -> list[AdSpec]:
    base = os.path.dirname(os.path.abspath(path))
    with open(path) as fh:
        data = json.load(fh)
    ads = []
    for a in data["ads"]:
        rd = lambda p: _bgra(cv2.imread(os.path.join(base, p), cv2.IMREAD_UNCHANGED))  # noqa: E731
        img = rd(a["image"])
        ads.append(AdSpec(
            id=a["id"], type=a.get("type", "ground"), image=img,
            center=tuple(a["center"]), size=tuple(a["size"]), rotation_deg=a.get("rotation_deg", 0.0),
            height=a.get("height", 0.0), opacity=a.get("opacity", 0.92),
            front_image=rd(a["front_image"]) if a.get("front_image") else None,
            side_color=tuple(a.get("side_color", (40, 40, 40))), shade=a.get("shade", 1.0),
            keep_lines=a.get("keep_lines", 0.85), start=a.get("start", 0.0), end=a.get("end"),
            fade=a.get("fade", 0.5)))
    return ads


# ---------------------------------------------------------------------------
def _local_to_world(ad: AdSpec, pts_local):
    th = np.radians(ad.rotation_deg)
    c, s = np.cos(th), np.sin(th)
    p = np.asarray(pts_local, float)
    x = ad.center[0] + c * p[:, 0] - s * p[:, 1]
    y = ad.center[1] + s * p[:, 0] + c * p[:, 1]
    return np.column_stack([x, y, p[:, 2]])


def footprint_world(ad: AdSpec) -> np.ndarray:
    w, d = ad.size
    d = 0.05 if ad.type == "billboard" else d
    loc = [(-w / 2, -d / 2, 0), (w / 2, -d / 2, 0), (w / 2, d / 2, 0), (-w / 2, d / 2, 0)]
    return _local_to_world(ad, loc)


def _faces(ad: AdSpec):
    """List of (corners3d[TL,TR,BR,BL], texture BGRA, brightness, outward normal)."""
    w, d = ad.size
    h = ad.height
    th = np.radians(ad.rotation_deg)
    rot = lambda n: (np.cos(th) * n[0] - np.sin(th) * n[1], np.sin(th) * n[0] + np.cos(th) * n[1], n[2])  # noqa: E731

    def solid(color):
        t = np.zeros((4, 4, 4), np.uint8)
        t[..., :3] = color
        t[..., 3] = 255
        return t

    if ad.type == "ground":
        loc = [(-w / 2, d / 2, 0), (w / 2, d / 2, 0), (w / 2, -d / 2, 0), (-w / 2, -d / 2, 0)]
        return [(_local_to_world(ad, loc), ad.image, 1.0, rot((0, 0, 1)))]
    front_tex = ad.front_image if ad.front_image is not None else ad.image
    if ad.type == "billboard":
        d = 0.05
        loc = [(-w / 2, -d / 2, h), (w / 2, -d / 2, h), (w / 2, -d / 2, 0), (-w / 2, -d / 2, 0)]
        return [(_local_to_world(ad, loc), front_tex, 1.0, rot((0, -1, 0)))]
    sc = ad.side_color
    shade = lambda k: tuple(int(c * k) for c in sc)  # noqa: E731
    faces = [
        ([(-w / 2, d / 2, h), (w / 2, d / 2, h), (w / 2, -d / 2, h), (-w / 2, -d / 2, h)], ad.image, 1.0, (0, 0, 1)),
        ([(-w / 2, -d / 2, h), (w / 2, -d / 2, h), (w / 2, -d / 2, 0), (-w / 2, -d / 2, 0)], front_tex, 0.9, (0, -1, 0)),
        ([(w / 2, d / 2, h), (-w / 2, d / 2, h), (-w / 2, d / 2, 0), (w / 2, d / 2, 0)], solid(shade(0.55)), 1.0, (0, 1, 0)),
        ([(-w / 2, d / 2, h), (-w / 2, -d / 2, h), (-w / 2, -d / 2, 0), (-w / 2, d / 2, 0)], solid(shade(0.7)), 1.0, (-1, 0, 0)),
        ([(w / 2, -d / 2, h), (w / 2, d / 2, h), (w / 2, d / 2, 0), (w / 2, -d / 2, 0)], solid(shade(0.45)), 1.0, (1, 0, 0)),
    ]
    return [(_local_to_world(ad, c), t, b, rot(n)) for c, t, b, n in faces]


def _premult(tex_bgra: np.ndarray) -> np.ndarray:
    t = tex_bgra.astype(np.float32)
    a = t[..., 3:4] / 255.0
    return np.concatenate([t[..., :3] * a, a], axis=2)


def render_layer(ad: AdSpec, cam: Camera) -> Layer | None:
    """Rasterise one ad to a premultiplied layer, or None if it is not visible."""
    W, H = cam.size
    faces = _faces(ad)
    C = cam.center
    visible = []
    for corners, tex, bright, normal in faces:
        centre = corners.mean(axis=0)
        if ad.type == "billboard" or np.dot(normal, C - centre) > 0:
            visible.append((corners, tex, bright))
    all_pts = np.concatenate([c for c, _, _ in visible])
    if np.any(cam.depth(all_pts) <= 0.1):
        return None
    # shadow footprint for raised objects
    fp = footprint_world(ad)
    shadow_poly = None
    if ad.type in ("box", "billboard") and ad.height > 0:
        off = np.array([0.45, 0.25, 0.0]) * ad.height
        shadow_poly = np.concatenate([fp, fp + off])
        all_pts = np.concatenate([all_pts, shadow_poly])
    px = cam.project(all_pts)
    if np.any(~np.isfinite(px)):
        return None
    x0 = max(int(np.floor(px[:, 0].min())) - 3, 0)
    y0 = max(int(np.floor(px[:, 1].min())) - 3, 0)
    x1 = min(int(np.ceil(px[:, 0].max())) + 3, W)
    y1 = min(int(np.ceil(px[:, 1].max())) + 3, H)
    if x1 <= x0 or y1 <= y0:
        return None
    cw, ch = x1 - x0, y1 - y0
    pm = np.zeros((ch * SS, cw * SS, 3), np.float32)
    a = np.zeros((ch * SS, cw * SS), np.float32)

    def blend(src_pm, src_a):
        nonlocal pm, a
        pm = src_pm + pm * (1 - src_a[..., None])
        a = src_a + a * (1 - src_a)

    if shadow_poly is not None:
        sp = cam.project(shadow_poly)
        hull = cv2.convexHull(((sp - [x0, y0]) * SS).astype(np.float32)).astype(np.int32)
        m = np.zeros_like(a)
        cv2.fillConvexPoly(m, hull, 1.0, lineType=cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 2.5 * SS) * 0.4
        blend(np.zeros_like(pm), m)

    order = sorted(visible, key=lambda f: -np.linalg.norm(f[0].mean(axis=0) - C))
    for corners, tex, bright in order:
        dst = ((cam.project(corners) - [x0, y0]) * SS).astype(np.float32)
        th_, tw_ = tex.shape[:2]
        src = np.float32([[0, 0], [tw_, 0], [tw_, th_], [0, th_]])
        M = cv2.getPerspectiveTransform(src, dst)
        # pre-shrink very large textures to avoid aliasing
        out_w = np.linalg.norm(dst[1] - dst[0])
        if out_w > 0 and tw_ > 2 * out_w:
            k = (2 * out_w) / tw_
            tex = cv2.resize(tex, (max(int(tw_ * k), 2), max(int(th_ * k), 2)), interpolation=cv2.INTER_AREA)
            th_, tw_ = tex.shape[:2]
            src = np.float32([[0, 0], [tw_, 0], [tw_, th_], [0, th_]])
            M = cv2.getPerspectiveTransform(src, dst)
        t = _premult(tex)
        t[..., :3] *= bright
        warped = cv2.warpPerspective(t, M, (cw * SS, ch * SS), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        blend(warped[..., :3], warped[..., 3])

    pm = cv2.resize(pm, (cw, ch), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, (cw, ch), interpolation=cv2.INTER_AREA)
    if a.max() < 1e-4:
        return None
    return Layer(x0, y0, pm, a)
