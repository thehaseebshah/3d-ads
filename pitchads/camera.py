"""Pinhole camera calibrated from pitch landmarks.

Because the panoramic camera is fixed, this is solved once per video. Given >= 4
(ideally >= 6, well spread) landmark correspondences we solve for focal length and
pose with a planar PnP, which then lets us project *any* 3D point, so ads can be
flat on the grass or true 3D objects.

Not modelled yet: lens distortion. For strongly curved stitched panoramas,
undistort/rectify the video first (or extend `project` with cv2.projectPoints).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np


def _project(K, rvec, tvec, pts):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    pc = pts @ R.T + np.asarray(tvec, float).reshape(1, 3)
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]], axis=1)
    uv[z <= 1e-6] = np.nan
    return uv


@dataclass
class Camera:
    K: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    size: tuple  # (width, height)
    rmse: float = 0.0

    # ---- geometry -------------------------------------------------------
    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, float))[0]

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ np.asarray(self.tvec, float).ravel()

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    def project(self, pts) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 3)
        return _project(self.K, self.rvec, self.tvec, pts)

    def depth(self, pts) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 3)
        return (pts @ self.R.T + np.asarray(self.tvec, float).reshape(1, 3))[:, 2]

    def ground_homography(self) -> np.ndarray:
        """3x3 H mapping ground (x, y) metres -> image pixels."""
        R = self.R
        return self.K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(self.tvec, float).ravel()])

    def image_to_ground(self, uv) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        Hi = np.linalg.inv(self.ground_homography())
        p = np.column_stack([uv, np.ones(len(uv))]) @ Hi.T
        return p[:, :2] / p[:, 2:3]

    # ---- construction ---------------------------------------------------
    @classmethod
    def look_at(cls, position, target, focal, size):
        C = np.asarray(position, float)
        f = np.asarray(target, float) - C
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(f, right)
        R = np.stack([right, down, f])
        w, h = size
        K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
        return cls(K, cv2.Rodrigues(R)[0].ravel(), (-R @ C), (w, h))

    @classmethod
    def fit(cls, world, image, size, focal: float | None = None) -> "Camera":
        """Solve pose (and focal length unless given) from ground-plane landmarks."""
        world = np.asarray(world, float)
        if world.shape[1] == 2:
            world = np.column_stack([world, np.zeros(len(world))])
        image = np.asarray(image, float).reshape(-1, 2)
        if len(world) < 4:
            raise ValueError("need at least 4 landmark correspondences")
        w, h = size

        def solve(f):
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
            _, rvs, tvs, _ = cv2.solvePnPGeneric(world, image, K, None, flags=cv2.SOLVEPNP_IPPE)
            best = None
            for rv, tv in zip(rvs, tvs):
                R = cv2.Rodrigues(rv)[0]
                if (-R.T @ tv.ravel())[2] <= 0:  # camera must be above the pitch
                    continue
                e = np.sqrt(np.mean(np.sum((_project(K, rv, tv, world) - image) ** 2, axis=1)))
                if np.isfinite(e) and (best is None or e < best[0]):
                    best = (e, rv.ravel(), tv.ravel())
            return K, best

        if focal is None:
            cands = np.geomspace(0.15 * w, 5 * w, 160)
            errs = [(solve(f)[1] or (np.inf,))[0] for f in cands]
            f0 = cands[int(np.argmin(errs))]
            cands = np.geomspace(f0 * 0.93, f0 * 1.07, 80)
            errs = [(solve(f)[1] or (np.inf,))[0] for f in cands]
            focal = float(cands[int(np.argmin(errs))])
        K, best = solve(focal)
        if best is None:
            raise ValueError("calibration failed: no valid pose (check landmark names / pixel positions)")
        _, rv, tv = best
        rv, tv = cv2.solvePnPRefineLM(world, image, K, None, rv.reshape(3, 1), tv.reshape(3, 1))
        rv, tv = rv.ravel(), tv.ravel()
        rmse = float(np.sqrt(np.mean(np.sum((_project(K, rv, tv, world) - image) ** 2, axis=1))))
        return cls(K, rv, tv, (w, h), rmse)

    # ---- io -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"image_size": list(self.size), "K": self.K.tolist(), "rvec": list(map(float, self.rvec)),
                "tvec": list(map(float, self.tvec)), "rmse_px": self.rmse}

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(np.array(d["K"], float), np.array(d["rvec"], float), np.array(d["tvec"], float),
                   tuple(d["image_size"]), d.get("rmse_px", 0.0))

    def save(self, path, pitch=None):
        d = self.to_dict()
        if pitch is not None:
            d["pitch"] = {"kind": pitch.kind, "length": pitch.length, "width": pitch.width}
        with open(path, "w") as fh:
            json.dump(d, fh, indent=2)

    @classmethod
    def load(cls, path) -> "Camera":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
