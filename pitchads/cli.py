"""Command line: pitchads {landmarks,calibrate,render,demo}."""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

from .ads import load_ads
from .background import build_plate
from .camera import Camera
from .compositor import Compositor, render_video
from .pitch import get_pitch


def draw_pitch_overlay(frame, cam: Camera, spec, color=(0, 255, 255)):
    out = frame.copy()
    for pl in spec.polylines():
        px = cam.project(np.column_stack([pl, np.zeros(len(pl))]))
        ok = np.isfinite(px).all(axis=1)
        if ok.sum() > 1:
            cv2.polylines(out, [np.round(px[ok]).astype(np.int32)], False, color, 1, cv2.LINE_AA)
    return out


def cmd_landmarks(a):
    spec = get_pitch(a.pitch, a.length, a.width)
    for n, (x, y) in spec.landmarks.items():
        print(f"{n:28s} x={x:7.2f} y={y:6.2f}")


def cmd_calibrate(a):
    with open(a.landmarks) as fh:
        d = json.load(fh)
    spec = get_pitch(d.get("pitch", "football"), d.get("length"), d.get("width"))
    names = [n for n in d["landmarks"] if n in spec.landmarks]
    unknown = set(d["landmarks"]) - set(names)
    if unknown:
        print("ignoring unknown landmarks:", ", ".join(sorted(unknown)), file=sys.stderr)
    world = [spec.landmarks[n] for n in names]
    img = [d["landmarks"][n] for n in names]
    cam = Camera.fit(world, img, tuple(d["image_size"]), a.focal)
    cam.save(a.out, spec)
    print(f"calibrated from {len(names)} landmarks: focal={cam.focal:.0f}px  reprojection RMSE={cam.rmse:.2f}px")
    print(f"camera position (m): {np.round(cam.center, 1).tolist()}")
    if len(names) < 6:
        print("warning: fewer than 6 landmarks; focal length may be unreliable", file=sys.stderr)
    if a.video:
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = cap.read()
        if ok:
            p = os.path.splitext(a.out)[0] + "_check.png"
            cv2.imwrite(p, draw_pitch_overlay(fr, cam, spec))
            print("wrote overlay for visual check:", p)


def cmd_render(a):
    cam = Camera.load(a.camera)
    ads = load_ads(a.ads)
    render_video(a.video, cam, ads, a.out, a.bg_frames, a.threshold,
                 progress=lambda i: print(f"  {i} frames", end="\r"))
    print(f"\nwrote {a.out}")


def cmd_demo(a):
    from .synthetic import generate, make_demo_ads

    out = a.out
    spec = get_pitch(a.pitch)
    print("1/4 generating synthetic panoramic match ...")
    video = generate(out, a.pitch, a.seconds)
    ads_json = make_demo_ads(out, spec)
    print("2/4 calibrating camera from landmark clicks ...")
    ns = argparse.Namespace(landmarks=os.path.join(out, "landmarks.json"), out=os.path.join(out, "camera.json"),
                            focal=None, video=video)
    cmd_calibrate(ns)
    truth = Camera.load(os.path.join(out, "truth_camera.json"))
    est = Camera.load(os.path.join(out, "camera.json"))
    print(f"    true focal={truth.focal:.0f}px  estimated={est.focal:.0f}px")
    print("3/4 rendering ads into the video ...")
    ads = load_ads(ads_json)
    render_video(video, est, ads, os.path.join(out, "output.mp4"), progress=lambda i: print(f"  {i} frames", end="\r"))
    print("\n4/4 writing before/after stills ...")
    plate = build_plate(video)
    comp = Compositor(est, ads, plate)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for fi in (60, 150):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            cv2.imwrite(os.path.join(out, f"before_{fi}.png"), fr)
            cv2.imwrite(os.path.join(out, f"after_{fi}.png"), comp.render(fr, fi / fps))
    print("done ->", out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="pitchads", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("landmarks", help="list named pitch landmarks (metres)")
    s.add_argument("--pitch", default="football", choices=["football", "futsal"])
    s.add_argument("--length", type=float)
    s.add_argument("--width", type=float)
    s.set_defaults(fn=cmd_landmarks)

    s = sub.add_parser("calibrate", help="solve the camera from landmark pixel positions")
    s.add_argument("landmarks", help='JSON: {"pitch","image_size":[w,h],"landmarks":{name:[u,v]}}')
    s.add_argument("--out", default="camera.json")
    s.add_argument("--focal", type=float, help="focal length in px if known (otherwise estimated)")
    s.add_argument("--video", help="write a pitch-line overlay on its first frame to verify the fit")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("render", help="composite ads into a video")
    s.add_argument("video")
    s.add_argument("--camera", required=True)
    s.add_argument("--ads", required=True)
    s.add_argument("--out", default="output.mp4")
    s.add_argument("--bg-frames", type=int, default=30)
    s.add_argument("--threshold", type=float, default=12.0, help="foreground sensitivity (lower = more sensitive)")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("demo", help="end-to-end run on synthetic footage")
    s.add_argument("--out", default="demo_out")
    s.add_argument("--pitch", default="football", choices=["football", "futsal"])
    s.add_argument("--seconds", type=float, default=10.0)
    s.set_defaults(fn=cmd_demo)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
