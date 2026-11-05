import cv2
import numpy as np
import pytest

from pitchads.ads import AdSpec
from pitchads.camera import Camera
from pitchads.compositor import Compositor
from pitchads.pitch import get_pitch
from pitchads.synthetic import make_camera


@pytest.mark.parametrize("kind", ["football", "futsal"])
def test_calibration_recovers_camera(kind):
    spec = get_pitch(kind)
    truth = make_camera(spec)
    rng = np.random.default_rng(0)
    names = list(spec.landmarks)
    world = np.array([[*spec.landmarks[n], 0.0] for n in names])
    px = truth.project(world)
    ok = (px[:, 0] > 0) & (px[:, 0] < truth.size[0]) & (px[:, 1] > 0) & (px[:, 1] < truth.size[1])
    cam = Camera.fit(world[ok, :2], px[ok] + rng.normal(0, 0.5, px[ok].shape), truth.size)
    assert abs(cam.focal - truth.focal) / truth.focal < 0.03
    assert np.linalg.norm(cam.center - truth.center) < 1.0
    assert cam.rmse < 1.5


def test_image_to_ground_roundtrip():
    cam = make_camera(get_pitch("football"))
    pts = np.array([[20.0, 10.0], [80.0, 50.0]])
    uv = cam.project(np.column_stack([pts, [0, 0]]))
    assert np.allclose(cam.image_to_ground(uv), pts, atol=1e-6)


def _scene():
    cam = make_camera(get_pitch("football"))
    plate = np.full((cam.size[1], cam.size[0], 3), (50, 120, 55), np.uint8)
    tex = np.full((100, 300, 4), 255, np.uint8)
    tex[..., :3] = (0, 0, 255)
    ad = AdSpec("a", "ground", tex, (52.5, 20), (20, 6), opacity=1.0, keep_lines=0.0)
    return cam, plate, ad


def test_ground_ad_is_painted_and_player_occludes_it():
    cam, plate, ad = _scene()
    comp = Compositor(cam, [ad], plate)
    cx, cy = cam.project([[52.5, 20, 0]])[0].astype(int)
    out = comp.render(plate.copy(), 1.0)
    assert out[cy, cx, 2] > 200 and out[cy, cx, 0] < 60  # red paint over green

    frame = plate.copy()
    cv2.rectangle(frame, (cx - 4, cy - 14), (cx + 4, cy + 2), (230, 200, 40), -1)  # "player"
    out = comp.render(frame, 1.0)
    assert tuple(out[cy - 6, cx]) == (230, 200, 40)  # player pixels untouched


def test_ad_fades_by_time():
    cam, plate, ad = _scene()
    ad.start, ad.end = 2.0, 4.0
    comp = Compositor(cam, [ad], plate)
    cx, cy = cam.project([[52.5, 20, 0]])[0].astype(int)
    assert np.array_equal(comp.render(plate.copy(), 1.0)[cy, cx], plate[cy, cx])
    assert comp.render(plate.copy(), 3.0)[cy, cx, 2] > 200
