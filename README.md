# pitchads — virtual 3D ads for football & futsal video

Adds ads that look painted on the grass (or standing on it as 3D blocks / boards) to **recorded match
footage from a fixed panoramic camera**. Ads keep correct perspective, sit behind players, pick up the
grass lighting, and can fade in/out on a timeline.

## Quick start

```bash
pip install -e ".[dev]"
python -m pitchads demo            # synthetic football match -> demo_out/
python -m pitchads demo --pitch futsal
pytest
```

`demo_out/` contains `input.mp4`, `output.mp4`, `before_*/after_*.png`, and `camera_check.png`
(pitch model reprojected on the video — use it to verify a calibration).

## Using your own footage

1. **Calibrate (once per camera position).** Open one frame, find pitch landmarks and write their pixel
   positions to JSON (`python -m pitchads landmarks` lists the names and metric positions):
   ```json
   {"pitch": "football", "image_size": [3840, 1080],
    "landmarks": {"corner_near_left": [212, 1004], "centre": [1920, 610], "halfway_far": [1925, 288]}}
   ```
   Use 6+ well-spread points (corners, halfway line, penalty boxes). Then:
   ```bash
   python -m pitchads calibrate landmarks.json --out camera.json --video match.mp4
   ```
   Check the printed RMSE (aim < 2 px) and `camera_check.png`.
2. **Describe the ads** in `ads.json` (positions in metres, see `pitchads/ads.py` for fields):
   ```json
   {"ads": [{"id": "a", "type": "ground", "image": "a.png", "center": [30, 34], "size": [18, 6]},
            {"id": "b", "type": "box", "image": "b.png", "center": [52, 60], "size": [7, 3], "height": 2.2,
             "start": 3, "end": 8}]}
   ```
   Types: `ground`, `box`, `billboard`. Texture left→right = +x, top = far side; use `rotation_deg: 180`
   for ads meant to be read from the far touchline.
3. **Render:** `python -m pitchads render match.mp4 --camera camera.json --ads ads.json --out out.mp4`

## How it works

| Stage | Module | Idea |
|---|---|---|
| Calibration | `camera.py` | Planar PnP (IPPE) over landmark correspondences, focal length found by 1-D search. Gives a full 3D camera, so any 3D ad can be projected. |
| Placement | `ads.py` | Ads live in pitch metres; faces are perspective-warped (2x supersampled) into premultiplied layers, cached because the camera is fixed. |
| Occlusion | `background.py`, `compositor.py` | Median "empty pitch" plate; Lab distance with down-weighted lightness (ignores shadows). Ground ads sit under *everything* standing on the grass; 3D ads are only occluded by people whose feet are nearer the camera than the object. |
| Integration | `compositor.py` | Grass lighting/mowing stripes modulate ground ads, pitch lines show through, soft edges, timeline fades. |

## Known limits / next steps

- **No lens distortion model** — stitched panoramas with strong curvature need rectifying first (or add
  distortion to `Camera.project` and mesh-warp faces).
- Foreground is colour-difference vs. a median plate: fine for a static camera, but it will struggle with
  changing light, heavy shadows, or players standing still the whole clip. Replace with a person
  segmentation model (e.g. YOLO-seg / SAM) for production quality.
- 3D occlusion uses each person's foot point vs. the object's centre depth (merged player blobs are approximated).
- Calibration is manual landmarks; automatic pitch-line detection would remove that step.
- Moving/PTZ cameras (live broadcast) need per-frame tracking — the `Camera` model already supports it.
