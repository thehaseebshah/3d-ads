"""Image-to-3D teaser generation on Modal: TRELLIS.2, Hunyuan3D-2.1, TripoSR.

    modal run gen3d/modal_app.py::trellis2   --image path/to/img.png
    modal run gen3d/modal_app.py::hunyuan    --image path/to/img.png
    modal run gen3d/modal_app.py::triposr    --image path/to/img.png
    modal run gen3d/modal_app.py::all        --image path/to/img.png

Results land in ``gen3d_out/<model>/`` (teaser gif/mp4, mesh/glb, intermediates).
First run of each model builds a CUDA image (10-40 min, cached afterwards) and downloads
weights into the ``gen3d-hf-cache`` volume (cached afterwards).

Secrets: TRELLIS.2 needs the gated ``facebook/dinov3-vitl16-pretrain-lvd1689m`` weights ->
accept the licence on Hugging Face, then ``modal secret create huggingface HF_TOKEN=hf_...``.
"""

from __future__ import annotations

import io
import os
import pathlib

import modal

APP_NAME = "gen3d-teasers"
app = modal.App(APP_NAME)

HF_CACHE = modal.Volume.from_name("gen3d-hf-cache", create_if_missing=True)
CACHE_DIR = "/cache"
SECRETS = [modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])]

CUDA_BASE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
# No GPU is visible while building the image, so tell torch which archs to compile for.
BUILD_ENV = {
    "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
    "CUDA_HOME": "/usr/local/cuda",
    "MAX_JOBS": "8",
    "HF_HOME": CACHE_DIR,
    "PYTHONUNBUFFERED": "1",
}
SYS_PKGS = [
    "git", "wget", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libegl1",
    "libjpeg-dev", "libxi6", "libxrender1", "libsm6", "libxxf86vm1", "libxfixes3",
    "libosmesa6-dev", "freeglut3-dev", "ffmpeg",
]


def _base(py: str) -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_BASE, add_python=py)
        .apt_install(*SYS_PKGS)
        .env(BUILD_ENV)
        .pip_install("wheel", "setuptools", "packaging", "imageio[ffmpeg]", "pillow")
    )


# --------------------------------------------------------------------------- TripoSR
triposr_image = (
    _base("3.10")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
    .run_commands("git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR")
    # torchmcubes compiles CUDA kernels, so it needs torch present and nvcc (devel image).
    .pip_install("setuptools", "cmake", "numpy<2")
    .run_commands(
        "pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git",
        "cd /opt/TripoSR && grep -v torchmcubes requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt 'onnxruntime' 'numpy<2'",
    )
)


@app.function(image=triposr_image, gpu="A10G", timeout=1800, volumes={CACHE_DIR: HF_CACHE})
def triposr_infer(image_bytes: bytes, n_views: int = 36, mc_resolution: int = 256) -> dict[str, bytes]:
    import sys

    sys.path.insert(0, "/opt/TripoSR")
    import imageio
    import numpy as np
    import rembg
    import torch
    from PIL import Image
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    dev = "cuda:0"
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(dev)

    img = remove_background(Image.open(io.BytesIO(image_bytes)).convert("RGB"), rembg.new_session())
    img = resize_foreground(img, 0.85)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    img = Image.fromarray((arr * 255).astype(np.uint8))

    with torch.no_grad():
        codes = model([img], device=dev)
        frames = model.render(codes, n_views=n_views, height=384, width=384, return_type="pil")[0]
        mesh = model.extract_mesh(codes, True, resolution=mc_resolution)[0]

    out: dict[str, bytes] = {}
    buf = io.BytesIO()
    imageio.mimsave(buf, [np.array(f) for f in frames], format="GIF", duration=1000 / 20, loop=0)
    out["teaser.gif"] = buf.getvalue()
    for name, fmt in (("mesh.glb", "glb"), ("mesh.obj", "obj")):
        out[name] = mesh.export(file_type=fmt)
        if isinstance(out[name], str):
            out[name] = out[name].encode()
    b = io.BytesIO()
    img.save(b, "PNG")
    out["input_preprocessed.png"] = b.getvalue()
    return out


# --------------------------------------------------------------------------- TRELLIS.2
trellis_image = (
    _base("3.10")
    .env({"ATTN_BACKEND": "flash-attn", "OPENCV_IO_ENABLE_OPENEXR": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    .run_commands("git clone --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2")
    # Mirrors setup.sh --basic (conda/sudo bits dropped; libjpeg-dev is apt-installed above).
    .pip_install(
        "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless", "ninja", "trimesh",
        "transformers", "tensorboard", "pandas", "lpips", "zstandard", "kornia", "timm", "pillow-simd",
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
    )
    .run_commands("pip install flash-attn==2.7.3 --no-build-isolation")
    .run_commands(
        "mkdir -p /tmp/ext && cd /tmp/ext"
        " && git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git && pip install ./nvdiffrast --no-build-isolation"
        " && git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git && pip install ./nvdiffrec --no-build-isolation"
        " && git clone --recursive https://github.com/JeffreyXiang/CuMesh.git && pip install ./CuMesh --no-build-isolation"
        " && git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git && pip install ./FlexGEMM --no-build-isolation"
        " && cp -r /opt/TRELLIS.2/o-voxel /tmp/ext/o-voxel && pip install ./o-voxel --no-build-isolation"
    )
)


@app.function(image=trellis_image, gpu="A100-80GB", timeout=3600, volumes={CACHE_DIR: HF_CACHE}, secrets=SECRETS)
def trellis2_infer(image_bytes: bytes, num_frames: int = 90, resolution: int = 768) -> dict[str, bytes]:
    os.chdir("/opt/TRELLIS.2")
    import sys

    sys.path.insert(0, "/opt/TRELLIS.2")
    import cv2
    import imageio
    import numpy as np
    import o_voxel
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils

    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device="cuda"))
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    mesh = pipeline.run(Image.open(io.BytesIO(image_bytes)))[0]
    mesh.simplify(16777216)  # nvdiffrast limit

    frames = render_utils.make_pbr_vis_frames(
        render_utils.render_video(mesh, envmap=envmap, resolution=resolution, num_frames=num_frames),
        resolution=resolution)
    out: dict[str, bytes] = {}
    tmp = pathlib.Path("/tmp/trellis_out"); tmp.mkdir(exist_ok=True)
    imageio.mimsave(tmp / "teaser.mp4", frames, fps=15)
    # Shaded-only GIF (first panel of the PBR visualisation grid is the shaded view).
    small = [cv2.resize(np.asarray(f), (480, int(480 * np.asarray(f).shape[0] / np.asarray(f).shape[1]))) for f in frames]
    imageio.mimsave(tmp / "teaser.gif", small, duration=1000 / 15, loop=0)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=500000, texture_size=2048, remesh=True, remesh_band=1, remesh_project=0, verbose=True)
    glb.export(str(tmp / "asset.glb"), extension_webp=True)
    for f in tmp.iterdir():
        out[f.name] = f.read_bytes()
    return out


# --------------------------------------------------------------------------- Hunyuan3D-2.1
hunyuan_image = (
    _base("3.10")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .run_commands("git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /opt/Hunyuan3D-2.1")
    .run_commands(
        # Drop the China mirror index lines and the nightly tensorboard pin; keep everything else as upstream.
        "cd /opt/Hunyuan3D-2.1 && grep -v -e '^--extra-index-url' -e '^tb_nightly' requirements.txt > /tmp/req.txt"
        " && pip install -r /tmp/req.txt tensorboard pyrender",
        "cd /opt/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && pip install -e . --no-build-isolation",
        "cd /opt/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh",
        "mkdir -p /opt/Hunyuan3D-2.1/hy3dpaint/ckpt && wget -q "
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth "
        "-P /opt/Hunyuan3D-2.1/hy3dpaint/ckpt",
    )
)


@app.function(image=hunyuan_image, gpu="A100-80GB", timeout=3600, volumes={CACHE_DIR: HF_CACHE}, secrets=SECRETS)
def hunyuan_infer(image_bytes: bytes, max_num_view: int = 6, resolution: int = 512, n_frames: int = 60) -> dict[str, bytes]:
    root = "/opt/Hunyuan3D-2.1"
    os.chdir(root)
    import sys

    sys.path[:0] = [f"{root}/hy3dshape", f"{root}/hy3dpaint", root]
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    import imageio
    import numpy as np
    from PIL import Image

    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except Exception as e:  # upstream demo treats this as optional
        print("torchvision_fix skipped:", e)
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    work = pathlib.Path("/tmp/hy_out"); work.mkdir(exist_ok=True)
    in_png = work / "input.png"
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = BackgroundRemover()(img)  # -> RGBA
    img.save(in_png)

    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    mesh = shape(image=img.convert("RGBA"))[0]
    white_glb = work / "shape.glb"
    mesh.export(str(white_glb))
    del shape

    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.realesrgan_ckpt_path = f"{root}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = f"{root}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = f"{root}/hy3dpaint/hunyuanpaintpbr"
    paint = Hunyuan3DPaintPipeline(conf)
    textured = paint(mesh_path=str(white_glb), image_path=str(in_png), output_mesh_path=str(work / "textured.glb"))

    out: dict[str, bytes] = {}
    # Turntables: untextured shape vs PBR-textured glb (pyrender honours metallic-roughness).
    try:
        for tag, path in (("shape", white_glb), ("pbr", textured)):
            frames = _turntable(str(path), n_frames)
            imageio.mimsave(work / f"turntable_{tag}.gif", frames, duration=1000 / 20, loop=0)
        # Side-by-side teaser: white shape | PBR textured.
        a = imageio.mimread(work / "turntable_shape.gif"); b = imageio.mimread(work / "turntable_pbr.gif")
        both = [np.concatenate([np.asarray(x)[..., :3], np.asarray(y)[..., :3]], axis=1) for x, y in zip(a, b)]
        imageio.mimsave(work / "teaser.gif", both, duration=1000 / 20, loop=0)
    except Exception as e:
        print("turntable render failed (mesh/texture outputs are still returned):", repr(e))
    for f in work.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".glb", ".obj", ".gif", ".png", ".jpg", ".mtl"}:
            out[str(f.relative_to(work))] = f.read_bytes()
    return out


def _turntable(glb_path: str, n: int, size: int = 512):
    import numpy as np
    import pyrender
    import trimesh

    scene_tm = trimesh.load(glb_path, force="scene")
    scene = pyrender.Scene.from_trimesh_scene(scene_tm, ambient_light=[0.35] * 3, bg_color=[1, 1, 1, 1])
    lo, hi = scene_tm.bounds
    centre, radius = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4)
    cam_node = scene.add(cam, pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=np.eye(4))
    r = pyrender.OffscreenRenderer(size, size)
    frames = []
    for i in range(n):
        th = 2 * np.pi * i / n
        eye = centre + radius * 2.6 * np.array([np.sin(th), 0.25, np.cos(th)])
        z = (eye - centre) / np.linalg.norm(eye - centre)
        x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
        scene.set_pose(cam_node, pose)
        frames.append(r.render(scene)[0])
    r.delete()
    return frames


# --------------------------------------------------------------------------- local entrypoints
def _save(model: str, files: dict[str, bytes]) -> None:
    out = pathlib.Path("gen3d_out") / model
    for name, data in files.items():
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    print(f"[{model}] wrote {len(files)} files to {out}/: {sorted(files)}")


@app.local_entrypoint()
def triposr(image: str, n_views: int = 36):
    """TripoSR: animated GIF of the image-to-3D result."""
    _save("triposr", triposr_infer.remote(pathlib.Path(image).read_bytes(), n_views))


@app.local_entrypoint()
def trellis2(image: str, num_frames: int = 90):
    """TRELLIS.2: PBR turntable teaser (mp4 + gif) and textured GLB."""
    _save("trellis2", trellis2_infer.remote(pathlib.Path(image).read_bytes(), num_frames))


@app.local_entrypoint()
def hunyuan(image: str, resolution: int = 512):
    """Hunyuan3D-2.1: shape-vs-PBR-textured turntable teaser plus textured GLB/maps."""
    _save("hunyuan3d21", hunyuan_infer.remote(pathlib.Path(image).read_bytes(), resolution=resolution))


@app.local_entrypoint()
def all(image: str):
    """Run all three models in parallel on the same input image."""
    data = pathlib.Path(image).read_bytes()
    calls = {
        "triposr": triposr_infer.spawn(data),
        "trellis2": trellis2_infer.spawn(data),
        "hunyuan3d21": hunyuan_infer.spawn(data),
    }
    for name, call in calls.items():
        try:
            _save(name, call.get())
        except Exception as e:  # keep the other models' results if one fails
            print(f"[{name}] FAILED: {e!r}")
