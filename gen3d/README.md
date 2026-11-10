# gen3d — image-to-3D teasers on Modal

| Model | Output | GPU |
|---|---|---|
| TRELLIS.2 (`microsoft/TRELLIS.2-4B`) | PBR turntable teaser (mp4 + gif) + textured GLB | A100-80GB |
| Hunyuan3D-2.1 | shape-vs-PBR turntable teaser + textured GLB / texture maps | A100-80GB |
| TripoSR | animated GIF + mesh | A10G |

## One-time setup (on your Mac)

```bash
pip install modal
modal setup                      # browser login, creates ~/.modal.toml
```

TRELLIS.2 and Hunyuan3D pull gated Hugging Face weights (DINOv3). Accept the licence for
`facebook/dinov3-vitl16-pretrain-lvd1689m` on huggingface.co, then:

```bash
modal secret create huggingface HF_TOKEN=hf_xxx
```

## Run

```bash
modal run gen3d/modal_app.py::triposr  --image my.png     # cheapest; try this first
modal run gen3d/modal_app.py::trellis2 --image my.png
modal run gen3d/modal_app.py::hunyuan  --image my.png
modal run gen3d/modal_app.py::all      --image my.png     # all three in parallel
```

Outputs go to `gen3d_out/<model>/` (git-ignored). The first run of each model builds a CUDA image
(10-40 min; flash-attn / nvdiffrast / custom_rasterizer compile) and downloads weights into the
`gen3d-hf-cache` volume; later runs start in about a minute.

## Status

The scripts were written against the upstream repos' current READMEs/demo scripts but **have not been
executed yet** (no Modal credentials or GPU in the authoring environment). Expect to iterate on the first
image build per model — most likely spots: flash-attn / torchmcubes compilation, the Hunyuan
pyrender (OSMesa) turntable, and TRELLIS.2 frame layout in the gif.
Input: a single object on a plain or removable background works best.
