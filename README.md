# Whole-Body UMI: Transferring UMI Manipulation Skills to Humanoid Whole-Body Manipulation via Real-Time Motion Generation

**Yuxuan Nai, Leixin Chang, Liangjing Yang, Shuo Yang, Zhongyu Li**

[Project Website](https://wholebody-umi.github.io/) | [arXiv:2609.22829](https://arxiv.org/abs/2609.22829) | [Model Weights](https://huggingface.co/EasonNai/WholeBody-UMI) | [Citation](#citation)

WholeBody-UMI generates whole-body motion for the Unitree G1 humanoid from
bimanual palm trajectories. This repository provides model inference and
MuJoCo visualization at 30 fps, using single-primitive rollout with overlapping
segments.

## Open-source status

- [x] Inference
- [ ] Training
- [ ] Deploy

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git LFS,
then clone the repository, download the G1 assets, and set up the environment:

```bash
git clone https://github.com/WholeBody-UMI/WholeBody-UMI.git
cd WholeBody-UMI
git lfs install
git lfs pull
uv sync --locked --extra visualization
```

The Python version and dependencies are pinned by `.python-version` and `uv.lock`.

## Model weights

Download `wbumi.pt` from [Hugging Face](https://huggingface.co/EasonNai/WholeBody-UMI):

```bash
uvx hf download EasonNai/WholeBody-UMI wbumi.pt --local-dir checkpoints
```

Both commands below use `checkpoints/wbumi.pt` by default.

## Generate motion

```bash
uv run --locked wholebody-umi-generate \
  --input examples/umi_walk_pnp_bottle.npz \
  --output outputs/motion.npz \
  --device cuda:0
```

Use `--device cpu` for CPU inference and `--checkpoint PATH` for another weight location.

## Visualization

```bash
uv run --locked --extra visualization wholebody-umi-demo \
  --input examples/umi_walk_pnp_bottle.npz \
  --device cuda:0
```

The MuJoCo window loops the generated motion; close it to exit.
Add `--video outputs/demo.mp4` to record a video. Headless recording requires
`MUJOCO_GL=egl` and a working EGL driver.
Playback displays generated poses kinematically, without IK refinement or a dynamics controller.

## Input and output

Input is a `.npy` array or an `.npz` bundle containing `keypoints` with shape
`(T, 2, 7)`. Hands are ordered left, right; each pose is
`[x, y, z, qw, qx, qy, qz]`, with positions in metres and unit wxyz quaternions.
Trajectories must be sampled at 30 fps and contain more than 8 frames.
An optional aligned `feat` array provides reference initialization; otherwise,
the model uses a standing seed. Hand-only trajectories must be aligned with
the standing seed's coordinate frame and initial pose.

The output `.npz` contains `qpos`, `motion_features`, `keypoints`, and `fps`.
`qpos` stores the root pose and G1 joint angles for playback.

## Citation

If you use Whole-Body UMI in your research, please cite our paper:

```bibtex
@misc{nai2026wholebodyumi,
  title         = {{Whole-Body UMI}: Transferring {UMI} Manipulation Skills to Humanoid
                   Whole-Body Manipulation via Real-Time Motion Generation},
  author        = {Yuxuan Nai and Leixin Chang and Liangjing Yang and
                   Shuo Yang and Zhongyu Li},
  year          = {2026},
  eprint        = {2609.22829},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.22829}
}
```
