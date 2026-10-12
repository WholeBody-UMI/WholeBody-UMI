"""Generate G1 whole-body motion from bimanual palm trajectories."""

import argparse
from pathlib import Path

from wholebody_umi.config import CONFIG
from wholebody_umi.inference.api import generate_motion
from wholebody_umi.io import load_trajectory, save_motion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Palm keypoints .npy or .npz")
    parser.add_argument(
        "--checkpoint", default="checkpoints/wbumi.pt", help="Exported inference weights"
    )
    parser.add_argument("--output", required=True, help="Output .npz bundle")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--init", choices=["auto", "reference", "standing"], default="auto")
    parser.add_argument("--ode-steps", type=int, default=CONFIG.ode_steps)
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    try:
        keypoints, reference_features = load_trajectory(args.input)
        qpos, motion_features, fps = generate_motion(
            args.checkpoint,
            keypoints,
            reference_features,
            device=args.device,
            init=args.init,
            ode_steps=args.ode_steps,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    output = save_motion(output, qpos, motion_features, keypoints, fps)
    print(f"Saved {len(qpos)} frames at {fps:g} fps to {output}")


if __name__ == "__main__":
    main()
