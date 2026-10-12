"""Generate whole-body motion and visualize it in MuJoCo."""

import argparse
from pathlib import Path

from wholebody_umi.config import CONFIG
from wholebody_umi.inference.api import generate_motion
from wholebody_umi.io import load_trajectory, save_motion
from wholebody_umi.visualization import visualize_motion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Palm keypoints .npy or .npz")
    parser.add_argument(
        "--checkpoint", default="checkpoints/wbumi.pt", help="Exported inference weights"
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--init", choices=["auto", "reference", "standing"], default="auto")
    parser.add_argument("--ode-steps", type=int, default=CONFIG.ode_steps)
    parser.add_argument(
        "--video",
        type=Path,
        nargs="?",
        const=Path("outputs/demo.mp4"),
        metavar="PATH.mp4",
        help="Record one pass to MP4 (default path: outputs/demo.mp4)",
    )
    parser.add_argument("--output", type=Path, help="Also save full generated motion to .npz")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels")
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=CONFIG.history_length,
        help="Hide initial frames during playback/recording (default: 8)",
    )
    args = parser.parse_args()
    if args.video is not None and args.video.suffix.lower() != ".mp4":
        parser.error("--video must end in .mp4")
    if args.output is not None and args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even integers")
    try:
        keypoints, reference_features = load_trajectory(args.input)
        if not 0 <= args.skip_frames < len(keypoints):
            raise ValueError("--skip-frames must leave at least one frame to display")
        print(f"Generating {len(keypoints)} frames on {args.device}...", flush=True)
        qpos, motion_features, fps = generate_motion(
            args.checkpoint,
            keypoints,
            reference_features,
            device=args.device,
            init=args.init,
            ode_steps=args.ode_steps,
        )
        if args.output is not None:
            output = save_motion(args.output, qpos, motion_features, keypoints, fps)
            print(f"Saved {len(qpos)} frames at {fps:g} fps to {output}")
        visualize_motion(
            qpos,
            fps,
            video=args.video,
            width=args.width,
            height=args.height,
            skip_frames=args.skip_frames,
        )
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
