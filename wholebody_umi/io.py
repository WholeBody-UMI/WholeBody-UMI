"""Read hand trajectories and save generated whole-body motion."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from wholebody_umi.motion.layout import REFERENCE_FEATURE_DIM


def validate_trajectory(
    keypoints: np.ndarray,
    reference_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate metres/wxyz hand poses and optional raw reference features."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (2, 7):
        raise ValueError(f"Expected keypoints (T, 2, 7), got {keypoints.shape}")
    if not np.isfinite(keypoints).all():
        raise ValueError("Keypoints must contain only finite values")
    norms = np.linalg.norm(keypoints[..., 3:7], axis=-1)
    if not np.allclose(norms, 1.0, atol=0.01, rtol=0):
        raise ValueError("Palm quaternions must be unit quaternions in wxyz order")
    if reference_features is not None:
        reference_features = np.asarray(reference_features, dtype=np.float32)
        expected_shape = (len(keypoints), REFERENCE_FEATURE_DIM)
        if reference_features.shape != expected_shape:
            raise ValueError(
                f"Expected aligned feat {expected_shape}, got {reference_features.shape}"
            )
        if not np.isfinite(reference_features).all():
            raise ValueError("Reference features must contain only finite values")
    return keypoints, reference_features


def load_trajectory(
    path: Union[str, Path],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read hand-only .npy or .npz with keypoints and optional ``feat`` (T, 70)."""
    path = Path(path).expanduser()
    if path.suffix == ".npy":
        return validate_trajectory(np.load(path, allow_pickle=False))
    if path.suffix != ".npz":
        raise ValueError("Input must be a .npy trajectory or .npz bundle")
    with np.load(path, allow_pickle=False) as bundle:
        if "keypoints" not in bundle:
            raise ValueError("Input .npz must contain keypoints (T, 2, 7)")
        reference_features = bundle["feat"] if "feat" in bundle else None
        return validate_trajectory(bundle["keypoints"], reference_features)


def save_motion(
    path: Union[str, Path],
    qpos: np.ndarray,
    motion_features: np.ndarray,
    keypoints: np.ndarray,
    fps: float,
) -> Path:
    """Save the full generated motion and its input in the public .npz format."""
    path = Path(path).expanduser()
    if path.suffix != ".npz":
        raise ValueError("Output must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=qpos,
        motion_features=motion_features,
        keypoints=keypoints,
        fps=np.float32(fps),
    )
    return path
