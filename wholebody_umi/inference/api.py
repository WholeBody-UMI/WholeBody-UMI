"""Public API for whole-body motion generation."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from wholebody_umi.checkpoint import load_model
from wholebody_umi.config import CONFIG
from wholebody_umi.inference.generator import generate_clip
from wholebody_umi.io import validate_trajectory


def generate_motion(
    checkpoint: Union[str, Path],
    keypoints: np.ndarray,
    initial_features: Optional[np.ndarray] = None,
    *,
    device: str = "cpu",
    init: str = "auto",
    ode_steps: int = CONFIG.ode_steps,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return world-frame qpos, motion features, and the frame rate.

    Motion features have shape (T, CONFIG.motion_dim).

    Palm input is (T,2,7), left then right, xyz in metres plus wxyz rotation.
    Optional initial_features is the aligned raw (T,70) reference representation.
    See wholebody_umi.motion.layout for the reference and model feature layouts.
    """
    if init not in ("auto", "reference", "standing"):
        raise ValueError("init must be auto, reference or standing")
    if not isinstance(ode_steps, int) or isinstance(ode_steps, bool) or ode_steps < 1:
        raise ValueError("ode_steps must be a positive integer")
    keypoints, initial_features = validate_trajectory(keypoints, initial_features)
    if len(keypoints) <= CONFIG.history_length:
        raise ValueError("Input needs more than 8 history frames")
    reference_init = init == "reference" or (init == "auto" and initial_features is not None)
    if reference_init and initial_features is None:
        raise ValueError("Reference initialization requires an aligned feat array")
    model = load_model(checkpoint, device)
    return generate_clip(model, keypoints, initial_features, reference_init, device, ode_steps)
