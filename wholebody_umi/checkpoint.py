"""Load the exported inference weights."""

from pathlib import Path
from typing import Union

import torch

from wholebody_umi.models import MotionPrior

WEIGHT_FORMAT = "wholebody-umi-emft-ep0355-ema-v1"


def load_model(checkpoint: Union[str, Path], device: str = "cpu") -> MotionPrior:
    checkpoint = Path(checkpoint).expanduser()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != WEIGHT_FORMAT:
        raise ValueError("Use the exported WholeBody-UMI inference weights")
    model = MotionPrior()
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval()
