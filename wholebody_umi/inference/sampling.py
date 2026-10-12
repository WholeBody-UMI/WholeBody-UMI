"""Zero-noise, shared-time ODE sampling of one 40-frame primitive."""

import torch

from wholebody_umi.config import CONFIG
from wholebody_umi.motion.layout import FOOT_CONTACTS


@torch.no_grad()
def sample_primitive(
    model: torch.nn.Module,
    condition: torch.Tensor,
    history: torch.Tensor,
    lookahead: torch.Tensor,
    lookahead_valid: torch.Tensor,
    ode_steps: int,
) -> torch.Tensor:
    known = torch.zeros(
        1,
        CONFIG.window_length,
        CONFIG.motion_dim,
        device=condition.device,
        dtype=condition.dtype,
    )
    observed_mask = torch.zeros_like(known)
    known[:, : CONFIG.history_length] = history
    observed_mask[:, : CONFIG.history_length] = 1.0
    motion = observed_mask * known + (1.0 - observed_mask) * torch.zeros_like(known)
    dt = 1.0 / ode_steps
    for step in range(ode_steps):
        base = step / ode_steps
        next_time = min(base + dt, 1.0)
        time = torch.full((1,), base, device=condition.device, dtype=condition.dtype)
        prediction = model(
            motion, known, observed_mask, condition, time, lookahead, lookahead_valid
        )
        initial_estimate = (motion - base * prediction) / max(1.0 - base, 1e-4)
        motion = (1.0 - next_time) * initial_estimate + next_time * prediction
        motion = observed_mask * known + (1.0 - observed_mask) * motion
    motion[..., FOOT_CONTACTS] = motion[..., FOOT_CONTACTS].clamp(0.0, 1.0)
    return motion
