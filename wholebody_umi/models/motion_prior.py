"""Motion Prior forward pass with hand-trajectory conditioning."""

import math

import torch
from torch import nn

from wholebody_umi.config import CONFIG
from wholebody_umi.models.temporal_dit import TemporalDiTTransformer
from wholebody_umi.motion.layout import CONDITION_VELOCITY


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb_dim = CONFIG.time_emb_dim
        self.proj = nn.Sequential(
            nn.Linear(self.emb_dim, CONFIG.hidden_dim),
            nn.SiLU(),
            nn.Linear(CONFIG.hidden_dim, CONFIG.hidden_dim),
        )

    def forward(self, time) -> torch.Tensor:
        half = self.emb_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / (half - 1)
        )
        angles = time.unsqueeze(-1) * frequencies
        return self.proj(torch.cat([angles.sin(), angles.cos()], dim=-1))


def _projection(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, CONFIG.hidden_dim),
        nn.SiLU(),
        nn.Linear(CONFIG.hidden_dim, CONFIG.hidden_dim),
    )


class MotionPrior(nn.Module):
    """Fixed DiT model; parameter attribute names match the exported state dict."""

    def __init__(self) -> None:
        super().__init__()
        self.feat_proj = _projection(CONFIG.motion_dim)
        self.mask_proj = nn.Linear(CONFIG.motion_dim, CONFIG.hidden_dim, bias=False)
        self.ee_pos_emb = nn.Parameter(torch.empty(1, CONFIG.window_length, CONFIG.hidden_dim))
        self.ee_proj = _projection(CONFIG.condition_dim)
        self.ee_frame_proj = _projection(CONFIG.condition_dim)
        self.look_proj = _projection(CONFIG.condition_dim)
        self.look_pos_emb = nn.Parameter(
            torch.empty(
                1,
                CONFIG.lookahead_length // CONFIG.lookahead_stride,
                CONFIG.hidden_dim,
            )
        )
        self.time_emb = SinusoidalTimeEmbedding()
        self.pos_emb = nn.Parameter(torch.empty(1, CONFIG.window_length, CONFIG.hidden_dim))
        self.temporal_dit = TemporalDiTTransformer(CONFIG)
        self.out_proj = nn.Sequential(
            nn.Linear(CONFIG.hidden_dim, 2 * CONFIG.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * CONFIG.hidden_dim, CONFIG.motion_dim),
        )

    def forward(
        self, motion, history, observed_mask, condition, time, lookahead, lookahead_valid
    ) -> torch.Tensor:
        observed = observed_mask * history + (1.0 - observed_mask) * motion
        body_tokens = (
            self.feat_proj(observed)
            + self.ee_frame_proj(condition)
            + self.mask_proj(observed_mask)
            + self.pos_emb
        )
        hand_tokens = self.ee_proj(condition) + self.ee_pos_emb
        hold = condition[:, -1:].clone()
        hold[..., CONDITION_VELOCITY] = 0.0
        preview = lookahead[:, :: CONFIG.lookahead_stride]
        valid = lookahead_valid[:, :: CONFIG.lookahead_stride]
        preview = torch.where(valid.unsqueeze(-1).bool(), preview, hold)
        preview_tokens = self.look_proj(preview) + self.look_pos_emb
        padding_mask = torch.cat(
            [
                torch.zeros(
                    valid.shape[0], CONFIG.window_length, device=valid.device, dtype=torch.bool
                ),
                ~valid.bool(),
            ],
            dim=1,
        )
        hidden = self.temporal_dit(
            body_tokens,
            torch.cat([hand_tokens, preview_tokens], dim=1),
            self.time_emb(time),
            padding_mask,
        )
        velocity = self.out_proj(hidden) * (1.0 - observed_mask)
        remaining = torch.clamp(1.0 - time.view(-1, 1, 1), min=1e-4)
        prediction = observed + remaining * velocity
        return observed_mask * history + (1.0 - observed_mask) * prediction
