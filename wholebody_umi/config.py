"""Model architecture and motion rollout settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionPriorConfig:
    motion_dim: int = 73
    condition_dim: int = 50
    hidden_dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    ffn_mult: int = 4
    time_emb_dim: int = 256
    fps: int = 30
    history_length: int = 8
    future_length: int = 32
    window_length: int = 40
    lookahead_length: int = 32
    lookahead_stride: int = 4
    overlap: int = 8
    ode_steps: int = 32


CONFIG = MotionPriorConfig()
