"""Temporal DiT blocks and transformer."""

from torch import Tensor, nn

from wholebody_umi.models.cross_attention import (
    NormalizedTemporalCrossAttention,
    gate_residual,
    modulate,
)


class TemporalDiTBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.n_heads,
            batch_first=True,
        )
        self.context_attention = NormalizedTemporalCrossAttention(config)
        self.norm_mlp = nn.LayerNorm(config.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * config.ffn_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_dim * config.ffn_mult, config.hidden_dim),
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_dim, 9 * config.hidden_dim))

    def forward(self, body_tokens, context_tokens, condition, padding_mask) -> Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            context_shift,
            context_scale,
            context_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.ada_ln(condition).chunk(9, dim=-1)
        query = modulate(self.norm_attention(body_tokens), attention_shift, attention_scale)
        output, _ = self.attention(query, query, query, need_weights=False)
        body_tokens = body_tokens + gate_residual(output, attention_gate)
        body_tokens = self.context_attention(
            body_tokens,
            context_tokens,
            context_shift,
            context_scale,
            context_gate,
            padding_mask,
        )
        mlp_input = modulate(self.norm_mlp(body_tokens), mlp_shift, mlp_scale)
        return body_tokens + gate_residual(self.mlp(mlp_input), mlp_gate)


class TemporalDiTTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TemporalDiTBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_ada_ln = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_dim, 2 * config.hidden_dim)
        )

    def forward(self, body_tokens, context_tokens, condition, padding_mask) -> Tensor:
        hidden = body_tokens
        for block in self.blocks:
            hidden = block(hidden, context_tokens, condition, padding_mask)
        shift, scale = self.final_ada_ln(condition).chunk(2, dim=-1)
        return modulate(self.final_norm(hidden), shift, scale)
