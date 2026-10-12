"""Normalized cross-attention for hand-window and lookahead context."""

from torch import Tensor, nn


def modulate(tokens: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return tokens * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def gate_residual(residual: Tensor, gate: Tensor) -> Tensor:
    return residual * gate.unsqueeze(1)


class NormalizedTemporalCrossAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(config.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.n_heads,
            batch_first=True,
        )

    def forward(self, body_tokens, context_tokens, shift, scale, gate, padding_mask) -> Tensor:
        query = modulate(self.norm_query(body_tokens), shift, scale)
        context = self.norm_context(context_tokens)
        output, _ = self.attention(
            query,
            context,
            context,
            need_weights=False,
            key_padding_mask=padding_mask,
        )
        return body_tokens + gate_residual(output, gate)
