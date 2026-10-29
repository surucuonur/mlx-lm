# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    intermediate_size_mlp: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_local_experts: int
    num_experts_per_tok: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    max_position_embeddings: int
    rope_scaling: Optional[Dict[str, Any]] = None
    interleave_moe_layer_step: int = 1
    no_rope_layers: Optional[List[int]] = None
    use_qk_norm: bool = True
    attn_temperature_tuning: bool = False
    attention_bias: bool = False
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.no_rope_layers is None:
            self.no_rope_layers = [1] * self.num_hidden_layers
        if self.attn_temperature_tuning:
            raise ValueError("MobileMoE: attn_temperature_tuning is not supported.")


class QKNorm(nn.Module):
    """RMS norm without a learned scale."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, use_rope: bool):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        bias = args.attention_bias
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)

        self.use_rope = use_rope
        if use_rope:
            self.rope = initialize_rope(
                self.head_dim,
                args.rope_theta,
                traditional=True,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

        # The reference applies this norm after RoPE. RoPE does not change
        # the vector norm, so applying it before RoPE gives the same result.
        self.use_qk_norm = args.use_qk_norm and use_rope
        if self.use_qk_norm:
            self.q_norm = QKNorm(args.rms_norm_eps)
            self.k_norm = QKNorm(args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        num_experts = args.num_local_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.expert_bias = mx.zeros((num_experts,))
        self.experts = SwitchGLU(dim, args.intermediate_size, num_experts)
        self.shared_expert = MLP(dim, args.intermediate_size_mlp)

    def __call__(self, x: mx.array) -> mx.array:
        # The reference computes the router in float32.
        scores = mx.sigmoid(self.router(x.astype(mx.float32)))

        # The bias changes which experts are selected, not their weights.
        k = self.top_k
        biased = scores + self.expert_bias
        inds = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
        scores = scores * self.routed_scaling_factor

        y = self.experts(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_expert(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, bool(args.no_rope_layers[layer_idx]))

        step = args.interleave_moe_layer_step
        if layer_idx % step == step - 1:
            self.feed_forward = MoE(args)
        else:
            self.feed_forward = MLP(args.hidden_size, args.intermediate_size_mlp)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.feed_forward(self.post_attention_layernorm(h))


class MobileMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MobileMoEModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Split the fused [E, D, 2F] gate/up tensor and use the
        # [E, out, in] layout of SwitchGLU.
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.feed_forward.experts"
            if f"{prefix}.gate_up_proj" in weights:
                gate_up = weights.pop(f"{prefix}.gate_up_proj")
                gate, up = mx.split(gate_up, 2, axis=-1)
                weights[f"{prefix}.gate_proj.weight"] = mx.swapaxes(gate, 1, 2)
                weights[f"{prefix}.up_proj.weight"] = mx.swapaxes(up, 1, 2)
            if f"{prefix}.down_proj" in weights:
                down = weights.pop(f"{prefix}.down_proj")
                weights[f"{prefix}.down_proj.weight"] = mx.swapaxes(down, 1, 2)
        return weights

    @property
    def layers(self):
        return self.model.layers
