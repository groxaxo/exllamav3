from __future__ import annotations

from typing_extensions import override

import torch

from ..model.config import Config
from ..model.model import Model
from ..util.file import no_default
from ..util.rope import RopeSettings, RopeStyle
from ..modules import Embedding, RMSNorm, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn


def _cfg_key(prefix: str | None, key: str) -> str:
    return f"{prefix}->{key}" if prefix else key


def _init_gemma4_text_config(config: Config, prefix: str | None) -> None:
    key = lambda name: _cfg_key(prefix, name)

    config.vocab_size = config.read_cfg(
        int,
        [key("vocab_size"), "vocab_size"],
        config.vocab_size or 262144,
    )
    config.num_hidden_layers = config.read_cfg(int, key("num_hidden_layers"), no_default)
    config.tie_word_embeddings = config.read_cfg(bool, [key("tie_word_embeddings"), "tie_word_embeddings"], True)

    config.hidden_size = config.read_cfg(int, key("hidden_size"), no_default)
    config.intermediate_size = config.read_cfg(int, key("intermediate_size"), no_default)
    config.num_q_heads = config.read_cfg(int, key("num_attention_heads"), no_default)
    config.num_kv_heads = config.read_cfg(int, key("num_key_value_heads"), no_default)
    config.head_dim = config.read_cfg(int, key("head_dim"), config.hidden_size // config.num_q_heads)
    config.num_global_kv_heads = config.read_cfg(
        int,
        key("num_global_key_value_heads"),
        config.num_kv_heads,
    )
    config.global_head_dim = config.read_cfg(int, key("global_head_dim"), config.head_dim)
    config.attention_k_eq_v = config.read_cfg(bool, key("attention_k_eq_v"), False)

    config.rms_norm_eps = config.read_cfg(float, key("rms_norm_eps"), 1e-6)
    config.max_position_embeddings = config.read_cfg(
        int,
        key("max_position_embeddings"),
        131072,
    )
    config.sliding_window = config.read_cfg(int, key("sliding_window"), 512)
    config.final_logit_softcapping = float(
        config.read_cfg([int, float], key("final_logit_softcapping"), 0.0) or 0.0
    )

    config.enable_moe_block = config.read_cfg(bool, key("enable_moe_block"), False)
    config.use_double_wide_mlp = config.read_cfg(bool, key("use_double_wide_mlp"), False)
    assert not config.enable_moe_block, "Gemma4 MoE blocks are not supported in exllamav3"
    assert not config.use_double_wide_mlp, "Gemma4 double-wide MLP is not supported in exllamav3"

    layer_types = config.read_cfg(list, key("layer_types"), None)
    if layer_types is None:
        layer_types = [
            "sliding_attention" if (idx + 1) % 6 else "full_attention"
            for idx in range(config.num_hidden_layers)
        ]
    assert len(layer_types) == config.num_hidden_layers, \
        "Length of layer_types key doesn't match number of hidden layers"
    for layer_type in layer_types:
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError(f"Unknown layer type in layer_types: {layer_type}")
    if layer_types[-1] != "full_attention":
        layer_types[-1] = "full_attention"
    config.layer_types = layer_types
    config.swa_pattern = [
        config.sliding_window if layer_type == "sliding_attention" else -1
        for layer_type in config.layer_types
    ]

    rope_parameters = config.read_cfg(dict, key("rope_parameters"), None) or {}
    local_rope = {
        "rope_type": "default",
        "rope_theta": 10000.0,
        **rope_parameters.get("sliding_attention", {}),
    }
    global_rope = {
        "rope_type": "proportional",
        "rope_theta": 1000000.0,
        "partial_rotary_factor": 0.25,
        **rope_parameters.get("full_attention", {}),
    }

    config.rope_settings_local = RopeSettings(
        head_dim=config.head_dim,
        rope_theta=float(local_rope.get("rope_theta", 10000.0)),
        rope_scaling=local_rope,
        partial_rotary_factor=float(local_rope.get("partial_rotary_factor", 1.0)),
        max_position_embeddings=config.max_position_embeddings,
        original_max_position_embeddings=config.max_position_embeddings,
        rope_style=RopeStyle.NEOX,
        override_type=local_rope.get("rope_type"),
    )
    config.rope_settings_global = RopeSettings(
        head_dim=config.global_head_dim,
        rope_theta=float(global_rope.get("rope_theta", 1000000.0)),
        rope_scaling=global_rope,
        partial_rotary_factor=float(global_rope.get("partial_rotary_factor", 0.25)),
        max_position_embeddings=config.max_position_embeddings,
        original_max_position_embeddings=config.max_position_embeddings,
        rope_style=RopeStyle.NEOX,
        override_type=global_rope.get("rope_type"),
    )

    config.layer_head_dims = []
    config.layer_num_kv_heads = []
    config.layer_use_k_as_v = []
    for layer_type in config.layer_types:
        is_sliding = layer_type == "sliding_attention"
        use_k_as_v = config.attention_k_eq_v and not is_sliding
        config.layer_head_dims.append(config.head_dim if is_sliding else config.global_head_dim)
        config.layer_num_kv_heads.append(
            config.num_global_kv_heads if use_k_as_v else config.num_kv_heads
        )
        config.layer_use_k_as_v.append(use_k_as_v)


class Gemma4Config(Config):
    arch_string = "Gemma4ForConditionalGeneration"

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": Gemma4Model}, **kwargs)
        _init_gemma4_text_config(self, "text_config")

    def default_max_position_embeddings(self):
        return 131072


class Gemma4TextConfig(Config):
    arch_string = "Gemma4ForCausalLM"

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": Gemma4TextModel}, **kwargs)
        _init_gemma4_text_config(self, None)

    def default_max_position_embeddings(self):
        return 131072


class Gemma4Model(Model):
    config_class = Gemma4Config

    def __init__(self, config: Gemma4Config | Gemma4TextConfig, key_prefix: str = "model.language_model.", **kwargs):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config=config,
                key=key_prefix + "embed_tokens",
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                normalize=True,
            )
        ]

        self.first_block_idx = len(self.modules)

        self.modules += [
            TransformerBlock(
                config=config,
                key=key_prefix + f"layers.{idx}",
                attn_norm=RMSNorm(
                    config=config,
                    key=key_prefix + f"layers.{idx}.input_layernorm",
                    rms_norm_eps=config.rms_norm_eps,
                ),
                attn=Attention(
                    config=config,
                    key=key_prefix + f"layers.{idx}.self_attn",
                    layer_idx=idx,
                    hidden_size=config.hidden_size,
                    head_dim=config.layer_head_dims[idx],
                    num_q_heads=config.num_q_heads,
                    num_kv_heads=config.layer_num_kv_heads[idx],
                    rope_settings=(
                        config.rope_settings_local
                        if config.swa_pattern[idx] != -1
                        else config.rope_settings_global
                    ),
                    sm_scale=1.0,
                    logit_softcapping=0.0,
                    sliding_window=config.swa_pattern[idx],
                    key_q="q_proj",
                    key_k="k_proj",
                    key_v="k_proj" if config.layer_use_k_as_v[idx] else "v_proj",
                    key_o="o_proj",
                    qmap="block.attn",
                    q_norm=RMSNorm(
                        config=config,
                        key=key_prefix + f"layers.{idx}.self_attn.q_norm",
                        rms_norm_eps=config.rms_norm_eps,
                    ),
                    k_norm=RMSNorm(
                        config=config,
                        key=key_prefix + f"layers.{idx}.self_attn.k_norm",
                        rms_norm_eps=config.rms_norm_eps,
                    ),
                    v_norm_eps=config.rms_norm_eps,
                ),
                attn_post_norm=RMSNorm(
                    config=config,
                    key=key_prefix + f"layers.{idx}.post_attention_layernorm",
                    rms_norm_eps=config.rms_norm_eps,
                    out_dtype=torch.float,
                ),
                mlp_norm=RMSNorm(
                    config=config,
                    key=key_prefix + f"layers.{idx}.pre_feedforward_layernorm",
                    rms_norm_eps=config.rms_norm_eps,
                ),
                mlp=GatedMLP(
                    config=config,
                    key=key_prefix + f"layers.{idx}.mlp",
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    key_up="up_proj",
                    key_gate="gate_proj",
                    key_down="down_proj",
                    qmap="block.mlp",
                    activation_fn="gelu",
                ),
                mlp_post_norm=RMSNorm(
                    config=config,
                    key=key_prefix + f"layers.{idx}.post_feedforward_layernorm",
                    rms_norm_eps=config.rms_norm_eps,
                    out_dtype=torch.float,
                ),
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.last_kv_module_idx = len(self.modules) - 1

        self.modules += [
            RMSNorm(
                config=config,
                key=key_prefix + "norm",
                rms_norm_eps=config.rms_norm_eps,
                out_dtype=torch.half,
            ),
            Linear(
                config=config,
                key="lm_head",
                qbits_key="head_bits",
                alt_key=key_prefix + "embed_tokens",
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                qmap="block",
                softcap=config.final_logit_softcapping,
                caps={"logits_output": True},
            ),
        ]

        self.logit_layer_idx = len(self.modules) - 1

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<bos><start_of_turn>user\n"
        if system_prompt:
            p += f"{system_prompt}\n\n"
        p += f"{prompt}\n"
        p += "<start_of_turn>model\n"
        return p


class Gemma4TextModel(Gemma4Model):
    config_class = Gemma4TextConfig

    def __init__(self, config: Gemma4TextConfig, **kwargs):
        super().__init__(config, key_prefix="model.", **kwargs)
