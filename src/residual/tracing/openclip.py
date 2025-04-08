import functools
from typing import Any, Mapping, Optional, Sequence, Union

import torch
from open_clip.model import VisionTransformer
from setuptools import monkey
from torch import nn

from residual.nn.encoder import OpenCLIPVisionEncoder
from residual.residual import OutputProj
from residual.tracing.tracer import ResidualTracer
from residual.tracing.tracing_op import TracingOp
from residual.tracing.utils import (
    apply_ln_residual,
    forward_MHA_traced,
    layer_heads_proj,
    single_head_proj,
)

model_name2info = {
    "openclip_b": dict(
        num_layers=12,
        num_heads_per_layer=12,
        head_dim=64,
        num_tokens=197,
        residual_dim=512,
    ),
    "openclip_l": dict(
        num_layers=24,
        num_heads_per_layer=16,
        head_dim=64,
        num_tokens=197,
        residual_dim=768,
    ),
    "clip_b": dict(
        num_layers=12,
        num_heads_per_layer=12,
        head_dim=64,
        num_tokens=197,
        residual_dim=512,
    ),
    "clip_l": dict(
        num_layers=24,
        num_heads_per_layer=16,
        head_dim=64,
        num_tokens=197,
        residual_dim=768,
    ),
}


class OpenCLIPOutputProj(OutputProj):
    @classmethod
    def from_encoder(
        cls,
        encoder: OpenCLIPVisionEncoder,
    ):
        model: VisionTransformer = encoder.model

        num_layers, num_heads_per_layer = (
            model_name2info[encoder.name]["num_layers"],
            model_name2info[encoder.name]["num_heads_per_layer"],
        )

        unit_normalization_factor = 2 * num_layers + 1
        head_normalization_factor = unit_normalization_factor * num_heads_per_layer

        return OpenCLIPOutputProj(
            mha_out_projs=nn.ModuleList(
                [rab.attn.out_proj for rab in model.transformer.resblocks]
            ),
            final_ln=model.ln_post,
            final_projection=model.proj,
            unit_normalization_factor=unit_normalization_factor,
            head_normalization_factor=head_normalization_factor,
            num_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer,
        )

    def __init__(
        self,
        mha_out_projs: nn.ModuleList,
        final_ln: nn.LayerNorm,
        final_projection: nn.Parameter,
        unit_normalization_factor: float,
        head_normalization_factor: float,
        num_layers: int,
        num_heads_per_layer: int,
    ):
        super().__init__()

        self.mha_out_projs = mha_out_projs
        self.final_ln = final_ln
        self.final_projection = final_projection

        self.register_buffer(
            "unit_normalization_factor", torch.tensor(unit_normalization_factor)
        )
        self.register_buffer(
            "head_normalization_factor", torch.tensor(head_normalization_factor)
        )
        self.register_buffer("num_layers", torch.tensor(num_layers, dtype=torch.int))
        self.register_buffer(
            "num_heads_per_layer", torch.tensor(num_heads_per_layer, dtype=torch.int)
        )

    @torch.no_grad()
    def project(
        self,
        unit_type2encodings: Union[Mapping[str, torch.Tensor], torch.Tensor],
    ):
        if "pre_ln" in unit_type2encodings:
            # this is the usual Tracker case, online
            preln = unit_type2encodings["pre_ln"]
            preln_mean = preln.mean(dim=-1)
            preln_std = preln.std(dim=-1, unbiased=False)
        else:
            # this is the offline case, we had the preln_mean and preln_std stored by set_spaces
            preln_mean = self.preln_mean
            preln_std = self.preln_std

        if "head" in unit_type2encodings:
            head_encodings = []
            for layer_index in range(self.num_layers):
                mha_out_proj = self.mha_out_projs[layer_index]

                head_encoding = unit_type2encodings["head"]
                assert head_encoding.size(2) % self.num_layers == 0, (
                    "The number of heads must be divisible by the number of layers"
                    f"({head_encoding.size(2)} % {self.num_layers} != 0)"
                )
                heads_per_layer = head_encoding.size(2) // self.num_layers

                head_encoding = unit_type2encodings["head"][
                    :,
                    :,
                    layer_index * heads_per_layer : (layer_index + 1) * heads_per_layer,
                    :,
                ]

                # project raw head encodings using the layer's output projection
                head_encoding = layer_heads_proj(
                    raw_heads=head_encoding,
                    out_proj_weight=mha_out_proj.weight,
                    out_proj_bias=mha_out_proj.bias,
                )

                head_encodings.append(head_encoding)
            unit_type2encodings["head"] = torch.cat(head_encodings, dim=2)

        # apply layer normalization and final projection
        for unit_type in ("emb", "head", "mlp"):
            if unit_type not in unit_type2encodings:
                continue

            unit_type2encodings[unit_type] = (
                apply_ln_residual(
                    x=unit_type2encodings[unit_type],
                    mean=preln_mean,
                    std=preln_std,
                    gamma=self.final_ln.weight,
                    beta=self.final_ln.bias,
                    eps=self.final_ln.eps,
                    norm=self.unit_normalization_factor
                    if unit_type != "head"
                    else self.head_normalization_factor,
                )
                @ self.final_projection
            )

        return unit_type2encodings

    @torch.no_grad()
    def project_unit(
        self,
        unit_type: str,
        unit_encoding: torch.Tensor,
        unit_info: Mapping[str, Any],
    ) -> torch.Tensor:
        if unit_type == "head":
            head_idx, layer_idx = unit_info["head_idx"], unit_info["layer_idx"]

            mha_out_proj = self.mha_out_projs[layer_idx]

            # project raw head encodings using the layer's output projection
            unit_encoding = single_head_proj(
                raw_head_encoding=unit_encoding,
                head_index=head_idx,
                num_heads=self.num_heads_per_layer,
                out_proj_weight=mha_out_proj.weight,
                out_proj_bias=mha_out_proj.bias,
            )
        unit_encoding = (
            apply_ln_residual(
                x=unit_encoding,
                mean=self.preln_mean,
                std=self.preln_std,
                gamma=self.final_ln.weight,
                beta=self.final_ln.bias,
                eps=self.final_ln.eps,
                norm=self.unit_normalization_factor
                if unit_type != "head"
                else self.head_normalization_factor,
            )
            @ self.final_projection
        )

        return unit_encoding


class OpenCLIPTracer(ResidualTracer):
    def _enter(self):
        # generic patch for F.multi_head_attention_forward to get the attention heads 'head'
        if "head" in self.unit_types:
            torch.backends.mha.set_fastpath_enabled(
                False
            )  # TODO: hack to force the use of the (patched) Python implementation
            replacement = functools.partial(forward_MHA_traced, tracer=self)
            replacement.__name__ = "multi_head_attention_forward"
            monkey.patch_func(
                replacement=replacement,
                target_mod=torch.nn.functional,
                func_name="multi_head_attention_forward",
            )

    def emb_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        emb = output
        emb = self.encoder.pooling_fn(emb, dim=1)

        emb = emb.view(emb.size(0), emb.size(1), 1, emb.size(2))

        self._buffer["emb"].append(emb)

        return output

    def pre_ln_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        pre_ln = input[0]
        pre_ln = self.encoder.pooling_fn(pre_ln, dim=1)

        self._buffer["pre_ln"].append(pre_ln)

        return output

    def mlp_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        mlp = output
        mlp = self.encoder.pooling_fn(mlp, dim=1)

        mlp = mlp.view(mlp.size(0), mlp.size(1), 1, mlp.size(2))

        self._buffer["mlp"].append(mlp)

        return output

    def _exit(self):
        torch.nn.functional.multi_head_attention_forward = (
            monkey.get_unpatched_function(
                torch.nn.functional.multi_head_attention_forward
            )
        )

    def __init__(
        self,
        module_name: str,
        encoder: OpenCLIPVisionEncoder,
        raw: bool,
        tracer_ops: Optional[Sequence[TracingOp]] = None,
    ) -> None:
        if module_name not in model_name2info:
            raise ValueError(f"Unknown model name: {module_name}")

        info = model_name2info[module_name]
        self.info = info

        super().__init__(
            module_name=module_name,
            encoder=encoder,
            unit_types=("emb", "head", "mlp", "pre_ln"),
            raw=raw,
            out_proj=OpenCLIPOutputProj.from_encoder(encoder=encoder),
            glob2fn={
                # head will be handled by the patched multi_head_attention_forward in _enter
                "model.ln_pre": self.emb_hook,
                "model.transformer.resblocks.*.mlp": self.mlp_hook,
                "model.ln_post": self.pre_ln_hook,
            },
            tracer_ops=tracer_ops,
        )
        self.encoder: OpenCLIPVisionEncoder = self.encoder
