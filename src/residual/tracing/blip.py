from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BlipVisionModel
from transformers.modeling_outputs import BaseModelOutput

from residual.nn.encoder import HFVisionEncoder
from residual.residual import OutputProj
from residual.tracing.tracer import ResidualTracer
from residual.tracing.tracing_op import TracingOp
from residual.tracing.utils import apply_ln_residual, layer_heads_proj, single_head_proj

model_name2info = {
    "blip_l_flickr": dict(
        num_layers=24,
        num_heads_per_layer=16,
        head_dim=64,
        num_tokens=577,
        residual_dim=256,
    )
}


class BLIPOutProj(OutputProj):
    @classmethod
    def from_encoder(
        cls,
        encoder: HFVisionEncoder,
    ):
        model: BlipVisionModel = encoder.model

        num_layers, num_heads_per_layer = (
            model_name2info[encoder.name]["num_layers"],
            model_name2info[encoder.name]["num_heads_per_layer"],
        )
        unit_normalization_factor = 2 * num_layers + 1
        head_normalization_factor = unit_normalization_factor * num_heads_per_layer

        return BLIPOutProj(
            mha_out_projs=nn.ModuleList(
                layer.self_attn.projection for layer in model.encoder.layers
            ),
            final_ln=model.post_layernorm,
            final_projection=encoder.vision_proj,
            num_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer,
            unit_normalization_factor=unit_normalization_factor,
            head_normalization_factor=head_normalization_factor,
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
    def project(self, unit_type2encodings: Mapping[str, torch.Tensor]):
        preln = unit_type2encodings["pre_ln"]
        preln_mean = preln.mean(dim=-1)
        preln_std = preln.std(dim=-1, unbiased=False)

        if "head" in unit_type2encodings:
            head_encodings = []
            for layer_index in range(self.num_layers):
                mha_out_proj = self.mha_out_projs[layer_index]

                head_encoding = unit_type2encodings["head"]
                assert head_encoding.size(2) % self.num_layers == 0
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
            unit_norm = (
                self.unit_normalization_factor
                if unit_type != "head"
                else self.head_normalization_factor
            )
            unit_encoding = unit_type2encodings[unit_type]

            unit_encoding = apply_ln_residual(
                x=unit_encoding,
                mean=preln_mean,
                std=preln_std,
                gamma=self.final_ln.weight,
                beta=self.final_ln.bias,
                eps=self.final_ln.eps,
                norm=unit_norm,
            )

            unit_encoding = F.linear(
                unit_encoding,
                self.final_projection.weight,
                self.final_projection.bias / unit_norm,
            )

            unit_type2encodings[unit_type] = unit_encoding

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

        unit_norm = (
            self.unit_normalization_factor
            if unit_type != "head"
            else self.head_normalization_factor
        )

        unit_encoding = apply_ln_residual(
            x=unit_encoding,
            mean=self.preln_mean,
            std=self.preln_std,
            gamma=self.final_ln.weight,
            beta=self.final_ln.bias,
            eps=self.final_ln.eps,
            norm=unit_norm,
        )

        unit_encoding = F.linear(
            unit_encoding,
            self.final_projection.weight,
            self.final_projection.bias / unit_norm,
        )

        return unit_encoding


class BLIPTracer(ResidualTracer):
    def _enter(self):
        pass

    def head_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        raw_heads = input[0]  # (n, t, d)

        raw_heads = self.encoder.pooling_fn(raw_heads, dim=1)

        raw_heads = torch.stack(raw_heads.split(self.info["head_dim"], dim=-1), dim=2)

        self._buffer["head"].append(raw_heads)

        return output

    def emb_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        emb = output
        emb = self.encoder.pooling_fn(emb, dim=1)

        emb = emb.view(emb.size(0), emb.size(1), 1, emb.size(2))

        self._buffer["emb"].append(emb)

        return output

    def pre_ln_hook(
        self, module: nn.Module, input: torch.Tensor, output: BaseModelOutput
    ):
        pre_ln = output.last_hidden_state

        pre_ln = self.encoder.pooling_fn(pre_ln, dim=1)

        self._buffer["pre_ln"].append(pre_ln)

        return output

    def mlp_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        n, t, d = output.shape
        mlp = output

        mlp = self.encoder.pooling_fn(mlp, dim=1)

        mlp = mlp.view(n, mlp.shape[1], 1, d)

        self._buffer["mlp"].append(mlp)

        return output

    def _exit(self):
        pass

    def __init__(
        self,
        module_name: str,
        encoder: HFVisionEncoder,
        raw: bool,
        tracer_ops: Optional[Sequence[TracingOp]] = None,
    ) -> None:
        self.info = model_name2info[module_name]

        super().__init__(
            module_name=module_name,
            encoder=encoder,
            unit_types=("emb", "head", "mlp", "pre_ln"),
            raw=raw,
            out_proj=BLIPOutProj.from_encoder(encoder=encoder),
            glob2fn=(
                {
                    "model.embeddings": self.emb_hook,
                    "model.encoder.layers.*.mlp": self.mlp_hook,
                    "model.encoder.layers.*.self_attn.projection": self.head_hook,
                    "model.encoder": self.pre_ln_hook,
                }
            ),
            tracer_ops=tracer_ops,
        )
        self.encoder: HFVisionEncoder
