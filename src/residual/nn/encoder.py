from typing import Any, Callable, Mapping, Optional

import gin
import torch
from torch import nn
from transformers import BlipForImageTextRetrieval, CLIPModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


def identity_pooling(x, *args, **kwargs):
    return x


@gin.configurable
def cls_pooling(x: torch.Tensor, dim: int, keep_dim: bool = True) -> torch.Tensor:
    x = x.select(dim=dim, index=0)

    if keep_dim:
        x = x.unsqueeze(dim)

    return x


@gin.configurable
def avg_pooling(
    x: torch.Tensor, dim: int, exclude_cls: bool = True, keep_dim: bool = True
) -> torch.Tensor:
    if exclude_cls:
        x = x.select(dim=dim, index=slice(1, None))

    x = x.mean(dim=dim, keepdim=keep_dim)

    return x


class Encoder(nn.Module):
    def __init__(
        self,
        name: str,
        encoding_dim: int,
        model: nn.Module,
        collate_fn: Callable,
        preprocess: Callable,
        pooling_fn: Optional[Callable],
    ):
        super().__init__()
        self.name = name
        self.model = model
        self.collate_fn = collate_fn
        self.preprocess = preprocess
        self.pooling_fn = pooling_fn if pooling_fn is not None else identity_pooling
        self._encoding_dim = encoding_dim

    @property
    def encoding_dim(self):
        return self._encoding_dim

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        raise NotImplementedError

    def properties(self) -> Mapping[str, Any]:
        raise NotImplementedError


class OpenCLIPVisionEncoder(Encoder):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        collate_fn: Callable,
        preprocess: Callable,
        pooling_fn: Optional[Callable] = None,
    ):
        super().__init__(
            name=name,
            encoding_dim=model.output_dim,
            model=model,
            collate_fn=collate_fn,
            preprocess=preprocess,
            pooling_fn=pooling_fn,
        )

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        return self.encode_image(x)

    def encode_image(self, x):
        pooled, tokens = self.model(x)

        if self.pooling_fn == identity_pooling:
            return pooled

        tokens = tokens @ self.model.proj

        tokens = torch.cat([pooled.unsqueeze(1), tokens], dim=1)
        tokens = self.pooling_fn(tokens, dim=1)
        return tokens

    def properties(self) -> Mapping[str, Any]:
        return {}


class OpenCLIPTextEncoder(Encoder):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        collate_fn: Callable,
        preprocess: Callable,
        pooling_fn=None,
    ):
        super().__init__(
            name=name,
            encoding_dim=model.visual.output_dim,
            model=model,
            collate_fn=collate_fn,
            preprocess=preprocess,
            pooling_fn=pooling_fn,
        )

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        return self.encode_text(x)

    def encode_text(self, x):
        return self.model.encode_text(x)

    def properties(self) -> Mapping[str, Any]:
        return {}


class HFVisionEncoder(Encoder):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        collate_fn: Callable,
        preprocess: Callable,
        pooling_fn: Optional[Callable] = None,
    ):
        if "blip" in name:
            model: BlipForImageTextRetrieval
            encoding_dim = model.config.vision_config.hidden_size
            vision_model = model.vision_model
        elif "clip" in name:
            encoding_dim = model.config.projection_dim
            vision_model = model.vision_model
        elif "vit" in name or "dinov2" in name:
            encoding_dim = model.config.hidden_size
            vision_model = model
        else:
            breakpoint()
            raise NotImplementedError
        super().__init__(
            name=name,
            encoding_dim=encoding_dim,
            model=vision_model,
            collate_fn=collate_fn,
            preprocess=preprocess,
            pooling_fn=pooling_fn,
        )
        if "blip" in name:
            self.vision_proj = model.vision_proj
        if "clip" in name:
            self.vision_proj = model.visual_projection

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        if "blip" in self.name:
            vision_out = self.model(x, return_dict=True)
            vision_out = vision_out["last_hidden_state"]
            vision_out = self.pooling_fn(vision_out, dim=1)

            return self.vision_proj(vision_out)
        elif "clip" in self.name:
            vision_out = self.model(x, return_dict=True)
            vision_out = vision_out["last_hidden_state"]
            vision_out = self.pooling_fn(vision_out, dim=1)
            vision_out = self.model.post_layernorm(vision_out)

            return self.vision_proj(vision_out)
        else:
            vision_out = self.model(
                x, output_hidden_states=True, output_attentions=False
            )
            vision_out = vision_out["last_hidden_state"]
            vision_out = self.pooling_fn(vision_out, dim=1)

            return vision_out

    def encode_image(self, x):
        return self(x)

    def properties(self) -> Mapping[str, Any]:
        return {}


class HFTextEncoder(Encoder):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        collate_fn: Callable,
        preprocess: Callable,
        pooling_fn: Optional[Callable] = None,
    ):
        if "blip" in name:
            encoding_dim = model.config.vision_config.hidden_size
            text_model = model.text_encoder
        elif "clip" in name:
            encoding_dim = model.config.projection_dim
            text_model = model.text_model
        else:
            raise NotImplementedError

        super().__init__(
            name=name,
            encoding_dim=encoding_dim,
            model=text_model,
            collate_fn=collate_fn,
            preprocess=preprocess,
            pooling_fn=pooling_fn,
        )
        if "blip" in name:
            self.text_proj = model.text_proj
        elif "clip" in name:
            self.text_proj = model.text_projection
        else:
            raise NotImplementedError

    def forward(self, x) -> torch.Tensor:
        if "blip" in self.name:
            encodings = self.model(**x, output_hidden_states=False).last_hidden_state[
                :, 0, ...
            ]
        else:
            encodings: BaseModelOutputWithPooling = self.model(
                **x, output_hidden_states=False
            ).pooler_output

        encodings = self.text_proj(encodings)
        return encodings

    def encode_text(self, x):
        return self(x)

    def properties(self) -> Mapping[str, Any]:
        return {}
