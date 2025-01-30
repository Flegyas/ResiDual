from typing import Any, Callable, Mapping, Optional

import gin
import torch
from torch import nn
from transformers import BlipForImageTextRetrieval
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
def all_but_cls_pooling(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.index_select(dim, torch.arange(1, x.size(dim), device=x.device))

    return x


@gin.configurable
def avg_pooling(
    x: torch.Tensor, dim: int, exclude_cls: bool = True, keep_dim: bool = True
) -> torch.Tensor:
    if exclude_cls:
        x = x.index_select(dim, torch.arange(1, x.size(dim), device=x.device))

    x = x.mean(dim=dim, keepdim=keep_dim)

    return x


@gin.configurable
def token_selection_pooling(
    x: torch.Tensor, dim: int, token_index: int, keep_dim: bool = True
) -> torch.Tensor:
    x = x.select(dim=dim, index=token_index)
    if keep_dim:
        x = x.unsqueeze(dim)

    return x


_name2pooling_fn = {
    "identity": identity_pooling,
    "cls": cls_pooling,
    "all_but_cls": all_but_cls_pooling,
    "avg": avg_pooling,
    "token_selection": token_selection_pooling,
}


def get_pooling_fn(pooling_fn_name: str) -> Callable:
    return _name2pooling_fn[pooling_fn_name]


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
    def logit_scale(self):
        return self.model.logit_scale

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

        if self.pooling_fn == identity_pooling or self.pooling_fn == cls_pooling:
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
        elif "clip" in name:
            self.vision_proj = model.visual_projection
        else:
            pass

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        output = None
        if "blip" in self.name:
            vision_out = self.model(x, return_dict=True)
            vision_out = vision_out["last_hidden_state"]
            vision_out = self.pooling_fn(vision_out, dim=1)

            output = self.vision_proj(vision_out)
        elif "clip" in self.name:
            vision_out = self.model(x, return_dict=True)
            vision_out = vision_out["last_hidden_state"]
            vision_out = self.pooling_fn(vision_out, dim=1)
            vision_out = self.model.post_layernorm(vision_out)

            output = self.vision_proj(vision_out)
        else:
            vision_out = self.model(
                x, output_hidden_states=True, output_attentions=False
            )
            vision_out = vision_out["last_hidden_state"]
            vision_out = self.pooling_fn(vision_out, dim=1)

            output = vision_out

        return output.squeeze(dim=1)

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
