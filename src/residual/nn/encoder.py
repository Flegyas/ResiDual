from typing import Any, Callable, Mapping

import torch
from torch import nn
from transformers.models.clip.configuration_clip import CLIPConfig


class Encoder(nn.Module):
    def __init__(
        self,
        name: str,
        encoding_dim: int,
        model: nn.Module,
        collate_fn: Callable,
        preprocess: Callable,
        cls_only: bool,
    ):
        super().__init__()
        self.name = name
        self.model = model
        self.collate_fn = collate_fn
        self.preprocess = preprocess
        self.cls_only = cls_only
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
        cls_only: bool,
    ):
        super().__init__(
            name=name,
            encoding_dim=model.output_dim,
            model=model,
            collate_fn=collate_fn,
            preprocess=preprocess,
            cls_only=cls_only,
        )

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        return self.encode_image(x)

    def encode_image(self, x):
        pooled, tokens = self.model(x)
        if self.cls_only:
            return pooled

        tokens = tokens @ self.model.proj

        tokens = torch.cat([pooled.unsqueeze(1), tokens], dim=1)
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
        cls_only: bool = False,
    ):
        super().__init__(
            name=name,
            encoding_dim=model.visual.output_dim,
            model=model,
            collate_fn=collate_fn,
            preprocess=preprocess,
            cls_only=cls_only,
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
        cls_only: bool,
    ):
        super().__init__(
            name=name,
            encoding_dim=model.config.image_text_hidden_size
            if not isinstance(model.config, CLIPConfig)
            else model.config.vision_config.projection_dim,
            model=model
            if ("blip" not in name and "clip" not in name)
            else model.vision_model,
            collate_fn=collate_fn,
            preprocess=preprocess,
            cls_only=cls_only,
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
            vision_out = vision_out["last_hidden_state"][
                :, 0 if self.cls_only else slice(None)
            ]
            return self.vision_proj(vision_out)
        elif "clip" in self.name:
            vision_out = self.model(x, return_dict=True)
            vision_out = vision_out["last_hidden_state"][
                :, 0 if self.cls_only else slice(None)
            ]
            vision_out = self.model.post_layernorm(vision_out)
            return self.vision_proj(vision_out)
        else:
            out = self.model(x, output_hidden_states=True, output_attentions=False)
            out = out["last_hidden_state"][:, 0 if self.cls_only else slice(None)]
            return out

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
        cls_only: bool,
    ):
        super().__init__(
            name=name,
            encoding_dim=model.config.hidden_size,
            model=model.text_encoder if "blip" in name else model,
            collate_fn=collate_fn,
            preprocess=preprocess,
            cls_only=cls_only,
        )
        if "blip" in name:
            self.text_proj = model.text_proj
        else:
            raise NotImplementedError

    def forward(self, x) -> torch.Tensor:
        encodings = self.model(**x, output_hidden_states=False)[0][
            :, 0 if self.cls_only else slice(None)
        ]
        encodings = self.text_proj(encodings)
        return encodings

    def encode_text(self, x):
        return self(x)

    def properties(self) -> Mapping[str, Any]:
        return {}
