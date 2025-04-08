from collections import defaultdict
from typing import Mapping, Sequence, Type

import pandas as pd
import torch
from torch import nn

from residual.nn.encoder import Encoder
from residual.residual import OutputProj, Residual
from residual.tracing.tracing_op import TracingOp


def register_tracer(tracer_cls):
    if tracer_cls.__name__ in _tracer_registry:
        raise ValueError(f"Tracer {tracer_cls.__name__} already registered")

    _tracer_registry[tracer_cls.__name__] = tracer_cls

    return tracer_cls


class TracerMeta(type):
    def __init__(cls, name, bases, dct):
        # Automatically ensure that all classes using TracerMeta are also subclasses of nn.Module
        # if not any(isinstance(base, nn.Module) for base in bases):
        #     raise TypeError(f"{name} must inherit from nn.Module to use TracerMeta.")

        # Register any subclass of ResidualTracer
        if any(base.__name__ == "ResidualTracer" for base in bases):
            register_tracer(cls)

        super().__init__(name, bases, dct)


class ResidualTracer(nn.Module, metaclass=TracerMeta):
    @property
    def name(self) -> str:
        return self.encoder.name

    @property
    def encoding_dim(self) -> int:
        return self.encoder.encoding_dim

    @property
    def collate_fn(self):
        return self.encoder.collate_fn

    @property
    def pooling_fn(self) -> str:
        return self.encoder.pooling_fn

    def _resolve_glob(self, glob_pattern: str) -> Sequence[nn.Module]:
        """
        Resolve a glob-like pattern (e.g., `transformer.resblocks.*.mlp`) into all matching modules.

        Args:
            glob_pattern: The glob-like pattern to resolve.

        Returns:
            A list of resolved modules matching the pattern.
        """
        parts = glob_pattern.split(".")
        current_modules = [
            ("", self.encoder)
        ]  # Start with the root module and an empty path

        for part in parts:
            next_modules = []
            for prefix, module in current_modules:
                if part == "*":
                    # Match all children or items in a list
                    if isinstance(module, (list, tuple)):  # Handle lists or tuples
                        next_modules.extend(
                            (f"{prefix}.{i}" if prefix else str(i), item)
                            for i, item in enumerate(module)
                        )
                    else:
                        next_modules.extend(
                            (f"{prefix}.{name}" if prefix else name, child)
                            for name, child in module.named_children()
                        )
                elif hasattr(module, part):
                    # Match specific attribute
                    next_modules.append(
                        (f"{prefix}.{part}" if prefix else part, getattr(module, part))
                    )
                else:
                    raise ValueError(f"Could not resolve {part} in {prefix}")

            current_modules = next_modules

        return [module for _, module in current_modules]

    def __init__(
        self,
        encoder: Encoder,
        module_name: str,
        raw: bool,
        out_proj: OutputProj,
        unit_types: Sequence[str],
        glob2fn: Mapping[str, callable] = None,
        tracer_ops: Sequence[TracingOp] = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.module_name = module_name
        self.raw = raw
        self.out_proj = out_proj
        self.unit_types = set(unit_types)
        self.glob2fn = glob2fn or {}
        self._ops = tracer_ops or []

        self._buffer = defaultdict(list)

        self.residual_composition = self.get_residual_composition()

    # def parameters(self, recurse: bool = True):
    #     """Exclude module from the parameters."""
    #     return (
    #         p
    #         for name, p in super(ResidualTracer, self).named_parameters(recurse)
    #         if not name.startswith("module.")
    #     )

    # def state_dict(self, destination=None, prefix="", keep_vars=False):
    #     """Exclude `module` from the state_dict."""
    #     state_dict = super(ResidualTracer, self).state_dict(
    #         destination, prefix, keep_vars
    #     )
    #     keys_to_remove = [k for k in state_dict.keys() if k.startswith("module.")]
    #     for key in keys_to_remove:
    #         del state_dict[key]
    #     return state_dict

    def forward(self, x: torch.Tensor):
        return self.encode(
            x=x,
            keys=None,
            return_residual=True,
        )["residual"].encoding.squeeze(dim=1)

    def encode(
        self,
        x: torch.Tensor,
        keys: Sequence[str],
        return_residual: bool,
        # *model_args,
        # **model_kwargs,
    ):
        model_out = self.encoder(x)

        unit_type2encodings = self.get_residual_units()
        assert self._buffer == {}, "Buffer not empty"

        for op in self._ops:
            op(tracer=self, unit_type2encodings=unit_type2encodings, keys=keys)

        result = dict(model_out=model_out, unit_type2encodings=unit_type2encodings)

        if return_residual:
            result["residual"] = self.to_residual(unit_type2encodings)

        return result

    def to_residual(self, unit_type2encodings: Mapping[str, torch.Tensor]):
        if self.raw:
            if self.out_proj is not None:
                unit_type2encodings = self.out_proj.project(unit_type2encodings)
            else:
                raise ValueError(
                    "Raw encodings require an output projection to get the residual"
                )

        residual = []
        # IMPORTANT: follow the order of the residual composition,
        # otherwise the residual tensor will be misaligned with everything else
        for unit_type in self.residual_composition["type"].unique():
            assert unit_type in unit_type2encodings, f"Missing {unit_type} encodings"

            unit_encodings = unit_type2encodings[unit_type]
            if unit_encodings.dim() != 4:
                raise ValueError(
                    f"Expected tensor of shape (n, t, u, d) for {unit_type}, got {unit_encodings.dim()} : {unit_encodings.shape}"
                )

            residual.append(unit_encodings)
        residual = torch.cat(residual, dim=2)

        return Residual(encoding=residual, encoding_info=self.residual_composition)

    def _enter(self):
        pass

    def __enter__(self):
        self.hooks = [
            module.register_forward_hook(fn)
            for glob, fn in self.glob2fn.items()
            for module in self._resolve_glob(glob)
        ]

        self._enter()

        return self

    def get_residual_units(self):
        unit_type2encodings = {
            unit_type: encodings for unit_type, encodings in self._buffer.items()
        }

        # unit_type2encodings = {
        #     unit_type: [encoding.mean(dim=1, keepdim=True) for encoding in encodings]
        #     for unit_type, encodings in unit_type2encodings.items()
        # }

        unit_type2encodings = {
            unit_type: torch.cat(
                encodings, dim=2
            )  # stack encodings along the residual dimension (n, t, u, d)
            if len(encodings) > 1
            else encodings[0]
            for unit_type, encodings in unit_type2encodings.items()
        }

        if not self.raw:
            unit_type2encodings = self.out_proj.project(
                unit_type2encodings=unit_type2encodings
            )

        # flush buffer
        self._buffer = defaultdict(list)

        return unit_type2encodings

    def __del__(self):
        if hasattr(self, "hooks"):
            for hook in self.hooks:
                hook.remove()

    def _exit(self):
        raise NotImplementedError

    def get_residual_composition(self) -> pd.DataFrame:
        composition = []
        # emb
        composition.append(
            {
                "type": "emb",
                "layer_idx": None,
                "head_idx": None,
                "unit_idx": 0,
            }
        )
        # mlp
        for layer_idx in range(self.info["num_layers"]):
            prev_attn_units = layer_idx * self.info["num_heads_per_layer"]
            composition.append(
                {
                    "type": "mlp",
                    "layer_idx": layer_idx,
                    "head_idx": None,
                    "unit_idx": 1  # emb
                    + layer_idx,  # number of previous mlp units
                }
            )
            # head
            for head_idx in range(self.info["num_heads_per_layer"]):
                composition.append(
                    {
                        "type": "head",
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "unit_idx": 1  # emb
                        + self.info["num_layers"]  # number of previous mlp units...
                        + prev_attn_units  # number of previous attn units
                        + head_idx,  # number of previous heads
                    }
                )

        composition = pd.DataFrame(composition)

        composition["unit_idx"] = composition["unit_idx"].astype("Int64")
        composition["layer_idx"] = composition["layer_idx"].astype("Int64")
        composition["head_idx"] = composition["head_idx"].astype("Int64")

        composition = composition.sort_values(by="unit_idx")

        return composition

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            raise exc_val
        else:
            for hook in self.hooks:
                hook.remove()
            self.hooks = []

            for op in self._ops:
                op.exit(self)

            self._exit()

        return False


_tracer_registry: Mapping[str, ResidualTracer] = {}


def get_registered_tracer(name: str) -> Type[ResidualTracer]:
    if name not in _tracer_registry:
        raise ValueError(f"Tracer for {name} not registered")

    return _tracer_registry[name]
