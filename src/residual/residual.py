from abc import abstractmethod
from pathlib import Path
from typing import Any, Generator, Hashable, Mapping, Optional, Sequence, Union

import gin
import pandas as pd
import torch
from latentis import PROJECT_ROOT
from latentis.space import Space
from torch import nn
from tqdm import tqdm

from residual.data.encode import ENCODINGS_DIR
from residual.nn.utils import pca_fn

_col2dtype = {
    "unit_idx": "Int64",
    "layer_idx": "Int64",
    "head_idx": "Int64",
    "type": "category",
}


class OutputProj(nn.Module):
    @classmethod
    @abstractmethod
    def from_encoder(cls, encoder: nn.Module):
        raise NotImplementedError

    @abstractmethod
    def project(
        self,
        unit_type2encodings: Mapping[str, torch.Tensor],
    ):
        """Project the encodings of the units to a common space (the output one).

        Args:
            unit_type2encodings: A mapping from unit types to encodings or a single tensor of encodings.

        Returns:
            The projected encodings.
        """
        raise NotImplementedError

    @abstractmethod
    def project_unit(
        self,
        unit_type: str,
        unit_encoding: torch.Tensor,
        unit_info: Mapping[str, Any],
    ):
        """Project the encodings of a single unit to a common space (the output one).

        Args:
            unit_type: The type of the unit.
            unit_encoding: The encoding of the unit.
            unit_info: The information of the unit.

        Returns:
            The projected unit.
        """
        raise NotImplementedError

    def set_spaces(
        self,
        unit_type2space: Mapping[str, Space],
        offsets: Optional[Sequence[int]] = None,
    ):
        """Set the spaces for the output projection.

        Args:
            unit_type2space: A mapping from unit types to spaces.
            offsets: The offsets of the samples to use.
        """
        preln = unit_type2space["pre_ln"]
        preln = preln[offsets or slice(None)]
        preln_mean = preln.mean(dim=-1)
        preln_std = preln.std(dim=-1, unbiased=False)

        self.register_buffer("preln_mean", preln_mean)
        self.register_buffer("preln_std", preln_std)

        return self


class Residual(nn.Module):
    @classmethod
    def read_unit_shapes(cls, source_dir: Path) -> int:
        unit_type2space = {
            unit_space_dir.name: Space.load_from_disk(unit_space_dir)
            for unit_space_dir in source_dir.glob("*/")
            if unit_space_dir.is_dir()
        }
        num_samples = [space.shape[0] for space in unit_type2space.values()]
        if not all(num == num_samples[0] for num in num_samples):
            raise ValueError(
                "The number of samples in the residual units is not consistent."
            )

        return {unit_type: space.shape for unit_type, space in unit_type2space.items()}

    @classmethod
    def read_composition(cls, source_dir: Path) -> pd.DataFrame:
        return pd.read_csv(
            source_dir / "composition.tsv",
            sep="\t",
            dtype=_col2dtype,
        )

    @classmethod
    @torch.no_grad()
    def load_output(
        cls,
        source_dir: Path,
        device: torch.device,
        as_tensor_device: Optional[torch.device] = None,
        offsets: Optional[Sequence[int]] = None,
        verbose: bool = False,
    ):
        output_encoding = 0

        stream = tqdm(
            cls.stream(
                source_dir=source_dir,
                device=device,
                offsets=offsets,
                as_tensor_device=as_tensor_device,
            ),
            desc="Loading output",
            disable=not verbose,
        )
        for unit, _unit_info in stream:
            output_encoding += unit

        return output_encoding

    @classmethod
    def load(
        cls,
        source_dir: Path,
        device: torch.device,
        offsets: Optional[Sequence[int]] = None,
    ):
        encoding = []

        for unit, _unit_info in cls.stream(
            source_dir=source_dir,
            device=device,
            offsets=offsets,
            as_tensor_device=device,
        ):
            encoding.append(unit)

        encoding = torch.stack(encoding, dim=2)  # (n, t, u, d)
        residual_composition = cls.read_composition(source_dir=source_dir)

        return Residual(
            encoding=encoding,
            encoding_info=residual_composition,
        )

    @classmethod
    def stream(
        cls,
        source_dir: Path,
        device: torch.device,
        offsets: Optional[Sequence[int]] = None,
        filter_fn: Optional[callable] = None,
        as_tensor_device: Optional[torch.device] = None,
        token_index: Optional[int] = None,
    ) -> Generator:
        unit_type2space = {
            unit_space_dir.name: Space.load_from_disk(unit_space_dir)
            for unit_space_dir in source_dir.glob("*/")
            if unit_space_dir.is_dir()
        }
        if as_tensor_device is not None:
            unit_type2space = {
                unit_type: unit_space.as_tensor(device=as_tensor_device)[
                    offsets if offsets is not None else slice(None),
                    token_index if token_index is not None else slice(None),
                ]
                for unit_type, unit_space in unit_type2space.items()
            }
            offsets = None

        unit_type2num_units = {
            unit_type: torch.Size(unit_space.shape[2:-1]).numel()
            for unit_type, unit_space in unit_type2space.items()
        }
        residual_composition = cls.read_composition(source_dir)

        assert len(residual_composition) == sum(
            num_units
            for unit_type, num_units in unit_type2num_units.items()
            if unit_type in residual_composition["type"].unique()
        ), (
            "Residual composition does not match the number of units in the residual.",
            residual_composition.shape,
            unit_type2num_units,
        )

        if (source_dir / "out_proj.pt").exists():
            output_projection: OutputProj = torch.load(
                source_dir / "out_proj.pt", map_location=device, weights_only=False
            ).eval()
            output_projection.set_spaces(unit_type2space, offsets=offsets)
            output_projection.to(device)

        # check that the unit_types are not interleaved in the composition
        all_unit_types = residual_composition["type"]
        assert list(all_unit_types) == sorted(
            all_unit_types, key=lambda x: all_unit_types.tolist().index(x)
        ), "Unit types are interleaved in the composition."

        ordered_unit_types = residual_composition["type"].unique()

        infos = iter(residual_composition.to_dict(orient="records"))

        for unit_type in ordered_unit_types:
            if unit_type not in unit_type2space:
                raise ValueError(
                    f"Unit type {unit_type} is not present in the residual."
                )

            unit_space = unit_type2space[unit_type]
            for local_unit_idx in range(unit_space.shape[2]):
                unit_info = next(infos)
                assert unit_info["type"] == unit_type, (
                    "Unit types are not ordered correctly.",
                    unit_info["type"],
                    unit_type,
                )

                if filter_fn and not filter_fn(unit_info):
                    continue

                unit = unit_space[
                    offsets if offsets is not None else slice(None),
                    token_index if token_index is not None else slice(None),
                    local_unit_idx,
                    :,
                ].to(device)

                if output_projection is not None:
                    unit = output_projection.project_unit(
                        unit_type=unit_type, unit_encoding=unit, unit_info=unit_info
                    )

                yield unit, unit_info

    def __init__(
        self,
        encoding: torch.Tensor,  # (n, t, u, d) = (num_samples, num_tokens, num_residual_units, encoding_dim)
        encoding_info: pd.DataFrame,
    ):
        super().__init__()
        self.encoding = encoding
        self.info = encoding_info
        self.unit_types = set(sorted(self.info["type"].unique()))

    def get_unit_indices(
        self,
        unit_type: str = None,
    ):
        if unit_type is None or unit_type == "all":
            return self.info["unit_idx"]

        return self.info[self.info["type"] == unit_type]["unit_idx"].tolist()

    def select(self, sample_ids: Sequence[Hashable]):
        try:
            sample_ids = [int(key) for key in sample_ids]
            offsets = torch.as_tensor(
                [self.key2offset[int(key)] for key in sample_ids], dtype=torch.long
            )
        except Exception:
            sample_ids = [str(key) for key in sample_ids]
            offsets = torch.as_tensor(
                [self.key2offset[key] for key in sample_ids], dtype=torch.long
            )

        offsets = [self.key2offset[key] for key in sample_ids]
        return self.encoding[offsets, ...]

    @property
    def num_layers(self) -> int:
        return self.info["layer_idx"].nunique()

    @property
    def num_heads(self) -> int:
        return self.info["head_idx"].nunique()

    def size(self, dim: Optional[int] = None) -> torch.Size:
        return self.encoding.size(dim)

    def num_units(self, unit_type: str = None) -> int:
        if unit_type is None:
            return self.encoding.shape[2]

        return self.info["type"].eq(unit_type).sum()

    def _check_unit_info(self, infos: Sequence[Mapping[str, int]]):
        if infos is None:
            return []

        if isinstance(infos, dict):
            infos = [infos]

        infos = pd.DataFrame(infos)
        infos = infos.drop_duplicates()

        for column, dtype in zip(infos.columns, infos.dtypes, strict=True):
            # map the columns to the correct dtype
            infos[column] = infos[column].astype(dtype)
            if column == "unit_idx":
                continue
            # add the columns that are not present in the infos
            if column not in infos.columns:
                infos[column] = None

        # assert set(infos.columns) == set(self.info.columns)

        return infos

    def _check_unit_types(self, types: Sequence[str]):
        if types is None:
            return []

        if types == "all":
            types = self.unit_types

        if isinstance(types, str):
            types = {types}

        if not types.issubset(self.unit_types):
            raise ValueError(
                f"Invalid unit types: {types}. Available unit types: {self.unit_types}"
            )

        return types

    def _get_keep_mask(
        self, keep_units: Union[pd.DataFrame, torch.Tensor, Sequence[Mapping[str, int]]]
    ) -> torch.Tensor:
        keep_mask = torch.zeros(self.encoding.shape[2], dtype=torch.bool)

        # ablation by df entries
        if isinstance(keep_units, pd.DataFrame):
            raise NotImplementedError("Almost there")
            keep_units = self._check_unit_info(keep_units.to_dict(orient="records"))

            matched_units = self.info.merge(keep_units, how="inner").index.tolist()
            assert len(matched_units) == len(keep_units), (
                "Some units in keep_units were not found in the residual.",
                keep_units[~keep_units.index.isin(matched_units)].to_dict(
                    orient="records"
                ),
            )

            keep_mask[matched_units] = True
        # ablation by global indices
        if isinstance(keep_units, torch.Tensor):
            assert keep_units.dtype == torch.long
            assert keep_units.dim() == 1
            assert len(set(keep_units)) == keep_units.shape[0]
            assert keep_units.max() < self.encoding.shape[2]

            keep_mask[keep_units] = True
        # ablation by unit type
        elif (
            isinstance(keep_units, str)
            or isinstance(keep_units, (set, list, tuple))
            and all(isinstance(unit, str) for unit in keep_units)
        ):
            keep_units = self._check_unit_types(keep_units)

            for unit_type in keep_units:
                unit_indices = self.info[self.info["type"] == unit_type].index
                keep_mask[unit_indices] = True
        # ablation by info
        elif (
            isinstance(keep_units, dict)
            or isinstance(keep_units, Sequence)
            and all(isinstance(unit, dict) for unit in keep_units)
        ):
            keep_units = self._check_unit_info(keep_units)
            matched_units = self.info.merge(keep_units, how="inner")[
                "unit_idx"
            ].tolist()
            assert len(matched_units) == len(keep_units), (
                "Some units in keep_units were not found in the residual.",
                keep_units[~keep_units.index.isin(matched_units)].to_dict(
                    orient="records"
                ),
            )
            keep_mask[matched_units] = True
        else:
            raise ValueError(
                "keep_units must be a string, a dictionary, a sequence of strings, or a sequence of dictionaries."
            )

        return keep_mask

    def ablate(
        self,
        keep_units: Union[
            pd.DataFrame, torch.Tensor, Sequence[Union[str, Mapping[str, int]]]
        ],
        ablation: Optional[str],  # "zero", "mean"
        return_kept_indices: bool = False,
        aggregation: str = "sum",  # "cat", "sum"
        token_index: Optional[int] = None,
    ):
        """
        Ablates units from the residual encoding.

        Args:
            keep_units: Units to keep. Can be specified by a DataFrame, a tensor of global indices, a sequence of unit types, or a sequence of dictionaries.
            ablation: Ablation type. Can be "zero" or "mean".
            return_kept_indices: Whether to return the indices of the kept units.
            aggregation: Aggregation method for the ablated units. Can be "cat" or "sum".

        Returns:
            Ablated residual encoding.
        """
        token_slice = (
            slice(token_index, token_index + 1, 1)
            if token_index is not None
            else slice(None)
        )
        keep_mask = self._get_keep_mask(keep_units)

        if ablation is None and not keep_mask.all():
            raise ValueError("Ablation type must be provided when units are ablated.")

        if ablation is None or ablation == "zero":
            ablated_residual = torch.empty(
                self.encoding.shape[0],
                self.encoding.shape[1] if token_index is None else 1,
                0,
                self.encoding.shape[3],
                device=self.encoding.device,
            )
        elif ablation == "mean":
            if aggregation == "cat":
                raise NotImplementedError
            elif aggregation == "sum":
                ablated_residual = self.encoding[:, token_slice, ~keep_mask, :].mean(
                    dim=0, keepdim=True
                )
            else:
                raise ValueError(f"Invalid return mode: {aggregation}")
        else:
            raise ValueError(f"Invalid ablation type: {ablation}")

        if aggregation == "cat":
            residual = torch.cat(
                [self.encoding[:, token_slice, keep_mask, :], ablated_residual], dim=2
            )
        elif aggregation == "sum":
            memory_efficient: bool = True
            if memory_efficient:
                residual = torch.zeros(
                    self.encoding.shape[0],
                    self.encoding.shape[1] if token_index is None else 1,
                    self.encoding.shape[3],
                    device=self.encoding.device,
                )

                for unit_idx in keep_mask.nonzero(as_tuple=False).squeeze(dim=-1):
                    residual.add_(self.encoding[:, token_slice, unit_idx, :])

                residual.add_(ablated_residual.sum(dim=2))

                residual = residual.unsqueeze(dim=2)
            else:
                residual = self.encoding[:, token_slice, keep_mask, :].sum(
                    dim=2, keepdim=True
                ) + ablated_residual.sum(dim=2, keepdim=True)
        else:
            raise ValueError(f"Invalid return mode: {aggregation}")

        residual = (
            residual.view(residual.shape[0], residual.shape[2], residual.shape[3])
            if token_index is not None
            else residual
        )

        if return_kept_indices:
            return residual, keep_mask.nonzero(as_tuple=False).squeeze(dim=-1)

        return residual


@torch.no_grad()
def get_head_basis(
    device: torch.device,
    source_dir: Path,
):
    layer_head2basis = {}
    stream = Residual.stream(
        source_dir=source_dir,
        filter_fn=lambda info: info["type"] == "head",
        device=device,
        as_tensor_device=device,
    )
    for head_encoding, head_info in tqdm(stream):
        if not head_encoding.shape[1] == 1:
            raise ValueError(
                f"There is more than one token in the head encoding: {head_encoding.shape}"
            )
        encoding_pca = pca_fn(
            head_encoding.view(-1, head_encoding.shape[-1]),
            k=head_encoding.shape[-1],
            return_weights=True,
            return_variance=True,
            return_mean=True,
        )
        layer_idx = head_info["layer_idx"]
        head_idx = head_info["head_idx"]

        layer_head2basis[(layer_idx, head_idx)] = encoding_pca

    return layer_head2basis


@torch.no_grad()
@gin.configurable
def heads2pca(
    encoder_name: str,
    dataset: str,
    split: str,
    device: torch.device,
    embeds_dir=ENCODINGS_DIR,
    dst_dir=PROJECT_ROOT / "residual_pca",
):
    print(gin.operative_config_str())
    if dst_dir is not None:
        dst_path = dst_dir / f"{encoder_name}_{dataset}_{split}.pt"
        if dst_path.exists():
            print(f"File {dst_path} already exists. Loading from that.")
            return torch.load(dst_path, weights_only=True)["layer_head2pca"]
        dst_dir.mkdir(parents=True, exist_ok=True)
    else:
        dst_path = None

    source_dir: Path = embeds_dir / dataset / split / encoder_name

    layer_head2pca = get_head_basis(
        source_dir=source_dir,
        device=device,
    )
    layer_head2pca = {
        (layer, head): {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in basis.items()
        }
        for (layer, head), basis in layer_head2pca.items()
    }
    info = gin.operative_config_str()

    if dst_path is not None:
        torch.save({"info": info, "layer_head2pca": layer_head2pca}, dst_path)

    return layer_head2pca
