from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import gin
import torch
from latentis.space import Space
from latentis.space.vector_source import HDF5Source
from torch import nn

from residual.decomposition.ipca import IncrementalPCA


class TracingOp(nn.Module):
    def __init__(self):
        super().__init__()


@gin.configurable
class SerializeResidualOp(TracingOp):
    def __init__(
        self,
        max_cached: int,
        root_dir: Path,
        metadata: Mapping[str, Any],
    ):
        super().__init__()

        self.max_cached = max_cached
        self.cached = 0
        self.target_dir = (
            root_dir
            / "encodings"
            / metadata["dataset"]
            / metadata["split"]
            / metadata["encoder"]
        )

        # target_dir = root_dir / dataset_name / split / f"{encoder_name}"
        # if (
        #     target_dir.exists()
        #     and target_dir.is_dir()
        #     and any(target_dir.glob("**/*") or [])
        # ):
        #     if not overwrite:
        #         print(f"Directory {target_dir} already exists. Skipping.")
        #         continue
        #     else:
        #         shutil.rmtree(target_dir)

        self.metadata = metadata

        self.initialized = False

        self.unit_type2space = None

    def exit(self, tracer):
        for unit_type, unit_space in self.unit_type2space.items():
            unit_space.save_to_disk(self.target_dir / unit_type)

        tracer.residual_composition.to_csv(
            self.target_dir / "composition.tsv", sep="\t", index=False
        )

        if tracer.out_proj is not None:
            torch.save(tracer.out_proj.cpu(), self.target_dir / "out_proj.pt")

    def build_space(self, unit_type: str, unit_shape: Sequence[int]):
        n, *t, u, d = unit_shape
        chunk_size = (
            (min(self.metadata["dataset_size"] // 4, 10_000),)
            + ((1,) if t else ())
            + (1, d)
        )

        return Space(
            vector_source=HDF5Source(
                shape=unit_shape,
                root_dir=self.target_dir / unit_type,
                h5py_params=dict(
                    # compression="gzip",
                    # compression_opts=4,
                    maxshape=(None, *unit_shape[1:]),
                    chunks=chunk_size,
                ),
            ),
            metadata=self.metadata,
        )

    def _init(self, unit_type2encodings: Mapping[str, torch.Tensor]):
        target_dir = self.target_dir
        if target_dir is not None:
            if target_dir.exists() and any(target_dir.glob("**/*") or []):
                raise ValueError(
                    f"Target directory {target_dir} already exists and is not empty"
                )
            target_dir.mkdir(parents=True, exist_ok=True)
            space_init = self.build_space

        else:
            space_init = lambda *_: Space(  # noqa: E731
                vector_source=None, metadata=self.metadata
            )

        self.unit_type2space: Mapping[str, Space] = {
            unit_type: space_init(
                unit_type=unit_type,
                unit_shape=unit_encodings.shape,
            )
            for unit_type, unit_encodings in unit_type2encodings.items()
        }

        self.initialized = True

    def forward(
        self,
        tracer,
        unit_type2encodings: Mapping[str, torch.Tensor],
        keys: Sequence[str] = None,
    ):
        if not self.initialized:
            self._init(unit_type2encodings)

        flush = self.cached >= self.max_cached
        self.cached += next(iter(unit_type2encodings.values())).shape[0]

        for unit_type, unit_space in self.unit_type2space.items():
            unit_space.add_vectors(
                vectors=unit_type2encodings[unit_type], keys=keys, write=flush
            )

        if flush:
            self.cached = 0


@gin.configurable
class SerializePCAOp(TracingOp):
    def __init__(
        self,
        root_dir: Path,
        metadata: Mapping[str, Any],
        k: int = None,
    ):
        super().__init__()

        if k is not None and k <= 0:
            raise ValueError("k must be a positive integer value or None")

        self.k = k

        self.target_dir = (
            root_dir
            / "pca_encodings"
            / metadata["dataset"]
            / metadata["split"]
            / metadata["encoder"]
        )

        self.metadata = metadata
        self.unit_type2ipca = None
        self.initialized = False

    def exit(self, tracer):
        self.target_dir.mkdir(parents=True, exist_ok=True)

        for unit_type, unit_ipca in self.unit_type2ipca.items():
            unit_ipca: IncrementalPCA
            torch.save(
                unit_ipca.cpu().state_dict(), self.target_dir / f"{unit_type}_ipca.pt"
            )

        tracer.residual_composition.to_csv(
            self.target_dir / "composition.tsv", sep="\t", index=False
        )

        # if tracer.out_proj is not None:
        #     torch.save(tracer.out_proj.cpu(), self.target_dir / "out_proj.pt")

    def _init(self, unit_type2encoding: Mapping[str, torch.Tensor]):  # noqa: F821
        target_dir = self.target_dir
        if target_dir.exists() and any(target_dir.glob("**/*") or []):
            raise ValueError(
                f"Target directory {target_dir} already exists and is not empty"
            )

        self.unit_type2ipca: Mapping[str, Space] = {
            unit_type: IncrementalPCA(k=self.k)
            for unit_type in unit_type2encoding.keys()
        }

        self.initialized = True

    def forward(
        self,
        tracer,
        unit_type2encodings: Mapping[str, torch.Tensor],
        keys: Sequence[str] = None,
    ):
        if not self.initialized:
            self._init(unit_type2encodings)

        for unit_type, unit_ipca in self.unit_type2ipca.items():
            encoding = unit_type2encodings[unit_type]
            n, *t, u, d = encoding.shape
            t = t[0] if t else 1
            encoding = encoding.view(n * t, u, d)
            encoding = encoding.permute(1, 0, 2)  # (u, n * t, d)
            unit_ipca.update(encoding)
