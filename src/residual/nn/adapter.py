from abc import abstractmethod
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import gin
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from residual.data.encode import ENCODINGS_DIR
from residual.nn.encoder import Encoder
from residual.nn.utils import KPruning, Pruning, pca_fn
from residual.residual import Residual


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x: torch.Tensor):
        raise NotImplementedError

    def get_unit_lambdas(self):
        raise NotImplementedError

    @abstractmethod
    def properties(self) -> Mapping[str, Any]:
        raise NotImplementedError


class SpectralAdapter(Adapter):
    def properties(self) -> Mapping[str, Any]:
        return {"n_lambdas": self.lambdas.numel()}

    def __init__(
        self,
        mean: torch.Tensor,
        basis: torch.Tensor,
        tune_lambdas: bool,
        lambda_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        inverse_basis = (
            torch.linalg.inv(basis @ basis.T) @ basis
            if basis.shape[0] != basis.shape[1]
            else basis.T
        )

        self.register_buffer("basis", basis)
        self.register_buffer("inverse_basis", inverse_basis)
        self.register_buffer("mean", mean)

        lambdas = None
        if isinstance(lambda_init, torch.Tensor):
            assert lambda_init.numel == basis.shape[0]
            lambdas = lambda_init
        elif isinstance(lambda_init, str):
            if lambda_init == "randn":
                lambdas = torch.randn(basis.shape[0], dtype=torch.float32) + 1
            elif lambda_init == "ones":
                lambdas = torch.ones(basis.shape[0], dtype=torch.float32)
            else:
                raise ValueError(f"Invalid lambda_init string: {lambda_init}")
        else:
            raise ValueError(f"Invalid lambda_init type: {type(lambda_init)}")

        self.lambdas = nn.Parameter(
            data=lambdas,
            requires_grad=tune_lambdas,
        )

    def get_unit_lambdas(self):
        return [self.lambdas.data]

    # def loss(self):
    #     return {"l1": self.lambdas.norm(p=1) * 0.001}

    def forward(self, x):
        x = x - self.mean

        x = torch.einsum("nd,dk->nk", x, self.basis * self.lambdas.unsqueeze(1))
        x = torch.einsum("nk,kd->nd", x, self.inverse_basis)

        # x = x + self.mean

        return x

    def encode(self, x):
        return self(x)

    def normalize_lambdas(self):
        self.lambdas.data = F.normalize(self.lambdas.data, p=2, dim=-1)
        return self.lambdas


class ResiDual(Adapter):
    def properties(self) -> Mapping[str, Any]:
        return self._properties

    def __init__(
        self,
        residual_types: Sequence[str],
        n_subspaces: int,
        residual_indices: torch.Tensor,
        bases: torch.Tensor,
        basis_lengths: torch.Tensor,
        means: torch.Tensor,
        component_padding: int,
        tune_lambdas: bool,
        lambda_init: Union[str, torch.Tensor] = "randn",
        ablation: Optional[str] = None,
        properties: Mapping[str, Any] = {},
    ):
        super().__init__()

        self.residual_types = residual_types

        assert ablation in {None, "sum", "mean", "zero"}, ablation

        self.register_buffer("residual_indices", residual_indices.to(torch.long))

        # assert residual_indices.shape[0]  == len(bases) == len(means), (
        #     residual_indices.shape,
        #     len(bases),
        #     len(means),
        # )

        residual_mask = torch.zeros(n_subspaces, dtype=torch.bool, requires_grad=False)
        residual_mask[self.residual_indices] = True
        assert residual_mask.sum() == self.residual_indices.shape[0]
        assert residual_mask.dtype == torch.bool
        assert (
            bases.ndim == 3
        ), f"bases must be (n_subspaces, n_components, n_features). Found: {bases.shape}"
        inverse_residual_bases = (
            torch.linalg.inv(bases @ bases.mT) @ bases
            if bases.shape[1] != bases.shape[2]
            else bases.mT
        )
        self.register_buffer("inverse_residual_bases", inverse_residual_bases)

        self.register_buffer("residual_mask", residual_mask)
        self.register_buffer("residual_indices", residual_indices)
        self.register_buffer("residual_bases", bases)
        self.register_buffer("residual_basis_lengths", basis_lengths)
        self.register_buffer("residual_means", means)
        self.register_buffer("component_padding", torch.tensor(component_padding))

        all_basis_components = basis_lengths.sum()
        lambdas = None
        if isinstance(lambda_init, torch.Tensor):
            assert lambda_init.numel() == all_basis_components, (
                lambda_init.numel(),
                all_basis_components,
            )
            lambdas = lambda_init
        elif isinstance(lambda_init, str):
            if lambda_init == "randn":
                lambdas = torch.randn(all_basis_components, dtype=torch.float32) + 1
            elif lambda_init == "ones":
                lambdas = torch.ones(all_basis_components, dtype=torch.float32)
            else:
                raise ValueError(f"Invalid lambda_init string: {lambda_init}")
        else:
            raise ValueError(f"Invalid lambda_init type: {type(lambda_init)}")

        self.lambdas = nn.Parameter(
            data=lambdas,
            requires_grad=tune_lambdas,
        )

        self.ablation = ablation

        self._properties = properties

    # def loss(self):
    #     return {"l1": self.lambdas.norm(p=1) * 1e-4}

    def get_unit_lambdas(self):
        return self.lambdas.data.split(self.residual_basis_lengths.tolist(), dim=0)

    def encode(self, x: torch.Tensor):
        adapted_residual = x[:, self.residual_indices, :]

        unit_lambdas = self.lambdas
        # unit_lambdas = F.dropout(unit_lambdas, p=0.5, training=self.training)
        unit_lambdas = unit_lambdas.split(self.residual_basis_lengths.tolist(), dim=0)
        unit_lambdas = pad_sequence(
            unit_lambdas, batch_first=True, padding_value=self.component_padding
        )
        unit_lambdas = F.pad(
            unit_lambdas,
            (0, self.residual_bases.shape[1] - unit_lambdas.shape[1]),
            value=self.component_padding,
        )

        adapted_residual = adapted_residual - self.residual_means
        # project onto the adapted basis
        adapted_residual = torch.einsum(
            "nrd,rkd->nrk",
            adapted_residual,
            self.residual_bases * unit_lambdas.unsqueeze(-1),
        )
        # project back to the original space using the inverse basis
        adapted_residual = torch.einsum(
            "nrk,rkd->nrd", adapted_residual, self.inverse_residual_bases
        )

        # adapted_residual = adapted_residual + self.residual_means

        return adapted_residual

    def forward(self, x: torch.Tensor):
        adapted_residual = self.encode(x)

        adapted_residual = adapted_residual.sum(dim=1)

        ablated_residual = x[:, ~self.residual_mask, :]

        if self.ablation is None or self.ablation == "sum":
            adapted_residual += ablated_residual.sum(dim=1)
        elif self.ablation == "mean":
            adapted_residual += ablated_residual.mean(dim=0, keepdim=True).sum(dim=1)
        elif self.ablation == "zero":
            pass
        else:
            raise ValueError(f"Invalid ablation: {self.ablation}")
        return adapted_residual


class CoarseAdapter(Adapter):
    def properties(self) -> Mapping[str, Any]:
        return self._properties

    def __init__(
        self,
        n_subspaces: int,
        residual_indices: torch.Tensor,
        tune_lambdas: bool,
        lambda_init: Union[str, torch.Tensor] = "randn",
        ablation: bool = False,
        l1_weight: float = 0,
    ):
        super().__init__()
        assert ablation in {None, "sum", "mean", "zero"}, ablation

        self.register_buffer("residual_indices", residual_indices.to(torch.long))
        residual_mask = torch.zeros(n_subspaces, dtype=torch.bool, requires_grad=False)
        residual_mask[self.residual_indices] = True
        assert residual_mask.sum() == self.residual_indices.shape[0]
        assert residual_mask.dtype == torch.bool
        self.register_buffer("residual_mask", residual_mask)
        self.ablation = ablation

        lambdas = None
        if isinstance(lambda_init, torch.Tensor):
            assert lambda_init.numel == residual_indices.shape[0]
            lambdas = lambda_init
        elif isinstance(lambda_init, str):
            if lambda_init == "randn":
                lambdas = (
                    torch.randn(residual_indices.shape[0], dtype=torch.float32) + 1
                )
            elif lambda_init == "ones":
                lambdas = torch.ones(residual_indices.shape[0], dtype=torch.float32)
            else:
                raise ValueError(f"Invalid lambda_init string: {lambda_init}")
        else:
            raise ValueError(f"Invalid lambda_init type: {type(lambda_init)}")

        self.lambdas = nn.Parameter(
            data=lambdas,
            requires_grad=tune_lambdas,
        )

        self.register_buffer("l1_weight", torch.tensor(l1_weight))

        self._properties = {
            "ablation": ablation,
            "lambda_init": lambda_init,
            "tune_lambdas": tune_lambdas,
            "l1_weight": l1_weight,
        }

    def encode(self, x: torch.Tensor):
        adapted_residual = x[:, self.residual_indices, :]

        return torch.einsum("nrd,r->nd", adapted_residual, self.lambdas)

    def forward(self, x: torch.Tensor):
        adapted_residual = self.encode(x)

        ablated_residual = x[:, ~self.residual_mask, :]

        if self.ablation is None or self.ablation == "sum":
            adapted_residual += ablated_residual.sum(dim=1)
        elif self.ablation == "mean":
            adapted_residual += ablated_residual.mean(dim=0, keepdim=True).sum(dim=1)
        elif self.ablation == "zero":
            pass
        else:
            raise ValueError(f"Invalid ablation: {self.ablation}")

        return adapted_residual

    def loss(self):
        if self.l1_weight != 0:
            return {"l1": self.lambdas.norm(p=1) * self.l1_weight}
        else:
            return {}

    def get_unit_lambdas(self):
        return [self.lambdas.data]


@gin.configurable
def build_residual_coarse_adapter(
    encoder_name: str,
    residual_types: Sequence[str],
    dataset_name: str,
    ablation: Optional[str],
    lambda_init: Union[str, torch.Tensor],
    l1_weight: float,
):
    residual_path = ENCODINGS_DIR / dataset_name / "train" / encoder_name
    residual_composition = Residual.read_composition(source_dir=residual_path)

    residual_indices = torch.as_tensor(
        residual_composition[residual_composition["type"].isin(residual_types)][
            "unit_idx"
        ].values
    )
    n_subspaces = len(residual_composition)

    adapter = CoarseAdapter(
        n_subspaces=n_subspaces,
        residual_indices=residual_indices,
        tune_lambdas=True,
        lambda_init=lambda_init,
        ablation=ablation,
        l1_weight=l1_weight,
    )

    return adapter


@gin.configurable
def build_proj_out_adapter(encoder: "Encoder"):
    d = encoder.encoding_dim
    return nn.Linear(in_features=d, out_features=d, bias=True)


@gin.configurable
def build_spectral_out_adapter(
    encoder_name: str,
    dataset_name: str,
    lambda_init: Union[str, torch.Tensor],
    device: torch.device,
):
    model_out = Residual.load(
        source_dir=ENCODINGS_DIR / dataset_name / "train" / encoder_name, device="cpu"
    ).encoding.sum(dim=(1, 2))

    model_out_pca = pca_fn(
        x=model_out.to(device),
        k=model_out.shape[0],
        return_variance=True,
        return_mean=True,
    )

    return SpectralAdapter(
        mean=model_out_pca["mean"],
        basis=model_out_pca["components"],
        tune_lambdas=True,
        lambda_init=lambda_init,
    )


@gin.configurable
def build_residual_spectral_adapter(
    encoder_name: str,
    residual_types: Sequence[str],
    dataset_name: str,
    ablation: Optional[str],
    basis_pruning: Optional[Union[int, Callable]],
    component_padding: int,
    lambda_init: Union[str, torch.Tensor],
    device: torch.device,
    tune_lambdas: bool = True,
):
    # lambdas = init_lambdas(
    #     encoder=_encoder,
    #     train_dataloader=train_dataloader,
    #     classifier=classifier,
    #     num_classes=num_classes,
    #     device=device,
    # )
    assert (
        isinstance(basis_pruning, (int, Pruning)) or basis_pruning is None
    ), f"Invalid basis_pruning: {basis_pruning}"
    if isinstance(basis_pruning, int):
        basis_pruning = KPruning(k=basis_pruning)

    residual_path = ENCODINGS_DIR / dataset_name / "train" / encoder_name
    stream = Residual.stream(
        source_dir=residual_path,
        filter_fn=lambda info: info["type"] in residual_types,
        device=device,
        token_index=0,
    )

    residual_shapes = Residual.read_unit_shapes(source_dir=residual_path)
    residual_composition = Residual.read_composition(source_dir=residual_path)

    unit_index2pca = {
        unit_info["unit_idx"]: pca_fn(
            x=unit_encoding,
            k=residual_shapes[unit_info["type"]][-1],
            return_variance=True,
            return_mean=True,
            return_weights=True,
        )
        for unit_encoding, unit_info in stream
    }
    unit_bases = []
    unit_means = []
    unit_lambdas = None if lambda_init != "eigen" else torch.empty(0, device=device)

    basis_lengths = []
    for _unit_idx, unit_pca in unit_index2pca.items():
        k = unit_pca["components"].shape[0]
        k = basis_pruning(unit_pca) if basis_pruning is not None else k
        unit_bases.append(unit_pca["components"])
        unit_means.append(unit_pca["mean"])
        basis_lengths.append(k)

        if lambda_init == "eigen":
            unit_lambdas = torch.cat([unit_lambdas, unit_pca["weights"][:k]])

    unit_bases = pad_sequence(
        [unit_pca["components"] for unit_pca in unit_index2pca.values()],
        batch_first=True,
        padding_value=0,
    )

    basis_lengths = torch.as_tensor(basis_lengths, dtype=torch.long, device=device)

    unit_means = torch.stack(unit_means, dim=0)

    num_subspaces = len(residual_composition)

    adapter = ResiDual(
        residual_types=residual_types,
        n_subspaces=num_subspaces,
        residual_indices=torch.as_tensor(list(unit_index2pca.keys()), device=device),
        bases=unit_bases,
        basis_lengths=basis_lengths,
        means=unit_means,
        tune_lambdas=tune_lambdas,
        component_padding=component_padding,
        lambda_init=lambda_init if lambda_init != "eigen" else unit_lambdas,
        ablation=ablation,
        properties={
            "ablation": ablation,
            "lambda_init": lambda_init,
            "component_padding": component_padding,
            "tune_lambdas": tune_lambdas,
            "residual_types": residual_types,
            **(
                {f"basis_pruning.{k}": v for k, v in basis_pruning.properties().items()}
                if basis_pruning is not None
                else {}
            ),
        },
    )

    return adapter
