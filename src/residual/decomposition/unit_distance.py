import argparse
import hashlib
import itertools
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

import gin
import torch
import torch.nn.functional as F
from latentis import PROJECT_ROOT
from tqdm import tqdm


def relative_avg_cosine(
    x_unit: torch.Tensor,
    y_unit: torch.Tensor,
    anchors: torch.Tensor,
    normalize: bool = True,
):
    if normalize:
        anchors = F.normalize(anchors - anchors.mean(dim=0), p=2, dim=-1)
        x_unit = F.normalize(x_unit - x_unit.mean(dim=0), p=2, dim=-1)
        y_unit = F.normalize(y_unit - y_unit.mean(dim=0), p=2, dim=-1)

    x_rel = x_unit @ anchors.T
    y_rel = y_unit @ anchors.T

    x_rel = x_rel.mean(dim=0)
    y_rel = y_rel.mean(dim=0)

    return F.cosine_similarity(x_rel, y_rel, dim=-1)


def relative_avg_correlation(
    x_unit: torch.Tensor,
    y_unit: torch.Tensor,
    anchors: torch.Tensor,
    normalize: bool = False,
):
    if normalize:
        anchors = F.normalize(anchors - anchors.mean(dim=0), p=2, dim=-1)
        x_unit = F.normalize(x_unit - x_unit.mean(dim=0), p=2, dim=-1)
        y_unit = F.normalize(y_unit - y_unit.mean(dim=0), p=2, dim=-1)

    x_rel = x_unit @ anchors.T
    y_rel = y_unit @ anchors.T

    x_rel = x_rel.mean(dim=0)
    y_rel = y_rel.mean(dim=0)

    x_rel = x_rel - x_rel.mean(dim=0)
    y_rel = y_rel - y_rel.mean(dim=0)

    return F.cosine_similarity(x_rel, y_rel, dim=-1)


def avg_cosine(
    x_unit: torch.Tensor, y_unit: torch.Tensor, normalize: bool = False, **kwargs
):
    if normalize:
        x_unit = F.normalize(x_unit - x_unit.mean(dim=0), p=2, dim=-1)
        y_unit = F.normalize(y_unit - y_unit.mean(dim=0), p=2, dim=-1)

    x_unit = x_unit.mean(dim=0)
    y_unit = y_unit.mean(dim=0)

    return F.cosine_similarity(x_unit, y_unit, dim=-1)


def avg_correlation(x_unit: torch.Tensor, y_unit: torch.Tensor, **kwargs):
    x_unit = x_unit.mean(dim=0)
    y_unit = y_unit.mean(dim=0)

    x_unit = x_unit - x_unit.mean(dim=0)
    y_unit = y_unit - y_unit.mean(dim=0)

    return F.cosine_similarity(x_unit, y_unit, dim=-1)


def euclidean_avg(x_unit: torch.Tensor, y_unit: torch.Tensor, **kwargs):
    x_unit = x_unit.mean(dim=0)
    y_unit = y_unit.mean(dim=0)

    return (x_unit - y_unit).norm(p=2)


def get_k(evr, threshold):
    return (evr > threshold).float().argmax(dim=-1) + 1


@torch.no_grad()
@gin.configurable
def normalized_spectral_cosine(
    X: torch.Tensor, Y: torch.Tensor, weights_x=None, weights_y=None
):
    k_x, d_x = X.shape
    k_y, d_y = Y.shape

    C = (X @ Y.T).abs()  # abs cosine similarity
    if weights_x is None and weights_y is None:
        weight_matrix = torch.eye(k_x, k_y, device=X.device)
    else:
        weight_matrix = torch.outer(weights_x, weights_y)
        assert weight_matrix.shape == C.shape, (weight_matrix.shape, C.shape)
    weight_matrix = weight_matrix / weight_matrix.diag().norm(p=2)

    C = C * weight_matrix

    k = min(k_x, k_y)
    spectral_cosines = torch.zeros(k)

    # Find the k largest entries in the cosine similarity matrix
    for i in range(k):
        # Find the maximum entry in the cosine similarity matrix
        max_over_rows, max_over_rows_indices = torch.max(C, dim=1)
        max_index = torch.argmax(max_over_rows)
        max_value = max_over_rows[max_index]
        max_coords = (max_index, max_over_rows_indices[max_index])

        spectral_cosines[i] = max_value

        # Avoid reselecting the same row or column
        C[max_coords[0], :] = -torch.inf
        C[:, max_coords[1]] = -torch.inf

    return spectral_cosines.norm(p=2) / weight_matrix.diag().norm(p=2)


@gin.configurable
def threshold_pruning(pca, threshold: float):
    k = get_k(pca["explained_variance_ratio"], threshold=threshold)
    return k_pruning(pca, k=k)


@gin.configurable
def k_pruning(pca, k: int):
    return pca["components"][:k, :]


@gin.configurable
def get_weights(pca, k, mode: str):
    if mode == "explained_variance":
        cumulative_explained_variance = pca["explained_variance_ratio"][:k]
        explained_variance = torch.zeros_like(cumulative_explained_variance)
        explained_variance[0] = cumulative_explained_variance[0]
        explained_variance[1:] = (
            cumulative_explained_variance[1:] - cumulative_explained_variance[:-1]
        )
        return explained_variance
    elif mode == "singular":
        return pca["weights"][:k]
    elif mode == "eigs":
        return pca["weights"][:k] ** 2
    else:
        raise ValueError(f"Invalid mode: {mode}")


@torch.no_grad()
@gin.configurable
def compute_spectral_distances(
    x_layer_head2pca: Mapping[Tuple[int, int], Dict[str, torch.Tensor]],
    y_layer_head2pca: Mapping[Tuple[int, int], Dict[str, torch.Tensor]],
    x_basis_pruning: Callable[[torch.Tensor], torch.Tensor],
    y_basis_pruning: Callable[[torch.Tensor], torch.Tensor],
    distance_fn,
    weighted: bool = False,
    filter_fn: Callable = lambda *_: True,
):
    x_num_layers = len(set(layer for layer, _ in x_layer_head2pca.keys()))
    x_num_heads = len(set(head for _, head in x_layer_head2pca.keys()))

    y_num_layers = len(set(layer for layer, _ in y_layer_head2pca.keys()))
    y_num_heads = len(set(head for _, head in y_layer_head2pca.keys()))

    pbar = tqdm(total=x_num_layers * x_num_heads * y_num_layers * y_num_heads)

    distances = (
        torch.zeros(x_num_layers, x_num_heads, y_num_layers, y_num_heads) - torch.inf
    )
    for ((x_layer, x_head), x_pca), ((y_layer, y_head), y_pca) in itertools.product(
        x_layer_head2pca.items(), y_layer_head2pca.items()
    ):
        if not filter_fn(x_layer, x_head, y_layer, y_head):
            continue
        x_basis = x_basis_pruning(pca=x_pca)
        y_basis = y_basis_pruning(pca=y_pca)
        if weighted:
            x_weights = get_weights(pca=x_pca, k=x_basis.shape[0])
            y_weights = get_weights(pca=y_pca, k=y_basis.shape[0])
        else:
            x_weights = None
            y_weights = None

        dist = distance_fn(
            X=x_basis, Y=y_basis, weights_x=x_weights, weights_y=y_weights
        )
        distances[x_layer, x_head, y_layer, y_head] = float(dist)

        pbar.update(1)
        pbar.set_description(
            f"X: Layer {x_layer}, Head {x_head} | Y: Layer {y_layer}, Head {y_head} | Dist: {dist:.2f}"
        )

    return distances


def score_unit_correlation(
    residual: torch.Tensor,
    property_encoding: Optional[torch.Tensor] = None,
    method="pearson",
):
    """
    Computes Pearson or Spearman correlation between residuals and property encodings.

    A modified version of https://arxiv.org/abs/2406.01583

    Args:
        residual (torch.Tensor): Tensor of shape (n, r, d) representing residuals.
        property_encoding (torch.Tensor): Tensor of shape (k, d) representing property encodings.
        method (str): Correlation method, either "pearson" or "spearman". Default is "pearson".

    Returns:
        torch.Tensor: Correlation values for each residual.
    """
    output = residual.sum(dim=1)  # Summing over the r dimension -> (n, d)

    if property_encoding is not None:
        # Orthogonalize property encodings
        property_basis = torch.linalg.qr(property_encoding.T).Q.T
        out_projs = (
            output @ property_basis.T
        )  # Projecting onto the property directions -> (n, k)

        # Project each individual residual
        unit_projs = residual @ property_basis.T  # (n, r, d) @ (d, k) -> (n, r, k)
    else:
        # If property encodings are not provided, use the identity matrix as basis
        out_projs = output
        unit_projs = residual

    # If Spearman correlation is selected, rank the projections
    if method == "spearman":
        out_projs = torch.argsort(torch.argsort(out_projs, dim=0), dim=0).float()
        unit_projs = torch.argsort(torch.argsort(unit_projs, dim=0), dim=0).float()

    # Mean-center the projections for Pearson correlation only
    if method == "pearson":
        out_projs = out_projs - out_projs.mean(dim=0)
        unit_projs = unit_projs - unit_projs.mean(dim=0, keepdims=True)

    # Compute the covariance between out_projs and unit_projs
    covar = (out_projs[:, None, :] * unit_projs).mean(dim=0)  # (r, k)

    # Normalize to get the Pearson or Spearman correlation
    std_out_projs = out_projs.std(dim=0)  # Standard deviation of out_projs along n
    std_unit_projs = unit_projs.std(
        dim=0
    )  # Standard deviation of unit_projs for each residual

    # Compute final correlation by normalizing covariance
    correlation = covar / (std_out_projs[None, :] * std_unit_projs)  # (r, k)

    return correlation.mean(
        dim=-1
    )  # Average over k to return final result for each residual


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument(
        "--param", action="append", help="Gin parameter overrides.", default=[]
    )

    args = parser.parse_args()
    print(args)
    config_file = Path(args.cfg)
    assert config_file.exists(), f"Config file {config_file} does not exist."

    cfg = gin.parse_config_files_and_bindings([config_file], bindings=args.param)

    distances = compute_spectral_distances()

    gin_config_str = gin.config_str()

    config_hash = hashlib.sha256(gin_config_str.encode("utf-8")).hexdigest()[:8]

    output_dir = PROJECT_ROOT / "results" / "head2head" / config_hash
    output_dir.mkdir(exist_ok=True, parents=True)

    torch.save(distances, output_dir / "distances.pt")
    (output_dir / "cfg.txt").write_text(gin_config_str, encoding="utf-8")
#
