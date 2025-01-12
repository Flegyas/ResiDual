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
    memory_friendly: bool = False,
    chunk_size: int = 10,
):
    """
    Computes Pearson or Spearman correlation between residuals and property encodings
    in either a standard or a more memory-friendly manner (if `memory_friendly=True`).

    A modified version of https://arxiv.org/abs/2406.01583

    Args:
        residual (torch.Tensor): Tensor of shape (n, r, d) representing residuals.
        property_encoding (torch.Tensor): Tensor of shape (k, d) representing property encodings.
        method (str): Correlation method, either "pearson" or "spearman". Default is "pearson".
        memory_friendly (bool): If True, perform the correlation computation in chunks along `r`.
        chunk_size (int): Size of each chunk to use if `memory_friendly=True`.

    Returns:
        torch.Tensor: Correlation values of shape (r,) for each residual.
    """
    device = residual.device
    dtype = residual.dtype

    # 1) Sum over the r dimension to get (n, d)
    output = residual.sum(dim=1)  # shape (n, d)
    n, r, d = residual.shape

    # 2) Handle property encodings: either an orthonormal basis or identity
    if property_encoding is not None:
        # Orthogonalize property encodings -> shape (k, d)
        property_basis = torch.linalg.qr(property_encoding.T).Q.T  # (k, d)
        k = property_basis.shape[0]

        # Project "output" onto property directions -> (n, k)
        out_projs = output @ property_basis.T

        # We will need to project "residual" as well, but possibly in chunks
        def project_residual_chunk(start, end):
            # Project the slice residual[:, start:end, :] onto property basis
            return residual[:, start:end, :] @ property_basis.T  # (n, chunk_size, k)

    else:
        # If not provided, treat as identity: out_projs = output, etc.
        k = d
        out_projs = output

        def project_residual_chunk(start, end):
            return residual[:, start:end, :]  # (n, chunk_size, d)

    # 3) Spearman rank transform helper (ranks along n-dim=0 for each [r, k])
    def spearman_rank_transform_2d(x_2d: torch.Tensor) -> torch.Tensor:
        """
        Rank-transform a (n, k) 2D tensor along dim=0 independently for each column.
        """
        # argsort(argsort(...)) along dim=0 => "rank" for each column
        # shape is preserved: (n, k)
        return torch.argsort(torch.argsort(x_2d, dim=0), dim=0).float()

    def spearman_rank_transform_3d(x_3d: torch.Tensor) -> torch.Tensor:
        """
        Rank-transform a (n, chunk_size, k) 3D tensor along dim=0
        independently for each [r, k] slice. Returns same shape.
        """
        # We do this by flattening over r,k -> but PyTorch supports 3D along dim=0
        # so we can directly do:
        #   x_3d -> argsort along n for each (r, k)
        # The shape is (n, chunk_size, k); we want to rank each [r, k] along n.
        return torch.argsort(torch.argsort(x_3d, dim=0), dim=0).float()

    # 4) If Spearman, rank-transform out_projs immediately (shape (n, k)).
    if method == "spearman":
        out_projs = spearman_rank_transform_2d(out_projs)

    # 5) For Pearson, we will need the mean later
    if method == "pearson":
        # Mean-center out_projs
        out_projs = out_projs - out_projs.mean(dim=0, keepdim=True)

    # 6) Precompute standard deviation of out_projs (shape (k,))
    std_out_projs = out_projs.std(dim=0)  # (k,)

    # 7) We'll accumulate covariance and std of unit_projs chunk by chunk
    covar = torch.empty(r, k, device=device, dtype=dtype)
    std_unit_projs = torch.empty(r, k, device=device, dtype=dtype)

    # Process chunks of the r dimension
    start_idx = 0
    while start_idx < r:
        end_idx = min(start_idx + chunk_size, r)
        # size = end_idx - start_idx

        # 7a) Project the chunk of residual onto property basis (or identity)
        chunk_projs = project_residual_chunk(start_idx, end_idx)  # shape (n, size, k/d)

        # 7b) If Spearman, rank-transform the chunk
        if method == "spearman":
            chunk_projs = spearman_rank_transform_3d(chunk_projs)  # (n, size, k)

        # 7c) If Pearson, mean-center the chunk
        if method == "pearson":
            # subtract mean over n dimension, keeping shape (1, size, k)
            chunk_projs = chunk_projs - chunk_projs.mean(dim=0, keepdim=True)

        # 7d) Compute covariance for the chunk:
        # covar_chunk[r_in_chunk, k] = (1/n) * sum over n of out_projs[:, k] * chunk_projs[:, r_in_chunk, k]
        # We can use einsum or manual summation
        # shape of out_projs is (n, k), shape of chunk_projs is (n, size, k)
        # => covar_chunk = (size, k)
        covar_chunk = torch.einsum("nk,nrk->rk", out_projs, chunk_projs) / float(n)

        # 7e) Compute std for each (r_in_chunk, k)
        # shape => (n, size, k) -> std over n => (size, k)
        std_chunk = chunk_projs.std(dim=0)  # shape (size, k)

        # 7f) Store in the big buffers
        covar[start_idx:end_idx, :] = covar_chunk
        std_unit_projs[start_idx:end_idx, :] = std_chunk

        start_idx = end_idx

        if not memory_friendly:
            # If we're NOT in memory-friendly mode, we only do one pass anyway;
            # but you can break early or skip chunking entirely if you like.
            break

    if not memory_friendly:
        # If memory_friendly=False, we never actually chunked anything above.
        # We just do the original (un-chunked) computation for the entire r in one pass.
        # So let's replicate that logic exactly here.

        # 1) Project entire residual: shape (n, r, k)
        if property_encoding is not None:
            unit_projs = residual @ property_basis.T
        else:
            unit_projs = residual  # shape (n, r, d)

        if method == "spearman":
            unit_projs = torch.argsort(torch.argsort(unit_projs, dim=0), dim=0).float()

        if method == "pearson":
            unit_projs = unit_projs - unit_projs.mean(dim=0, keepdim=True)

        # 2) Covariance: shape (r, k)
        covar = torch.einsum("nk,nrk->rk", out_projs, unit_projs) / float(n)

        # 3) std of unit_projs: shape (r, k)
        std_unit_projs = unit_projs.std(dim=0)

    # 8) Compute correlation = covar / (std_out_projs.unsqueeze(0) * std_unit_projs)
    #    shape => (r, k)
    # We unsqueeze std_out_projs => shape (1, k) for broadcasting
    correlation = covar / (std_out_projs.unsqueeze(0) * std_unit_projs)

    # 9) Mean over k dimension => shape (r,)
    return correlation.mean(dim=-1)


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
