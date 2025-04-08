from typing import Optional

import gin
import torch
from torch import nn


def pca_fn(
    x: torch.Tensor,
    k: Optional[int] = None,
    return_weights: bool = False,
    return_recon: bool = False,
    return_variance: bool = False,
    return_mean: bool = False,
):
    """Perform Principal Component Analysis (PCA) on the input data.

    Args:
        x (torch.Tensor): Input data of shape (n_samples, dim).
        k (int): Number of components.
        return_weights (bool, optional): Whether to return the singular values (weights). Defaults to False.
        return_recon (bool, optional): Whether to return the reconstruction of the input data. Defaults to False.
        return_variance (bool, optional): Whether to return the explained variance and explained variance ratio. Defaults to False.
        return_mean (bool, optional): Whether to return the mean of the input data. Defaults to False.

    Returns:
        dict: Dictionary containing the components, weights, explained variance, explained variance ratio, and reconstruction of the input data.
    """
    k = k or x.shape[1]

    x_mean = torch.mean(x, dim=0)
    # Center the data by subtracting the mean of each dimension
    x_centered = x - x_mean

    # Compute the SVD of the centered data
    U, S, Vt = torch.linalg.svd(x_centered, full_matrices=False)

    # Select the number of components we want to keep
    components = Vt[:k]

    result = {}
    result["components"] = components

    # Optionally return the singular values (weights) and eigenvalues
    if return_weights:
        singular_values = S

        eigenvalues = (singular_values**2) / (x.shape[0] - 1)

        participation_ratio = eigenvalues.sum() ** 2 / torch.sum(eigenvalues**2)

        result["weights"] = singular_values[:k]
        result["eigenvalues"] = eigenvalues[:k]
        result["participation_ratio"] = participation_ratio

    # Optionally return the explained variance and explained variance ratio
    if return_variance:
        n_samples = x.shape[0]
        explained_variance = (S**2) / (n_samples - 1)
        explained_variance_ratio = explained_variance / torch.sum(explained_variance)

        result["explained_variance"] = torch.cumsum(explained_variance[:k], dim=0)
        result["explained_variance_ratio"] = torch.cumsum(
            explained_variance_ratio[:k], dim=0
        )

    # Optionally return the reconstruction of the input data
    if return_recon:
        recon = x_centered @ components.T @ components + x_mean
        result["recon"] = recon

    # Optionally return the mean of the input data
    if return_mean:
        result["mean"] = result["mu"] = x_mean

    return result


@torch.no_grad()
@gin.configurable
def get_k_by_evr(
    pca_data,
    threshold: float,
):
    evr = pca_data["explained_variance_ratio"]
    return (evr > threshold).float().argmax(dim=-1) + 1


class Pruning(nn.Module):
    def forward(self, pca_out) -> int:
        raise NotImplementedError

    def properties(self):
        raise NotImplementedError


@gin.configurable
class KPruning(Pruning):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, pca_out) -> int:
        return self.k

    def properties(self):
        return {"k": self.k, "type": self.__class__.__name__}


@gin.configurable
class EVRPruning(Pruning):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, pca_out) -> int:
        return get_k_by_evr(
            pca_data=pca_out,
            threshold=self.threshold,
        )

    def properties(self):
        return {"threshold": self.threshold, "type": self.__class__.__name__}
