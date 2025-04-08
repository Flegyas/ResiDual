from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from sklearn.decomposition import PCA
from torch import nn

from residual.nn.utils import pca_fn


class IncrementalPCA(nn.Module):
    @classmethod
    def from_file(cls, path: Path, device: torch.device = "cpu"):
        state_dict = torch.load(path, map_location=device, weights_only=True)
        ipca = cls()
        ipca.load_state_dict(state_dict)
        return ipca

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        """Custom load_state_dict to properly handle buffers"""

        loaded_k = state_dict.get("k", torch.empty(0))
        loaded_mu = state_dict.get("mu")
        loaded_cov = state_dict.get("cov_matrix")
        loaded_total_samples = state_dict.get("total_samples")

        if loaded_k.numel() == 0:
            # If it has 0 elements, interpret that as k=None
            self.k = None
        else:
            # Otherwise, interpret the single value as the actual k
            self.k = loaded_k.item()

        self.mu = loaded_mu
        self.cov_matrix = loaded_cov
        self.total_samples = loaded_total_samples

        super().load_state_dict(state_dict, strict=False)

    def __init__(self, k: Optional[int] = None):
        """
        Initialize the Incremental PCA object.
        Args:
            k (int, optional): Number of principal components to compute. If None, compute all components.
        """
        super().__init__()

        self.register_buffer("k", torch.tensor(k) if k is not None else None)
        self.register_buffer("mu", None)
        self.register_buffer("cov_matrix", None)
        self.register_buffer("total_samples", torch.tensor(0))

    def update(self, X: torch.Tensor):
        """Update the running mean and covariance matrix with new data X.

        Args:
            X (torch.Tensor): New data points of shape (r, n, d) or (n, d).
        """
        # Handle both (n, d) and (r, n, d) cases
        if X.ndim == 2:
            X = X.unsqueeze(0)

        r, n, d = X.shape
        batch_mean = torch.mean(X, dim=1)
        X_centered = X - batch_mean[:, None, :]

        if self.mu is None:
            self.mu = batch_mean
            self.cov_matrix = torch.einsum("rni,rnj->rij", X_centered, X_centered)
        else:
            # Update mean
            delta = batch_mean - self.mu
            total_samples = self.total_samples + n
            self.mu += delta * n / total_samples

            # Update covariance matrix
            self.cov_matrix += torch.einsum(
                "rni,rnj->rij", X_centered, X_centered
            )  # (r, d, d)
            self.cov_matrix += torch.einsum("ri,rj->rij", delta, delta) * (
                self.total_samples * n / total_samples
            )

        self.total_samples += n

    def compute(self):
        """
        Compute the principal components using the accumulated mean and covariance matrix.
        Returns:
            dict: Contains eigenvalues, eigenvectors, and mean for each of the r datasets.
        """
        if self.total_samples == 0:
            raise ValueError("No data to compute PCA. Please call update() with data.")

        # Normalize covariance matrix
        cov_matrix_norm = self.cov_matrix / (
            self.total_samples - 1
        )  # Normalize for unbiased estimation

        # Perform eigen decomposition for each dataset in parallel
        eigenvalues, eigenvectors = torch.linalg.eigh(
            cov_matrix_norm
        )  # Shapes: (r, d), (r, d, d)
        idx = torch.argsort(eigenvalues, descending=True, dim=1)
        eigenvalues = torch.gather(eigenvalues, dim=1, index=idx)
        eigenvectors = torch.stack(
            [eigenvectors[i, :, idx[i]] for i in range(eigenvectors.size(0))]
        )

        # If n_components is specified, truncate
        if self.k is not None:
            eigenvalues = eigenvalues[:, : self.k]  # (r, k)
            eigenvectors = eigenvectors[:, :, : self.k]  # (r, d, k)

        return dict(
            eigs=eigenvalues.squeeze(0),
            components=eigenvectors.permute(0, 2, 1).squeeze(0),
            mu=self.mu.squeeze(0),
        )


# Function to compute PCA directly on the full ∂å†å for comparison
def full_pca(X, n_components=None):
    """
    Compute PCA directly on the full dataset.
    Args:
        X (torch.Tensor): Data of shape (n_samples, n_features).
        n_components (int, optional): Number of components to compute. If None, compute all components.
    Returns:
        eigenvalues (torch.Tensor): Eigenvalues sorted in descending order.
        eigenvectors (torch.Tensor): Corresponding eigenvectors sorted by eigenvalues.
    """
    # Center the data
    X_centered = X - torch.mean(X, dim=0)

    # Compute covariance matrix
    cov_matrix = torch.mm(X_centered.T, X_centered) / (X.size(0) - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # If n_components is specified, truncate
    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    return eigenvalues, eigenvectors.T


def assert_allclose_with_info(tensor1, tensor2, atol=1e-5, message=""):
    assert torch.allclose(
        tensor1, tensor2, atol=atol
    ), f"{message}: {tensor1} vs {tensor2}. Max Error: {torch.abs(tensor1 - tensor2).max()}"


# Test Incremental PCA against Full PCA and Sklearn PCA
def test_incremental_pca():
    torch.manual_seed(42)  # For reproducibility

    u, n, d = 5, 10_000, 768
    k = 5

    # Generate synthetic dataset
    X = torch.randn(u, n, d)

    # Split dataset into random batch sizes for incremental updates
    batch_limits = torch.randperm(n)[:10].sort().values
    batch_sizes = torch.cat([batch_limits, torch.tensor([n])]) - torch.cat(
        [torch.tensor([0]), batch_limits]
    )
    # batch_sizes = batch_sizes[batch_sizes > 100].tolist()

    batches = torch.split(X, tuple(batch_sizes), dim=1)

    # Initialize Incremental PCA
    ipca = IncrementalPCA(k=k)

    # Incrementally update with batches
    for batch in batches:
        ipca.update(batch)

    # Compute Incremental PCA results
    ipca_result = ipca.compute()
    inc_eigenvalues = ipca_result["eigs"]
    inc_eigenvectors = ipca_result["components"]

    for unit_idx in range(u):
        unit = X[unit_idx]

        # Compute Full PCA results
        full_eigenvalues, full_eigenvectors = full_pca(unit, n_components=k)

        # Compute PCA using sklearn for validation
        sklearn_pca = PCA(n_components=k)
        sklearn_pca.fit(unit.numpy())
        sklearn_eigenvalues = torch.tensor(
            sklearn_pca.explained_variance_, dtype=torch.float32
        )
        sklearn_eigenvectors = torch.tensor(
            sklearn_pca.components_.T, dtype=torch.float32
        )

        # Compute PCA using custom pca_fn for validation
        pca = pca_fn(x=unit, k=k, return_weights=True)
        pca_fn_eigenvalues = pca["eigenvalues"]

        atol = 1e-3

        # Compare eigenvalues
        assert_allclose_with_info(
            inc_eigenvalues[unit_idx],
            full_eigenvalues,
            atol=atol,
            message="Eigenvalues mismatch between Incremental and Full PCA",
        )
        assert_allclose_with_info(
            inc_eigenvalues[unit_idx],
            sklearn_eigenvalues,
            atol=atol,
            message="Eigenvalues mismatch between Incremental PCA and Sklearn",
        )
        assert_allclose_with_info(
            inc_eigenvalues[unit_idx].abs(),
            pca_fn_eigenvalues.abs(),
            atol=atol,
            message="Eigenvalues mismatch between Incremental PCA and pca_fn",
        )

        # Compare eigenvectors (allow sign ambiguity)
        assert_allclose_with_info(
            torch.abs(inc_eigenvectors[unit_idx]),
            torch.abs(full_eigenvectors),
            atol=atol,
            message="Eigenvectors mismatch between Incremental and Full PCA",
        )
        assert_allclose_with_info(
            torch.abs(inc_eigenvectors[unit_idx]),
            torch.abs(sklearn_eigenvectors.T),
            atol=atol,
            message="Eigenvectors mismatch between Incremental PCA and Sklearn",
        )
        assert_allclose_with_info(
            torch.abs(inc_eigenvectors[unit_idx]),
            torch.abs(pca["components"]),
            atol=atol,
            message="Eigenvectors mismatch between Incremental PCA and pca_fn",
        )

    print(
        "All tests passed! Incremental PCA matches Full PCA, Sklearn PCA, and pca_fn."
    )


# Run the test
if __name__ == "__main__":
    test_incremental_pca()
