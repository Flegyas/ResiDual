import pytest
import torch

from residual.decomposition.unit_distance import score_unit_correlation


@pytest.mark.parametrize("method", ["pearson", "spearman"])
@pytest.mark.parametrize("use_property_encoding", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 2, 8, 16, 64])
def test_score_unit_correlation_equivalence(method, use_property_encoding, chunk_size):
    """
    Test that 'memory_friendly=True' yields exactly the same output
    as 'memory_friendly=False' for both Pearson and Spearman correlations,
    with or without a property_encoding, and for various chunk sizes.
    """

    # 1) Create dummy data of random shape
    #    - n: number of samples
    #    - r: number of residual units
    #    - d: dimensionality of each residual
    n, r, d = 32, 16, 8
    torch.manual_seed(0)  # for reproducibility
    residual = torch.randn(n, r, d)

    # 2) Optionally create a property encoding of shape (k, d)
    if use_property_encoding:
        k = 4  # number of properties
        property_encoding = torch.randn(k, d)
    else:
        property_encoding = None

    # 3) Compute the correlation with memory_friendly=False (the "reference" output)
    ref_output = score_unit_correlation(
        residual,
        property_encoding=property_encoding,
        method=method,
        memory_friendly=False,
    )

    # 4) Compute the correlation with memory_friendly=True
    test_output = score_unit_correlation(
        residual,
        property_encoding=property_encoding,
        method=method,
        memory_friendly=True,
        chunk_size=chunk_size,
    )

    # 5) Check for exact match (within floating point tolerance)
    #    For float32, you might need a slightly looser tolerance like (rtol=1e-5, atol=1e-6)
    assert torch.allclose(ref_output, test_output, atol=1e-5), (
        f"Mismatch found for method={method}, use_property_encoding={use_property_encoding}, "
        f"chunk_size={chunk_size}. "
        f"ref_output={ref_output}, test_output={test_output}"
    )
