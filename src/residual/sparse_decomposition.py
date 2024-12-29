from abc import abstractmethod
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


class Projection(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(
        self,
        X: torch.Tensor,
        dictionary: torch.Tensor,
        descriptors: list,
        k: int,
        device,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    @property
    def key(self):
        return self.name


class PCA(Projection):
    def __init__(self, k: int, compute_evr: bool = True):
        super().__init__("pca")
        self.k = k
        self.compute_evr = compute_evr

    @property
    def key(self):
        return f"{self.name}_{self.k}"

    def forward(self, X: torch.Tensor, *args, **kwargs):
        return pca(*args, X=X, k=self.k, compute_evr=self.compute_evr, **kwargs)


def pca(X, k, compute_evr: bool = True, *args, **kwargs):
    # Center the data by subtracting the mean of each feature
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean

    # Compute the SVD of the centered data
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    # Select the number of components we want to keep
    components = Vt[:k]  # Transpose Vt to get right singular vectors in columns
    # explained_variance = S**2 / (X.size(0) - 1)  # Variance explained by each singular value
    # explained_variance_ratio = explained_variance[:k] / explained_variance.sum()

    if compute_evr:
        std_orig = torch.std(X, dim=0) ** 2

        evr = torch.zeros(k)
        l2 = torch.zeros(k)
        cosine = torch.zeros(k)
        for i in range(1, k + 1):
            filtering = components[:i].T @ components[:i]
            recon_i = X_centered @ filtering + X_mean
            std_recon = torch.std(recon_i, dim=0) ** 2
            evr[i - 1] = std_recon.sum(dim=-1) / std_orig.sum(dim=-1)
            cosine[i - 1] = F.cosine_similarity(X, recon_i).mean().item()
            l2[i - 1] = F.mse_loss(X, recon_i).item()

    recon = X_centered @ components.T @ components + torch.mean(X, dim=0)
    results = []
    chosen = -torch.ones(k, dtype=torch.long)
    weights = S[:k]

    return dict(
        recon=recon,
        results=results,
        chosen=chosen,
        weights=weights,
        order=None,
        components=components,
        evr=evr,
        l2=l2,
        cosine=cosine,
    )


class OMP(Projection):
    def __init__(self, k: int):
        super().__init__("omp")
        self.k = k

    def forward(
        self,
        X: torch.Tensor,
        dictionary: torch.Tensor,
        descriptors: list,
        device,
        *args,
        **kwargs,
    ):
        x_pc = pca(X, 1)["components"][0]
        omp_result = omp(
            *args,
            X=x_pc,
            orig_X=X,
            dictionary=dictionary,
            descriptors=descriptors,
            k=self.k,
            device=device,
            **kwargs,
        )
        return omp_result


@torch.no_grad()
def omp(
    X: torch.Tensor,
    orig_X: torch.Tensor,
    dictionary: torch.Tensor,
    descriptors: list,
    k: int,
    device,
    compute_evr: bool = False,
    *args,
    **kwargs,
):
    assert (
        dictionary.shape[1] == X.shape[0]
    ), f"Dictionary: {dictionary.shape[1]}, X: {X.shape[0]}"
    assert (
        len(descriptors) == dictionary.shape[0]
    ), f"descriptors: {len(descriptors)}, Dictionary: {dictionary.shape[0]}"

    X = X.to(device)
    dictionary = dictionary.to(device)
    # dictionary = torch.nn.functional.normalize(dictionary, dim=1)
    chosen = []
    notchosen = torch.ones(dictionary.shape[0]).to(device)
    results = []
    recon = torch.zeros_like(X)  # +X_mean
    residual = X.clone()
    evr = torch.zeros(k)
    cosine = torch.zeros(k)
    l2 = torch.zeros(k)
    std_orig = torch.std(orig_X, dim=0) ** 2
    for j in range(k):
        cross = residual @ dictionary.T
        cross = cross * notchosen
        proj_std = cross.abs()
        atom_idx = proj_std.argmax()
        chosen.append(atom_idx.item())
        notchosen[atom_idx] = 0
        results.append(descriptors[atom_idx])
        current_atoms = torch.index_select(
            dictionary, 0, torch.as_tensor(chosen).to(device)
        )
        lstsq_weights = torch.linalg.lstsq(
            current_atoms.T.double(), X.double()
        ).solution.float()
        recon = current_atoms.T @ lstsq_weights
        residual = X - recon
        recon_X = (
            orig_X
            @ (recon / recon.norm()).reshape(-1, 1)
            @ (recon / recon.norm()).reshape(1, -1)
        )
        std_recon = torch.std(recon_X, dim=0) ** 2
        evr[j] = std_recon.sum(dim=-1) / std_orig.sum(dim=-1)
        cosine[j] = F.cosine_similarity(orig_X, recon_X).mean().item()
        l2[j] = F.mse_loss(orig_X, recon_X).item()
    results = np.array(results, dtype="object")
    chosen = torch.as_tensor(chosen, dtype=torch.long)
    return dict(
        recon=recon,
        results=results,
        chosen=chosen,
        weights=lstsq_weights.abs().cpu(),
        order=None,
        evr=evr,
        l2=l2,
        cosine=cosine,
    )


class SOMP(Projection):
    def __init__(
        self,
        k: int,
        criterion="l1",
        pc: Optional[int] = None,
        compute_evr: bool = False,
    ):
        super().__init__(name="somp")
        self.k = k
        self.criterion = criterion
        self.pc = pc
        self.compute_evr = compute_evr

    @property
    def key(self):
        return f"{self.name}_{self.k}_{self.criterion}_{self.pc}"

    def forward(
        self,
        X: torch.Tensor,
        dictionary: torch.Tensor,
        descriptors: list,
        device,
        *args,
        **kwargs,
    ):
        orig_X = X
        # X_mean = torch.mean(X, dim=0)
        # X_centered = X - X_mean

        if self.pc is not None:
            pca_out = pca(X, self.pc)
            weights = pca_out["weights"].unsqueeze(1)
            X = pca_out["components"] * weights**2

        result = somp(
            X=X.double(),
            orig_X=orig_X.double(),
            pc=self.pc,
            dictionary=dictionary.double(),
            descriptors=descriptors,
            k=self.k,
            device=device,
            criterion=self.criterion,
            compute_evr=self.compute_evr,
        )

        return result


@torch.no_grad()
def somp(
    X: torch.Tensor,
    orig_X: torch.Tensor,
    pc,
    dictionary: torch.Tensor,
    descriptors: list,
    k: int,
    device,
    criterion="l1",
    centering: bool = True,
    compute_evr: bool = False,
    *args,
    **kwargs,
):
    assert dictionary.shape[0] >= k, f"Dictionary: {dictionary.shape[0]}, k: {k}"
    assert (
        dictionary.shape[1] == X.shape[1]
    ), f"Dictionary: {dictionary.shape[1]}, X: {X.shape[1]}"
    assert (
        len(descriptors) == dictionary.shape[0]
    ), f"descriptors: {len(descriptors)}, Dictionary: {dictionary.shape[0]}"
    X = X.to(device)
    orig_X_mean = orig_X.mean(dim=0)
    orig_X_centered = orig_X - orig_X_mean
    dictionary = dictionary.to(device)

    std_orig = torch.std(orig_X, dim=0) ** 2
    if centering:
        X_mean = X.mean(dim=0, keepdims=True)
        X = X - X_mean
    else:
        X_mean = torch.zeros_like(X)

    chosen = []
    notchosen = torch.ones(dictionary.shape[0]).to(device)
    results = []
    recon = torch.zeros_like(X)  # +X_mean
    residual = X.clone()
    evr = torch.zeros(k)
    l2 = torch.zeros(k)
    cosine = torch.zeros(k)
    for i in range(k):
        cross = residual @ dictionary.T
        cross = cross * notchosen

        if criterion == "l1":
            proj_scores = torch.sum(cross.abs(), dim=0)
        elif criterion == "std":
            proj_scores = torch.std(cross, dim=0)
        else:
            raise ValueError(f"Criterion {criterion} not recognized")

        atom_idx = proj_scores.argmax()

        chosen.append(atom_idx.item())
        notchosen[atom_idx] = 0
        results.append(descriptors[atom_idx])
        current_atoms = torch.index_select(
            dictionary, 0, torch.as_tensor(chosen).to(device)
        )
        lstsq_weights = torch.linalg.lstsq(current_atoms.T, X.T).solution
        recon = (current_atoms.T @ lstsq_weights).T

        residual = X - recon
        if pc is None:
            std_recon = torch.std((X_mean + recon), dim=0) ** 2
            evr[i] = std_recon.sum(dim=-1) / std_orig.sum(dim=-1)
            cosine[i] = F.cosine_similarity(orig_X, X_mean + recon).mean().item()
            l2[i] = F.mse_loss(orig_X, X_mean + recon).item()
        else:
            u, _, v = torch.linalg.svd(recon, full_matrices=False)
            somp_pcs = u @ v
            X_recon = orig_X_centered @ somp_pcs.T @ somp_pcs + orig_X_mean
            std_recon = torch.std(X_recon, dim=0) ** 2
            evr[i] = std_recon.sum(dim=-1) / std_orig.sum(dim=-1)
            cosine[i] = F.cosine_similarity(orig_X, X_recon).mean().item()
            l2[i] = F.mse_loss(orig_X, X_recon).item()

    results = np.asarray(results, dtype=object)
    weights = lstsq_weights.norm(dim=1).cpu()
    # weights = lstsq_weights.mean(dim=1).cpu()
    order = torch.argsort(weights, descending=True).cpu().numpy()
    chosen = torch.tensor(chosen).cpu()

    recon = X_mean + recon
    residual = X - recon

    return dict(
        recon=recon.cpu().float(),
        residual=residual.cpu().float(),
        results=results,
        chosen=chosen,
        weights=weights.float(),
        weights_full=lstsq_weights.float().T,
        order=order,
        evr=evr.float(),
        l2=l2.float(),
        cosine=cosine.float(),
    )


@torch.no_grad()
def smp(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    descriptors: list,
    k: int,
    device,
    criterion="l1",
    *args,
    **kwargs,
):
    assert dictionary.shape[0] >= k, f"Dictionary: {dictionary.shape[0]}, k: {k}"
    assert (
        dictionary.shape[1] == X.shape[1]
    ), f"Dictionary: {dictionary.shape[1]}, X: {X.shape[1]}"
    assert (
        len(descriptors) == dictionary.shape[0]
    ), f"descriptors: {len(descriptors)}, Dictionary: {dictionary.shape[0]}"

    X = X.to(device)
    dictionary = dictionary.to(device)

    # u, s, vh = torch.linalg.svd(X, full_matrices=False)
    # vh = vh[:rank]
    # dictionary = ((vh.T @ torch.linalg.inv(vh @ vh.T) @ vh) @ dictionary.T).T

    # dictionary = torch.nn.functional.normalize(dictionary, dim=1)
    X_mean = X.mean(dim=0, keepdims=True)
    X = X - X_mean
    chosen = []
    notchosen = torch.ones(dictionary.shape[0]).to(device)
    results = []
    recon = torch.zeros_like(X) + X_mean
    for _ in range(k):
        cross = X @ dictionary.T
        cross = cross * notchosen
        if criterion == "l1":
            proj_std = torch.sum(cross.abs(), dim=0)
        else:
            proj_std = torch.std(cross, dim=0)

        atom_idx = proj_std.argmax()
        chosen.append(atom_idx.item())
        notchosen[atom_idx] = 0

        results.append(descriptors[atom_idx])
        w = cross[:, atom_idx]
        X = X - torch.outer(
            w / torch.norm(dictionary[atom_idx]) ** 2, dictionary[atom_idx]
        )
        recon = recon + torch.outer(
            w / torch.norm(dictionary[atom_idx]) ** 2, dictionary[atom_idx]
        )
        dict_cov = dictionary @ dictionary.T
        dictionary = dictionary - torch.outer(
            dict_cov[:, atom_idx] / torch.norm(dictionary[atom_idx]) ** 2,
            dictionary[atom_idx],
        )

    # results = np.asarray(results, dtype=object)
    # results=results[torch.argsort(lstsq_weights.norm(dim=1), descending=True).cpu().numpy()]
    return recon, results, chosen


class Textspan(Projection):
    def __init__(self, k: int, rank: int):
        super().__init__("textspan")
        self.k = k
        self.rank = rank

    def forward(
        self,
        X: torch.Tensor,
        dictionary: torch.Tensor,
        descriptors: list,
        device,
        *args,
        **kwargs,
    ):
        return textspan(
            *args,
            X=X.double(),
            dictionary=dictionary.double(),
            descriptors=descriptors,
            k=self.k,
            rank=self.rank,
            device=device,
            **kwargs,
        )


@torch.no_grad()
def textspan(
    X,
    dictionary: torch.Tensor,
    descriptors,
    k: int,
    rank: int,
    device,
    *args,
    **kwargs,
):
    assert dictionary.shape[0] >= k, f"Dictionary: {dictionary.shape[0]}, k: {k}"
    assert (
        dictionary.shape[1] == X.shape[1]
    ), f"Dictionary: {dictionary.shape[1]}, X: {X.shape[1]}"
    assert (
        len(descriptors) == dictionary.shape[0]
    ), f"descriptors: {len(descriptors)}, Dictionary: {dictionary.shape[0]}"

    dictionary = F.normalize(dictionary)

    X = X.to(device)
    dictionary = dictionary.to(device)

    u, s, vh = torch.linalg.svd(X, full_matrices=False)
    vh = vh[:rank]
    dictionary = ((vh.T @ torch.linalg.inv(vh @ vh.T) @ vh) @ dictionary.T).T
    # u, s, vh = torch.linalg.svd(dictionary, full_matrices=False)
    # vh = dictionary.clone()
    # vh = vh[:rank]
    # X = ((vh.T @ torch.linalg.inv(vh @ vh.T) @ vh) @ X.T).T
    '''
    dist = ((new_dictionary - dictionary) ** 2).mean(1)
    dist = torch.nn.functional.cosine_similarity(new_dictionary, dictionary)
    top = torch.topk(dist, 5)
    least = torch.topk(dist, 5, largest=False)
    print("Most similar (cosine): ")
    for i, ind in enumerate(top.indices):
        print(texts[ind], top.values[i].item())
    print("\nMost different (cosine): ")
    for i, ind in enumerate(least.indices):
        print(texts[ind], least.values[i].item())
    """
    u, s, vh = torch.linalg.svd(new_dictionary, full_matrices=False)
    vh = vh[:20]

    dictionary = ((vh.T @ vh) @ new_dictionary.T).T
    """
    dictionary = new_dictionary
    '''
    # dictionary = torch.nn.functional.normalize(dictionary, dim=1)
    orig_X = X.clone()
    std_orig = torch.std(orig_X, dim=0) ** 2
    X_mean = X.mean(dim=0, keepdims=True)
    X = X - X_mean
    chosen = []
    results = []
    recon = torch.zeros_like(X) + X_mean
    weights = []
    evr = torch.zeros(k)
    l2 = torch.zeros(k)
    cosine = torch.zeros(k)
    for i in range(k):
        cross = X @ dictionary.T
        proj_std = torch.std(cross, dim=0)
        # proj_std = torch.sum(cross.abs(), dim=0)
        atom_idx = proj_std.argmax()
        # topn = torch.topk(proj_std, 5).indices
        # for t in topn:
        # print(texts[t])
        chosen.append(atom_idx.item())
        results.append(descriptors[atom_idx])
        w = cross[:, atom_idx]
        weights.append(w.cpu())
        X = X - torch.outer(
            w / torch.norm(dictionary[atom_idx]) ** 2, dictionary[atom_idx]
        )
        recon = recon + torch.outer(
            w / torch.norm(dictionary[atom_idx]) ** 2, dictionary[atom_idx]
        )
        dict_cov = dictionary @ dictionary.T

        dictionary = dictionary - torch.outer(
            dict_cov[:, atom_idx] / torch.norm(dictionary[atom_idx]) ** 2,
            dictionary[atom_idx],
        )
        std_recon = torch.std(recon, dim=0) ** 2
        evr[i] = std_recon.sum(dim=-1) / std_orig.sum(dim=-1)
        cosine[i] = F.cosine_similarity(orig_X, X_mean + recon).mean().item()
        l2[i] = F.mse_loss(orig_X, X_mean + recon).item()
    weights = torch.stack(weights, dim=0)
    weights = weights.norm(dim=1)
    chosen = torch.tensor(chosen)

    return dict(
        recon=recon,
        results=results,
        chosen=chosen,
        weights=weights,
        order=None,
        evr=evr,
        l2=l2,
        cosine=cosine,
    )


@torch.no_grad()
def replace_with_iterative_removal(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    descriptors: list,
    k: int,
    rank,
    device,
):
    assert dictionary.shape[0] >= k, f"Dictionary: {dictionary.shape[0]}, k: {k}"
    assert (
        dictionary.shape[1] == X.shape[1]
    ), f"Dictionary: {dictionary.shape[1]}, X: {X.shape[1]}"
    assert (
        len(descriptors) == dictionary.shape[0]
    ), f"descriptors: {len(descriptors)}, Dictionary: {dictionary.shape[0]}"

    results = []
    atoms = []
    X = X.double().to(device)
    dictionary = dictionary.numpy()
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    vh = vh[:rank]
    dictionary = (
        vh.T.dot(np.linalg.inv(vh.dot(vh.T)).dot(vh)).dot(dictionary.T).T
    )  # Project the text to the span of W_OV

    # original_data = np.copy(X)
    X = torch.from_numpy(X).double().to(device)
    mean_data = X.mean(dim=0, keepdim=True)
    X = X - mean_data
    reconstruct = einops.repeat(mean_data, "A B -> (C A) B", C=X.shape[0])
    reconstruct = reconstruct.detach().cpu()  # .numpy()
    dictionary = torch.from_numpy(dictionary).double().to(device)
    for _i in range(k):
        projection = X @ dictionary.T
        projection_std = projection.std(axis=0).detach().cpu().numpy()
        top_n = np.argmax(projection_std)
        atoms.append(top_n)
        results.append(descriptors[top_n])
        text_norm = dictionary[top_n] @ dictionary[top_n].T
        reconstruct += (
            (
                (X @ dictionary[top_n] / text_norm)[:, np.newaxis]
                * dictionary[top_n][np.newaxis, :]
            )
            .detach()
            .cpu()
            .numpy()
        )
        X = X - (
            (X @ dictionary[top_n] / text_norm)[:, np.newaxis]
            * dictionary[top_n][np.newaxis, :]
        )

        dictionary = (
            dictionary
            - (dictionary @ dictionary[top_n] / text_norm)[:, np.newaxis]
            * dictionary[top_n][np.newaxis, :]
        )

    return reconstruct, results, atoms
