import functools
import itertools
from typing import Any, Mapping, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset
from latentis import PROJECT_ROOT
from latentis.space import Space
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from tqdm import tqdm

from residual.data.dataset import get_dataset
from residual.data.encode import ENCODINGS_DIR
from residual.decomposition.unit_distance import (
    score_unit_correlation,
)
from residual.nn.classifier import CentroidClassifier
from residual.residual import Residual
from residual.sparse_decomposition import SOMP

datasets = [
    "dtd",
    "cifar10",
    "cifar100",
    "eurosat",
    "gtsrb",
    "mnist",
    "resisc45",
    "stanford_cars",
    "sun397",
    "svhn",
]
models = [
    "vit_l",
    "dinov2_l",
    "openclip_b",
    "openclip_l",
    "clip_b",
    "clip_l",
    "blip_l_flickr",
]


def evaluate(dataloader, num_classes: int, device: torch.device, desc: str):
    metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    for preds, target in dataloader:
        metric.update(preds=preds.to(device), target=target.to(device))

    return metric.compute()


@torch.no_grad()
def encoding_predict(batch: Tuple[torch.Tensor, torch.Tensor], classifier, device):
    encoding, target_y = batch[0]
    encoding = encoding.to(device, non_blocking=True)
    target_y = target_y.to(device, non_blocking=True)

    if encoding.ndim != 2:
        raise ValueError(
            f"Expected residual to have shape (num_samples, encoding_dim), got {encoding.shape=}"
        )

    logits = classifier(encoding)

    preds = torch.argmax(logits, dim=-1)

    return preds, target_y


def supervised_score_heads(
    residual: torch.Tensor,
    dataset: Dataset,
    classifier: nn.Module,
    device: torch.device,
):
    assert (
        residual.ndim == 3
    ), f"Expected residual to have shape (n, u, d), got {residual.shape=}"
    unit_scores = torch.zeros(residual.shape[1])
    for unit_idx in range(residual.shape[1]):
        head_unit_score = evaluate(
            DataLoader(
                TensorDataset(
                    residual[:, unit_idx, :].unsqueeze(0),
                    torch.as_tensor(dataset["y"], dtype=torch.long).unsqueeze(0),
                ),
                collate_fn=functools.partial(
                    encoding_predict,
                    classifier=classifier,
                    device=device,
                ),
                batch_size=1,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
            ),
            num_classes=dataset.features["y"].num_classes,
            device=device,
            desc="fit head unit",
        ).item()
        unit_scores[unit_idx] = head_unit_score

    return unit_scores


def cosine_score(residual: torch.Tensor, device: torch.device, absolute: bool = False):
    residual = residual.to(device)
    residual = residual - residual.mean(dim=0)
    full_residual = F.normalize(residual.sum(dim=1), p=2, dim=-1)
    residual = F.normalize(residual, p=2, dim=-1)
    unit_scores = torch.einsum("nrd,nd->nr", residual, full_residual)
    unit_scores = unit_scores.mean(dim=0)

    return (unit_scores.abs() if absolute else unit_scores).cpu()


def correlation_score(residual: torch.Tensor, device: torch.device, property_enc=None):
    unit_scores = score_unit_correlation(
        residual=residual.to(device),
        property_encoding=property_enc
        if property_enc is not None
        else residual.sum(dim=1).to(device),
        method="pearson",
        memory_friendly=residual.size(0) > 10_000,
    ).cpu()

    return unit_scores


def decompose(encoding: torch.Tensor, dictionary: Mapping[str, Any]):
    decomposition = SOMP(k=64)
    return decomposition(
        X=encoding,
        dictionary=F.normalize(dictionary["encodings"]),
        descriptors=dictionary["dictionary"],
        device=encoding.device,
    )


@torch.no_grad()
def compute_ablations(
    encoder_name: str,
    dataset_name: str,
    classifier: CentroidClassifier,
    test_dataset: Dataset,
    decomp_dictionary,
    device: torch.device,
):
    fit_dataset = get_dataset(dataset_name, split="val")
    fit_residual: Residual = Residual.load(
        source_dir=ENCODINGS_DIR / dataset_name / "val" / encoder_name,
        device=device,
    )

    if len(fit_residual.size()) != 4:
        raise ValueError(
            f"Expected residual to have shape (num_samples, num_tokens, num_units, unit_dim), got {fit_residual.shape=}"
        )
    # if fit_residual.size()[0] != len(fit_dataset):
    #     raise ValueError(
    #         f"Expected residual to have the same number of samples as the dataset, got {fit_residual.size()[0]=} and {len(fit_dataset)=}"
    #     )
    if fit_residual.size()[1] != 1:
        raise ValueError(
            f"Expected residual to have only one token per sample, got {fit_residual.size(1)=}"
        )

    # base ablations
    ablations = [
        dict(keep_units="head", ablation="zero", type="heads"),
        dict(keep_units="head", ablation="mean", type="heads"),
        dict(keep_units={"mlp", "emb"}, ablation="zero", type="layers"),
        dict(keep_units={"mlp", "emb"}, ablation="mean", type="layers"),
        dict(keep_units="all", ablation=None, type="units"),
    ]

    num_layers, num_heads = fit_residual.num_layers, fit_residual.num_heads
    # layer-wise ablations
    for layer_idx in range(num_layers):
        layer_head_units = [
            dict(layer_idx=layer_idx, head_idx=head_idx, type="head")
            for head_idx in range(num_heads)
        ]
        for ablation_mode in ("zero", "mean"):
            ablations.append(
                dict(
                    keep_units=layer_head_units,
                    ablation=ablation_mode,
                    type=f"layer{layer_idx}_heads",
                )
            )
            ablations.append(
                dict(
                    keep_units=[
                        dict(layer_idx=layer_idx, head_idx=head_idx, type="head")
                        for layer_idx in range(0, layer_idx + 1)
                        for head_idx in range(num_heads)
                    ],
                    ablation=ablation_mode,
                    type=f"until_layer{layer_idx}_heads",
                )
            )

    out_fit_residual = fit_residual.encoding.sum(dim=(1, 2)).to(device)
    fit_residual.cpu()
    num_head_units = fit_residual.num_units("head")

    greedy_topks = [
        ("1", 1),
        ("1%", max(2, int(num_head_units * 0.01))),
        ("5%", max(1, int(num_head_units * 0.05))),
        ("10%", max(1, int(num_head_units * 0.1))),
        ("25%", max(1, int(num_head_units * 0.25))),
        ("50%", max(1, int(num_head_units * 0.5))),
        ("75%", max(1, int(num_head_units * 0.75))),
    ]

    residual_head_units, unit_idx = fit_residual.ablate(
        keep_units="head",
        ablation="zero",
        aggregation="cat",
        return_kept_indices=True,
        token_index=0,
    )
    assert (
        unit_idx.numel()
        == fit_residual.num_units("head")
        == residual_head_units.shape[1]
    )

    random_heads = lambda seed, **kwargs: torch.randn(  # noqa: E731
        num_head_units,
        generator=torch.Generator().manual_seed(seed),
    )

    fit_sample_ids = Space.load_from_disk(
        path=ENCODINGS_DIR / dataset_name / "val" / encoder_name / "head"
    ).keys
    supervised_selection = functools.partial(
        supervised_score_heads,
        dataset=fit_dataset.filter(lambda x: x["sample_id"] in fit_sample_ids),
        classifier=classifier,
    )
    task_conditioned_corr = functools.partial(
        correlation_score,
        property_enc=classifier.centroids.T,
    )
    full_out_corr = functools.partial(
        correlation_score,
        property_enc=out_fit_residual,
    )
    n_random_seeds = 10

    random_selections = [
        functools.partial(random_heads, seed=seed) for seed in range(n_random_seeds)
    ]

    for selection_method, method_name in zip(
        (
            supervised_selection,
            task_conditioned_corr,
            correlation_score,
            full_out_corr,
            *random_selections,
        ),
        (
            "supervised",
            "corr_task",
            "corr",
            "corr_full_out",
            *[f"random_{i}" for i in range(n_random_seeds)],
        ),
        strict=True,
    ):
        unit_scores = selection_method(residual=residual_head_units, device=device)
        # head_indices = fit_residual.get_unit_indices(unit_type="head")

        # unit_scores = unit_scores[head_indices]
        # unit_idx = unit_idx[head_indices]

        for topk_name, topk in greedy_topks:
            topk_units = unit_scores.topk(topk).indices
            topk_units = unit_idx[topk_units]
            for ablation_mode in ("zero", "mean"):
                ablations.append(
                    dict(
                        keep_units=topk_units,
                        ablation=ablation_mode,
                        type=f"greedy_{topk_name}_{method_name}_heads",
                    )
                )

    out_fit_residual.cpu()
    del out_fit_residual
    del fit_residual

    test_sample_ids = Space.load_from_disk(
        path=ENCODINGS_DIR / dataset_name / "test" / encoder_name / "head"
    ).keys
    test_target_y = test_dataset.filter(lambda x: x["sample_id"] in test_sample_ids)[
        "y"
    ]
    test_residual: Residual = Residual.load(
        source_dir=ENCODINGS_DIR / dataset_name / "test" / encoder_name,
        device=device if len(test_dataset) < 20_000 else "cpu",
    )

    ablation_results = []
    for ablation in ablations:
        ablation_type = ablation.pop("type")
        ablated_residual, residual_indices = test_residual.ablate(
            **ablation, return_kept_indices=True, token_index=0
        )
        summed_residual = ablated_residual.sum(dim=1)
        ablated_residual_score = evaluate(
            DataLoader(
                TensorDataset(
                    summed_residual.unsqueeze(0),
                    torch.as_tensor(test_target_y, dtype=torch.long).unsqueeze(0),
                ),
                collate_fn=functools.partial(
                    encoding_predict,
                    classifier=classifier,
                    device=device,
                ),
                batch_size=1,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
            ),
            num_classes=test_dataset.features["y"].num_classes,
            device=device,
            desc=ablation_type,
        ).item()

        if decomp_dictionary is not None:
            ablated_residual_decomp = decompose(
                encoding=summed_residual, dictionary=decomp_dictionary
            )
        else:
            ablated_residual_decomp = None

        ablation_results.append(
            dict(
                ablated_shape=ablated_residual.shape,
                score=ablated_residual_score,
                decomp=ablated_residual_decomp,
                type=ablation_type,
                residual_indices=residual_indices,
                **ablation,
            )
        )

    return ablation_results


@torch.no_grad()
def exp3(
    encoder_name: str,
    dataset_name: str,
    device,
):
    print(f"Running experiment 3 for {encoder_name} on {dataset_name}.")

    test_dataset = get_dataset(dataset_name, split="test")
    classifier = (
        CentroidClassifier(encoder_name=encoder_name, dataset_name=dataset_name)
        .to(device)
        .eval()
    )

    dictionary_path = PROJECT_ROOT / "dictionaries" / "textspan" / f"{encoder_name}.pt"
    if dictionary_path.exists():
        decomp_dictionary = torch.load(
            dictionary_path, weights_only=False, map_location=device
        )
    else:
        decomp_dictionary = None

    ablations = compute_ablations(
        encoder_name=encoder_name,
        dataset_name=dataset_name,
        classifier=classifier,
        test_dataset=test_dataset,
        decomp_dictionary=decomp_dictionary,
        device=device,
    )

    return dict(
        model_name=encoder_name,
        dataset_name=dataset_name,
        ablations=ablations,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pbar = tqdm(total=len(datasets) * len(models), desc="Exp3", position=0)
    for dataset_name, model_name in itertools.product(datasets, models):
        pbar.update(1)
        pbar.set_description(f"Dataset: {dataset_name}, Model: {model_name}")

        output_dir = PROJECT_ROOT / "results" / "exp3"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{dataset_name}_{model_name}.pt"

        if output_file.exists():
            print(f"Skipping {output_file} as it already exists.")
            continue

        data = exp3(encoder_name=model_name, dataset_name=dataset_name, device=device)

        torch.save(data, output_file)
