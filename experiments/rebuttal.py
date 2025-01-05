import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from latentis import PROJECT_ROOT
from latentis.space import Space
from tqdm import tqdm

from residual.decomposition.ipca import IncrementalPCA
from residual.decomposition.unit_distance import normalized_spectral_cosine
from residual.nn.utils import pca_fn
from residual.residual import Residual
from residual.sparse_decomposition import SOMP


def mlp_somp():
    # SOMP on MLPs (to show their entanglement)

    stream = Residual.stream(
        source_dir=PROJECT_ROOT / "encodings" / "imagenet" / "train" / "openclip_l",
        device=device,
        filter_fn=lambda x: x["type"] == "mlp",
        token_index=0,
    )

    decomp_dictionary = torch.load(
        PROJECT_ROOT / "dictionaries" / "textspan" / "openclip_l.pt",
        weights_only=False,
        map_location=device,
    )

    output = []
    for unit, unit_info in tqdm(stream, desc="SOMP on MLPs"):
        decomposition = SOMP(k=10)
        decomp_out = decomposition(
            X=unit,
            dictionary=F.normalize(decomp_dictionary["encodings"]),
            descriptors=decomp_dictionary["dictionary"],
            device=device,
        )
        unit_info["decomposition"] = [str(x) for x in decomp_out["results"]]
        output.append(unit_info)

    out_file = PROJECT_ROOT / "rebuttal" / "mlp_somp_openclip_l_imagenet_train.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    json.dump(
        output,
        out_file.open("w", encoding="utf-8"),
        indent=4,
    )


def cls_vs_others():
    pca_others_file = (
        PROJECT_ROOT / "pca_encodings/imagenet/train/openclip_l/head_ipca.pt"
    )
    others_pca: torch.Tensor = IncrementalPCA.from_file(
        path=pca_others_file, device=device
    ).compute()

    stream = Residual.stream(
        source_dir=PROJECT_ROOT / "encodings" / "imagenet" / "train" / "openclip_l",
        device=device,
        filter_fn=lambda x: x["type"] == "head",
        token_index=0,
    )

    result = []
    for (head, head_info), (head_others_pcs, head_others_eigs) in tqdm(
        zip(
            stream,
            zip(others_pca["components"], others_pca["eigs"], strict=True),
            strict=True,
        ),
        desc="PCA(CLS) vs PCA(Others)",
    ):
        head_cls_pca = pca_fn(x=head, return_weights=True, return_variance=True)
        score = normalized_spectral_cosine(
            X=head_cls_pca["components"],
            Y=head_others_pcs,
            weights_x=head_cls_pca["eigenvalues"],
            weights_y=head_others_eigs,
        )
        head_info["cls_vs_others_spectral_cosine_weighted"] = score.item()

        score = normalized_spectral_cosine(
            X=head_cls_pca["components"],
            Y=head_others_pcs,
        )
        head_info["cls_vs_others_spectral_cosine_unweighted"] = score.item()

        pc_dists = head_cls_pca["components"] @ head_others_pcs.T
        head_info["cls_vs_others_pc_cos_sim_mean"] = pc_dists.diag().mean().item()

        result.append(head_info)

    out_file = (
        PROJECT_ROOT / "rebuttal" / "cls_vs_others_openclip_l_imagenet_train.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    json.dump(
        result,
        out_file.open("w", encoding="utf-8"),
        indent=4,
    )


def residual_somp():
    # SOMP on head representations from ResiDual (to analyze their specialization)

    encoder_name: str = "openclip_l"

    decomp_dictionary = torch.load(
        PROJECT_ROOT / "dictionaries" / "textspan" / f"{encoder_name}.pt",
        weights_only=False,
        map_location=device,
    )

    for dataset in ("mnist", "gtsrb"):
        dataset_output = defaultdict(list)
        for exp_type in ("residual_fine", "residual_full"):
            head_residual = Space.load_from_disk(
                PROJECT_ROOT
                / "optimized_encodings"
                / dataset
                / "test"
                / f"{encoder_name}_{exp_type}",
            )

            output = []
            for local_unit_idx in tqdm(
                range(head_residual.shape[1]), desc=f"SOMP on {exp_type} {dataset}"
            ):
                unit_info = {
                    "local_unit_idx": local_unit_idx,
                }
                unit = head_residual[:, local_unit_idx, :].to(device)
                decomposition = SOMP(k=10)
                decomp_out = decomposition(
                    X=unit,
                    dictionary=F.normalize(decomp_dictionary["encodings"]),
                    descriptors=decomp_dictionary["dictionary"],
                    device=device,
                )
                unit_info["decomposition"] = [str(x) for x in decomp_out["results"]]
                output.append(unit_info)

            dataset_output[exp_type] = output

        out_file = (
            PROJECT_ROOT
            / "rebuttal"
            / f"{exp_type}_head_somp_{encoder_name}_{dataset}_test.json"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)

        json.dump(
            dataset_output,
            out_file.open("w", encoding="utf-8"),
            indent=4,
        )


if __name__ == "__main__":
    device = "cuda"

    # mlp_somp()
    # cls_vs_others()
    residual_somp()
