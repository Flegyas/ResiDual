import argparse
import itertools
import random
from pathlib import Path
from typing import Sequence

import gin
import pandas as pd
import torch
from latentis import PROJECT_ROOT
from tqdm import tqdm

from residual.data.encode import ENCODINGS_DIR
from residual.decomposition.unit_distance import (
    avg_correlation,
    avg_cosine,
    compute_spectral_distances,
    euclidean_avg,
    relative_avg_correlation,
    relative_avg_cosine,
)
from residual.residual import Residual, heads2pca


@gin.configurable
def diagonal_filter(layer_idx1, head_idx1, layer_idx2, head_idx2):
    return layer_idx1 == layer_idx2 and head_idx1 == head_idx2


def run(
    distance_type2fn,
    encoder_name: str,
    dataset1: str,
    dataset2: str,
    anchor_dataset: str,
    split: str,
    num_anchors: int,
    filter_fn: callable,
    device: torch.device,
):
    anchor_residual_dir = ENCODINGS_DIR / anchor_dataset / split / encoder_name
    anchor_offsets = random.sample(
        range(Residual.read_unit_shapes(source_dir=anchor_residual_dir)["head"][0]),
        num_anchors,
    )

    anchor_residual_stream = Residual.stream(
        source_dir=anchor_residual_dir,
        filter_fn=lambda x: x["type"] == "head",
        offsets=anchor_offsets,
        token_index=0,
        device=device,
    )

    random.seed(42)

    x_source_dir = ENCODINGS_DIR / dataset1 / split / encoder_name
    y_source_dir = ENCODINGS_DIR / dataset2 / split / encoder_name

    x_residual_composition: pd.DataFrame = Residual.read_composition(x_source_dir)
    x_num_layers = x_residual_composition[x_residual_composition["type"] == "head"][
        "layer_idx"
    ].nunique()
    x_num_heads = x_residual_composition[x_residual_composition["type"] == "head"][
        "head_idx"
    ].nunique()

    y_residual_composition: pd.DataFrame = Residual.read_composition(y_source_dir)
    y_num_layers = y_residual_composition[y_residual_composition["type"] == "head"][
        "layer_idx"
    ].nunique()
    y_num_heads = y_residual_composition[y_residual_composition["type"] == "head"][
        "head_idx"
    ].nunique()

    assert x_num_layers == y_num_layers
    assert x_num_heads == y_num_heads

    x_residual_stream = Residual.stream(
        source_dir=x_source_dir,
        filter_fn=lambda x: x["type"] == "head",
        token_index=0,
        device=device,
    )
    y_residual_stream = Residual.stream(
        source_dir=y_source_dir,
        filter_fn=lambda x: x["type"] == "head",
        token_index=0,
        device=device,
    )

    # num_layers = x_residual_stream.num_layers
    # num_heads = x_residual_stream.num_heads

    distance_type2distances = {
        distance_type: (
            torch.zeros(
                (x_num_layers, x_num_heads, x_num_layers, x_num_heads),
                dtype=torch.float32,
            )
            - torch.inf
        )
        for distance_type in distance_type2fn.keys()
    }

    for (x_unit, x_unit_info), (y_unit, y_unit_info), (
        anchor_unit,
        anchor_unit_info,
    ) in tqdm(
        zip(
            x_residual_stream,
            y_residual_stream,
            anchor_residual_stream,
            strict=True,
        ),
        total=x_num_layers * x_num_heads,
    ):
        if not filter_fn(
            layer_idx1=x_unit_info["layer_idx"],
            head_idx1=x_unit_info["head_idx"],
            layer_idx2=y_unit_info["layer_idx"],
            head_idx2=y_unit_info["head_idx"],
        ):
            continue
        assert all(
            x_unit_info[k] == y_unit_info[k] == anchor_unit_info[k]
            for k in x_unit_info.keys()
        ), (
            x_unit_info,
            y_unit_info,
            anchor_unit_info,
        )

        for distance_name, fn in distance_type2fn.items():
            distance = fn(
                x_unit=x_unit,
                y_unit=y_unit,
                anchors=anchor_unit,
            )
            distance_type2distances[distance_name][
                x_unit_info["layer_idx"],
                x_unit_info["head_idx"],
                y_unit_info["layer_idx"],
                y_unit_info["head_idx"],
            ] = distance.abs()

    return distance_type2distances


@gin.configurable
def exp2(
    encoders: Sequence[str],
    datasets: Sequence[str],
    anchor_dataset: str,
    split: str,
    num_anchors: int,
    filter_fn: callable,
    device: torch.device,
):
    pbar = tqdm(total=len(datasets) * len(encoders), desc="Exp2", position=0)
    for (dataset1, dataset2), encoder_name in itertools.product(
        itertools.product(["imagenet"], datasets), encoders
    ):
        output_dir = PROJECT_ROOT / "results" / "exp2"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{dataset1}_{dataset2}_{encoder_name}.pt"

        if output_file.exists():
            print(f"Skipping {output_file} as it already exists.")
            continue

        pbar.update(1)
        pbar.set_description(
            f"Dataset1: {dataset1}, Dataset2: {dataset2} Encoder: {encoder_name}"
        )

        gin_config_str = gin.operative_config_str()

        spectral_distances = compute_spectral_distances(
            x_layer_head2pca=heads2pca(
                encoder_name=encoder_name,
                dataset=dataset1,
                split=split,
                device=device,
            ),
            y_layer_head2pca=heads2pca(
                encoder_name=encoder_name,
                dataset=dataset2,
                split=split,
                device=device,
            ),
        )
        distance_name2distances = run(
            dataset1=dataset1,
            dataset2=dataset2,
            distance_type2fn={
                "rel_cosine": relative_avg_cosine,
                "abs_cosine": avg_cosine,
                "rel_correlation": relative_avg_correlation,
                "abs_correlation": avg_correlation,
                "euclidean": euclidean_avg,
            },
            encoder_name=encoder_name,
            anchor_dataset=anchor_dataset,
            split=split,
            num_anchors=num_anchors,
            filter_fn=filter_fn,
            device=device,
        )

        torch.save(
            dict(
                dataset1=dataset1,
                dataset2=dataset2,
                encoder_name=encoder_name,
                spectral_distances=spectral_distances,
                **distance_name2distances,
                cfg=gin_config_str,
            ),
            output_file,
        )


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

    config_file = Path(args.cfg)
    assert config_file.exists(), f"Config file {config_file} does not exist."

    cfg = gin.parse_config_files_and_bindings(
        [config_file],
        finalize_config=True,
        bindings=None,
    )

    exp2()
