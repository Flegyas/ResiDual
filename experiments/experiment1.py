import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import gin
import torch
from latentis import PROJECT_ROOT
from tqdm import tqdm

from residual.data.dataset import get_dataset
from residual.data.encode import ENCODINGS_DIR
from residual.intrinsic_dimensionality import twoNN
from residual.nn.utils import pca_fn
from residual.residual import Residual


@torch.no_grad()
def run(
    encoder_name: str,
    dataset_name: str,
    split: str,
    device: torch.device,
    sample_limit: int = 50_000,
):
    dataset = get_dataset(dataset=dataset_name, split=split)

    residual_dir: Path = ENCODINGS_DIR / dataset_name / split / encoder_name

    num_samples = len(dataset)
    if num_samples > sample_limit:
        offsets = torch.randperm(
            num_samples, generator=torch.Generator().manual_seed(42)
        )[:sample_limit]
        offsets = torch.sort(offsets).values
    else:
        offsets = None

    result = defaultdict(dict)

    for unit, unit_info in tqdm(
        Residual.stream(
            source_dir=residual_dir,
            offsets=offsets,
            # filter_fn=lambda x: x["type"] == "head",
            device=device,
            as_tensor_device=device,
        ),
        desc="Processing units",
        position=1,
    ):
        unit = unit.view(unit.shape[0], -1)
        if (unit.shape[0] != num_samples) and (unit.shape[0] != sample_limit):
            raise ValueError(
                f"Expected {num_samples} or 50_000 samples, got {unit.shape[0]}."
            )

        unit_pca = pca_fn(
            x=unit, k=unit.shape[-1], return_weights=True, return_variance=True
        )
        unit_dists = torch.cdist(unit, unit)
        unit.cpu()
        del unit

        unit_id_twonn = twoNN(X=unit_dists, distances=True)

        layer_idx = unit_info["layer_idx"]
        head_idx = unit_info["head_idx"]

        result[(layer_idx, head_idx, unit_info["type"])] = dict(
            pca_s=unit_pca["weights"].cpu(),
            pca_evr=unit_pca["explained_variance_ratio"].cpu(),
            id_twonn=unit_id_twonn.item(),
        )

    return result


@gin.configurable
def exp1(
    models: Sequence[str],
    datasets: Sequence[str],
    split: str,
    device: torch.device,
):
    pbar = tqdm(total=len(datasets) * len(models), desc="Exp1", position=0)
    for dataset_name, model_name in itertools.product(datasets, models):
        pbar.update(1)
        pbar.set_description(f"Dataset: {dataset_name}, Model: {model_name}")

        output_dir = PROJECT_ROOT / "results" / "exp1"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{dataset_name}_{model_name}.pt"

        if output_file.exists():
            print(f"Skipping {output_file} as it already exists.")
            continue

        data = run(
            encoder_name=model_name,
            dataset_name=dataset_name,
            split=split,
            device=device,
        )

        torch.save(
            dict(
                encoder_name=model_name,
                dataset_name=dataset_name,
                data=data,
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

    cfg = gin.parse_config_files_and_bindings(
        [config_file],
        finalize_config=True,
        bindings=None,
    )

    exp1()
