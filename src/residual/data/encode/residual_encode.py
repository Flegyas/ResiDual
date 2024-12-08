import argparse
import itertools
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Type

import gin
import torch
from latentis import PROJECT_ROOT
from latentis.space import Space
from torch.utils.data import DataLoader
from tqdm import tqdm

from residual.data.dataset import get_dataset
from residual.data.encode import ENCODINGS_DIR
from residual.nn.encoder import Encoder
from residual.nn.model_registry import get_vision_encoder
from residual.residual import Residual
from residual.tracing.tracer import ResidualTracer, get_registered_tracer


def clean_dirs():
    for directory in ENCODINGS_DIR.iterdir():
        if directory.is_file():
            continue

        for encodings_dir in directory.glob("*/*"):
            encodings_data = list(encodings_dir.glob("*"))

            encodings_data = [f for f in encodings_data if f.is_dir()]
            for space in encodings_data:
                try:
                    Space.load_from_disk(space)
                except Exception as e:
                    print(f"Removing {encodings_dir} because of {e}")
                    # shutil.rmtree(encodings_dir)
                    break


@torch.no_grad()
def _encode(
    model_name: str,
    encoder: Encoder,
    tracer_type: Type[ResidualTracer],
    dataset_size: int,
    dataloader: DataLoader,
    flush_every: int,
    target_dir: Path,
    device: str = "cuda",
    tracer_args: Mapping[str, Any] = {},
    check_residual: bool = True,
):
    with tracer_type(
        module_name=model_name,
        encoder=encoder,
        target_dir=target_dir,
        dataset_size=dataset_size,
        **tracer_args,
    ) as residual_tracer:
        residual_tracer: ResidualTracer

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # if i == 20:
            #     break
            x = batch["x"]
            # y = batch["y"]
            encode_out = residual_tracer.encode(
                x=x.to(device),
                flush=i % flush_every == 0,
                return_residual=check_residual,
                keys=batch["sample_id"],
            )

            if check_residual:
                model_out = encode_out["model_out"]
                residual: Residual = encode_out["residual"]

                residual_sum = (
                    residual.encoding.sum(dim=1)
                    if residual_tracer.cls_only
                    else residual.encoding.sum(dim=2)
                )
                assert torch.allclose(
                    model_out,
                    residual_sum,
                    atol=1e-5 if "dino" not in model_name else 1e-4,
                )


@gin.configurable
def encode(
    encoder_tracer: Mapping[str, str],
    cls_only: bool,
    datasets: Sequence[str],
    splits: Sequence[str],
    overwrite: bool,
    batch_size: int,
    flush_every: int,
    num_workers: int,
    device: str,
    tracer_args: Mapping[str, Any] = {},
    check_residual: bool = False,
    root_dir: Optional[Path] = None,
):
    if root_dir is None:
        root_dir = PROJECT_ROOT / "encodings"

    pbar = tqdm(total=len(encoder_tracer) * len(datasets) * len(splits))
    for (encoder_name, tracer_name), dataset_name, split in itertools.product(
        encoder_tracer.items(), datasets, splits
    ):
        pbar.set_description(f"{encoder_name} {dataset_name} {split}")

        target_dir = root_dir / dataset_name / split / f"{encoder_name}"
        if (
            target_dir.exists()
            and target_dir.is_dir()
            and any(target_dir.glob("**/*") or [])
        ):
            if not overwrite:
                print(f"Directory {target_dir} already exists. Skipping.")
                continue
            else:
                shutil.rmtree(target_dir)

        encoder = get_vision_encoder(name=encoder_name, cls_only=cls_only)

        device = torch.device(device)

        dataset = get_dataset(dataset=dataset_name, split=split)
        dataset_size = len(dataset)
        dataloader = DataLoader(
            dataset,
            collate_fn=encoder.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
            num_workers=num_workers,
        )
        encoder: Encoder = encoder.to(device)
        encoder.eval()

        target_dir.mkdir(parents=True, exist_ok=True)

        _tracer_args = tracer_args.copy()
        if "metadata" not in _tracer_args:
            _tracer_args["metadata"] = {
                "dataset": dataset_name,
                "split": split,
                "encoder": encoder_name,
            }
        else:
            if "dataset" in _tracer_args["metadata"]:
                raise ValueError("metadata should not already contain 'dataset' key")
            if "split" in _tracer_args["metadata"]:
                raise ValueError("metadata should not already contain 'split' key")
            if "model" in _tracer_args["metadata"]:
                raise ValueError("metadata should not already contain 'model' key")

            _tracer_args["metadata"].update(
                {"dataset": dataset_name, "split": split, "model": encoder_name}
            )

        tracer_type = get_registered_tracer(name=tracer_name)

        _encode(
            encoder=encoder,
            model_name=encoder_name,
            dataset_size=dataset_size,
            dataloader=dataloader,
            flush_every=flush_every,
            tracer_type=tracer_type,
            tracer_args=_tracer_args,
            device=device,
            target_dir=target_dir,
            check_residual=check_residual,
        )

        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument(
        "--param", action="append", help="Gin parameter overrides.", default=[]
    )

    args = parser.parse_args()
    config_file = Path(args.cfg)
    assert config_file.exists(), f"Config file {config_file} does not exist."

    cfg = gin.parse_config_files_and_bindings(
        [config_file], bindings=args.param, finalize_config=False
    )

    encode()
