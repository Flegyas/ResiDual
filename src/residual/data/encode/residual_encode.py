import argparse
import itertools
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Type

import gin
import torch
from latentis import PROJECT_ROOT
from torch.utils.data import DataLoader
from tqdm import tqdm

from residual.data.dataset import get_dataset
from residual.nn.encoder import Encoder, HFVisionEncoder
from residual.residual import Residual
from residual.tracing.tracer import ResidualTracer, get_registered_tracer
from residual.tracing.tracing_op import TracingOp


@torch.no_grad()
def _encode(
    residual_tracer: ResidualTracer,
    dataloader: DataLoader,
    device: str = "cuda",
    check_residual: bool = True,
    batch_limit: Optional[int] = None,
):
    with residual_tracer:
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if batch_limit is not None and i >= batch_limit:
                break

            x = batch["x"]
            # y = batch["y"]
            encode_out = residual_tracer.encode(
                x=x.to(device),
                return_residual=check_residual,
                keys=batch["sample_id"],
            )

            if check_residual:
                model_out = encode_out["model_out"]
                if model_out.ndim == 2:
                    model_out = model_out.unsqueeze(1)

                residual: Residual = encode_out["residual"]

                residual_dims = residual.encoding.ndim
                if residual_dims > 3:
                    residual_sum = residual.encoding.sum(
                        dim=tuple(range(2, residual_dims - 1))
                    )

                assert torch.allclose(
                    model_out,
                    residual_sum,
                    atol=1e-4
                    if not isinstance(residual_tracer.encoder, HFVisionEncoder)
                    else 1e-4,
                )


@gin.configurable
def encode(
    encoder_tracer: Mapping[str, str],
    get_encoder_fn: Callable,
    pooling_fn: Callable,
    datasets: Sequence[str],
    splits: Sequence[str],
    batch_size: int,
    num_workers: int,
    device: str,
    tracer_args: Mapping[str, Any] = {},
    check_residual: bool = False,
    tracer_ops: Sequence[Type[TracingOp]] = None,
    root_dir: Optional[Path] = None,
    batch_limit: Optional[int] = None,
):
    if root_dir is None:
        root_dir = PROJECT_ROOT / "encodings"

    tracer_ops = tracer_ops or []

    pbar = tqdm(total=len(encoder_tracer) * len(datasets) * len(splits))
    for (encoder_name, tracer_name), dataset_name, split in itertools.product(
        encoder_tracer.items(), datasets, splits
    ):
        pbar.set_description(f"{encoder_name} {dataset_name} {split}")

        encoder = get_encoder_fn(name=encoder_name, pooling_fn=pooling_fn)

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

        _tracer_args = tracer_args.copy()
        metadata = {
            "dataset": dataset_name,
            "split": split,
            "encoder": encoder_name,
            "dataset_size": dataset_size,
        }

        tracer_type = get_registered_tracer(name=tracer_name)

        _tracer_ops = [tracer_op(metadata=metadata) for tracer_op in tracer_ops]

        if any(
            (hasattr(tracer_op, "target_dir") and tracer_op.target_dir.exists())
            for tracer_op in _tracer_ops
        ):
            print(
                "Some output directory for tracing ops already exists. "
                "Skipping the whole experiment. (ok, maybe it would be better to skip only this tracer op)."
            )
            continue

        _encode(
            residual_tracer=tracer_type(
                module_name=encoder_name,
                encoder=encoder,
                tracer_ops=_tracer_ops,
                **_tracer_args,
            ),
            dataloader=dataloader,
            device=device,
            check_residual=check_residual,
            batch_limit=batch_limit,
        )

        pbar.update(1)
    pbar.close()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument(
        "--param", action="append", help="Gin parameter overrides.", default=[]
    )

    args = parser.parse_args()
    config_file = Path(args.cfg)
    assert config_file.exists(), f"Config file {config_file} does not exist."

    _cfg = gin.parse_config_files_and_bindings(
        [config_file], bindings=args.param, finalize_config=False
    )

    encode()


if __name__ == "__main__":
    run()
