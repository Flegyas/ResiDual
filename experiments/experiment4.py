import itertools

import gin
from latentis import PROJECT_ROOT
from tqdm import tqdm

from residual.nn import train
from residual.nn.train import *  # noqa

if __name__ == "__main__":
    datasets = [
        # "imagenet",
        "gtsrb",
        "mnist",
        "cifar10",
        "cifar100",
        "eurosat",
        "dtd",
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

    exp_config_files = list((PROJECT_ROOT / "conf" / "train").glob("*.gin"))
    pbar = tqdm(total=len(datasets) * len(models) * len(exp_config_files), position=0)
    for encoder_name, dataset_name, exp_config_file in itertools.product(
        models, datasets, exp_config_files
    ):
        pbar.set_description(
            f"Dataset: {dataset_name}, Model: {encoder_name}, exp_type: {exp_config_file.name}"
        )

        bindings = [
            f"encoder_name='{encoder_name}'",
            f"dataset_name='{dataset_name}'",
            # 'run.trainer_args={"max_epochs": 1, "fast_dev_run": False, "log_every_n_steps": 5, "limit_train_batches": 0.1, "limit_val_batches": 0.1}',
        ]
        if exp_config_file.stem == "finetune" and "blip" in encoder_name:
            bindings.append("run.batch_size=32")

        cfg = gin.parse_config_files_and_bindings(
            [exp_config_file],
            bindings=bindings,
            finalize_config=False,
        )

        exp_name = gin.query_parameter("run.exp_type")

        train.run()

        pbar.update(1)
