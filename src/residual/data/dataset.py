import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Image,
    load_dataset,
    load_from_disk,
)
from latentis import PROJECT_ROOT
from latentis.data.dataset import HFDatasetView
from latentis.data.processor import ImageNet
from PIL import Image as PILImage
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from wilds import get_dataset as wilds_get_dataset
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset

from residual.data.data_registry import (
    dataset2classes_templates,
    eurosat_classname2dataset_class,
)


def _generate_synthetic_data(
    n_samples: int, shape: tuple, k_classes: int, range_min=-1, range_max=1
):
    # Samples per class
    samples_per_class = n_samples // k_classes

    # Initialize data and labels
    data = torch.zeros((n_samples, *shape))
    labels = torch.zeros(n_samples, dtype=torch.long)

    # Generate class centers within a wider range to ensure better separation
    class_centers = torch.linspace(range_min + 0.8, range_max - 0.8, k_classes)

    for class_idx in range(k_classes):
        # Define a center for the class with an increased margin between classes
        center = (
            torch.randn(*shape) * 0.05 + class_centers[class_idx]
        )  # Small random shift around each center

        # Generate samples for the current class from a Gaussian distribution with a small spread
        samples = torch.randn(samples_per_class, *shape) * 0.1 + center

        # Ensure that samples are naturally constrained by controlling the spread and center positions
        data[class_idx * samples_per_class : (class_idx + 1) * samples_per_class] = (
            samples
        )
        labels[class_idx * samples_per_class : (class_idx + 1) * samples_per_class] = (
            class_idx
        )

    return data, labels


def get_dataset(dataset: str, split: Optional[str] = None):
    if not (PROJECT_ROOT / "data" / dataset).exists():
        if dataset not in _dataset2build_fn:
            raise ValueError(f"Unknown dataset: {dataset}")
        build_fn = _dataset2build_fn[dataset]
        data = build_fn()
    else:
        data = load_from_disk(str(PROJECT_ROOT / "data" / dataset))

    if split is not None:
        data = data[split]

    if not isinstance(data, torch.Tensor) and (
        (
            isinstance(data, DatasetDict)
            and "sample_id" not in data["train"].column_names
        )
        or (isinstance(data, Dataset) and "sample_id" not in data.column_names)
    ):
        data = data.map(
            lambda x, i: {"sample_id": str(i)}, with_indices=True, batched=True
        )

    return data


def read_label_mapping(dataset_name: str):
    if dataset_name == "imagenet":
        file = PROJECT_ROOT / "data" / "ImageNet_mapping.tsv"
        data = pd.read_csv(file, sep="\t")
        data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
        data = data.to_dict(orient="records")
        data = {x["synset_id"]: x for x in data}
        return data
        lines = file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1000
        result = {}
        for label_index, line in enumerate(lines):
            synset_id, *lemmas = line.split(" ")
            label_name = " ".join(lemmas)
            pos, *offset = synset_id
            offset = "".join(offset).rjust(8, "0")
            synset_id = f"{pos}{offset}"
            result[synset_id] = dict(
                synset_id=synset_id, label_name=label_name, index=label_index
            )
        return result
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def build_imagenet():
    imagenet_view: HFDatasetView = ImageNet.build().run()["dataset_view"]
    data = imagenet_view.hf_dataset
    data = data.rename_columns({"image": "x", "label": "y"})
    fit_data = data["train"].train_test_split(
        test_size=0.2, seed=42, stratify_by_column="y"
    )
    data = DatasetDict(
        {
            "train": fit_data["train"],
            "val": fit_data["test"],
            "test": data["test"],
        }
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )
    data.save_to_disk(str(PROJECT_ROOT / "data" / "imagenet"))

    return data


def build_sketch():
    data = load_dataset("songweig/imagenet_sketch", trust_remote_code=True)
    data = data.rename_columns({"image": "x", "label": "y"})
    data = data["train"].train_test_split(
        test_size=0.2, seed=42, stratify_by_column="y"
    )
    train_data = data["train"]
    data = data["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="y")
    data = DatasetDict(
        {
            "train": train_data,
            "val": data["train"],
            "test": data["test"],
        }
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )
    data.save_to_disk(str(PROJECT_ROOT / "data" / "sketch"))

    return data


def build_gtsrb():
    data = load_dataset("bazyl/GTSRB")
    data = data.cast_column("Path", Image())
    data = data.rename_columns({"ClassId": "y", "Path": "x"})
    data = data.cast_column(
        "y",
        ClassLabel(
            num_classes=len(set(data["train"]["y"])),
            names=dataset2classes_templates["gtsrb"][0],
        ),
    )

    fit_data = data["train"].train_test_split(
        test_size=0.1, seed=42, stratify_by_column="y"
    )
    data = DatasetDict(
        {
            "train": fit_data["train"],
            "val": fit_data["test"],
            "test": data["test"],
        }
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )
    data.save_to_disk(str(PROJECT_ROOT / "data" / "gtsrb"))
    return data


def build_random():
    to_pil = ToPILImage()

    with torch.random.fork_rng():
        torch.manual_seed(42)
        images, labels = _generate_synthetic_data(
            n_samples=10_000, shape=(3, 224, 224), k_classes=10
        )
        images = [to_pil(image) for image in images]
        data = Dataset.from_dict(dict(x=images, y=labels))
        data = data.class_encode_column(column="y")
        data = data.cast_column("x", Image())
        data = data.train_test_split(test_size=0.2, seed=42, stratify_by_column="y")
        train_data = data["train"]
        test_data = data["test"].train_test_split(
            test_size=0.5, seed=42, stratify_by_column="y"
        )
        data = DatasetDict(
            train=train_data,
            val=test_data["train"],
            test=test_data["test"],
        )

        data = data.map(
            lambda batch, indices: {
                "sample_id": [str(i) for i in indices],
            },
            batched=True,
            with_indices=True,
        )

        data.save_to_disk(str(PROJECT_ROOT / "data" / "random"))

    return data


def build_mnist():
    data = load_dataset("mnist")
    data = data.rename_columns({"image": "x", "label": "y"})
    fit_data = data["train"].train_test_split(
        test_size=0.1, seed=42, stratify_by_column="y"
    )
    data = DatasetDict(
        {
            "train": fit_data["train"],
            "val": fit_data["test"],
            "test": data["test"],
        }
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )
    data.save_to_disk(str(PROJECT_ROOT / "data" / "mnist"))
    return data


def build_cifar10():
    data = load_dataset("cifar10")
    data = data.rename_columns({"img": "x", "label": "y"})
    fit_data = data["train"].train_test_split(
        test_size=0.1, seed=42, stratify_by_column="y"
    )
    data = DatasetDict(
        {
            "train": fit_data["train"],
            "val": fit_data["test"],
            "test": data["test"],
        }
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )
    data.save_to_disk(str(PROJECT_ROOT / "data" / "cifar10"))

    return data


def build_cifar100():
    data = load_dataset("cifar100")
    data = data.rename_columns({"img": "x", "fine_label": "y"})
    fit_data = data["train"].train_test_split(
        test_size=0.1, seed=42, stratify_by_column="y"
    )
    data = DatasetDict(
        {
            "train": fit_data["train"],
            "val": fit_data["test"],
            "test": data["test"],
        }
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )
    data.save_to_disk(str(PROJECT_ROOT / "data" / "cifar100"))

    return data


def build_svhn():
    data = load_dataset("ufldl-stanford/svhn", "cropped_digits")
    data = data.rename_columns({"image": "x", "label": "y"})
    del data["extra"]
    fit_data = data["train"].train_test_split(
        test_size=0.2, seed=42, stratify_by_column="y"
    )
    data = DatasetDict(
        {
            "train": fit_data["train"],
            "val": fit_data["test"],
            "test": data["test"],
        }
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )
    data.save_to_disk(str(PROJECT_ROOT / "data" / "svhn"))

    return data


def build_eurosat():
    data = load_dataset("mikewang/EuroSAT", trust_remote_code=True)

    class_names = list(eurosat_classname2dataset_class.values())
    data = data.cast_column(
        "class",
        ClassLabel(
            num_classes=len(class_names),
            names=class_names,
        ),
    )

    split_data = data["train"].train_test_split(
        test_size=0.2, seed=42, stratify_by_column="class"
    )
    train_data = split_data["train"]
    split_data = split_data["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="class"
    )
    val_data = split_data["train"]
    test_data = split_data["test"]

    data = DatasetDict(
        train=train_data,
        val=val_data,
        test=test_data,
    )

    data = data.rename_columns({"image_id": "sample_id", "class": "y"})
    data = data.map(
        lambda batch: {
            "sample_id": [str(x) for x in batch["sample_id"]],
        },
        batched=True,
    )

    data = data.map(
        lambda batch: {
            "x": [PILImage.open(x) for x in batch["image_path"]],
        },
        batched=True,
        remove_columns=["image_path"],
    )

    data.save_to_disk(str(PROJECT_ROOT / "data" / "eurosat"))

    return data


def build_sun397():
    data = load_dataset("1aurent/SUN397")

    split_data = data["train"].train_test_split(
        test_size=0.2, seed=42, stratify_by_column="label"
    )
    train_data = split_data["train"]
    split_data = split_data["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="label"
    )
    val_data = split_data["train"]
    test_data = split_data["test"]

    data = DatasetDict(
        train=train_data,
        val=val_data,
        test=test_data,
    )

    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )

    data = data.rename_columns({"image": "x", "label": "y"})

    data.save_to_disk(str(PROJECT_ROOT / "data" / "sun397"))

    return data


def build_dtd():
    data = load_dataset("tanganke/dtd")
    data = data.rename_columns({"image": "x", "label": "y"})
    fit_data = data["train"].train_test_split(
        test_size=0.2, seed=42, stratify_by_column="y"
    )
    data = DatasetDict(
        {
            "train": fit_data["train"],
            "val": fit_data["test"],
            "test": data["test"],
        }
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )
    data.save_to_disk(str(PROJECT_ROOT / "data" / "dtd"))

    return data


def build_resisc45():
    data = load_dataset("timm/resisc45")
    data = data.rename_columns({"image": "x", "label": "y", "image_id": "sample_id"})
    data = DatasetDict(
        {
            "train": data["train"],
            "val": data["validation"],
            "test": data["test"],
        }
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )
    data.save_to_disk(str(PROJECT_ROOT / "data" / "resisc45"))
    return data


def build_stanford_cars():
    train_data = load_dataset("tanganke/stanford_cars", split="train")
    test_data = load_dataset("tanganke/stanford_cars", split="test")

    data = DatasetDict(
        train=train_data,
        test=test_data,
    )

    data = data.rename_columns({"image": "x", "label": "y"})

    fit_data = data["train"].train_test_split(
        test_size=0.2, seed=42, stratify_by_column="y"
    )

    data = DatasetDict(
        {
            "train": fit_data["train"],
            "val": fit_data["test"],
            "test": data["test"],
        }
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )
    data.save_to_disk(str(PROJECT_ROOT / "data" / "stanford_cars"))

    return data


def build_pacs():
    data = load_dataset("flwrlabs/pacs")
    data = data.rename_columns({"image": "x", "label": "y"})

    split_data = data["train"].train_test_split(
        test_size=0.2, seed=42, stratify_by_column="y"
    )
    train_data = split_data["train"]
    split_data = split_data["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="y"
    )
    val_data = split_data["train"]
    test_data = split_data["test"]

    data = DatasetDict(
        train=train_data,
        val=val_data,
        test=test_data,
    )
    data = data.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )

    data.save_to_disk(str(PROJECT_ROOT / "data" / "pacs"))

    return data


def build_waterbirds():
    dataset: WaterbirdsDataset = wilds_get_dataset("waterbirds", download=True)

    root_dir = Path(dataset.data_dir)
    metadata_path = root_dir / "metadata.csv"

    metadata = pd.read_csv(metadata_path)

    splits = {
        "train": metadata[metadata["split"] == 0],
        "val": metadata[metadata["split"] == 1],
        "test": metadata[metadata["split"] == 2],
    }

    # Function to process each split without transformations
    def process_split(df):
        data = {"image": [], "label": [], "background": [], "filename": []}

        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_path = root_dir / row["img_filename"]
            label = int(row["y"])  # Bird type: 0 (landbird), 1 (waterbird)
            background = int(row["place"])  # 0 (land), 1 (water)

            img = PILImage.open(image_path)

            data["image"].append(img)
            data["label"].append(label)
            data["background"].append(background)
            data["filename"].append(row["img_filename"])

        return Dataset.from_dict(data)

    hf_dataset = DatasetDict(
        {
            "train": process_split(splits["train"]),
            "val": process_split(splits["val"]),
            "test": process_split(splits["test"]),
        }
    )

    hf_dataset = hf_dataset.rename_columns({"image": "x", "label": "y"})
    hf_dataset = hf_dataset.cast_column(
        "y",
        ClassLabel(
            num_classes=2,
            names=["landbird", "waterbird"],
        ),
    )
    hf_dataset = hf_dataset.cast_column(
        "background", ClassLabel(num_classes=2, names=["land", "water"])
    )

    hf_dataset = hf_dataset.map(
        lambda batch, indices: {
            "sample_id": [str(i) for i in indices],
        },
        batched=True,
        with_indices=True,
    )

    hf_dataset.save_to_disk(str(PROJECT_ROOT / "data" / "waterbirds"))

    shutil.rmtree(root_dir)

    return hf_dataset


_dataset2build_fn = {
    "imagenet": build_imagenet,
    "sketch": build_sketch,
    "gtsrb": build_gtsrb,
    "svhn": build_svhn,
    "random": build_random,
    "mnist": build_mnist,
    "cifar10": build_cifar10,
    "cifar100": build_cifar100,
    "sun397": build_sun397,
    "eurosat": build_eurosat,
    "dtd": build_dtd,
    "resisc45": build_resisc45,
    "stanford_cars": build_stanford_cars,
    "pacs": build_pacs,
    "waterbirds": build_waterbirds,
}


def register_dataset(name: str, build_fn):
    if name in _dataset2build_fn:
        raise ValueError(f"Dataset {name} is already registered")
    _dataset2build_fn[name] = build_fn


if __name__ == "__main__":
    # print(get_dataset(dataset="gtsrb")["train"].features["y"])
    build_waterbirds()
