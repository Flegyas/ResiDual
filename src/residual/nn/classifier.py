import itertools
from pathlib import Path
from typing import Sequence

import gin
import torch
import torch.nn.functional as F
from datasets import Dataset
from latentis import PROJECT_ROOT
from torch import nn
from transformers import ViTForImageClassification

from residual.data.data_registry import dataset2classes_templates
from residual.data.dataset import get_dataset
from residual.nn.encoder import Encoder
from residual.nn.model_registry import get_text_encoder
from residual.residual import Residual


@gin.configurable
class LinearClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, bias: bool):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features, out_features=num_classes, bias=bias
        )

    def forward(self, x):
        return self.fc(x)


@gin.configurable
class ViTClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
    ):
        super().__init__()
        vit_model = ViTForImageClassification.from_pretrained(encoder_name)
        self.linear: nn.Linear = vit_model.classifier

    @property
    def num_classes(self):
        return self.linear.weight.shape[1]

    def forward(self, x: torch.Tensor):
        return self.linear(x)


@gin.configurable
class CentroidClassifier(nn.Module):
    # @classmethod
    # def from_tensor(cls, centroids: torch.Tensor):
    #     model = cls.__new__(cls)
    #     super(CentroidClassifier, model).__init__()
    #     model.register_buffer("centroids", centroids)
    #     return model

    @classmethod
    def from_file(cls, file: Path, x_normalize: bool = True):
        class_encodings = torch.load(file, weights_only=True)

        return cls(
            class_names=class_encodings["classes"],
            centroids=class_encodings["class_encodings"],
            x_normalize=x_normalize,
        )

    def __init__(
        self,
        class_names: Sequence[str],
        centroids: torch.Tensor,
        x_normalize: bool = True,
    ):
        super().__init__()

        self.class_names = class_names
        self.register_buffer("centroids", centroids)
        self.register_buffer("x_normalize", torch.tensor(x_normalize, dtype=torch.bool))

    @property
    def num_classes(self):
        return self.centroids.shape[1]

    def forward(self, x: torch.Tensor):
        centroids = self.centroids

        if self.x_normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x @ centroids


@torch.no_grad()
@gin.configurable
def build_centroid_classifier(
    encoder_name: str,
    dataset_name: str,
    device: torch.device,
    root_dir=PROJECT_ROOT,
    x_normalize=True,
):
    output_dir = root_dir / "classifiers" / dataset_name

    try:
        classes, templates = dataset2classes_templates[dataset_name]  # noqa: F821
    except KeyError:
        raise ValueError(f"Dataset {dataset_name} not supported.") from None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{encoder_name}.pt"
    if not output_file.exists():
        encoder: Encoder = get_text_encoder(name=encoder_name).to(device)

        class_encodings = []
        for classname in classes:
            batch = encoder.preprocess([template(classname) for template in templates])

            class_embedding = encoder.encode_text(x=batch.to(device))
            class_embedding = F.normalize(class_embedding, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()

            class_encodings.append(class_embedding.detach().cpu())
        class_encodings = torch.stack(class_encodings, dim=1)

        class_encodings = class_encodings * encoder.logit_scale.exp().cpu()

        data = {
            "classes": classes,
            "class_encodings": class_encodings.detach().cpu(),
        }
        torch.save(data, output_file)

    return CentroidClassifier.from_file(
        output_file,
        x_normalize=x_normalize,
    )


def build_vision_prototypical_classifier(
    dataset_name: str, split: str, encoder_name: str, device: torch.device
):
    output_file = PROJECT_ROOT / "classifiers" / dataset_name / f"{encoder_name}.pt"
    if output_file.exists():
        print(f"Classifier for {dataset_name} {encoder_name} already exists. Skipping.")
        return

    dataset: Dataset = get_dataset(dataset=dataset_name, split=split)
    dataset = dataset.with_format("torch", columns=["y"], device=device)
    encoding = Residual.load_output(
        source_dir=PROJECT_ROOT / "encodings" / dataset_name / split / encoder_name,
        device=device,
        # as_tensor_device=device,
        verbose=True,
    )[:, 0, :]

    class_encodings = torch.stack(
        [
            encoding[dataset["y"] == i].mean(dim=0)
            for i in range(dataset.features["y"].num_classes)
        ],
        dim=0,
    )

    data = {
        "classes": dataset.features["y"].names,
        "class_encodings": class_encodings.detach().cpu().T,
    }
    torch.save(data, output_file)


if __name__ == "__main__":
    for dataset_name, encoder_name in itertools.product(
        [
            "gtsrb",
            "eurosat",
            "mnist",
            "svhn",
            "imagenet",
            "cifar10",
            "cifar100",
            "sun397",
            # "sketch",
            "dtd",
            "resisc45",
            "stanford_cars",
            # "pacs",
        ],
        ("vit_l", "dinov2_l"),
    ):
        build_vision_prototypical_classifier(
            dataset_name=dataset_name,
            split="train",
            encoder_name=encoder_name,
            device="cuda",
        )
