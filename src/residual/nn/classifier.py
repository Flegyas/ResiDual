import itertools
from datasets import Dataset
import gin
import torch
from latentis import PROJECT_ROOT
from torch import nn
from transformers import ViTForImageClassification

from residual.data.dataset import get_dataset
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
    @classmethod
    def from_tensor(cls, centroids: torch.Tensor):
        model = cls.__new__(cls)
        super(CentroidClassifier, model).__init__()
        model.register_buffer("centroids", centroids)
        return model

    def __init__(
        self,
        encoder_name: str,
        dataset_name: str,
    ):
        super().__init__()
        class_encodings = torch.load(
            PROJECT_ROOT / "classifiers" / dataset_name / f"{encoder_name}.pt",
            weights_only=True,
        )
        self.class_names = class_encodings["classes"]
        self.register_buffer("centroids", class_encodings["class_encodings"])

    @property
    def num_classes(self):
        return self.centroids.shape[1]

    def forward(self, x: torch.Tensor):
        centroids = self.centroids

        # x = F.normalize(x, p=2, dim=-1)

        return x @ centroids


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
