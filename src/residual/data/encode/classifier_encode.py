import itertools
from pathlib import Path
from typing import Sequence

import torch
from latentis import PROJECT_ROOT
from tqdm import tqdm
from residual.data.data_registry import dataset2classes_templates
from residual.nn.model_registry import get_text_encoder


@torch.no_grad()
def generate_class_encodings(
    encoder_name: str,
    classes: Sequence[str],
    templates: Sequence[str],
    output_dir: Path,
    device: torch.device,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{encoder_name}.pt"
    if output_file.exists():
        return

    encoder = get_text_encoder(name=encoder_name).to(device)

    class_encodings = []
    for classname in classes:
        batch = encoder.preprocess([template(classname) for template in templates])
        class_embeddings = encoder.encode_text(x=batch.to(device))
        class_embedding = torch.nn.functional.normalize(class_embeddings, dim=-1).mean(
            dim=0
        )
        class_embedding /= class_embedding.norm()
        class_encodings.append(class_embedding.detach().cpu())
    class_encodings = torch.stack(class_encodings, dim=1)

    data = {
        "classes": classes,
        "class_encodings": class_encodings.detach().cpu(),
    }
    torch.save(data, output_file)


if __name__ == "__main__":
    encoders = [
        "blip_l_flickr",
        "openclip_b",
        "openclip_l",
        "clip_b",
        "clip_l",
    ]

    datasets = dataset2classes_templates.keys()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tqdm(total=len(encoders) * len(datasets)) as pbar:
        for encoder_name, dataset_name in itertools.product(encoders, datasets):
            pbar.set_description(f"{encoder_name} on {dataset_name}")
            output_dir = PROJECT_ROOT / "classifiers" / dataset_name

            try:
                classes, templates = dataset2classes_templates[dataset_name]
            except KeyError:
                raise ValueError(f"Dataset {dataset_name} not supported.") from None

            generate_class_encodings(
                encoder_name=encoder_name,
                classes=classes,
                templates=templates,
                output_dir=output_dir,
                device=device,
            )
            pbar.update(1)
