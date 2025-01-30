import itertools
from pathlib import Path
from typing import Sequence

import torch
from latentis import PROJECT_ROOT
from tqdm import tqdm

from residual.data.data_registry import dataset2classes_templates
from residual.nn.classifier import build_centroid_classifier
from residual.nn.model_registry import get_text_encoder


@torch.no_grad()
def generate_dictionary_encodings(
    encoder_name: str,
    dictionary: Sequence[str],
    output_dir: Path,
    device: torch.device,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{encoder_name}.pt"
    if output_file.exists():
        return

    encoder = get_text_encoder(name=encoder_name).to(device)

    batch = encoder.preprocess(dictionary)
    dict_encodings = encoder.encode_text(x=batch.to(device))

    data = {
        "dictionary": dictionary,
        "encodings": dict_encodings.detach().cpu(),
    }
    torch.save(data, output_file)


def get_textspan_dict():
    dictionary = PROJECT_ROOT / "data" / "textspan_dictionary.txt"
    dictionary = dictionary.read_text().splitlines()
    return dictionary


if __name__ == "__main__":
    encoders = [
        # "blip_l_flickr",
        "openclip_b",
        "openclip_l",
        # "clip_b",
        "clip_l",
        # "hf_clip_l",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dictionaries = {"textspan": get_textspan_dict()}

    with tqdm(total=len(encoders) * len(dictionaries)) as pbar:
        for encoder_name, [dictionary_name, dictionary] in itertools.product(
            encoders, dictionaries.items()
        ):
            pbar.set_description(f"{encoder_name} on {dictionary_name}")
            output_dir = PROJECT_ROOT / "dictionaries" / dictionary_name

            generate_dictionary_encodings(
                encoder_name=encoder_name,
                dictionary=dictionary,
                output_dir=output_dir,
                device=device,
            )
            pbar.update(1)

    datasets = dataset2classes_templates.keys()
    with tqdm(total=len(encoders) * len(datasets)) as pbar:
        for encoder_name, dataset_name in itertools.product(encoders, datasets):
            pbar.set_description(f"{encoder_name} on {dataset_name}")

            build_centroid_classifier(
                encoder_name=encoder_name,
                dataset_name=dataset_name,
                device=device,
            )
            pbar.update(1)
