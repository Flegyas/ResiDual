from collections import defaultdict
from functools import partial
from typing import Any, Callable, Mapping, Type, Union

import gin
import open_clip
import torch
from tokenizers.implementations import BaseTokenizer
from torch.utils.data import default_collate
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    BaseImageProcessor,
    BlipForImageTextRetrieval,
    BlipProcessor,
    CLIPModel,
    PreTrainedModel,
    ViTImageProcessor,
    ViTModel,
)

from residual.nn.encoder import (
    Encoder,
    HFTextEncoder,
    HFVisionEncoder,
    OpenCLIPTextEncoder,
    OpenCLIPVisionEncoder,
    get_pooling_fn,
)

modality2model_name2entry = {}


def load_hf(
    hf_name: str,
    processor_cls: Type[BaseImageProcessor | BaseTokenizer],
    model_cls: Type[PreTrainedModel],
    model_kwargs: Mapping[str, Any] = None,
    processor_kwargs: Mapping[str, Any] = None,
):
    # , padding=True, return_tensors="pt")
    processor = processor_cls.from_pretrained(hf_name)
    if processor_kwargs:
        processor = partial(processor, **processor_kwargs)

    return dict(
        model=model_cls.from_pretrained(hf_name, **(model_kwargs or {})).eval(),
        processor=processor,
    )


def load_openclip(model_name: str, pretrained: str, modality: str):
    model, train_processor, val_processor = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained
    )
    processor = val_processor

    if modality == "vision":
        model.visual.output_tokens = True
        model = model.visual.eval()
    elif modality == "text":
        model = model.eval()
        processor = open_clip.get_tokenizer(model_name)
    else:
        raise ValueError(f"Modality {modality} not supported")

    return dict(model=model, processor=processor)


# Vision
modality2model_name2entry["vision"] = defaultdict(dict)

# Hugging Face models
modality2model_name2entry["vision"]["vit_l"] = dict(
    library="transformers",
    load_modules_fn=partial(
        load_hf,
        hf_name="google/vit-large-patch16-224",
        processor_cls=ViTImageProcessor,
        model_cls=ViTModel,
    ),
)
modality2model_name2entry["vision"]["vit_b"] = dict(
    library="transformers",
    load_modules_fn=partial(
        load_hf,
        hf_name="google/vit-base-patch16-224",
        processor_cls=ViTImageProcessor,
        model_cls=ViTModel,
    ),
)
modality2model_name2entry["vision"]["blip_l_flickr"] = dict(
    library="transformers",
    load_modules_fn=partial(
        load_hf,
        hf_name="Salesforce/blip-itm-large-flickr",
        processor_cls=BlipProcessor,
        model_cls=BlipForImageTextRetrieval,
    ),
)
modality2model_name2entry["vision"]["dinov2_l"] = dict(
    library="transformers",
    load_modules_fn=partial(
        load_hf,
        hf_name="facebook/dinov2-large",
        processor_cls=AutoImageProcessor,
        model_cls=AutoModel,
    ),
)
modality2model_name2entry["vision"]["hf_clip_l"] = dict(
    library="transformers",
    load_modules_fn=partial(
        load_hf,
        hf_name="openai/clip-vit-large-patch14-336",
        processor_cls=AutoImageProcessor,
        model_cls=AutoModel,
    ),
)

# OpenCLIP models
modality2model_name2entry["vision"]["openclip_b"] = dict(
    library="open_clip",
    load_modules_fn=partial(
        load_openclip,
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        modality="vision",
    ),
)
modality2model_name2entry["vision"]["openclip_l"] = dict(
    library="open_clip",
    load_modules_fn=partial(
        load_openclip,
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
        modality="vision",
    ),
)
modality2model_name2entry["vision"]["clip_b"] = dict(
    library="open_clip",
    load_modules_fn=partial(
        load_openclip,
        model_name="ViT-B-16",
        pretrained="openai",
        modality="vision",
    ),
)
modality2model_name2entry["vision"]["clip_l"] = dict(
    library="open_clip",
    load_modules_fn=partial(
        load_openclip,
        model_name="ViT-L-14",
        pretrained="openai",
        modality="vision",
    ),
)


def _collate_fn_image(samples, preprocess, library: str):
    if library == "transformers":
        images = preprocess(
            [sample["x"].convert("RGB") for sample in samples], return_tensors="pt"
        )
        images = images["pixel_values"]
        x = images
    elif library == "open_clip":
        images = [preprocess(sample["x"]) for sample in samples]
        x = torch.stack(images, dim=0)
    else:
        raise ValueError(f"Library {library} not supported")

    labels = torch.tensor([sample["y"] for sample in samples])
    sample_ids = [str(sample["sample_id"]) for sample in samples]

    return dict(x=x, y=labels, sample_id=sample_ids)


@gin.configurable
def get_vision_encoder(name: str, pooling_fn: Union[Callable, str]) -> Encoder:
    if name not in modality2model_name2entry["vision"]:
        raise ValueError(f"Model {name} not supported in registry")

    pooling_fn = pooling_fn if callable(pooling_fn) else get_pooling_fn(pooling_fn)

    entry = modality2model_name2entry["vision"][name]

    modules = entry["load_modules_fn"]()
    model = modules["model"]
    processor = modules["processor"]

    library = entry["library"]

    collate_fn = modules.get(
        "collate_fn", partial(_collate_fn_image, preprocess=processor, library=library)
    )

    if library == "open_clip":
        encoder_cls = OpenCLIPVisionEncoder
    elif library == "transformers":
        encoder_cls = HFVisionEncoder
    else:
        raise ValueError(f"Library {library} not supported")

    return encoder_cls(
        name=name,
        model=model,
        preprocess=processor,
        collate_fn=collate_fn,
        pooling_fn=pooling_fn,
    )


model_names = {
    "openclip_b": "OpenCLIP-B",
    "openclip_l": "OpenCLIP-L",
    "clip_l": "CLIP-L",
    "clip_b": "CLIP-B",
    "vit_l": "ViT-L",
    "dinov2_l": "DINOv2-L",
    "blip_l_coco": "BLIP-L-COCO",
    "blip_l_flickr": "BLIP-L",
    "openclip_h": "OpenCLIP-H",
}

# Text
modality2model_name2entry["text"] = defaultdict(dict)

# OpenCLIP models
modality2model_name2entry["text"]["openclip_b"] = dict(
    library="open_clip",
    load_modules_fn=partial(
        load_openclip,
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        modality="text",
    ),
)
modality2model_name2entry["text"]["openclip_l"] = dict(
    library="open_clip",
    load_modules_fn=partial(
        load_openclip,
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
        modality="text",
    ),
)
modality2model_name2entry["text"]["clip_b"] = dict(
    library="open_clip",
    load_modules_fn=partial(
        load_openclip,
        model_name="ViT-B-16",
        pretrained="openai",
        modality="text",
    ),
)
modality2model_name2entry["text"]["clip_l"] = dict(
    library="open_clip",
    load_modules_fn=partial(
        load_openclip,
        model_name="ViT-L-14",
        pretrained="openai",
        modality="text",
    ),
)

# Hugging Face models
modality2model_name2entry["text"]["blip_l_flickr"] = dict(
    library="transformers",
    load_modules_fn=partial(
        load_hf,
        hf_name="Salesforce/blip-itm-large-flickr",
        processor_cls=AutoTokenizer,
        model_cls=BlipForImageTextRetrieval,
        processor_kwargs=dict(padding=True, return_tensors="pt"),
    ),
)
modality2model_name2entry["text"]["hf_clip_l"] = dict(
    library="transformers",
    load_modules_fn=partial(
        load_hf,
        hf_name="openai/clip-vit-large-patch14-336",
        processor_cls=AutoTokenizer,
        model_cls=CLIPModel,
        processor_kwargs=dict(padding=True, return_tensors="pt"),
    ),
)


@gin.configurable
def get_text_encoder(name: str) -> Encoder:
    if name not in modality2model_name2entry["text"]:
        raise ValueError(f"Model {name} not supported in registry")

    entry = modality2model_name2entry["text"][name]

    modules = entry["load_modules_fn"]()
    model = modules["model"]
    processor = modules["processor"]

    library = entry["library"]

    collate_fn = modules.get("collate_fn", default_collate)

    if library == "open_clip":
        return OpenCLIPTextEncoder(
            name=name,
            model=model,
            preprocess=processor,
            collate_fn=collate_fn,
            pooling_fn=None,
        )
    elif library == "transformers":
        return HFTextEncoder(
            name=name,
            model=model,
            preprocess=processor,
            collate_fn=collate_fn,
            pooling_fn=...,
        )
    else:
        raise ValueError(f"Library {library} not supported")
