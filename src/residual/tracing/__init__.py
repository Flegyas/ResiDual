from .blip import BLIPTracer
from .dinov2 import Dinov2Tracer
from .hf_clip import HFCLIPTracer
from .openclip import OpenCLIPTracer
from .tracing_op import SerializeResidualOp, TracingOp
from .vit import ViTTracer

encoder_name2tracer = {
    "blip_l_flickr": BLIPTracer,
    "dinov2_l": Dinov2Tracer,
    "openclip_b": OpenCLIPTracer,
    "openclip_l": OpenCLIPTracer,
    "clip_b": OpenCLIPTracer,
    "clip_l": OpenCLIPTracer,
    "vit_l": ViTTracer,
    "hf_clip_l": HFCLIPTracer,
}

__all__ = [
    "BLIPTracer",
    "Dinov2Tracer",
    "OpenCLIPTracer",
    "ViTTracer",
    "HFCLIPTracer",
    "encoder_name2tracer",
    "TracingOp",
    "SerializeResidualOp",
]
