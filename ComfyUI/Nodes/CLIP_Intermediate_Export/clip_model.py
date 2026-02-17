from dataclasses import dataclass
from typing import Any, Dict, Tuple

# Defer heavy deps so the node module can import even if torch/open_clip are missing.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    import open_clip
except ImportError:  # pragma: no cover
    open_clip = None  # type: ignore


REQS = ["open_clip_torch>=2.24.0"]


@dataclass
class ClipModelBundle:
    model: Any
    tokenizer: Any
    preprocess: Any
    device: str


class OpenCLIPTextModelLoader:
    """
    Loads an OpenCLIP text encoder and tokenizer for downstream analysis.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "architecture": (
                    [
                        "ViT-B-32",
                        "ViT-B-16",
                        "ViT-L-14",
                        "ViT-L-14-336",
                    ],
                    {"tooltip": "OpenCLIP text tower architecture to load."},
                ),
                "pretrained": (
                    [
                        "openai",
                        "laion2b-s34b-b79k",
                        "laion2b-s32b-b82k",
                        "laion400m_e32",
                        "laion400m_e31",
                        "custom_or_other",
                    ],
                    {"tooltip": "Pretrained weights source. Use custom_or_other with checkpoint_path for local weights."},
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"tooltip": "auto selects CUDA if available, otherwise CPU."},
                ),
                "force_quick_gelu": ("BOOLEAN", {"default": True, "tooltip": "Keep enabled for compatibility with most CLIP checkpoints."}),
                "checkpoint_path": ("STRING", {"default": "", "file": True, "tooltip": "Optional local checkpoint file. If set, overrides pretrained choice."}),
                "cache_dir": ("STRING", {"default": "", "dir": True, "tooltip": "Optional cache folder for downloaded model files."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("clip_model",)
    FUNCTION = "load"
    CATEGORY = "CLIP Intermediate Export"

    def load(
        self,
        architecture: str,
        pretrained: str,
        checkpoint_path: str,
        cache_dir: str,
        device: str,
        force_quick_gelu: bool,
    ) -> Tuple[ClipModelBundle]:
        if torch is None:
            raise ImportError(
                "torch not installed. Install torch/torchvision with the right CUDA/CPU build "
                "(e.g., pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121)."
            )
        if open_clip is None:
            raise ImportError(
                "open_clip_torch not installed. Install via: pip install open_clip_torch"
            )
        resolved_device = (
            torch.device("cuda")
            if device == "auto" and torch.cuda.is_available()
            else torch.device(device)
        )
        resolved_cache = cache_dir or None
        pretrained_arg = checkpoint_path if checkpoint_path else pretrained
        model, _, preprocess = open_clip.create_model_and_transforms(
            architecture,
            pretrained=pretrained_arg,
            force_quick_gelu=force_quick_gelu,
            cache_dir=resolved_cache,
        )
        tokenizer = open_clip.get_tokenizer(architecture)
        model = model.to(resolved_device)
        model.eval()
        bundle = ClipModelBundle(
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            device=str(resolved_device),
        )
        return (bundle,)


NODE_CLASS_MAPPINGS = {
    "OpenCLIPTextModelLoader": OpenCLIPTextModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenCLIPTextModelLoader": "CIE: Load OpenCLIP (Text)",
}
