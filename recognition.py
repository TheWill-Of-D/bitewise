"""Offline inference with a pretrained Food-101 model or custom FoodSG weights."""

from __future__ import annotations

import io
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "models" / "food_classifier.pt"
LABELS = ROOT / "models" / "class_names.json"
PRETRAINED_DIR = ROOT / "models" / "food101"


def selected_backend() -> str:
    backend = os.environ.get("BITEWISE_MODEL", "auto").lower()
    if backend not in {"auto", "custom", "pretrained", "none"}:
        raise ValueError("BITEWISE_MODEL must be auto, custom, pretrained, or none.")
    if backend != "auto":
        return backend
    if WEIGHTS.is_file() and LABELS.is_file():
        return "custom"
    if all((PRETRAINED_DIR / name).is_file() for name in ("config.json", "model.safetensors")):
        return "pretrained"
    return "none"


def model_cache_key() -> str:
    backend = selected_backend()
    paths = [WEIGHTS, LABELS] if backend == "custom" else [PRETRAINED_DIR / "config.json", PRETRAINED_DIR / "model.safetensors"] if backend == "pretrained" else []
    return backend + ":" + ":".join(str(path.stat().st_mtime_ns) if path.is_file() else "missing" for path in paths)


def model_status() -> tuple[bool, str]:
    try:
        backend = selected_backend()
    except ValueError as exc:
        return False, str(exc)
    if backend == "none":
        return False, "Recognition is not configured. Run python setup_model.py to install the pretrained model."
    packages = ("torch", "torchvision", "safetensors") if backend == "pretrained" else ("torch", "torchvision")
    if any(importlib.util.find_spec(package) is None for package in packages):
        return False, "Install requirements-model.txt to enable local photo recognition."
    if backend == "custom":
        return WEIGHTS.is_file() and LABELS.is_file(), "Custom food model · EfficientNet-B0"
    ready = all((PRETRAINED_DIR / name).is_file() for name in ("config.json", "model.safetensors"))
    return ready, "Pretrained Food-101 · EfficientNet-B2" if ready else "Run python setup_model.py to install the pretrained model."


def open_image(image_bytes: bytes):
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(image_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


@lru_cache(maxsize=1)
def _load(backend=None, signature=None):
    import torch
    from torchvision import models, transforms

    backend = backend or selected_backend()
    # Small inference batches are often slower with dozens of CPU threads.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if backend == "pretrained":
        from safetensors.torch import load_file

        config = json.loads((PRETRAINED_DIR / "config.json").read_text())
        labels = config["class_names"]
        class_names = [label.replace("_", " ").capitalize() for label in labels]
        model = models.efficientnet_b2(weights=None)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=float(config["dropout"]), inplace=True),
            torch.nn.Linear(model.classifier[1].in_features, len(class_names)),
        )
        state = load_file(str(PRETRAINED_DIR / "model.safetensors"), device="cpu")
        model.load_state_dict({key.removeprefix("backbone."): value for key, value in state.items()}, strict=True)
        transform = models.EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()
    elif backend == "custom":
        class_names = json.loads(LABELS.read_text())
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
        checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        raise RuntimeError("No local food model is installed. Run python setup_model.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return torch, model, class_names, transform, device


def predict_image(image_bytes: bytes, top_k: int = 3) -> list[tuple[str, float]]:
    if top_k < 1:
        raise ValueError("top_k must be at least one.")
    torch, model, class_names, transform, device = _load(selected_backend(), model_cache_key())
    image = open_image(image_bytes)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor)[0], dim=0)
        scores, indices = torch.topk(probabilities, min(top_k, len(class_names)))
    return [(class_names[int(index)], float(score)) for score, index in zip(scores.cpu(), indices.cpu())]
