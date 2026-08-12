from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models


class ResUNet(nn.Module):
    """The ResNet34-based UNet architecture used by this project."""

    def __init__(self, n_classes: int = 1) -> None:
        super().__init__()
        # The checkpoint contains encoder weights, so inference must not download
        # ImageNet weights merely to construct the architecture.
        base_layers = list(models.resnet34(weights=None).children())
        self.layer0 = nn.Sequential(*base_layers[:3])
        self.layer1 = nn.Sequential(*base_layers[3:5])
        self.layer2 = base_layers[5]
        self.layer3 = base_layers[6]
        self.layer4 = base_layers[7]
        self.upsample4 = self._upsample(512, 256)
        self.upsample3 = self._upsample(256, 128)
        self.upsample2 = self._upsample(128, 64)
        self.upsample1 = self._upsample(64, 64)
        self.upsample0 = self._upsample(64, 64)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    @staticmethod
    def _upsample(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x0 = self.layer0(image)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.upsample4(x4) + x3
        x = self.upsample3(x) + x2
        x = self.upsample2(x) + x1
        x = self.upsample1(x) + x0
        return self.final_conv(self.upsample0(x))


def load_model(checkpoint: Union[str, Path], device: torch.device) -> nn.Module:
    """Load a state-dict checkpoint or a TorchScript .pt/.ptx model."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    try:
        scripted = torch.jit.load(str(checkpoint), map_location=device)
        return scripted.eval()
    except (RuntimeError, ValueError):
        pass

    payload = torch.load(str(checkpoint), map_location=device)
    if isinstance(payload, nn.Module):
        return payload.to(device).eval()
    if isinstance(payload, dict):
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
    else:
        raise TypeError("Checkpoint must contain a state dict, nn.Module, or TorchScript model")
    # Accept checkpoints saved from DataParallel.
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = ResUNet(n_classes=1).to(device)
    model.load_state_dict(state)
    return model.eval()


class Segmenter:
    def __init__(self, checkpoint: Union[str, Path], device: str = "auto",
                 cpu_threads: int = 0) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cpu":
            if cpu_threads > 0:
                torch.set_num_threads(cpu_threads)
            # Inter-op parallelism adds overhead for this single-model worker.
            torch.set_num_interop_threads(1)
        self.model = load_model(checkpoint, self.device)
        if self.device.type == "cpu":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray, threshold: float = 0.5,
                min_component_ratio: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(image_bgr, (256, 256)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(self.device, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).div_(255.0)
        tensor = (tensor - self.mean) / self.std
        if self.device.type == "cpu":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        probability = torch.sigmoid(self.model(tensor))[0, 0].float().cpu().numpy()
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = (probability >= threshold).astype(np.uint8) * 255
        mask = remove_small_components(mask, min_component_ratio)
        probability[mask == 0] = 0
        return probability, mask


def remove_small_components(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    # Be defensive at process/API boundaries: None used to cause an intermittent
    # int(None) exception and discard an otherwise valid inference frame.
    ratio = 0.001 if min_area_ratio is None else float(min_area_ratio)
    if not np.isfinite(ratio) or ratio < 0:
        ratio = 0.001
    minimum = max(1, round(mask.size * ratio))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum:
            filtered[labels == label] = 255
    return filtered


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color: tuple[int, int, int] = (0, 0, 255), alpha: float = 0.42) -> np.ndarray:
    result = image.copy()
    active = mask.astype(bool)
    if not np.any(active):
        return result
    colored = np.empty_like(image)
    colored[:] = color
    # OpenCV returns None when passed an empty boolean-indexed array. Blend full
    # images, then copy only active mask pixels into the result.
    blended = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)
    result[active] = blended[active]
    return result
