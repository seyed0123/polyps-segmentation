"""Run ResUNet inference on one image or a directory of images."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from tqdm import tqdm

try:
    from .model import Segmenter, overlay_mask
except ImportError:  # Supports: python infer/inference.py
    from model import Segmenter, overlay_mask


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("--output", "-o", type=Path, default=Path("infer/res"))
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-ratio", type=float, default=0.001)
    parser.add_argument("--save-masks", action="store_true", help="Also save binary masks")
    return parser.parse_args()


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input does not exist: {source}")
    return sorted(path for path in source.iterdir() if path.suffix.lower() in EXTENSIONS)


def main() -> None:
    args = parse_args()
    paths = image_paths(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    segmenter = Segmenter(args.model, args.device, args.cpu_threads)
    elapsed_total = 0.0

    for path in tqdm(paths, desc="Segmenting"):
        image = cv2.imread(str(path))
        if image is None:
            tqdm.write(f"Skipping unreadable image: {path}")
            continue
        started = time.perf_counter()
        _, mask = segmenter.predict(image, args.threshold, args.min_component_ratio)
        elapsed_total += time.perf_counter() - started
        cv2.imwrite(str(args.output / f"{path.stem}_overlay.png"), overlay_mask(image, mask))
        if args.save_masks:
            cv2.imwrite(str(args.output / f"{path.stem}_mask.png"), mask)

    if paths:
        print(f"Processed {len(paths)} image(s); inference average: {elapsed_total / len(paths):.3f}s")


if __name__ == "__main__":
    main()
