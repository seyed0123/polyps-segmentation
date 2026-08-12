"""Check a video for polyps and save a segmentation-overlay video."""
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from .model import Segmenter, overlay_mask
except ImportError:  # Supports: python infer/video_inference.py
    from model import Segmenter, overlay_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the input video")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output video path (default: infer/res/<input>_overlay.mp4)",
    )
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of sampled frames inferred concurrently (default: 2)",
    )
    parser.add_argument(
        "--inference-fps",
        type=float,
        default=20.0,
        help="Maximum frames per video-second sent to the model (default: 20)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-ratio", type=float, default=0.001)
    parser.add_argument(
        "--detection-ratio",
        type=float,
        default=0.001,
        help="Minimum fraction of frame pixels in the mask to count as a detection",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Do not draw the POLYP/No polyp label on output frames",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if not np.isfinite(args.inference_fps) or args.inference_fps <= 0.0:
        raise ValueError("--inference-fps must be greater than 0")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.min_component_ratio < 0.0:
        raise ValueError("--min-component-ratio must be non-negative")
    if not 0.0 <= args.detection_ratio <= 1.0:
        raise ValueError("--detection-ratio must be between 0 and 1")


def draw_status(frame: np.ndarray, detected: bool, coverage: float) -> None:
    label = f"POLYP ({coverage:.1%})" if detected else "No polyp"
    color = (0, 0, 255) if detected else (0, 200, 0)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                color, 2, cv2.LINE_AA)


def predict_frame(
    segmenter: Segmenter,
    frame: np.ndarray,
    threshold: float,
    min_component_ratio: float,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    _, mask = segmenter.predict(frame, threshold, min_component_ratio)
    return mask, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    validate_args(args)
    output = args.output or Path("infer/res") / f"{args.input.stem}_overlay.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The video does not report valid frame dimensions")

    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output}")

    segmenter = Segmenter(args.model, args.device, args.cpu_threads)
    processed = inferred_frames = detected_frames = 0
    maximum_coverage = 0.0
    inference_seconds = 0.0
    # For a 30 FPS video and the default 20 inference FPS, this schedules
    # frames 0, 2, 3, 5, 6, ... and preserves uniform sampling over time.
    effective_inference_fps = min(args.inference_fps, fps)
    inference_interval = fps / effective_inference_fps
    expected_inferences = (
        math.ceil(frame_count * effective_inference_fps / fps)
        if frame_count > 0 else None
    )
    next_inference_frame = 0.0
    # Each entry contains one inference future and all output frames that should
    # use that prediction. Keeping this queue bounded avoids loading the entire
    # video into memory while allowing several model calls to overlap.
    pending: deque[tuple[Future[tuple[np.ndarray, float]], list[np.ndarray]]] = deque()
    max_pending = max(2, args.workers * 2)

    def write_oldest() -> None:
        nonlocal inferred_frames, detected_frames, maximum_coverage, inference_seconds
        future, frames = pending.popleft()
        mask, elapsed = future.result()
        coverage = float(np.count_nonzero(mask)) / mask.size
        detected = coverage >= args.detection_ratio
        inferred_frames += 1
        detected_frames += int(detected)
        maximum_coverage = max(maximum_coverage, coverage)
        inference_seconds += elapsed
        for buffered_frame in frames:
            result = overlay_mask(buffered_frame, mask)
            if not args.no_label:
                draw_status(result, detected, coverage)
            writer.write(result)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            with tqdm(
                total=expected_inferences, unit="sample", desc="Analyzing video"
            ) as progress:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if processed >= next_inference_frame:
                        future = executor.submit(
                            predict_frame,
                            segmenter,
                            frame.copy(),
                            args.threshold,
                            args.min_component_ratio,
                        )
                        pending.append((future, [frame]))
                        next_inference_frame += inference_interval
                        if len(pending) >= max_pending:
                            write_oldest()
                            progress.update()
                    else:
                        pending[-1][1].append(frame)
                    processed += 1

                while pending:
                    write_oldest()
                    progress.update()
    finally:
        capture.release()
        writer.release()

    if processed == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("The input video contained no readable frames")

    print(f"Processed {processed} frame(s)")
    print(f"Analyzed by model: {inferred_frames} frame(s) at up to {args.inference_fps:g} FPS")
    print(f"Inference workers: {args.workers} ({args.cpu_threads} CPU threads each)")
    print(f"Polyp predicted: {'YES' if detected_frames else 'NO'}")
    print(
        f"Detected samples: {detected_frames}/{inferred_frames} "
        f"({detected_frames / inferred_frames:.1%})"
    )
    print(f"Maximum mask coverage: {maximum_coverage:.1%}")
    print(f"Average inference time: {inference_seconds / inferred_frames:.3f}s/sample")
    print(f"Annotated video: {output}")


if __name__ == "__main__":
    main()
