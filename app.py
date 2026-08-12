"""Native desktop webcam application for real-time polyp segmentation."""
from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import wave
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

# PyInstaller's multiprocessing dispatcher should run before heavy torch imports.
if __name__ == "__main__":
    mp.freeze_support()

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import winsound
except ImportError:  # winsound is available only on Windows.
    winsound = None


def windows_alert() -> None:
    try:
        winsound.Beep(1000, 180)
        time.sleep(0.08)
        winsound.Beep(1250, 260)
    except RuntimeError:
        # Fall back to the configured Windows system notification sound.
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)


def linux_alert() -> None:
    """Generate an audible two-tone WAV and play it on Linux."""
    sample_rate = 22_050
    parts = []
    for frequency, duration in ((1000, 0.18), (0, 0.08), (1250, 0.26)):
        count = int(sample_rate * duration)
        if frequency:
            timeline = np.arange(count, dtype=np.float32) / sample_rate
            samples = np.sin(2 * np.pi * frequency * timeline) * 0.4
        else:
            samples = np.zeros(count, dtype=np.float32)
        parts.append((samples * 32767).astype(np.int16))

    wav_data = io.BytesIO()
    with wave.open(wav_data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.concatenate(parts).tobytes())

    aplay = shutil.which("aplay")
    if aplay:
        result = subprocess.run(
            [aplay, "-q"], input=wav_data.getvalue(), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=False, timeout=3,
        )
        if result.returncode == 0:
            return
    canberra = shutil.which("canberra-gtk-play")
    if canberra:
        subprocess.run(
            [canberra, "-i", "dialog-warning", "-d", "Polyp detected"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=3,
        )


def bundled_file(filename: str) -> Path:
    """Find a data file both in source and in a PyInstaller bundle."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / filename


def default_capture_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path.home() / "Documents" / "PolypMonitorCaptures"
    return Path("captures")


def timestamp_stem(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def capture_worker(frame_queue: mp.Queue, inference_queue: mp.Queue,
                   status_queue: mp.Queue, stop_event: mp.Event, config: dict) -> None:
    """Continuously capture frames without ever blocking the GUI or inference."""
    cv2.setNumThreads(1)
    camera = cv2.VideoCapture(config["camera"])
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera_width"])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera_height"])
    camera.set(cv2.CAP_PROP_FPS, config["camera_fps"])
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        status_queue.put(("error", f"Could not open camera {config['camera']}"))
        return

    def camera_number(prop: int, fallback: float) -> float:
        value = camera.get(prop)
        if value is None or not np.isfinite(value) or value <= 0:
            return fallback
        return float(value)

    status_queue.put(("ready", {
        "width": round(camera_number(cv2.CAP_PROP_FRAME_WIDTH, config["camera_width"])),
        "height": round(camera_number(cv2.CAP_PROP_FRAME_HEIGHT, config["camera_height"])),
        "fps": camera_number(cv2.CAP_PROP_FPS, config["camera_fps"]),
    }))
    inference_interval = 1.0 / config["target_fps"]
    last_inference_frame = 0.0

    def replace_latest(destination: mp.Queue, value: np.ndarray) -> None:
        try:
            destination.put_nowait(value)
        except queue.Full:
            try:
                destination.get_nowait()
            except queue.Empty:
                pass
            try:
                destination.put_nowait(value)
            except queue.Full:
                pass

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                continue
            # Keep only the newest camera frame. Old video must never build latency.
            replace_latest(frame_queue, frame)
            now = time.monotonic()
            if now - last_inference_frame >= inference_interval:
                replace_latest(inference_queue, frame)
                last_inference_frame = now
    except Exception as exc:
        status_queue.put(("error", f"Camera error: {exc}"))
    finally:
        camera.release()


def inference_worker(input_queue: mp.Queue, output_queue: mp.Queue, config: dict) -> None:
    """Load and run PyTorch outside the GUI process."""
    # Keep torch out of the GUI and camera processes created with multiprocessing spawn.
    from infer.model import Segmenter, overlay_mask

    cv2.setNumThreads(1)
    try:
        segmenter = Segmenter(config["model"], config["device"], config["cpu_threads"])
        output_queue.put(("ready", None))
    except Exception as exc:
        output_queue.put(("error", f"Could not load model: {exc}"))
        return

    automatic_dir = Path(config["save_dir"]) / "automatic"
    automatic_dir.mkdir(parents=True, exist_ok=True)
    last_save = 0.0
    consecutive = 0

    while True:
        frame = input_queue.get()
        if frame is None:
            return
        try:
            started = time.perf_counter()
            _, mask = segmenter.predict(
                frame, config["threshold"], config["min_component_ratio"]
            )
            inference_ms = (time.perf_counter() - started) * 1000
            mask_ratio = float(np.count_nonzero(mask)) / mask.size
            detected = mask_ratio >= config["auto_save_ratio"]
            consecutive = consecutive + 1 if detected else 0
            auto_saved = False

            now = time.monotonic()
            if consecutive >= config["consecutive_frames"] and now - last_save >= config["cooldown"]:
                stem = timestamp_stem("polyp")
                cv2.imwrite(str(automatic_dir / f"{stem}_overlay.jpg"), overlay_mask(frame, mask))
                last_save = now
                auto_saved = True

            display = overlay_mask(frame, mask)
            output_queue.put_nowait(("frame", {
                "display": display,
                "mask_ratio": mask_ratio,
                "detected": detected,
                "auto_saved": auto_saved,
                "inference_ms": inference_ms,
            }))
        except queue.Full:
            pass
        except Exception as exc:
            try:
                details = traceback.format_exc(limit=4)
                output_queue.put_nowait(("error", f"Inference error: {exc}\n{details}"))
            except queue.Full:
                pass


class PolypMonitor:
    def __init__(self, root: tk.Tk, config: dict, camera_queue: mp.Queue,
                 camera_status: mp.Queue, camera_stop: mp.Event,
                 input_queue: mp.Queue, output_queue: mp.Queue,
                 camera_worker: mp.Process, inference_process: mp.Process) -> None:
        self.root = root
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.camera_queue = camera_queue
        self.camera_status = camera_status
        self.camera_stop = camera_stop
        self.camera_worker = camera_worker
        self.inference_process = inference_process
        self.running = True
        self.worker_ready = False
        self.latest_raw: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.frame_count = 0
        self.fps_started = time.perf_counter()
        self.was_detected = False

        self.manual_dir = Path(config["save_dir"]) / "manual"
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        self._build_window()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(1, self._tick)

    def _build_window(self) -> None:
        self.root.title("ResUNet Polyp Monitor")
        self.root.geometry("1280x800")
        self.root.minsize(900, 600)
        self.root.configure(bg="#071018")

        header = tk.Frame(self.root, bg="#0c1b26", height=64)
        header.pack(fill="x")
        tk.Label(header, text="ResUNet Polyp Monitor", bg="#0c1b26", fg="#e7f3f4",
                 font=("TkDefaultFont", 20, "bold")).pack(side="left", padx=20, pady=16)
        self.status = tk.Label(header, text="Loading model…", bg="#233744", fg="#dce8ec",
                               padx=14, pady=7, font=("TkDefaultFont", 11, "bold"))
        self.status.pack(side="right", padx=20)

        body = tk.Frame(self.root, bg="#071018")
        body.pack(fill="both", expand=True, padx=16, pady=16)
        side = tk.Frame(body, bg="#0c1b26", width=245)
        side.pack(side="right", fill="y", padx=(16, 0))
        side.pack_propagate(False)

        self.monitor = tk.Label(body, bg="#020609", cursor="crosshair",
                                text="Opening camera…", fg="#93a8b2")
        self.monitor.pack(side="left", fill="both", expand=True)
        self.monitor.bind("<Button-1>", self.save_manual)
        self.monitor.bind("<Configure>", lambda _event: self._redraw_latest())

        ttk.Button(side, text="Save original image", command=self.save_manual).pack(
            fill="x", padx=18, pady=(22, 8), ipady=7
        )
        ttk.Button(side, text="Test alert sound", command=self._play_alert).pack(
            fill="x", padx=18, pady=(0, 18), ipady=5
        )
        self.fps_text = self._metric(side, "Processed rate", "0.0 FPS")
        self.latency_text = self._metric(side, "Inference", "— ms")
        self.coverage_text = self._metric(side, "Mask area", "0.0%")
        tk.Label(side, text="Last action", bg="#0c1b26", fg="#7997a5").pack(
            anchor="w", padx=18, pady=(20, 3)
        )
        self.message = tk.Label(side, text="Waiting…", bg="#0c1b26", fg="#e7f3f4",
                                justify="left", wraplength=205)
        self.message.pack(anchor="w", padx=18)

    @staticmethod
    def _metric(parent: tk.Widget, label: str, initial: str) -> tk.Label:
        tk.Label(parent, text=label, bg="#0c1b26", fg="#7997a5").pack(
            anchor="w", padx=18, pady=(15, 2)
        )
        value = tk.Label(parent, text=initial, bg="#0c1b26", fg="#e7f3f4",
                         font=("TkDefaultFont", 16, "bold"))
        value.pack(anchor="w", padx=18)
        return value

    def _tick(self) -> None:
        if not self.running:
            return
        self._read_result()
        self._read_camera_status()
        try:
            # Queue size is one, but draining also handles a frame racing with get().
            while True:
                self.latest_raw = self.camera_queue.get_nowait()
        except queue.Empty:
            pass
        # Poll results frequently so a completed inference immediately feeds the next one.
        self.root.after(5, self._tick)

    def _read_camera_status(self) -> None:
        try:
            kind, payload = self.camera_status.get_nowait()
        except queue.Empty:
            return
        if kind == "error":
            self.message.configure(text=str(payload))
            self.status.configure(text="Camera error", bg="#84252b")
        else:
            self.message.configure(
                text=f"Camera {payload['width']}×{payload['height']} @ {payload['fps']:.0f} FPS"
            )

    def _read_result(self) -> None:
        try:
            kind, payload = self.output_queue.get_nowait()
        except queue.Empty:
            return
        if kind == "ready":
            self.worker_ready = True
            self.status.configure(text="Monitoring", bg="#233744")
            return
        if kind == "error":
            self.status.configure(text="Model error", bg="#84252b")
            self.message.configure(text=str(payload))
            return

        self.last_display = payload["display"]
        self._redraw_latest()
        detected = payload["detected"]
        self.status.configure(text="POLYP DETECTED" if detected else "Monitoring",
                              bg="#84252b" if detected else "#233744")
        self.coverage_text.configure(text=f"{payload['mask_ratio'] * 100:.1f}%")
        self.latency_text.configure(text=f"{payload['inference_ms']:.1f} ms")
        self.frame_count += 1
        now = time.perf_counter()
        if now - self.fps_started >= 1:
            self.fps_text.configure(text=f"{self.frame_count / (now - self.fps_started):.1f} FPS")
            self.frame_count = 0
            self.fps_started = now
        if detected and not self.was_detected:
            self._play_alert()
        self.was_detected = detected
        if payload["auto_saved"]:
            self.message.configure(text="Detection saved automatically")

    def _play_alert(self) -> None:
        """Play a non-blocking native alert when a new polyp appears."""
        if winsound is not None:
            # Run outside Tk's UI loop so two distinct beeps do not pause video.
            threading.Thread(target=windows_alert, daemon=True).start()
        else:
            threading.Thread(target=linux_alert, daemon=True).start()

    def _redraw_latest(self) -> None:
        display = getattr(self, "last_display", None)
        if display is None or self.monitor.winfo_width() < 2:
            return
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((self.monitor.winfo_width(), self.monitor.winfo_height()), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        self.monitor.configure(image=self.photo, text="")

    def save_manual(self, _event=None) -> None:
        if self.latest_raw is None:
            self.message.configure(text="No camera frame available yet")
            return
        path = self.manual_dir / f"{timestamp_stem('capture')}.jpg"
        if cv2.imwrite(str(path), self.latest_raw):
            self.message.configure(text=f"Saved {path}")
        else:
            self.message.configure(text="Could not save the frame")

    def close(self) -> None:
        self.running = False
        self.camera_stop.set()
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            pass
        self.inference_process.join(timeout=2)
        self.camera_worker.join(timeout=2)
        if self.inference_process.is_alive():
            self.inference_process.terminate()
        if self.camera_worker.is_alive():
            self.camera_worker.terminate()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(bundled_file("model.pth")))
    parser.add_argument("--save-dir", default=str(default_capture_dir()))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--target-fps", type=int, default=12)
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="PyTorch inference threads; try 4, 8, or 12 on CPU")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-ratio", type=float, default=0.001)
    parser.add_argument("--auto-save-ratio", type=float, default=0.02)
    parser.add_argument("--consecutive-frames", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    config = vars(parse_args())
    # Normalize values before serializing configuration to child processes.
    config["threshold"] = float(config["threshold"] if config["threshold"] is not None else 0.5)
    config["min_component_ratio"] = float(
        config["min_component_ratio"] if config["min_component_ratio"] is not None else 0.001
    )
    config["auto_save_ratio"] = float(
        config["auto_save_ratio"] if config["auto_save_ratio"] is not None else 0.02
    )
    context = mp.get_context("spawn")
    camera_queue = context.Queue(maxsize=1)
    camera_status = context.Queue(maxsize=2)
    camera_stop = context.Event()
    input_queue = context.Queue(maxsize=1)
    output_queue = context.Queue(maxsize=2)
    camera_process = context.Process(
        target=capture_worker,
        args=(camera_queue, input_queue, camera_status, camera_stop, config),
        daemon=True,
    )
    inference_process = context.Process(
        target=inference_worker, args=(input_queue, output_queue, config), daemon=True
    )
    camera_process.start()
    inference_process.start()
    root = tk.Tk()
    try:
        PolypMonitor(
            root, config, camera_queue, camera_status, camera_stop,
            input_queue, output_queue, camera_process, inference_process,
        )
        root.mainloop()
    except Exception as exc:
        camera_stop.set()
        camera_process.terminate()
        inference_process.terminate()
        messagebox.showerror("Polyp Monitor", str(exc))
        root.destroy()


if __name__ == "__main__":
    main()
