import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sirenn"))

import cv2
import inference as sirenn
import numpy as np
import pyaudio
import torch
from rich.console import Console
from rich.panel import Panel
from ultralytics import YOLO

from amroute_core import Detections


class VisionDetector:
    def __init__(
        self,
        detections,
        stop,
        model_path,
        source,
        conf,
        imgsz,
        output,
        cooldown,
        console,
    ):
        self.detections = detections
        self.stop = stop
        self.model_path = model_path
        self.source = source
        self.conf = conf
        self.imgsz = imgsz
        self.output = output
        self.cooldown = cooldown
        self.console = console

    def run(self):
        try:
            model = YOLO(self.model_path, verbose=False)
            self.console.print(
                f"[vision] loaded {self.model_path} | device: {model.device.type.upper()}"
            )
        except Exception as exc:
            self.console.print(f"[bold red][vision] failed to load model: {exc}[/]")
            return

        source = int(self.source) if self.source.isdigit() else self.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.console.print(
                f"[bold red][vision] could not open source: {self.source}[/]"
            )
            return

        writer = None
        fps = 0.0
        last_alert = 0.0
        prev_t = time.perf_counter()

        try:
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                result = model.predict(
                    frame, conf=self.conf, imgsz=self.imgsz, verbose=False
                )[0]

                now = time.perf_counter()
                inst_fps = 1.0 / max(now - prev_t, 1e-6)
                fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps
                prev_t = now

                detected = False
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf:
                        continue
                    detected = True
                    if now - last_alert >= self.cooldown:
                        label = result.names.get(int(box.cls[0]), str(int(box.cls[0])))
                        self.console.print(
                            f"[vision][{time.strftime('%H:%M:%S')}] {label} detected (conf={conf:.2f})"
                        )
                        last_alert = now
                if detected:
                    self.detections.ambulance_fire()

                annotated = result.plot()
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if writer is None and self.output:
                    h, w = annotated.shape[:2]
                    src_fps = cap.get(cv2.CAP_PROP_FPS)
                    out_fps = src_fps if src_fps > 1 else (fps if fps > 0 else 30.0)
                    writer = cv2.VideoWriter(
                        self.output,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        out_fps,
                        (w, h),
                    )
                if writer is not None:
                    writer.write(annotated)

                cv2.imshow("Ambulance Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as exc:
            self.console.print(f"[bold red][vision] error: {exc}[/]")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            self.stop.set()


class AudioDetector:
    def __init__(
        self,
        detections,
        stop,
        model_path,
        input_device,
        rms_threshold,
        hysteresis,
        console,
    ):
        self.detections = detections
        self.stop = stop
        self.model_path = model_path
        self.input_device = input_device
        self.rms_threshold = rms_threshold
        self.hysteresis = hysteresis
        self.console = console

    def run(self):
        p = pyaudio.PyAudio()

        try:
            selected = sirenn.choose_input_device(p, requested=self.input_device)
            if selected is None:
                default_info = p.get_default_input_device_info()
                selected = default_info["index"]
                self.console.print(
                    f"[audio] no input device with signal found; falling back to "
                    f"default input: idx {selected} ({default_info['name']})"
                )

            device_info = p.get_device_info_by_index(selected)
            capture_sr = int(device_info["defaultSampleRate"])
            block_size = capture_sr * sirenn.WINDOW_SECONDS
            self.console.print(
                f"[audio] selected input device: idx {selected} "
                f"({device_info['name']}) @ {capture_sr} Hz"
            )

            device = sirenn.get_device()
            self.console.print(f"[audio] using device: {device}")
            self.console.print(f"[audio] loading model from {self.model_path}...")
            model, temperature = sirenn.load_model(self.model_path, device)
            self.console.print("[audio] model loaded")

            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=capture_sr,
                input=True,
                input_device_index=selected,
                frames_per_buffer=int(capture_sr * sirenn.CHUNK_SECONDS),
            )
        except Exception as exc:
            self.console.print(f"[bold red][audio] failed to init: {exc}[/]")
            p.terminate()
            self.stop.set()
            return

        window_id = 0
        detect_streak = 0
        noise_floor = sirenn.NOISE_FLOOR_INIT
        cooldown_until = 0.0

        try:
            while not self.stop.is_set():
                data = sirenn.read_frames(
                    stream, block_size * 2, timeout=sirenn.STALL_TIMEOUT
                )
                if len(data) < block_size * 2:
                    self.console.print(
                        f"[audio] no audio received for {sirenn.STALL_TIMEOUT:.0f}s "
                        f"on device idx {selected} - is the mic muted?"
                    )
                    continue

                window_id += 1
                audio = np.frombuffer(data, dtype=np.int16) / 32768.0
                rms = float(np.sqrt(np.mean(audio**2)))
                peak = float(np.max(np.abs(audio)))

                if rms < self.rms_threshold or rms < sirenn.RELATIVE_GATE * noise_floor:
                    detect_streak = 0
                    noise_floor = (
                        1 - sirenn.NOISE_FLOOR_ALPHA
                    ) * noise_floor + sirenn.NOISE_FLOOR_ALPHA * rms
                    continue

                tensor = sirenn.preprocess_audio_array(audio, sr=capture_sr).to(device)
                with torch.no_grad():
                    logits = model(tensor) / temperature
                    probabilities = torch.softmax(logits, dim=1)
                    confidence, predicted_class = probabilities.max(dim=1)

                conf = confidence.item()
                class_id = int(predicted_class.item())
                p_siren = probabilities[0, sirenn.SIREN_CLASS].item()
                p_noise = probabilities[0, sirenn.NOISE_CLASS].item()
                is_detection = (
                    class_id == sirenn.SIREN_CLASS
                    and conf > sirenn.SIREN_THRESHOLD
                    and (p_siren - p_noise) > sirenn.NOISE_MARGIN
                )
                detect_streak = detect_streak + 1 if is_detection else 0
                fired = is_detection and detect_streak >= self.hysteresis

                if fired:
                    self.detections.siren_fire()
                    now = time.monotonic()
                    if now >= cooldown_until:
                        cooldown_until = now + sirenn.COOLDOWN_SECONDS
                        self.console.print(
                            f"[audio][bold yellow]SIREN DETECTED[/] | conf={conf:.4f} "
                            f"| siren_p={p_siren:.4f} | noise_p={p_noise:.4f} "
                            f"| rms={rms:.6f} | peak={peak:.6f} "
                            f"| streak={detect_streak}/{self.hysteresis}"
                        )
                else:
                    tag = sirenn.CLASS_LABELS.get(class_id, "unknown")
                    self.console.print(
                        f"[audio] {tag} | conf={conf:.4f} | siren_p={p_siren:.4f} "
                        f"| noise_p={p_noise:.4f} | streak={detect_streak}/{self.hysteresis}"
                    )
        except Exception as exc:
            self.console.print(f"[bold red][audio] error: {exc}[/]")
        finally:
            stream.close()
            p.terminate()


def fusion_monitor(detections, stop, window, fuse_cooldown, console):
    last_fuse = 0.0
    last_snapshot = None
    while not stop.is_set():
        time.sleep(0.5)
        snapshot = detections.snapshot()
        if snapshot is None or snapshot == last_snapshot:
            continue
        last_snapshot = snapshot
        now = time.monotonic()
        if now - last_fuse >= fuse_cooldown:
            last_fuse = now
            console.print(
                Panel.fit(
                    "[bold red]CONFIRMED AMBULANCE [/]\n"
                    f"siren + camera detection within {window:.1f}s "
                    f"(audio fires: {detections.siren_count}, "
                    f"vision fires: {detections.ambulance_count})",
                    border_style="red",
                )
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="amroute: fused ambulance detection - YOLO vision (brite) + "
        "sireNN audio run simultaneously",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    vision = parser.add_argument_group("vision (brite / YOLO)")
    vision.add_argument(
        "--source",
        default="0",
        help="webcam index, video file, or RTSP URL",
    )
    vision.add_argument(
        "--model",
        default=str(ROOT / "brite" / "best_YOLO_ambulance_detect.pt"),
        help="path to YOLO weights",
    )
    vision.add_argument(
        "--conf", type=float, default=0.25, help="vision confidence threshold"
    )
    vision.add_argument("--imgsz", type=int, default=640, help="inference image size")
    vision.add_argument(
        "--output", default=None, help="save annotated stream to this mp4 file"
    )
    vision.add_argument(
        "--cooldown", type=float, default=2.0, help="seconds between vision alerts"
    )

    audio = parser.add_argument_group("audio (sireNN / LSTM)")
    audio.add_argument(
        "--audio-model",
        default=str(ROOT / "sirenn" / "sireNN.pt"),
        help="path to the sireNN weights",
    )
    audio.add_argument(
        "--list-devices",
        action="store_true",
        help="probe all input audio devices and exit",
    )
    audio.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="PyAudio index of the mic to capture from",
    )
    audio.add_argument(
        "--rms-threshold",
        type=float,
        default=sirenn.RMS_THRESHOLD,
        help="windows with RMS energy below this are treated as silence and skipped",
    )
    audio.add_argument(
        "--hysteresis",
        type=int,
        default=sirenn.HYSTERESIS,
        help="number of consecutive siren windows required before firing",
    )

    fuse = parser.add_argument_group("fusion")
    mode = fuse.add_mutually_exclusive_group()
    mode.add_argument(
        "--audio-only",
        action="store_true",
        help="run only sireNN audio detection (no camera or fused alert)",
    )
    mode.add_argument(
        "--vision-only",
        action="store_true",
        help="run only YOLO vision detection (no microphone or fused alert)",
    )
    fuse.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="seconds between an audio and a vision detection that counts as fused",
    )
    fuse.add_argument(
        "--fuse-cooldown",
        type=float,
        default=15.0,
        help="seconds between fused alerts",
    )

    return parser.parse_args(argv)


def selected_mode(args) -> str:
    if args.audio_only:
        return "audio"
    if args.vision_only:
        return "vision"
    return "fused"


def validate_model_paths(args, parser_error) -> None:
    """Fail before opening hardware when a required model is unavailable."""
    mode = selected_mode(args)
    missing = []
    if mode in ("fused", "vision") and not Path(args.model).is_file():
        missing.append(f"vision model: {args.model}")
    if mode in ("fused", "audio") and not Path(args.audio_model).is_file():
        missing.append(f"audio model: {args.audio_model}")
    if missing:
        parser_error(
            "required model file(s) not found:\n  - "
            + "\n  - ".join(missing)
            + "\nSee README.md#model-files."
        )


def main():
    args = parse_args()
    console = Console()

    if args.list_devices:
        sirenn.list_devices_and_exit()
        return

    validate_model_paths(args, lambda message: raise_system_exit(message))

    mode = selected_mode(args)

    detections = Detections(args.window)
    stop = threading.Event()

    vision = VisionDetector(
        detections,
        stop,
        args.model,
        args.source,
        args.conf,
        args.imgsz,
        args.output,
        args.cooldown,
        console,
    )
    audio = AudioDetector(
        detections,
        stop,
        args.audio_model,
        args.input_device,
        args.rms_threshold,
        args.hysteresis,
        console,
    )

    details = {
        "fused": f"vision: YOLO on {args.source} | audio: sireNN on mic",
        "audio": "audio: sireNN on mic",
        "vision": f"vision: YOLO on {args.source}",
    }
    console.print(
        Panel.fit(
            f"[bold]amroute[/] - {mode} ambulance detection\n{details[mode]}",
            border_style="cyan",
        )
    )

    threads = []
    if mode in ("fused", "vision"):
        threads.append(threading.Thread(target=vision.run, name="vision", daemon=True))
    if mode in ("fused", "audio"):
        threads.append(threading.Thread(target=audio.run, name="audio", daemon=True))
    for thread in threads:
        thread.start()

    try:
        if mode == "fused":
            fusion_monitor(detections, stop, args.window, args.fuse_cooldown, console)
        else:
            while not stop.wait(0.5):
                pass
    except KeyboardInterrupt:
        console.print("[yellow]interrupt received, stopping...[/]")
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2)
        console.print(
            f"[bold]done[/] - audio fires: {detections.siren_count}, "
            f"vision fires: {detections.ambulance_count}"
        )


def raise_system_exit(message: str) -> None:
    raise SystemExit(f"amroute: error: {message}")


if __name__ == "__main__":
    main()
