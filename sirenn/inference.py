import argparse
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio
import torch
import torch.nn as nn
from rich import box
from rich.console import Console
from rich.progress import track
from rich.table import Table

import pandas as pd

import preprocess


class SireNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden.squeeze(0))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bandpass_filter(data, low=500, high=2000, sr=16000):
    return preprocess.bandpass_filter(data, sr=sr, low=low, high=high)


MODEL_SR = preprocess.MODEL_SR


def preprocess_audio(filepath, n_mfcc=80):
    features = preprocess.extract_features_from_path(filepath, n_mfcc=n_mfcc)
    return preprocess.to_model_input(features)


def preprocess_audio_array(audio, sr, n_mfcc=80):
    features = preprocess.extract_features_from_array(audio, sr, n_mfcc=n_mfcc)
    return preprocess.to_model_input(features)


def load_model(model_path="sireNN.pt", device=None, temperature=None):
    if device is None:
        device = get_device()
    model = SireNN(input_dim=80, hidden_dim=128, num_classes=3, dropout=0.5)
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = torch.compile(model)
    model.to(device)
    model.eval()
    if temperature is None:
        temperature = load_temperature()
    return model, temperature


def load_temperature(default=1.0):
    path = Path(__file__).resolve().parent / "temperature.json"
    try:
        import json

        with open(path) as f:
            return float(json.load(f)["temperature"])
    except (OSError, KeyError, ValueError):
        return default


def predict(model, audio_tensor, device=None, temperature=1.0):
    if device is None:
        device = get_device()

    with torch.no_grad():
        logits = model(audio_tensor.to(device)) / temperature
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = probabilities.max(dim=1)

    class_labels = {0: "traffic", 1: "siren", 2: "noise"}
    pred_id = int(predicted_class.item())
    conf = confidence.item()

    probs = probabilities[0].tolist()
    return {
        "prediction": class_labels[pred_id],
        "class_id": pred_id,
        "confidence": round(conf, 4),
        "probabilities": {
            "traffic": round(probs[0], 4),
            "siren": round(probs[1], 4),
            "noise": round(probs[2], 4),
        },
    }


def predict_over_directory(
    directory_path: Path,
    model_path: str,
    results_filepath: str | None = "results.csv",
    show_results: bool = True,
):
    assert directory_path.exists(), (
        "Directory " + str(directory_path.absolute()) + " does not exists"
    )
    audio_filepaths = list(directory_path.glob("*.wav"))
    assert len(audio_filepaths) > 1, "No audio files found in directory"

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}...")
    model, temperature = load_model(model_path, device)
    print("Model loaded and compiled successfully.\n")
    results = []

    if show_results:
        table = Table(title="sireNN results", box=box.HEAVY_HEAD)

        table.add_column("audio", justify="right")
        table.add_column("prediction")
        table.add_column(
            "class_id",
            justify="right",
        )
        table.add_column(
            "confidence",
            justify="right",
        )

    for audio_filepath in track(
        audio_filepaths,
        description="Predicting sireNN over directory",
        total=len(audio_filepaths),
    ):
        audio_tensor = preprocess_audio(audio_filepath)

        result: dict = predict(model, audio_tensor, device, temperature=temperature)
        if show_results:
            table.add_row(
                str(audio_filepath.stem),
                str(result["prediction"]),
                str(result["class_id"]),
                str(result["confidence"]),
            )
        results.append(result)

    results_df = pd.DataFrame.from_records(results)
    results_df.to_csv(results_filepath, index=False)
    if show_results:
        cns = Console()
        cns.print(table)


def run_wavefile_mode(model_path: str, audio_path: str, directory_path: str | None):
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}...")
    model, temperature = load_model(model_path, device)
    print("Model loaded and compiled successfully.\n")

    print(f"Processing audio: {audio_path}")
    audio_tensor = preprocess_audio(audio_path)
    print(f"Audio tensor shape: {audio_tensor.shape}\n")

    result = predict(model, audio_tensor, device, temperature=temperature)

    print("=" * 40)
    print("PREDICTION RESULTS")
    print("=" * 40)
    print(f"Predicted Class: {result['prediction'].upper()}")
    print(f"Class ID:        {result['class_id']}")
    print(f"Confidence:      {result['confidence']:.2%}")
    print(
        f"Probabilities:   Traffic={result['probabilities']['traffic']:.2%}, "
        f"Siren={result['probabilities']['siren']:.2%}, "
        f"Noise={result['probabilities']['noise']:.2%}"
    )
    print("=" * 40)

    if directory_path is not None:
        print("Running model over the directory")
        _ = predict_over_directory(
            directory_path=Path.cwd() / directory_path, model_path=model_path
        )


# Realtime prediction
SR = 48_000
WINDOW_SECONDS = 2
BLOCK_SIZE = SR * WINDOW_SECONDS
CLASS_LABELS = {0: "traffic", 1: "siren", 2: "noise"}
SIREN_CLASS = 1
NOISE_CLASS = 2
SIREN_THRESHOLD = 0.80
NOISE_MARGIN = 0.40
RMS_THRESHOLD = 0.01
HYSTERESIS = 2
SIGNAL_THRESHOLD = 1e-4
PROBE_SECONDS = 3
CHUNK_SECONDS = 0.1
STALL_TIMEOUT = 5.0
RELATIVE_GATE = 3.0
NOISE_FLOOR_ALPHA = 0.05
NOISE_FLOOR_INIT = 0.05
COOLDOWN_SECONDS = 5.0
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def read_frames(stream: pyaudio.Stream, num_bytes: int, timeout: float) -> bytes:
    frames_needed = num_bytes // 2
    buffer = b""
    deadline = time.monotonic() + timeout
    try:
        while len(buffer) // 2 < frames_needed:
            available = stream.get_read_available()
            if available > 0:
                n = min(available, frames_needed - len(buffer) // 2)
                buffer += stream.read(n, exception_on_overflow=False)
            else:
                if time.monotonic() > deadline:
                    break
                time.sleep(0.02)
    except Exception:
        pass
    return buffer


def save_window(
    cache_dir: Path, data: bytes, tag: str, timestamp: str, conf: float, rms: float,
    framerate: int = SR,
):
    path = cache_dir / f"{tag}_{timestamp}_conf_{conf:.4f}_rms_{rms:.4f}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(data)
    return path


def probe_input_device(
    p: pyaudio.PyAudio, index: int, seconds: float = PROBE_SECONDS
) -> float | None:
    device_info = p.get_device_info_by_index(index)
    sr = int(device_info["defaultSampleRate"])
    chunk = int(sr * CHUNK_SECONDS)
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sr,
        input=True,
        input_device_index=index,
        frames_per_buffer=chunk,
    )
    try:
        data = read_frames(stream, int(seconds * sr) * 2, timeout=seconds + 1.0)
    finally:
        stream.close()
    if len(data) < int(seconds * sr) * 2:
        return None
    audio = np.frombuffer(data, dtype=np.int16) / 32768.0
    return float(np.sqrt(np.mean(audio**2)))


def probe_devices(p: pyaudio.PyAudio) -> list[dict]:
    rows = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] <= 0:
            continue
        rms = None
        try:
            rms = probe_input_device(p, i)
        except Exception:
            pass
        rows.append(
            {
                "index": i,
                "name": info["name"],
                "sr": int(info["defaultSampleRate"]),
                "rms": rms,
            }
        )
    return rows


def print_device_table(rows: list[dict], selected: int | None = None):
    table = Table(title="Input devices (probed)", box=box.HEAVY_HEAD)
    table.add_column("idx", justify="right")
    table.add_column("name")
    table.add_column("sample rate", justify="right")
    table.add_column("signal (rms)", justify="right")
    for row in rows:
        marker = "  <- selected" if selected == row["index"] else ""
        if row["rms"] is None:
            status = "no data"
        elif row["rms"] < SIGNAL_THRESHOLD:
            status = "silence"
        else:
            status = "signal"
        rms_str = f"{row['rms']:.6f} {status}" if row["rms"] is not None else status
        table.add_row(str(row["index"]), row["name"], str(row["sr"]), rms_str + marker)
    Console().print(table)


def choose_input_device(p: pyaudio.PyAudio, requested: int | None) -> int | None:
    rows = probe_devices(p)
    if requested is not None:
        selected = requested
    else:
        selected = next(
            (
                row["index"]
                for row in rows
                if row["rms"] is not None and row["rms"] > SIGNAL_THRESHOLD
            ),
            None,
        )
    print_device_table(rows, selected)
    return selected


def run_mic_mode(
    model_path: str,
    cache_dir: Path,
    save_all: bool,
    rms_threshold: float,
    hysteresis: int,
    input_device: int | None,
):
    p = pyaudio.PyAudio()

    selected = choose_input_device(p, requested=input_device)
    if selected is None:
        default_info = p.get_default_input_device_info()
        selected = default_info["index"]
        print(
            f"\nNo input device with signal found; falling back to default input: "
            f"idx {selected} ({default_info['name']})"
        )

    device_info = p.get_device_info_by_index(selected)
    capture_sr = int(device_info["defaultSampleRate"])
    block_size = capture_sr * WINDOW_SECONDS
    print(
        f"\nSelected input device: idx {selected} ({device_info['name']}) "
        f"@ {capture_sr} Hz"
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving windows to: {cache_dir}")

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}...")
    model, temperature = load_model(model_path, device)
    print("Model loaded and compiled successfully.\n")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=capture_sr,
        input=True,
        input_device_index=selected,
        frames_per_buffer=int(capture_sr * CHUNK_SECONDS),
    )

    print("Listening for ambulance sirens...")
    window_id = 0
    detect_streak = 0
    saved_count = 0
    silence_warned = False
    noise_floor = NOISE_FLOOR_INIT
    cooldown_until = 0.0
    try:
        while True:
            data = read_frames(stream, block_size * 2, timeout=STALL_TIMEOUT)
            if len(data) < block_size * 2:
                print(
                    f"[w#{window_id}] no audio received for {STALL_TIMEOUT:.0f}s on "
                    f"device idx {selected} - is something playing or is the mic muted?"
                )
                continue

            window_id += 1
            audio = np.frombuffer(data, dtype=np.int16) / 32768.0
            rms = float(np.sqrt(np.mean(audio**2)))
            peak = float(np.max(np.abs(audio)))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            if rms < rms_threshold:
                detect_streak = 0
                noise_floor = (1 - NOISE_FLOOR_ALPHA) * noise_floor + NOISE_FLOOR_ALPHA * rms
                if not silence_warned:
                    print(
                        f"  (no signal on device idx {selected} - is the mic muted or "
                        f"unplugged? try --list-devices / --input-device)"
                    )
                    silence_warned = True
                print(f"[w#{window_id}] silence | rms={rms:.6f} | peak={peak:.6f}")
                continue

            if rms < RELATIVE_GATE * noise_floor:
                detect_streak = 0
                noise_floor = (1 - NOISE_FLOOR_ALPHA) * noise_floor + NOISE_FLOOR_ALPHA * rms
                print(
                    f"[w#{window_id}] background | rms={rms:.6f} | peak={peak:.6f} | "
                    f"floor={noise_floor:.4f}"
                )
                continue

            tensor = preprocess_audio_array(audio, sr=capture_sr)
            tensor = tensor.to(device)
            with torch.no_grad():
                logits = model(tensor) / temperature
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_class = probabilities.max(dim=1)

            conf = confidence.item()
            class_id = int(predicted_class.item())
            p_siren = probabilities[0, SIREN_CLASS].item()
            p_noise = probabilities[0, NOISE_CLASS].item()
            is_detection = (
                class_id == SIREN_CLASS
                and conf > SIREN_THRESHOLD
                and (p_siren - p_noise) > NOISE_MARGIN
            )
            detect_streak = detect_streak + 1 if is_detection else 0
            fired = is_detection and detect_streak >= hysteresis
            tag = CLASS_LABELS.get(class_id, "unknown")
            if is_detection:
                tag = "siren"

            saved = ""
            if save_all or is_detection:
                path = save_window(cache_dir, data, tag, timestamp, conf, rms, framerate=capture_sr)
                saved_count += 1
                saved = f" | saved: {path}"

            now = time.monotonic()
            if fired:
                if now >= cooldown_until:
                    cooldown_until = now + COOLDOWN_SECONDS
                    print(
                        f"[w#{window_id}] SIREN DETECTED | confidence={conf:.4f} | "
                        f"siren_p={p_siren:.4f} | noise_p={p_noise:.4f} | "
                        f"rms={rms:.6f} | peak={peak:.6f} | "
                        f"streak={detect_streak}/{hysteresis}{saved}"
                    )
                else:
                    print(
                        f"[w#{window_id}] siren (cooldown) | confidence={conf:.4f} | "
                        f"rms={rms:.6f} | peak={peak:.6f}"
                    )
            else:
                print(
                    f"[w#{window_id}] {tag} | confidence={conf:.4f} | "
                    f"siren_p={p_siren:.4f} | noise_p={p_noise:.4f} | "
                    f"rms={rms:.6f} | peak={peak:.6f} | "
                    f"streak={detect_streak}/{hysteresis}{saved}"
                )
    except KeyboardInterrupt:
        print(f"\nStopped after {window_id} windows, saved {saved_count} files.")
    finally:
        stream.close()
        p.terminate()


def list_devices_and_exit():
    p = pyaudio.PyAudio()
    print_device_table(probe_devices(p))
    p.terminate()


def main():
    parser = argparse.ArgumentParser(
        description="sireNN inference: ambulance siren detection on wave files or live mic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_group = parser.add_argument_group("mode")
    mode_group.add_argument(
        "--use-mic",
        action="store_true",
        help="run realtime prediction from the microphone instead of a wave file",
    )
    mode_group.add_argument(
        "--model",
        default="sireNN.pt",
        help="path to the model weights",
    )
    mode_group.add_argument(
        "--audio",
        default="./sounds/ambulance_and_traffic/sound_200.wav",
        help="wave file to predict on (ignored with --use-mic)",
    )
    mode_group.add_argument(
        "--directory",
        default=None,
        help="optional directory of wave files to batch predict over, writing results.csv (ignored with --use-mic)",
    )

    mic_group = parser.add_argument_group("mic observability")
    mic_group.add_argument(
        "--list-devices",
        action="store_true",
        help="probe all input devices and exit (diagnose which mic to use)",
    )
    mic_group.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="PyAudio index of the mic to capture from; default is to auto-probe and pick the first device with signal",
    )
    mic_group.add_argument(
        "--cache-dir",
        default=str(CACHE_DIR),
        help="directory where 2-second mic windows are saved as wav files",
    )
    mic_group.add_argument(
        "--save-all",
        action="store_true",
        help="save every 2-second mic window; by default only ambulance windows are saved",
    )
    mic_group.add_argument(
        "--rms-threshold",
        type=float,
        default=RMS_THRESHOLD,
        help="windows with RMS energy below this are treated as silence and skipped",
    )
    mic_group.add_argument(
        "--hysteresis",
        type=int,
        default=HYSTERESIS,
        help="number of consecutive ambulance windows required before SIREN DETECTED fires",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices_and_exit()
        return

    if args.use_mic:
        run_mic_mode(
            model_path=args.model,
            cache_dir=Path(args.cache_dir),
            save_all=args.save_all,
            rms_threshold=args.rms_threshold,
            hysteresis=args.hysteresis,
            input_device=args.input_device,
        )
    else:
        run_wavefile_mode(args.model, args.audio, args.directory)


if __name__ == "__main__":
    main()
