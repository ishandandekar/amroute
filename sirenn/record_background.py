import argparse
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio

from inference import (
    SIGNAL_THRESHOLD,
    CHUNK_SECONDS,
    choose_input_device,
    probe_devices,
)


def record_background(seconds, window_seconds, out_dir, input_device):
    p = pyaudio.PyAudio()

    selected = choose_input_device(p, requested=input_device)
    if selected is None:
        default_info = p.get_default_input_device_info()
        selected = default_info["index"]
        print(
            f"\nNo input device with signal found; falling back to default input: "
            f"idx {selected} ({default_info['name']})"
        )
    else:
        print(
            f"\nSelected input device: idx {selected} "
            f"({p.get_device_info_by_index(selected)['name']})"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_info = p.get_device_info_by_index(selected)
    capture_sr = int(device_info["defaultSampleRate"])
    window_frames = int(capture_sr * window_seconds)
    chunk_frames = int(capture_sr * CHUNK_SECONDS)
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=capture_sr,
        input=True,
        input_device_index=selected,
        frames_per_buffer=chunk_frames,
    )

    print(f"Recording {seconds}s of background; keep the room as quiet as you can...")
    frames_read = 0
    window_buf = bytearray()
    window_index = 0
    try:
        while frames_read < seconds * capture_sr:
            data = stream.read(chunk_frames, exception_on_overflow=False)
            window_buf += data
            frames_read += chunk_frames
            if len(window_buf) // 2 >= window_frames:
                payload = bytes(window_buf[: window_frames * 2])
                window_buf = bytearray(window_buf[window_frames * 2 :])
                window_index += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = out_dir / f"background_{timestamp}.wav"
                with wave.open(str(path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(capture_sr)
                    wf.writeframes(payload)
                audio = np.frombuffer(payload, dtype=np.int16) / 32768.0
                rms = float(np.sqrt(np.mean(audio**2)))
                print(f"  saved {path} (rms={rms:.6f})")
    finally:
        stream.close()
        p.terminate()
    print(f"\nDone: {window_index} background windows -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Record a quiet background clip and split it into windows "
        "for the 'noise' training class",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--out-dir", default="sounds/noise")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="probe all input devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        from rich.console import Console
        from rich.table import Table

        p = pyaudio.PyAudio()
        rows = probe_devices(p)
        table = Table(title="Input devices (probed)", box=None)
        table.add_column("idx", justify="right")
        table.add_column("name")
        table.add_column("sample rate", justify="right")
        table.add_column("signal (rms)", justify="right")
        for row in rows:
            if row["rms"] is None:
                status = "no data"
            elif row["rms"] < SIGNAL_THRESHOLD:
                status = "silence"
            else:
                status = "signal"
            rms_str = (
                f"{row['rms']:.6f} {status}" if row["rms"] is not None else status
            )
            table.add_row(str(row["index"]), row["name"], str(row["sr"]), rms_str)
        Console().print(table)
        p.terminate()
        return

    record_background(
        seconds=args.seconds,
        window_seconds=args.window,
        out_dir=args.out_dir,
        input_device=args.input_device,
    )


if __name__ == "__main__":
    main()
