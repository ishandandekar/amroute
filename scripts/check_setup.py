"""Dependency-light preflight checks for a local Am-Rout checkout."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DETECTION_MODULES = ("cv2", "librosa", "numpy", "pyaudio", "rich", "torch", "ultralytics")
DETECTION_MODELS = (
    ROOT / "brite" / "best_YOLO_ambulance_detect.pt",
    ROOT / "sirenn" / "sireNN.pt",
)


def check_detection(root: Path = ROOT) -> list[tuple[bool, str]]:
    checks = []
    for module in DETECTION_MODULES:
        checks.append((importlib.util.find_spec(module) is not None, f"Python module: {module}"))
    for model in DETECTION_MODELS:
        path = root / model.relative_to(ROOT)
        checks.append((path.is_file(), f"Model file: {path.relative_to(root)}"))
    return checks


def check_simulation(root: Path = ROOT) -> list[tuple[bool, str]]:
    wrapper = root / "sim" / "vendor" / "bin"
    sumo = shutil.which("sumo") or (wrapper / "sumo" if platform.system() == "Linux" else None)
    netconvert = shutil.which("netconvert") or (
        wrapper / "netconvert" if platform.system() == "Linux" else None
    )
    return [
        (importlib.util.find_spec("traci") is not None, "Python module: traci"),
        (importlib.util.find_spec("osmium") is not None, "Python module: osmium"),
        (bool(sumo and Path(sumo).is_file()), "SUMO executable"),
        (bool(netconvert and Path(netconvert).is_file()), "netconvert executable"),
        ((root / "sim" / "corridor" / "corridor.osm").is_file(), "Corridor OSM source"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="check whether Am-Rout is ready to run")
    parser.add_argument("--mode", choices=("all", "detection", "simulation"), default="all")
    args = parser.parse_args(argv)

    checks = [(sys.version_info >= (3, 12), "Python 3.12+")]
    if args.mode in ("all", "detection"):
        checks.extend(check_detection())
    if args.mode in ("all", "simulation"):
        checks.extend(check_simulation())

    for passed, label in checks:
        print(f"{'[ok]' if passed else '[missing]'} {label}")
    missing = sum(not passed for passed, _ in checks)
    if missing:
        print(f"\n{missing} required item(s) missing. See README.md for setup instructions.")
        return 1
    print("\nAm-Rout setup looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
