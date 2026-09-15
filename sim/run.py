"""SUMO smoke-test runner: launches sumo headless with TraCI, steps the sim, reads state."""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import traci

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKDIR = REPO_ROOT / "sim" / "smoke"
DEFAULT_CFG = "smoke.sumocfg"
DEFAULT_STEPS = 200


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def resolve_sumo_bin(env_bin: str | None) -> str:
    if env_bin:
        return env_bin
    wrapper = REPO_ROOT / "sim" / "vendor" / "bin" / "sumo"
    if wrapper.is_file():
        return str(wrapper)
    which = shutil.which("sumo")
    if which:
        return which
    raise FileNotFoundError("sumo binary not found; set SUMO_BIN or add sim/vendor/bin to PATH")


def get_version_summary() -> str:
    api_version, sumo_version = traci.getVersion()
    return f"traci_api={api_version} sumo_binary={sumo_version} traci_lib={traci.__version__}"


def check_version_alignment() -> None:
    _, sumo_version = traci.getVersion()
    sumo_release = sumo_version.split()[1] if len(sumo_version.split()) > 1 else sumo_version
    if sumo_release != traci.__version__:
        raise RuntimeError(
            f"TraCI version mismatch: sumo={sumo_release} traci={traci.__version__}; "
            "align by pinning the right traci in pyproject.toml"
        )


def run_smoke(
    workdir: Path,
    cfg: str,
    steps: int,
    sumo_bin: str,
) -> None:
    workdir = Path(workdir)
    cfg_path = workdir / cfg
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    port = find_free_port()
    cmd = [
        sumo_bin,
        "-c",
        cfg,
        "--remote-port",
        str(port),
        "--no-warnings",
    ]
    log_path = workdir / "run.log"

    print(f"[run.py] starting: {' '.join(cmd)} (cwd={workdir})")
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )

    sim_time = 0.0
    veh_counts: list[int] = []
    traci_version_info = ""
    try:
        traci.init(port=port, host="127.0.0.1", numRetries=60)
        traci_version_info = get_version_summary()
        check_version_alignment()
        for _ in range(steps):
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            veh_counts.append(len(traci.vehicle.getIDList()))
    finally:
        traci.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    max_vehicles = max(veh_counts) if veh_counts else 0
    print(f"[run.py] {traci_version_info}", end="")
    print(f"[run.py] simulated steps={steps} end_time={sim_time:.0f}s")
    print(f"[run.py] vehicles on net: max={max_vehicles} at_step={veh_counts.index(max_vehicles)}")

    if sim_time < steps:
        raise RuntimeError(f"sim ended early at {sim_time}s before {steps} steps")
    if max_vehicles == 0:
        raise RuntimeError("no vehicles were present on the network")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SUMO TraCI smoke test")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"sim steps (default {DEFAULT_STEPS})")
    ap.add_argument("--cfg", default=DEFAULT_CFG, help=f"config file in workdir (default {DEFAULT_CFG})")
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="directory with sumo cfg")
    ap.add_argument("--sumo-bin", default=None, help="sumo binary (default: SUMO_BIN env, then sim/vendor/bin)")
    args = ap.parse_args(argv)

    sumo_bin = resolve_sumo_bin(args.sumo_bin or None)
    try:
        run_smoke(args.workdir, args.cfg, args.steps, sumo_bin)
    except Exception as exc:
        print(f"[run.py] FAILED: {exc}", file=sys.stderr)
        return 1
    print("[run.py] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())