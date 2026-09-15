"""SUMO runners: smoke test + baseline EV telemetry (R=0) + preemption (knob R).

smoke     launch corridor/smoke net headless, step, read state  (M0 verify)
baseline  headless R=0 run: background traffic + 1 fixed EV, record EV travel
          time, # stops (red at signal vs total), speed profile per density
preempt   R-preemption run: a signal preempts green when the EV is within R m;
          --r inf holds every signal from the moment it becomes next (ceiling).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import traci
from rich.console import Console
from rich.table import Table

from control import PreemptionController, format_r

REPO_ROOT = Path(__file__).resolve().parent.parent
CORRIDOR_DIR = REPO_ROOT / "sim" / "corridor"
RESULTS_DIR = REPO_ROOT / "sim" / "results"
EV_ROUTE_PATH = CORRIDOR_DIR / "ev.route.json"
DEFAULT_WORKDIR = REPO_ROOT / "sim" / "smoke"
DEFAULT_CFG = "smoke.sumocfg"
DEFAULT_STEPS = 200

EV_ID = "ev"
STOP_SPEED = 0.1  # m/s below which the EV counts as stopped
RESUME_SPEED = 1.0  # m/s above which a stop episode ends
RED_DIST = 30.0  # m: max distance to a non-green signal for a "red stop" flag
GREEN_STATES = ("g", "G", "y", "Y")


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


def launch_sumo(workdir: Path, cfg: str, log_path: Path, sumo_bin: str):
    port = find_free_port()
    cmd = [sumo_bin, "-c", cfg, "--remote-port", str(port), "--no-warnings"]
    print(f"[run.py] starting: {' '.join(cmd)} (cwd={workdir})")
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    return proc, port


def stop_proc(proc) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


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

    proc, port = launch_sumo(workdir, cfg, workdir / "run.log", sumo_bin)

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
        stop_proc(proc)

    max_vehicles = max(veh_counts) if veh_counts else 0
    print(f"[run.py] {traci_version_info}", end="")
    print(f"[run.py] simulated steps={steps} end_time={sim_time:.0f}s")
    print(f"[run.py] vehicles on net: max={max_vehicles} at_step={veh_counts.index(max_vehicles)}")

    if sim_time < steps:
        raise RuntimeError(f"sim ended early at {sim_time}s before {steps} steps")
    if max_vehicles == 0:
        raise RuntimeError("no vehicles were present on the network")


# --------------------------------------------------------------------------
# Baseline (R=0) telemetry
# --------------------------------------------------------------------------

def load_ev_route() -> dict:
    if not EV_ROUTE_PATH.is_file():
        raise FileNotFoundError(f"EV route not found: {EV_ROUTE_PATH} (run sim/scenario.py first)")
    return json.loads(EV_ROUTE_PATH.read_text())


def collect_telemetry(sumo_bin: str, density: str, seed: int, results_dir: Path,
                      r: float | None = None):
    cfg = f"corridor.{density}.{seed}.sumocfg"
    workdir = CORRIDOR_DIR
    cfg_path = workdir / cfg
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}; run sim/scenario.py first")

    tag = "R0" if r is None else f"R{format_r(r)}"
    summary_name = "R0_baseline.csv" if r is None else "R_preempt.csv"
    ctl = PreemptionController(r) if r is not None else None

    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{tag}_{density}_{seed}.run.log"
    proc, port = launch_sumo(workdir, cfg, log_path, sumo_bin)

    route = load_ev_route()
    route_edges = set(route["edges"])

    samples: list[dict] = []
    stops: list[dict] = []
    cur_episode: dict | None = None
    first_seen = None
    arrival = None
    sim_time = 0.0
    occ_sum = 0.0
    occ_n = 0
    occ_peak = 0
    arrived_via_traci = False
    sim_end_hard = 0.0

    try:
        traci.init(port=port, host="127.0.0.1", numRetries=60)
        print(f"[run.py] {get_version_summary()}")
        check_version_alignment()
        sim_end_hard = traci.simulation.getEndTime()

        # corridor floor for concurrent-traffic occupancy (forward + reverse partners)
        all_edges = set(traci.edge.getIDList())
        edge_set = set(route_edges) | {rev for rev in ("-" + e for e in route_edges) if rev in all_edges}

        while True:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            if ctl is not None:
                ctl.step()

            present = EV_ID in traci.vehicle.getIDList()
            if present:
                if first_seen is None:
                    first_seen = sim_time
                speed = traci.vehicle.getSpeed(EV_ID)
                lane = traci.vehicle.getLaneID(EV_ID)
                pos = traci.vehicle.getLanePosition(EV_ID)
                dist = traci.vehicle.getDistance(EV_ID)
                try:
                    nxt = traci.vehicle.getNextTLS(EV_ID)
                    nxt = nxt[0] if nxt else None
                except traci.TraCIException:
                    nxt = None
                tls_id = tls_dist = tls_state = ""
                if nxt is not None:
                    tls_id, tls_dist, tls_state = nxt[0], float(nxt[2]), str(nxt[3])
                sample = {
                    "t": sim_time, "speed": speed, "lane": lane, "pos": pos,
                    "dist": dist, "tls": tls_id, "tls_dist": tls_dist, "tls_state": tls_state,
                }
                samples.append(sample)

                if speed < STOP_SPEED:
                    if cur_episode is None:
                        cur_episode = {"start": sim_time, "red": False}
                    if nxt is not None and tls_dist <= RED_DIST and tls_state not in GREEN_STATES:
                        cur_episode["red"] = True
                elif speed > RESUME_SPEED and cur_episode is not None:
                    cur_episode["end"] = sim_time
                    stops.append(cur_episode)
                    cur_episode = None

                if int(sim_time * 10) % 50 == 0:  # sample occupancy every ~5s
                    occ = sum(traci.edge.getLastStepVehicleNumber(e) for e in edge_set)
                    occ_sum += occ
                    occ_n += 1
                    occ_peak = max(occ_peak, occ)

            if EV_ID in traci.simulation.getArrivedIDList():
                arrival = sim_time
                arrived_via_traci = True
                break
            if not present and first_seen is not None:
                # EV vanished without an arrival record (e.g. teleport) -> bail
                arrival = sim_time
                arrived_via_traci = False
                break
            if sim_time >= sim_end_hard:
                break
    finally:
        if cur_episode is not None:
            cur_episode["end"] = sim_time
            stops.append(cur_episode)
        if ctl is not None:
            ctl.reset()
        traci.close()
        stop_proc(proc)

    metrics = summarize(density, seed, route, samples, stops, first_seen, arrival,
                        arrived_via_traci, occ_sum, occ_n, occ_peak, r=r)
    write_results(density, seed, results_dir, samples, metrics, tag=tag, summary=summary_name)
    if r is None:
        print_baseline_table(metrics)
    return metrics


def summarize(density, seed, route, samples, stops, first_seen, arrival,
              arrived_via_traci, occ_sum, occ_n, occ_peak, r: float | None = None):
    speeds = [s["speed"] for s in samples]
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    max_speed = max(speeds) if speeds else 0.0

    # theoretical free-flow time over the route edges using the net
    freeflow_s = None
    try:
        import sumolib
        net = sumolib.net.readNet(str(CORRIDOR_DIR / "corridor.net.xml"))
        tt = 0.0
        for eid in route["edges"]:
            e = net.getEdge(eid)
            tt += e.getLength() / min(e.getSpeed(), 33.33)
        freeflow_s = tt
    except Exception:
        freeflow_s = None

    completed = arrived_via_traci and arrival is not None and first_seen is not None
    travel_time = (arrival - first_seen) if completed else None

    metrics = {
        "density": density,
        "seed": seed,
        "depart_s": first_seen if first_seen is not None else "",
        "arrive_s": arrival if arrival is not None else "",
        "travel_time_s": travel_time if travel_time is not None else "",
        "freeflow_s": freeflow_s if freeflow_s is not None else "",
        "travel_ratio": (travel_time / freeflow_s) if (travel_time and freeflow_s) else "",
        "n_stops": len(stops),
        "n_red_stops": sum(1 for s in stops if s["red"]),
        "mean_speed_ms": mean_speed,
        "max_speed_ms": max_speed,
        "pct_vmax": (max_speed / 33.33 * 100) if max_speed else 0.0,
        "corridor_avg": (occ_sum / occ_n) if occ_n else 0.0,
        "corridor_peak": occ_peak,
        "completed": completed,
    }
    if r is not None:
        metrics["r"] = format_r(r)
    return metrics


def write_results(density, seed, results_dir, samples, metrics,
                  tag="R0", summary="R0_baseline.csv") -> None:
    profile = results_dir / f"{tag}_{density}_{seed}_profile.csv"
    with profile.open("w", newline="") as f:
        cols = ["t", "speed", "lane", "pos", "dist", "tls", "tls_dist", "tls_state"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in samples:
            w.writerow({k: s[k] for k in cols})

    baseline = results_dir / summary
    cols = list(metrics.keys())
    write_header = not baseline.is_file()
    with baseline.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow(metrics)


def print_baseline_table(m) -> None:
    console = Console()
    r_label = m.get("r")
    title = (
        f"R={r_label} preemption — {m['density'].upper()} / seed {m['seed']}"
        if r_label else
        f"R=0 baseline — {m['density'].upper()} / seed {m['seed']}"
    )
    table = Table(title=title + (" COMPLETED" if m["completed"] else " INCOMPLETE"))
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Travel time", f"{m['travel_time_s']:.1f}s" if m["travel_time_s"] != "" else "n/a")
    table.add_row("Free-flow estimate", f"{m['freeflow_s']:.1f}s" if m["freeflow_s"] != "" else "n/a")
    table.add_row("Travel / free-flow", f"{m['travel_ratio']:.2f}x" if m["travel_ratio"] != "" else "n/a")
    table.add_row("Stops", str(m["n_stops"]))
    table.add_row("Red stops (at signal)", str(m["n_red_stops"]))
    table.add_row("Mean speed", f"{m['mean_speed_ms']:.2f} m/s ({m['mean_speed_ms'] * 3.6:.0f} km/h)")
    table.add_row("Max speed", f"{m['max_speed_ms']:.2f} m/s ({m['max_speed_ms'] * 3.6:.0f} km/h)")
    table.add_row("Peak speed vs EV cap", f"{m['pct_vmax']:.0f}%")
    table.add_row("Corridor occupancy (avg/peak)", f"{m['corridor_avg']:.1f} / {m['corridor_peak']}")
    console.print(table)


def run_baseline(sumo_bin: str, density: str, seed: int, results_dir: Path):
    console = Console()
    console.print(f"[green]baseline[/green] density={density} seed={seed} link=R=0")
    collect_telemetry(sumo_bin, density, seed, results_dir)


def load_baseline_travel_time(results_dir: Path, density: str, seed: int):
    path = results_dir / "R0_baseline.csv"
    if not path.is_file():
        return None
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("density") == density and str(row.get("seed")) == str(seed):
                tt = row.get("travel_time_s")
                return float(tt) if tt not in ("", None) else None
    return None


def parse_r(text: str) -> float:
    if text in ("inf", "infinity", "∞"):
        return math.inf
    return float(text)


def print_preempt_table(m, baseline_tt) -> None:
    console = Console()
    r = m["r"]
    table = Table(title=f"R={r} preemption — {m['density'].upper()} / seed {m['seed']}"
                       + (" COMPLETED" if m["completed"] else " INCOMPLETE"))
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Travel time", f"{m['travel_time_s']:.1f}s" if m["travel_time_s"] != "" else "n/a")
    if baseline_tt is not None:
        saved = f"{baseline_tt - m['travel_time_s']:.1f}s" if m["travel_time_s"] != "" else "n/a"
        table.add_row("vs R=0 baseline (saved)", f"{saved} (base {baseline_tt:.1f}s)")
    if baseline_tt is not None and m["travel_time_s"] != "" and baseline_tt:
        table.add_row("Fraction of baseline", f"{m['travel_time_s'] / baseline_tt:.2f}x")
    table.add_row("Free-flow estimate", f"{m['freeflow_s']:.1f}s" if m["freeflow_s"] != "" else "n/a")
    table.add_row("Stops", str(m["n_stops"]))
    table.add_row("Red stops (at signal)", str(m["n_red_stops"]))
    table.add_row("Mean speed", f"{m['mean_speed_ms']:.2f} m/s ({m['mean_speed_ms'] * 3.6:.0f} km/h)")
    table.add_row("Max speed", f"{m['max_speed_ms']:.2f} m/s ({m['max_speed_ms'] * 3.6:.0f} km/h)")
    table.add_row("Corridor occupancy (avg/peak)", f"{m['corridor_avg']:.1f} / {m['corridor_peak']}")
    console.print(table)


def run_preempt(sumo_bin: str, density: str, seed: int, r: float, results_dir: Path):
    console = Console()
    console.print(f"[green]preempt[/green] density={density} seed={seed} R={format_r(r)}m")
    metrics = collect_telemetry(sumo_bin, density, seed, results_dir, r=r)
    if math.isinf(r) and metrics["n_red_stops"] != 0:
        raise AssertionError(
            f"R=inf demo failed: {metrics['n_red_stops']} red stops at signal "
            f"({density}/seed {seed}); expected 0 for an unbroken green corridor"
        )
    baseline_tt = load_baseline_travel_time(results_dir, density, seed)
    print_preempt_table(metrics, baseline_tt)
    return metrics


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="amroute SUMO runners (smoke + R=0 baseline + R preemption)")
    ap.add_argument("--sumo-bin", default=None, help="sumo binary (default: SUMO_BIN env, then sim/vendor/bin)")
    sub = ap.add_subparsers(dest="command")

    p_smoke = sub.add_parser("smoke", help="existing headless smoke test")
    p_smoke.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"sim steps (default {DEFAULT_STEPS})")
    p_smoke.add_argument("--cfg", default=DEFAULT_CFG, help=f"config in workdir (default {DEFAULT_CFG})")
    p_smoke.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="directory with sumo cfg")

    p_base = sub.add_parser("baseline", help="headless R=0 baseline with EV telemetry")
    p_base.add_argument("--density", choices=["low", "med", "high"], required=True)
    p_base.add_argument("--seed", type=int, default=1)
    p_base.add_argument("--results", type=Path, default=RESULTS_DIR, help="results dir")

    p_pre = sub.add_parser("preempt", help="signal-preemption run (knob R); --r inf = full corridor")
    p_pre.add_argument("--density", choices=["low", "med", "high"], required=True)
    p_pre.add_argument("--seed", type=int, default=1)
    p_pre.add_argument("--r", default="200", help="detection range R in meters (number or 'inf')")
    p_pre.add_argument("--results", type=Path, default=RESULTS_DIR, help="results dir")

    args = ap.parse_args(argv)
    sumo_bin = resolve_sumo_bin(args.sumo_bin or None)
    try:
        if args.command == "smoke":
            run_smoke(args.workdir, args.cfg, args.steps, sumo_bin)
        elif args.command == "baseline":
            run_baseline(sumo_bin, args.density, args.seed, args.results)
        elif args.command == "preempt":
            run_preempt(sumo_bin, args.density, args.seed, parse_r(args.r), args.results)
        else:
            ap.error("no subcommand; use 'smoke', 'baseline' or 'preempt'")
    except Exception as exc:
        print(f"[run.py] FAILED: {exc}", file=sys.stderr)
        return 1
    print("[run.py] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())