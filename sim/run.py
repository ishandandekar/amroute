"""SUMO runners: smoke test + baseline EV telemetry (R=0) + preemption (knob R).

smoke     launch corridor/smoke net headless, step, read state  (M0 verify)
baseline  headless R=0 run: background traffic + 1 fixed EV, record EV travel
          time, # stops (red at signal vs total), speed profile per density
preempt   R-preemption run: a signal preempts green when the EV is within R m;
          --r inf holds every signal from the moment it becomes next (ceiling).
          Lane-clearance shunt is ON by default for preempt (--no-shunt to
          disable): slow vehicles ahead of the EV inside the R zone are moved
          out of the EV's lane so a queued lane empties ahead of the ambulance.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import traci
from rich.console import Console
from rich.table import Table

from control import LaneClearanceController, PreemptionController, format_r
from scenario import build_many

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
                      r: float | None = None, shunt: bool = False,
                      legacy: bool = True):
    cfg = f"corridor.{density}.{seed}.sumocfg"
    workdir = CORRIDOR_DIR
    cfg_path = workdir / cfg
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}; run sim/scenario.py first")

    tag = "R0" if r is None else f"R{format_r(r)}"
    summary_name = "R0_baseline.csv" if r is None else "R_preempt.csv"
    route = load_ev_route()
    ctl = PreemptionController(r) if r is not None else None
    shunt_ctrl = None
    if r is not None and shunt:
        shunt_ctrl = LaneClearanceController(r, route["edges"],
                                             CORRIDOR_DIR / "corridor.net.xml")

    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{tag}_{density}_{seed}.run.log"
    proc, port = launch_sumo(workdir, cfg, log_path, sumo_bin)

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
    teleports = 0
    collisions = 0
    tele_id_set: set[str] = set()
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
            if shunt_ctrl is not None:
                shunt_ctrl.step()
            teleports += traci.simulation.getStartingTeleportNumber()
            tele_id_set.update(traci.simulation.getStartingTeleportIDList())
            try:
                collisions = max(collisions, len(traci.simulation.getCollisions()))
            except traci.TraCIException:
                pass

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

        if r is not None and (teleports or collisions):
            try:
                coll_log = list(traci.simulation.getCollisions())
            except traci.TraCIException:
                coll_log = []
            shunted = getattr(shunt_ctrl, "shunted_ids", set()) if shunt_ctrl else set()
            caused_by_shunt = any(
                (getattr(c, "collider", None) in shunted or getattr(c, "victim", None) in shunted)
                for c in coll_log
            ) or bool(tele_id_set & shunted)
            print(
                f"[run.py] collision/teleport warning: {teleports} teleports "
                f"{[v for v in tele_id_set if v not in shunted]} (shunt-caused={caused_by_shunt}); "
                f"{len(coll_log)} collisions {[(getattr(c,'time',None), getattr(c,'collider',None), getattr(c,'victim',None)) for c in coll_log]}",
                file=sys.stderr,
            )
    finally:
        if cur_episode is not None:
            cur_episode["end"] = sim_time
            stops.append(cur_episode)
        if ctl is not None:
            ctl.reset()
        if shunt_ctrl is not None:
            shunt_ctrl.reset()
        traci.close()
        stop_proc(proc)

    metrics = summarize(density, seed, route, samples, stops, first_seen, arrival,
                        arrived_via_traci, occ_sum, occ_n, occ_peak, r=r)
    metrics["teleports"] = teleports
    metrics["collisions"] = collisions
    if shunt_ctrl is not None:
        metrics.update({
            "n_shunt_req": shunt_ctrl.n_req,
            "n_shunt_done": shunt_ctrl.n_done,
            "n_shunt_failed": shunt_ctrl.n_failed,
        })
    else:
        metrics.update({"n_shunt_req": 0, "n_shunt_done": 0, "n_shunt_failed": 0})
    if r is not None and math.isinf(r):
        metrics["red_stop_violation"] = 1 if metrics["n_red_stops"] else 0
    write_results(density, seed, results_dir, samples, metrics,
                  tag=tag, summary=summary_name, legacy=legacy)
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
                  tag="R0", summary="R0_baseline.csv", legacy=True) -> None:
    profile = results_dir / f"{tag}_{density}_{seed}_profile.csv"
    with profile.open("w", newline="") as f:
        cols = ["t", "speed", "lane", "pos", "dist", "tls", "tls_dist", "tls_state"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in samples:
            w.writerow({k: s[k] for k in cols})

    if not legacy:
        return

    path = results_dir / summary
    fieldnames = list(metrics.keys())
    if path.exists() and path.stat().st_size > 0:
        with path.open() as f:
            existing = next(csv.reader(f))
        missing = [c for c in fieldnames if c not in existing]
        if missing:
            # extend the header in place; old rows keep empty cells for the new fields
            with path.open() as f:
                rows = list(csv.DictReader(f))
            fieldnames = existing + missing
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in rows:
                    w.writerow({k: row.get(k, "") for k in fieldnames})
        else:
            fieldnames = existing
    else:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow({k: metrics.get(k, "") for k in fieldnames})


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


def parse_r_list(text: str) -> list[float]:
    return [parse_r(t.strip()) for t in text.split(",") if t.strip()]


def parse_seed_list(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


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
    if "n_shunt_req" in m:
        table.add_row("Shunts (req/done/fail)",
                      f"{m['n_shunt_req']} / {m['n_shunt_done']} / {m['n_shunt_failed']}")
    if "teleports" in m:
        table.add_row("Teleports / collisions", f"{m['teleports']} / {m['collisions']}")
    table.add_row("Free-flow estimate", f"{m['freeflow_s']:.1f}s" if m["freeflow_s"] != "" else "n/a")
    table.add_row("Stops", str(m["n_stops"]))
    table.add_row("Red stops (at signal)", str(m["n_red_stops"]))
    table.add_row("Mean speed", f"{m['mean_speed_ms']:.2f} m/s ({m['mean_speed_ms'] * 3.6:.0f} km/h)")
    table.add_row("Max speed", f"{m['max_speed_ms']:.2f} m/s ({m['max_speed_ms'] * 3.6:.0f} km/h)")
    table.add_row("Corridor occupancy (avg/peak)", f"{m['corridor_avg']:.1f} / {m['corridor_peak']}")
    console.print(table)


def run_preempt(sumo_bin: str, density: str, seed: int, r: float, results_dir: Path,
                shunt: bool = True):
    console = Console()
    mode = "+shunt" if shunt else "preempt-only"
    console.print(f"[green]{mode}[/green] density={density} seed={seed} R={format_r(r)}m")
    metrics = collect_telemetry(sumo_bin, density, seed, results_dir, r=r, shunt=shunt)
    if math.isinf(r) and metrics["n_red_stops"] != 0:
        raise AssertionError(
            f"R=inf demo failed: {metrics['n_red_stops']} red stops at signal "
            f"({density}/seed {seed}); expected 0 for an unbroken green corridor"
        )
    baseline_tt = load_baseline_travel_time(results_dir, density, seed)
    print_preempt_table(metrics, baseline_tt)
    return metrics


# --------------------------------------------------------------------------
# Batch experiment (M4, issue #6): R x congestion x paired seeds -> results.csv
# --------------------------------------------------------------------------

# R knob values of the experiment matrix, in display order.
R_MATRIX: list[float] = [0.0, 50.0, 200.0, 500.0, math.inf]
DENSITY_MATRIX: list[str] = ["low", "med", "high"]
BATCH_COLUMNS = [
    "r", "density", "seed", "depart_s", "arrive_s", "travel_time_s", "freeflow_s",
    "travel_ratio", "n_stops", "n_red_stops", "mean_speed_ms", "max_speed_ms",
    "pct_vmax", "corridor_avg", "corridor_peak", "completed",
    "teleports", "collisions", "n_shunt_req", "n_shunt_done", "n_shunt_failed",
    "red_stop_violation",
]


def run_cell(sumo_bin: str, density: str, seed: int, r: float,
             results_dir: Path) -> dict:
    """Run one (R, congestion, seed) cell; returns its metrics dict (raises on error).

    R=0 cells run the unified telemetry with no preemption and no shunt, which
    reproduces the R=0 baseline exactly while keeping a uniform row schema.
    """
    shunt = r is not None and r > 0
    metrics = collect_telemetry(sumo_bin, density, seed, results_dir,
                                r=r, shunt=shunt, legacy=False)
    return metrics


def append_batch_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BATCH_COLUMNS)
        if fresh:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in BATCH_COLUMNS})


def batch_cells_done(path: Path) -> set[tuple[str, str, str]]:
    """Cells (r, density, seed) already recorded as completed in results.csv."""
    done: set[tuple[str, str, str]] = set()
    if not path.is_file():
        return done
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("completed") == "True" and row.get("r") not in (None, ""):
                done.add((row["r"], row["density"], row["seed"]))
    return done


def run_batch(sumo_bin: str, densities: list[str], r_values: list[float],
              seeds: list[int], results_dir: Path, parallel: int = 4,
              resume: bool = True, force_scenarios: bool = False) -> int:
    results_path = results_dir / "results.csv"
    console = Console()

    built = build_many(densities, seeds, force=force_scenarios)
    if built:
        console.print(f"[green]batch[/green] built {len(built)} missing scenario configs: "
                      + ", ".join(f"{d}/{s}" for d, s in built))
    else:
        console.print("[green]batch[/green] all scenario configs already present")

    cells = [(r, d, s) for r in r_values for d in densities for s in seeds]
    if resume:
        done = batch_cells_done(results_path)
        pending = [c for c in cells if (format_r(c[0]), c[1], str(c[2])) not in done]
        console.print(f"[green]batch[/green] matrix={len(cells)} cells, "
                      f"{len(pending)} pending ({len(done)} already completed; resume=on)")
        cells = pending
    else:
        console.print(f"[green]batch[/green] matrix={len(cells)} cells (resume=off)")

    if not cells:
        console.print("[green]batch[/green] nothing to run")
        return 0

    failures: list[tuple] = []
    red_violations = 0
    completed_failed = 0
    console.print(f"[green]batch[/green] running {len(cells)} cells "
                  f"(parallel={parallel}) -> {results_path}")

    args = [(sumo_bin, d, s, r, results_dir) for r, d, s in cells]
    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(run_cell, *a): a for a in args}
            for fut in as_completed(futures):
                _sumo_bin, d, s, r, _resdir = futures[fut]
                try:
                    metrics = fut.result()
                except Exception as exc:
                    failures.append((r, d, s, repr(exc)))
                    print(f"[run.py] FAILED cell R{format_r(r)} {d}/{s}: {exc}",
                          file=sys.stderr)
                    continue
                if not metrics.get("completed"):
                    completed_failed += 1
                if r is not None and math.isinf(r) and metrics.get("n_red_stops"):
                    red_violations += 1
                append_batch_row(results_path, metrics)
    else:
        for r, d, s in cells:
            try:
                metrics = run_cell(sumo_bin, d, s, r, results_dir)
            except Exception as exc:
                failures.append((r, d, s, repr(exc)))
                print(f"[run.py] FAILED cell R{format_r(r)} {d}/{s}: {exc}",
                      file=sys.stderr)
                continue
            if not metrics.get("completed"):
                completed_failed += 1
            if r is not None and math.isinf(r) and metrics.get("n_red_stops"):
                red_violations += 1
            append_batch_row(results_path, metrics)

    console.print("[green]batch[/green] done: "
                  f"{len(cells) - len(failures)} rows written, "
                  f"{len(failures)} failed, {completed_failed} incomplete arrivals, "
                  f"{red_violations} R=inf red-stop violations")
    for r, d, s, err in failures:
        console.print(f"  [red]FAILED[/red] R{format_r(r)} {d}/{s}: {err}")
    return 1 if (failures or red_violations) else 0


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
    p_pre.add_argument("--no-shunt", action="store_false", dest="shunt", default=True,
                       help="disable the lane-clearance shunt (default: enabled)")
    p_pre.add_argument("--results", type=Path, default=RESULTS_DIR, help="results dir")

    p_batch = sub.add_parser(
        "batch", help="run the R x congestion x seed experiment matrix -> results.csv")
    p_batch.add_argument("--density", default=",".join(DENSITY_MATRIX),
                         help=f"comma-separated densities (default: {','.join(DENSITY_MATRIX)})")
    p_batch.add_argument("--r", default=",".join(format_r(v) for v in R_MATRIX),
                         help=f"comma-separated R meters or 'inf' (default: {','.join(format_r(v) for v in R_MATRIX)})")
    p_batch.add_argument("--seeds", default="1-10",
                         help="seeds, e.g. '1', '1,2,3' or '1-10' (default: 1-10)")
    p_batch.add_argument("--parallel", type=int, default=4,
                         help="concurrent sumo processes (default 4; 1 = sequential)")
    p_batch.add_argument("--no-resume", action="store_false", dest="resume", default=True,
                         help="rerun cells already present in results.csv (default: skip them)")
    p_batch.add_argument("--force-scenarios", action="store_true",
                         help="rebuild (density, seed) scenarios even if cached")
    p_batch.add_argument("--results", type=Path, default=RESULTS_DIR, help="results dir")

    args = ap.parse_args(argv)
    sumo_bin = resolve_sumo_bin(args.sumo_bin or None)
    try:
        if args.command == "smoke":
            run_smoke(args.workdir, args.cfg, args.steps, sumo_bin)
        elif args.command == "baseline":
            run_baseline(sumo_bin, args.density, args.seed, args.results)
        elif args.command == "preempt":
            run_preempt(sumo_bin, args.density, args.seed, parse_r(args.r),
                        args.results, shunt=args.shunt)
        elif args.command == "batch":
            return run_batch(
                sumo_bin,
                [d.strip() for d in args.density.split(",") if d.strip()],
                parse_r_list(args.r),
                parse_seed_list(args.seeds),
                args.results,
                parallel=max(1, args.parallel),
                resume=args.resume,
                force_scenarios=args.force_scenarios,
            )
        else:
            ap.error("no subcommand; use 'smoke', 'baseline', 'preempt' or 'batch'")
    except Exception as exc:
        print(f"[run.py] FAILED: {exc}", file=sys.stderr)
        return 1
    print("[run.py] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())