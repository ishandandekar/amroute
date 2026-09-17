"""Scenario generator for the amroute baseline (M2, ticket #3).

Builds one headless scenario per congestion level:
  - background flows: synthetic random O/D trips via SUMO's randomTrips.py ->
    duarouter, scaled by a density knob (emission period: low/med/high);
  - a single emergency EV on the fixed pre-computed LBS Marg route (from
    sim/corridor/ev.route.json, derived from chosen.json node_path);

Writes per-density background routes, the EV route file and a matching .sumocfg
into sim/corridor/. The headless telemetry run is sim/run.py baseline.

Commands:
  scenario --density low --seed 1        build one scenario (period from knob)
  scenario --density low --seed 1 --period 5      override the density period
  scenario --all --seed 1                build all three congestion levels
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sumolib
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = REPO_ROOT / "sim"
CORRIDOR_DIR = SIM_DIR / "corridor"
RESULTS_DIR = SIM_DIR / "results"
NET_PATH = CORRIDOR_DIR / "corridor.net.xml"
CHOSEN_PATH = CORRIDOR_DIR / "chosen.json"
EV_ROUTE_PATH = CORRIDOR_DIR / "ev.route.json"
VENDOR_BIN = SIM_DIR / "vendor" / "bin"

DENSITY_PERIOD: Dict[str, float] = {"low": 20.0, "med": 3.0, "high": 1.9}
DENSITY_HELP = "low | med | high"
DEFAULT_BEGIN = 0.0
DEFAULT_END = 1500.0
DEFAULT_EV_DEPART = 120.0
STEP_LENGTH = 0.5
FRINGE_FACTOR = 10

EV_MAX_SPEED = 33.33  # 120 km/h, matches corridor free-flow speed cap (m/s)

_DRIVABLE = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential",
    "living_street", "service",
}


def resolve_tools() -> Tuple[Path, Path]:
    """Locate randomTrips.py (SUMO tools) and a runnable duarouter."""
    candidates = [
        os.environ.get("SUMO_TOOLS"),
        "/var/lib/flatpak/app/org.eclipse.sumo/x86_64/stable/active/files/share/sumo/tools",
        "/var/lib/flatpak/app/org.eclipse.sumo/x86_64/" + "stable/active/files/share/sumo/tools",
    ]
    tools_dir = None
    for c in candidates:
        if c and Path(c).joinpath("randomTrips.py").is_file():
            tools_dir = Path(c)
            break
    if tools_dir is None:
        raise FileNotFoundError("randomTrips.py not found; set SUMO_TOOLS to the sumo tools dir")

    duarouter = VENDOR_BIN / "duarouter"
    if duarouter.is_file():
        return tools_dir / "randomTrips.py", duarouter
    which = shutil.which("duarouter")
    if which:
        return tools_dir / "randomTrips.py", Path(which)
    raise FileNotFoundError("duarouter not found; add sim/vendor/bin/duarouter or set PATH")


def load_net() -> sumolib.net.Net:
    if not NET_PATH.is_file():
        raise FileNotFoundError(f"corridor net not found: {NET_PATH}")
    return sumolib.net.readNet(str(NET_PATH))


def corridor_edge_set(net) -> set:
    chosen = json.loads(CHOSEN_PATH.read_text())["chosen"]
    nodes = chosen["node_path"]
    edges = set()
    for a, b in zip(nodes, nodes[1:]):
        match = [e for e in net.getEdges()
                 if not e.getID().startswith(":")
                 and e.getFromNode().getID() == a and e.getToNode().getID() == b]
        if not match:
            raise ValueError(f"no edge between nodes {a} -> {b} in corridor net")
        edges.add(match[0].getID())
    return edges


def build_ev_route(cache: bool = True) -> Tuple[List[str], List[str]]:
    """Rebuild the EV edge sequence from chosen.json node_path (cache on disk)."""
    if EV_ROUTE_PATH.is_file() and cache:
        payload = json.loads(EV_ROUTE_PATH.read_text())
        return payload["edges"], payload["edge_str"].split()

    net = load_net()
    chosen = json.loads(CHOSEN_PATH.read_text())["chosen"]
    nodes = chosen["node_path"]
    edges: List[str] = []
    for a, b in zip(nodes, nodes[1:]):
        match = [e for e in net.getEdges()
                 if not e.getID().startswith(":")
                 and e.getFromNode().getID() == a and e.getToNode().getID() == b]
        if not match:
            raise ValueError(f"no edge between nodes {a} -> {b} in corridor net")
        edges.append(match[0].getID())

    if edges[0] != chosen["origin_edge"] or edges[-1] != chosen["destination_edge"]:
        raise ValueError("resolved EV route endpoints do not match chosen origin/destination")

    payload = {
        "origin_node": nodes[0],
        "destination_node": nodes[-1],
        "origin_edge": edges[0],
        "destination_edge": edges[-1],
        "n_edges": len(edges),
        "edges": edges,
        "edge_str": " ".join(edges),
    }
    if cache:
        EV_ROUTE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return edges, payload["edge_str"].split()


def tweak_background_rou(rou_path: Path) -> int:
    """Rewrite generated bg routes: define vType 'car' and assign it to vehicles."""
    tree = ET.parse(rou_path)
    root = tree.getroot()
    vtype = ET.Element("vType", {"id": "car", "vClass": "passenger", "maxSpeed": "25",
                                 "accel": "2.6", "decel": "4.5", "length": "5",
                                 "sigma": "0.5", "speedFactor": "1.0"})
    root.insert(0, vtype)
    n = 0
    for veh in root.findall("vehicle"):
        veh.set("type", "car")
        n += 1
    ET.ElementTree(root).write(rou_path, encoding="UTF-8", xml_declaration=True)
    return n


def gen_background(density: str, seed: int, period: float,
                   begin: float, end: float, quiet: bool = False) -> Path:
    """Run randomTrips.py -> duarouter for a density level; return bg route path."""
    randomtrips, duarouter = resolve_tools()
    trips = CORRIDOR_DIR / f"bg.{density}.{seed}.trips.xml"
    routes = CORRIDOR_DIR / f"bg.{density}.{seed}.rou.xml"

    env = dict(os.environ)
    env["DUAROUTER_BINARY"] = str(duarouter)
    if "SUMO_HOME" not in env:
        env["SUMO_HOME"] = str(randomtrips.parent)

    cmd = [
        sys.executable, str(randomtrips),
        "-n", NET_PATH.name,
        "-o", trips.name,
        "-r", routes.name,
        "--vehicle-class", "passenger",
        "--fringe-factor", str(FRINGE_FACTOR),
        "--seed", str(seed),
        "-b", str(begin),
        "-e", str(end),
        "-p", str(period),
        "--trip-attributes", 'departLane="best"',
        "--remove-loops",
    ]
    if not quiet:
        Console().print(f"[green]randomTrips[/green] density={density} seed={seed} period={period}s "
                        f"(cwd={CORRIDOR_DIR.name})")
    proc = subprocess.run(cmd, cwd=CORRIDOR_DIR, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.stdout and not quiet:
        print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"randomTrips failed (rc={proc.returncode})")
    trips.unlink(missing_ok=True)

    n = tweak_background_rou(routes)
    if n == 0:
        raise SystemExit(f"randomTrips produced no background vehicles for {density}")
    return routes


def write_ev_rou(edges: List[str], depart: float) -> Path:
    """Write the fixed EV into sim/corridor/ev.rou.xml."""
    path = CORRIDOR_DIR / "ev.rou.xml"
    line_edge_str = " ".join(edges)
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<routes>\n"
        '  <vType id="ev" vClass="emergency" maxSpeed="33.33" accel="4.0" '
        'decel="4.5" length="5" sigma="0.2" speedFactor="1.0"/>\n'
        f'  <route id="ev_route" edges="{line_edge_str}"/>\n'
        f'  <vehicle id="ev" type="ev" route="ev_route" depart="{depart}" '
        'departLane="best" departSpeed="max"/>\n'
        "</routes>\n"
    )
    path.write_text(content)
    return path


def write_cfg(density: str, seed: int, bg_rou: Path, ev_rou: Path,
              begin: float, end: float) -> Path:
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", value="corridor.net.xml")
    ET.SubElement(inp, "route-files", value=f"{bg_rou.name},{ev_rou.name}")
    time = ET.SubElement(cfg, "time")
    ET.SubElement(time, "begin", value=f"{begin:g}")
    ET.SubElement(time, "end", value=f"{end:g}")
    ET.SubElement(time, "step-length", value=f"{STEP_LENGTH:g}")
    proc = ET.SubElement(cfg, "processing")
    ET.SubElement(proc, "no-step-log", value="true")
    ET.SubElement(proc, "ignore-route-errors", value="true")
    ET.SubElement(proc, "time-to-teleport", value="3600")
    path = CORRIDOR_DIR / f"corridor.{density}.{seed}.sumocfg"
    ET.ElementTree(cfg).write(path, encoding="UTF-8", xml_declaration=True)
    return path


def summary(density: str, seed: int, period: float, bg_rou: Path, ev_edges: List[str],
            begin: float, end: float, ev_depart: float) -> None:
    n_vehs = sum(1 for _ in ET.parse(bg_rou).getroot().findall("vehicle"))
    chosen = json.loads(CHOSEN_PATH.read_text())["chosen"]
    table = Table(title=f"Scenario: {density.upper()} / seed {seed} (R=0 baseline)")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Density period", f"{period:g}s")
    table.add_row("Background vehicles", str(n_vehs))
    table.add_row("EV route edges", str(len(ev_edges)))
    table.add_row("Origin edge", chosen["origin_edge"])
    table.add_row("Destination edge", chosen["destination_edge"])
    table.add_row("Sim window", f"{begin:g}s .. {end:g}s")
    table.add_row("EV depart", f"{ev_depart:g}s")
    Console().print(table)
    Console().print(f"[green]scenario[/green] wrote {bg_rou.name} + ev.rou.xml + "
                    f"corridor.{density}.{seed}.sumocfg\n")


def build(density: str, seed: int, period: Optional[float],
          begin: float, end: float, ev_depart: float, quiet: bool = False) -> None:
    if density not in DENSITY_PERIOD:
        raise SystemExit(f"density must be one of {DENSITY_HELP}")
    eff_period = DENSITY_PERIOD[density] if period is None else period

    CORRIDOR_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ev_edges, _ = build_ev_route()
    bg_rou = gen_background(density, seed, eff_period, begin, end, quiet=quiet)
    ev_rou = write_ev_rou(ev_edges, ev_depart)
    cfg = write_cfg(density, seed, bg_rou, ev_rou, begin, end)
    if not quiet:
        summary(density, seed, eff_period, bg_rou, ev_edges, begin, end, ev_depart)
        Console().print(f"[green]scenario[/green] run with: "
                        f"{sys.executable} sim/run.py baseline --density {density} --seed {seed}")


def build_many(densities: List[str], seeds: List[int], force: bool = False,
               begin: float = DEFAULT_BEGIN, end: float = DEFAULT_END,
               ev_depart: float = DEFAULT_EV_DEPART) -> List[Tuple[str, int]]:
    """Build all (density, seed) scenario configs, skipping existing pairs.

    A pair is considered built when both its .sumocfg and its background route
    file exist; randomTrips is deterministic in its seed, so a cached file is
    equivalent to a rebuild. Returns the list of (density, seed) actually built.
    """
    built: List[Tuple[str, int]] = []
    for density in densities:
        for seed in seeds:
            cfg = CORRIDOR_DIR / f"corridor.{density}.{seed}.sumocfg"
            rou = CORRIDOR_DIR / f"bg.{density}.{seed}.rou.xml"
            if not force and cfg.is_file() and cfg.stat().st_size > 0 \
                    and rou.is_file() and rou.stat().st_size > 0:
                continue
            build(density, seed, None, begin, end, ev_depart, quiet=True)
            built.append((density, seed))
    return built


def parse_seed_list(text: str) -> List[int]:
    """Parse '1-10' or '1,2,3' (or a bare int) into a list of seeds."""
    seeds: List[int] = []
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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="amroute baseline scenario generator (R=0)")
    ap.add_argument("--density", choices=["low", "med", "high"], help=f"congestion level ({DENSITY_HELP})")
    ap.add_argument("--seed", type=int, default=1, help="random seed (repeatable)")
    ap.add_argument("--period", type=float, default=None, help="override background emission period (s)")
    ap.add_argument("--begin", type=float, default=DEFAULT_BEGIN, help=f"sim begin (default {DEFAULT_BEGIN:g})")
    ap.add_argument("--end", type=float, default=DEFAULT_END, help=f"sim end (default {DEFAULT_END:g})")
    ap.add_argument("--ev-depart", type=float, default=DEFAULT_EV_DEPART,
                    help=f"EV departure time (default {DEFAULT_EV_DEPART:g})")
    ap.add_argument("--all", action="store_true", help="build all three density levels")
    ap.add_argument("--seeds", default=None,
                    help="seeds to build, e.g. '1', '1,2,3' or '1-10' (default: --seed)")
    args = ap.parse_args(argv)
    if args.all:
        seeds = parse_seed_list(args.seeds) if args.seeds else [args.seed]
        for d in ["low", "med", "high"]:
            for s in seeds:
                build(d, s, None, args.begin, args.end, args.ev_depart, quiet=False)
        return 0
    if not args.density:
        ap.error("--density is required unless --all is used")
    if args.seeds:
        for s in parse_seed_list(args.seeds):
            build(args.density, s, args.period, args.begin, args.end, args.ev_depart,
                  quiet=False)
        return 0
    build(args.density, args.seed, args.period, args.begin, args.end, args.ev_depart,
          quiet=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())