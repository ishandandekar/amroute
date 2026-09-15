"""Corridor crop + candidate pick for the amroute SUMO study.

Pipeline: pyosmium bbox-crop the Bombay extract -> .osm -> netconvert -> corridor.net.xml,
then auto-detect arterial corridors with >=4 signalized intersections, pick one with a human.

Commands:
  crop        bbox-crop => corridor.net.xml (+ net stats)
  validate    headless sumo run to prove the corridor net loads
  candidates  auto-detect 3 candidate corridors (arterial + >=4 signals)
  choose      record the picked corridor in SPEC.md and chosen.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import osmium
import sumolib
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = REPO_ROOT / "sim"
PBF = SIM_DIR / "Bombay.osm.pbf"
CORRIDOR_DIR = SIM_DIR / "corridor"
REGION_NET = SIM_DIR / "scratch" / "mumbai.net.xml"
OSM_PATH = CORRIDOR_DIR / "corridor.osm"
NET_PATH = CORRIDOR_DIR / "corridor.net.xml"
CFG_PATH = CORRIDOR_DIR / "corridor.sumocfg"
ROU_PATH = CORRIDOR_DIR / "corridor.rou.xml"
CHOSEN_PATH = CORRIDOR_DIR / "chosen.json"
CROP_PATH = CORRIDOR_DIR / "crop.json"
SPEC_PATH = SIM_DIR / "SPEC.md"

BBOX_HELP = "minlon,minlat,maxlon,maxlat"

_DRIVABLE = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential",
    "living_street", "service",
}

_ARTERIAL = {
    "highway.motorway", "highway.motorway_link",
    "highway.trunk", "highway.trunk_link",
    "highway.primary", "highway.primary_link",
    "highway.secondary", "highway.secondary_link",
    "highway.tertiary", "highway.tertiary_link",
}


class _NodeFilter(osmium.SimpleHandler):
    def __init__(self, minlon: float, minlat: float, maxlon: float, maxlat: float) -> None:
        super().__init__()
        self.minlon, self.minlat, self.maxlon, self.maxlat = minlon, minlat, maxlon, maxlat
        self.keep: Dict[int, Tuple[float, float, Dict[str, str]]] = {}

    def node(self, nd) -> None:
        loc = nd.location
        if not loc.valid():
            return
        lon, lat = loc.lon, loc.lat
        if self.minlon <= lon <= self.maxlon and self.minlat <= lat <= self.maxlat:
            self.keep[nd.id] = (lon, lat, {t.k: t.v for t in nd.tags})


class _WayFilter(osmium.SimpleHandler):
    def __init__(self, keep_nodes: Dict[int, Tuple[float, float, Dict[str, str]]]) -> None:
        super().__init__()
        self.keep_nodes = keep_nodes
        self.ways: List[Tuple[int, List[int], Dict[str, str]]] = []

    def way(self, w) -> None:
        tags = {t.k: t.v for t in w.tags}
        if tags.get("highway") not in _DRIVABLE:
            return
        refs = [n.ref for n in w.nodes]
        inside = [r for r in refs if r in self.keep_nodes]
        if len(inside) < 2:
            return
        self.ways.append((w.id, inside, tags))


def write_osm(path: Path, nodes: Dict[int, Tuple[float, float, Dict[str, str]]],
              ways: List[Tuple[int, List[int], Dict[str, str]]]) -> None:
    root = ET.Element("osm", version="0.6", generator="amroute corridor.py")
    for nid, (lon, lat, tags) in nodes.items():
        elem = ET.SubElement(root, "node", id=str(nid), lat=f"{lat:.7f}", lon=f"{lon:.7f}")
        for k, v in tags.items():
            ET.SubElement(elem, "tag", k=k, v=v)
    for wid, refs, tags in ways:
        way = ET.SubElement(root, "way", id=str(wid))
        for r in refs:
            ET.SubElement(way, "nd", ref=str(r))
        for k, v in tags.items():
            ET.SubElement(way, "tag", k=k, v=v)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def resolve_sumo_bin() -> str:
    env_bin = os.environ.get("SUMO_BIN")
    if env_bin:
        return env_bin
    wrapper = SIM_DIR / "vendor" / "bin" / "sumo"
    if wrapper.is_file():
        return str(wrapper)
    which = shutil.which("sumo")
    if which:
        return which
    raise FileNotFoundError("sumo binary not found; set SUMO_BIN or add sim/vendor/bin to PATH")


def resolve_netconvert_bin() -> str:
    env_bin = os.environ.get("NETCONVERT_BIN")
    if env_bin:
        return env_bin
    wrapper = SIM_DIR / "vendor" / "bin" / "netconvert"
    if wrapper.is_file():
        return str(wrapper)
    which = shutil.which("netconvert")
    if which:
        return which
    raise FileNotFoundError("netconvert binary not found; set NETCONVERT_BIN or add sim/vendor/bin to PATH")


def parse_bbox(text: str) -> Tuple[float, float, float, float]:
    parts = [float(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"bbox must be {BBOX_HELP}")
    return parts[0], parts[1], parts[2], parts[3]


def net_stats(net_path: Path) -> Dict[str, int]:
    tree = ET.parse(net_path)
    root = tree.getroot()
    edges = int(len(root.findall("edge")))
    internal = 0
    for e in root.findall("edge"):
        if "internal" in e.get("function", "") or (e.get("id") or "").startswith(":"):
            internal += 1
    nodes = len(root.findall("junction"))
    tls = len(root.findall("tlLogic"))
    return {"edges": edges, "internal": internal, "nodes": nodes, "tlLogic": tls}


def crop(bbox: Tuple[float, float, float, float], osm_out: Path, net_out: Path) -> None:
    console = Console()
    minlon, minlat, maxlon, maxlat = bbox
    osm_out.parent.mkdir(parents=True, exist_ok=True)
    nf = _NodeFilter(minlon, minlat, maxlon, maxlat)
    nf.apply_file(str(PBF))
    wf = _WayFilter(nf.keep)
    wf.apply_file(str(PBF))
    osm_out.parent.mkdir(parents=True, exist_ok=True)
    write_osm(osm_out, nf.keep, wf.ways)
    console.print(f"[green]crop[/green] bbox={bbox} -> {len(nf.keep)} nodes, {len(wf.ways)} ways -> {osm_out.name}")

    nc = resolve_netconvert_bin()
    cmd = [
        nc,
        "--osm-files", str(osm_out),
        "--output-file", str(net_out),
        "--osm.skip-duplicates-check",
        "--roundabouts.guess",
        "--ramps.guess",
        "--tls.guess-signals",
        "--output.street-names",
        "--osm.extra-attributes", "name",
    ]
    console.print(f"[green]netconvert[/green] {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"netconvert failed (rc={proc.returncode})")

    stats = net_stats(net_out)
    console.print(f"[green]net[/green] {net_out.name}: {stats['edges']} edges ({stats['internal']} internal), "
                  f"{stats['nodes']} nodes, {stats['tlLogic']} tlLogic")
    CORRIDOR_DIR.mkdir(parents=True, exist_ok=True)
    CROP_PATH.write_text(json.dumps({"bbox": list(bbox)}) + "\n")
    if stats["tlLogic"] < 4:
        console.print("[yellow]warning[/yellow] fewer than 4 traffic lights in the crop", style="yellow")


def ensure_scaffold() -> None:
    CORRIDOR_DIR.mkdir(parents=True, exist_ok=True)
    if not ROU_PATH.exists():
        ET.ElementTree(ET.Element("routes")).write(ROU_PATH, encoding="UTF-8", xml_declaration=True)
    if not CFG_PATH.exists():
        cfg = ET.Element("configuration")
        inp = ET.SubElement(cfg, "input")
        ET.SubElement(inp, "net-file", value="corridor.net.xml")
        ET.SubElement(inp, "route-files", value="corridor.rou.xml")
        time = ET.SubElement(cfg, "time")
        ET.SubElement(time, "begin", value="0")
        ET.SubElement(time, "end", value="100")
        proc = ET.SubElement(cfg, "processing")
        ET.SubElement(proc, "no-step-log", value="true")
        ET.SubElement(proc, "ignore-route-errors", value="true")
        ET.ElementTree(cfg).write(CFG_PATH, encoding="UTF-8", xml_declaration=True)


def validate(net_path: Path) -> None:
    if not net_path.is_file():
        raise FileNotFoundError(f"corridor net not found: {net_path}")
    ensure_scaffold()
    sumo_bin = resolve_sumo_bin()
    cmd = [sumo_bin, "-c", CFG_PATH.name, "--no-warnings"]
    console = Console()
    console.print(f"[green]validate[/green] {' '.join(cmd)} (cwd={CFG_PATH.parent})")
    proc = subprocess.run(cmd, cwd=CFG_PATH.parent, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"sumo validation failed (rc={proc.returncode})")
    console.print(f"[green]validate[/green] OK: {net_path.name} loaded for 100s headless")


class Corridor:
    def __init__(self, name_seq: List[str], node_path: List[str], edge_ids: List[str],
                 signals: int, length_m: float, crow_m: float, first: str, last: str,
                 speed_kph: float = 0.0, lanes: int = 0) -> None:
        self.name_seq = name_seq
        self.node_path = node_path
        self.edge_ids = edge_ids
        self.signals = signals
        self.length_m = length_m
        self.crow_m = crow_m
        self.first = first
        self.last = last
        self.speed_kph = speed_kph
        self.lanes = lanes

    @property
    def strace(self) -> str:
        return " → ".join(n.strip() for n in self.name_seq if n.strip()) or "(unnamed)"

    @property
    def straightness(self) -> float:
        return self.crow_m / self.length_m if self.length_m else 0.0


def load_arterial_net(net_path: Path):
    net = sumolib.net.readNet(str(net_path))
    edges = []
    for e in net.getEdges():
        eid = e.getID()
        if eid.startswith(":"):
            continue
        typ = (e.getType() or "").strip()
        if typ in _ARTERIAL:
            edges.append(e)
    adj: Dict[str, List[Tuple[str, object]]] = {}
    pair: Dict[Tuple[str, str], object] = {}
    for e in edges:
        f = e.getFromNode().getID()
        t = e.getToNode().getID()
        adj.setdefault(f, []).append((t, e))
        adj.setdefault(t, []).append((f, e))
        pair[(f, t)] = e
        pair[(t, f)] = e
    return net, edges, adj, pair


def _signal_ids(net) -> List[str]:
    tls_nodes = net.getTrafficLights()
    return [n.getID() for n in tls_nodes]


def _components(adj: Dict[str, List[Tuple[str, object]]]) -> List[List[str]]:
    seen = set()
    comps = []
    for start in adj:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb, _ in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _path_between(adj, pair, a: str, b: str) -> Optional[List[str]]:
    if a == b:
        return [a]
    parent = {a: None}
    stack = [a]
    while stack:
        node = stack.pop()
        for nb, _ in adj[node]:
            if nb not in parent:
                parent[nb] = node
                stack.append(nb)
                if nb == b:
                    path = [b]
                    cur = b
                    while parent[cur] is not None:
                        cur = parent[cur]
                        path.append(cur)
                    return list(reversed(path))
    return None


def _crow_distance(net, a: str, b: str) -> float:
    na = net.getNode(a)
    nb = net.getNode(b)
    x1, y1 = na.getCoord()
    x2, y2 = nb.getCoord()
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _named_graph(named_edges: List[object]):
    same = defaultdict(list)
    for e in named_edges:
        f = e.getFromNode().getID()
        t = e.getToNode().getID()
        same[f].append((t, e))
        same[t].append((f, e))
    return same


def _path_stats(path: List[str], pair) -> Tuple[List[str], float, int, float]:
    edge_ids = []
    length = 0.0
    lanes_list = []
    speed = 0.0
    for u, v in zip(path, path[1:]):
        e = pair[(u, v)]
        edge_ids.append(e.getID())
        length += e.getLength()
        lanes_list.append(len(e.getLanes()))
        speed = max(speed, e.getSpeed())
    lanes_sorted = sorted(lanes_list)
    lanes = lanes_sorted[len(lanes_sorted) // 2]
    return edge_ids, length, lanes, speed * 3.6


def candidates(net_path: Path, top: int = 3) -> List[Corridor]:
    net, edges, adj, pair = load_arterial_net(net_path)
    signals = set(_signal_ids(net))

    by_name: Dict[str, List[object]] = {}
    for e in edges:
        by_name.setdefault(e.getName().strip(), []).append(e)

    found: List[Corridor] = []
    for name, named_edges in by_name.items():
        if not name:
            continue
        same = _named_graph(named_edges)
        for comp in _components(same):
            endpoints = [n for n in comp if len(same[n]) <= 1]
            if len(endpoints) < 2:
                continue
            best_path = None
            for i, a in enumerate(endpoints):
                for b in endpoints[i + 1:]:
                    path = _path_between(same, None, a, b)
                    if path and (best_path is None or len(path) > len(best_path)):
                        best_path = path
            if not best_path or len(best_path) < 2:
                continue
            sig = sum(1 for n in best_path if n in signals)
            if sig < 4:
                continue
            edge_ids, length, lanes, speed_kph = _path_stats(best_path, pair)
            crow = _crow_distance(net, best_path[0], best_path[-1])
            found.append(Corridor([name], best_path, edge_ids, sig, length, crow,
                                  edge_ids[0], edge_ids[-1], speed_kph, lanes))

    found.sort(key=lambda c: (c.signals, c.length_m), reverse=True)

    selected: List[Corridor] = []
    used_names = set()
    for c in found:
        key = c.name_seq[0] if c.name_seq else ""
        if key in used_names:
            continue
        used_names.add(key)
        selected.append(c)
        if len(selected) >= top:
            break
    return selected


def print_candidates(cands: List[Corridor]) -> None:
    console = Console()
    table = Table(title="Candidate corridors (arterial + >=4 signalized intersections)")
    table.add_column("#")
    table.add_column("Streets")
    table.add_column("Signals")
    table.add_column("Length")
    table.add_column("Km/h")
    table.add_column("Lanes")
    table.add_column("Origin")
    table.add_column("Destination")
    for i, c in enumerate(cands, 1):
        table.add_row(
            str(i),
            c.strace,
            str(c.signals),
            f"{c.length_m / 1000:.2f} km",
            f"{c.speed_kph:.0f}",
            str(c.lanes),
            c.first,
            c.last,
        )
    console.print(table)


def summary_line(c: Corridor) -> str:
    return (f"{c.strace}: {c.signals} signals, {c.length_m / 1000:.2f} km, "
            f"{c.speed_kph:.0f} km/h, {c.lanes} lanes, "
            f"O={c.first} D={c.last}")


def record_chosen(bbox: Tuple[float, float, float, float], chosen: Corridor,
                  alternatives: List[Corridor]) -> None:
    CORRIDOR_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "bbox": list(bbox),
        "region_net": str(REGION_NET),
        "chosen": {
            "streets": chosen.name_seq,
            "node_path": chosen.node_path,
            "signals": chosen.signals,
            "length_m": round(chosen.length_m, 1),
            "speed_kph": round(chosen.speed_kph, 1),
            "lanes": chosen.lanes,
            "origin_edge": chosen.first,
            "destination_edge": chosen.last,
        },
        "alternatives": [
            {"streets": c.name_seq, "signals": c.signals, "length_m": round(c.length_m, 1),
             "speed_kph": round(c.speed_kph, 1), "lanes": c.lanes,
             "origin_edge": c.first, "destination_edge": c.last}
            for c in alternatives
        ],
    }
    CHOSEN_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    spec = SPEC_PATH.read_text()
    section = (
        "## Corridor (M1)\n\n"
        "Chosen study corridor (issue #2). Candidates were auto-detected from the Bombay region "
        f"net `{REGION_NET.name}`; the corridor net was cropped to ~5 km² around the pick.\n\n"
        f"- **Corridor:** {chosen.strace}\n"
        f"- **BBox (lon,lat):** {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n"
        f"- **Signalized intersections on the route:** {chosen.signals}\n"
        f"- **Length:** {chosen.length_m / 1000:.2f} km\n"
        f"- **Speed / lanes:** {chosen.speed_kph:.0f} km/h, {chosen.lanes} lanes (median)\n"
        f"- **Origin edge:** `{chosen.first}`\n"
        f"- **Destination edge:** `{chosen.last}`\n"
        f"- **Net:** `sim/corridor/corridor.net.xml`; regenerate with "
        f"`uv run python sim/corridor.py crop {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}`\n\n"
        "Alternatives (kept, auto-detected):\n\n"
        + "\n".join(f"{i}. {summary_line(c)}" for i, c in enumerate(alternatives, 1))
        + "\n\n"
    )
    if "## Corridor (M1)" in spec:
        spec = re.sub(r"## Corridor \(M1\).*?(?=\n## |\Z)", section.rstrip(), spec, flags=re.S)
    else:
        spec = spec.replace("## Milestones", section + "## Milestones")
    SPEC_PATH.write_text(spec)


def choose(index: int, region_net: Path, corridor_net: Path) -> None:
    region_cands = candidates(region_net, top=10)
    if index < 1 or index > len(region_cands):
        print_candidates(region_cands)
        raise SystemExit(f"choose index must be 1..{len(region_cands)}")
    chosen_region = region_cands[index - 1]
    alternatives = [c for i, c in enumerate(region_cands) if i != index - 1]
    region_name = chosen_region.name_seq[0]
    corridor_cands = candidates(corridor_net, top=20)
    chosen = None
    for c in corridor_cands:
        if c.name_seq[0] == region_name:
            chosen = c
            break
    if chosen is None:
        chosen = chosen_region
    if CROP_PATH.is_file():
        bbox = tuple(json.loads(CROP_PATH.read_text())["bbox"])
    else:
        bbox = (0.0, 0.0, 0.0, 0.0)
    record_chosen(bbox, chosen, alternatives)
    Console().print(f"[green]choose[/green] recorded candidate {index}: {summary_line(chosen)}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="amroute corridor crop + candidate pick")
    ap.add_argument("--net", type=Path, default=NET_PATH, help="corridor net file (default sim/corridor/corridor.net.xml)")
    sub = ap.add_subparsers(dest="command", required=True)

    p_crop = sub.add_parser("crop", help="bbox-crop Bombay pbf -> corridor.net.xml")
    p_crop.add_argument("--bbox", required=True, help=BBOX_HELP)
    p_crop.add_argument("--osm-out", type=Path, default=OSM_PATH)
    p_crop.add_argument("--net-out", type=Path, default=NET_PATH)

    sub.add_parser("validate", help="headless sumo run on the corridor net")

    p_cand = sub.add_parser("candidates", help="auto-detect candidate corridors")
    p_cand.add_argument("--top", type=int, default=3)

    p_choose = sub.add_parser("choose", help="record the picked corridor")
    p_choose.add_argument("index", type=int, help="index into the region-net candidate list")
    p_choose.add_argument("--region-net", type=Path, default=REGION_NET,
                          help="net the candidate table was printed from (default sim/scratch/mumbai.net.xml)")
    p_choose.add_argument("--corridor-net", type=Path, default=NET_PATH,
                          help="final cropped corridor net to record O/D edges from")
    args = ap.parse_args(argv)

    if args.command == "crop":
        crop(parse_bbox(args.bbox), args.osm_out, args.net_out)
    elif args.command == "validate":
        validate(args.net)
    elif args.command == "candidates":
        print_candidates(candidates(args.net, top=args.top))
    elif args.command == "choose":
        choose(args.index, args.region_net, args.corridor_net)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())