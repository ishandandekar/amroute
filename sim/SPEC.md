# Am-Rout Green Corridor Simulation (SUMO) — Spec

Status: draft — converged design for the emergency-vehicle traffic simulation study.

## Research question

**How much ambulance response time does early siren detection buy?**

- **(A)** Quantify the payoff of early detection: given a detection horizon, how much sooner does
  the ambulance reach the incident vs. no intervention?
- **(C)** Prove that SUMO can model the dynamic at all: a green corridor (TLC preemption + lane
  clearance) under Mumbai-style congestion.

The simulation is an *offline scenario study*. The live detector (`sireNN` audio + YOLO vision,
fused in `detection.py`) is the motivation, not a runtime dependency. In principle the flow is
`DETECTION -> GREEN-CORRIDOR`; wiring the real detector in is a later phase, not part of this sim.

## Design decisions

| Branch | Decision |
| --- | --- |
| Coupling | Offline scenario generator. Live-detector wiring deferred to a later phase. |
| Scope | Crop the full Bombay net to a ~2–5 km² corridor, auto-picked from the network. |
| Mechanic | TLC preemption **plus** TraCI lane-clearance (shunt blocking vehicles out of the corridor ahead of the EV). |
| Trigger | Single knob **R** = detection range (m). Preempt a signal only when the EV is within R. R=∞ = full pre-knowledge ceiling. |
| Background traffic | Synthetic flow-based demand (`randomTrips`/`flowrouter`), scaled by a density knob (low / med / high). |
| EV | One ambulance per run, fixed pre-computed route, `vClass="emergency"`, constant speed profile. No adaptive routing (later phase). |
| Experiment | R × congestion matrix, 10 paired traffic seeds. One ambulance per run. |
| Metrics | EV travel time (median + IQR, vs. no-intervention baseline), # stops at red, % of Max Speed achieved, % of theoretical ceiling captured. |
| Tooling | `micromamba`/conda env for SUMO binaries; `traci` pip package in the project's `uv` venv. |

## Experiment matrix

R ∈ {0 m (no preemption), 50, 200, 500 m, ∞} × congestion ∈ {low, med, high} × 10 seeds = **150 runs**.

- Same 10 seeds reused across all cells → paired comparisons (early detection saves X s).
- R=0 is the no-intervention baseline. R=∞ is the pre-knowledge ceiling. The headline result is
  "travel time vs. R" per congestion level, plus "seconds saved" and "% of ceiling captured."

## Trigger model (knob R)

A signal on the EV's route preempts to green only when the EV is within **R meters** of it
(distances from the corridor the EV traverses). Sweeping R degrades the ceiling (R=∞) toward the
real detector's finite audible footprint (≈<200 m in a noisy city).

## Lane clearance (mechanic)

When the EV enters the preemption zone, a TraCI loop moves the blocking vehicles ahead of it out
of the lane (shoulder / adjacent lane where possible), approximating "cars pull over." This is
required for the EV to actually benefit — preemption alone doesn't help if the lane stays queued.

Open question: how the shunt behaves on a congested multi-lane arterial (could block crossing
edges, cascade into surrounding cells). This is where (C) gets proven or falsified — test early
in a GUI run before scaling.

## Directory layout

```
sim/
  Bombay.osm.pbf        # source extract (25 MB)
  convert_pbf.py        # pbf -> .osm helper (reuses pyosmium); use upstream .osm directly
  Bombay.net.xml        # full-city network (1 GB, regenerable; to be cropped)
  SPEC.md               # this file
  corridor.py           # bbox-crop (pyosmium), netconvert, auto-pick corridor candidate
  scenario.py           # background flows (density knob), EV route, .sumo.cfg
  traci_loop.py         # green-corridor control: preemption trigger (R) + lane-clearance
  run.py                # batch runner over the matrix x seeds (headless)
  analyze.py            # aggregate runs -> results.csv, tables, plots
  results/              # per-run logs, results, plots
```

## Tooling setup (M0)

1. `micromamba create -n sumo -c conda-forge sumo` (native binaries, no AUR system churn).
2. `uv add traci` in this workspace (bundles `sumolib`).
3. Keep pip `traci` version close to the conda `sumo` binary version (TraCI protocol drifts;
   a version mismatch fails on connect).
4. Smoke test: `traci.connect` a minimal `sumo --remote-port` session.

## Milestones

0. **M0 — Tooling:** conda SUMO env + `uv add traci`, TraCI connect + version match.
1. **M1 — Corridor:** bbox-crop Bombay `.pbf` -> netconvert -> `.net.xml`; auto-pick 3 candidate
   corridors (arterial + >=4 signalized intersections), pick one with the user.
2. **M2 — Baseline runs:** background flows + 1 fixed EV, R=0; verify travel-time measurement.
3. **M3 — Green corridor:** preemption trigger (R) + lane clearance; verify the EV never stops
   mid-corridor and TLCs hold green. Risk item: shunt behavior on a 4-lane arterial.
4. **M4 — Batch run:** 150-run matrix with paired seeds.
5. **M5 — Analysis:** travel time vs R per congestion, seconds saved, % of ceiling, plots.

## Risks / unknowns

- pip `traci` <-> conda `sumo` version drift.
- Lane-clearance shuttling behavior on a congested arterial (M3).
- Whole-city `Bombay.net.xml` is 1 GB and slow: always operate on the cropped corridor net
  after M1.
- Detection latencies of the real `sireNN` model are not in the loop for A/C; they enter only in
  the later live-detector wiring phase.