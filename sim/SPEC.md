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

Implemented as `LaneClearanceController` in `sim/control.py` (the green-corridor control
lives in `control.py`, not the `traci_loop.py` sketched in the original layout), composed
with the preemption trigger and active for every R>0 run (shunt is ON by default for
`run.py preempt`; `--no-shunt` disables it).

While the EV is on its route, a control step (0.5 s) finds the slow vehicles — slower than
a cutoff, on the lane the EV currently drives, ahead of it along the route, within R meters
of route distance — and moves them to the adjacent same-direction lane:

- preferred: a single safe (urgent/cooperative) `changeLane` into the adjacent lane;
- fallback after a short grace: a hard `moveTo` into a *free slot* of the adjacent lane
  (never an occupied spot), with the shunted car held at v=0 for a few seconds like a
  pulled-over vehicle, then released — so the queued lane physically empties ahead of the
  ambulance without rear-ending the adjacent lane's traffic.

Scoping: the target set is built only from the EV's route edges, only ahead of the EV,
and only within R — nothing off-route, on the opposite carriageway, or behind the EV is
ever touched, which preserves the "does not clear the whole net" property. Because the
drained lane is the EV's *current* lane and the corridor has exactly 2 facing lanes
everywhere on the route, the mechanic is that cars pull over onto the lane the ambulance
is not using; when the EV changes lanes (e.g. into a turn pocket) the drained lane moves
with it. Telemetry: `n_shunt_req` / `n_shunt_done` / `n_shunt_failed`, plus run-level
`teleports` / `collisions` counters used as the repeat-run robustness gate (target: zero
shunt-attributable teleports/collisions).

Residuals: a short junction-box stall remains on one congested mid-route node (the EV
waits out cross-traffic in the box even with its approach lane empty); it predates the
shunt (same spot, same duration class in preempt-only runs) and is not shunt-caused.
At small R (e.g. 50 m) the hard-shove frequently finds no deposit slot — the adjacent
lane within the zone is itself queued — so failures rise (\(\approx\) 75% of requests) and
the EV keeps its stall; the shunt only pays off where the neighbour lane has room, which
at these densities is from roughly R=200 upward.

## Directory layout

```
sim/
  Bombay.osm.pbf        # source extract (25 MB)
  convert_pbf.py        # pbf -> .osm helper (reuses pyosmium); use upstream .osm directly
  Bombay.net.xml        # full-city network (1 GB, regenerable; to be cropped)
  SPEC.md               # this file
  corridor.py           # bbox-crop (pyosmium), netconvert, auto-pick corridor candidate
  scenario.py           # background flows (density knob), EV route, .sumo.cfg
  control.py            # green-corridor control: preemption trigger (R) + lane-clearance
  run.py                # batch runner over the matrix x seeds (headless)
  results/              # per-run logs + results.csv for the 150-run matrix
  analysis/
    analyze.py          # aggregate results.csv -> aggregated.csv, tables, plots
    aggregated.csv      # medians + IQR per (R, congestion), completion, paired savings
    report.md           # M5 written finding (next to this spec)
    figures/            # tt_vs_r / saved_vs_r / ceiling / completion / grid PNGs
```

## Tooling setup (M0)

1. `micromamba create -n sumo -c conda-forge sumo` (native binaries, no AUR system churn).
2. `uv add traci` in this workspace (bundles `sumolib`).
3. Keep pip `traci` version close to the conda `sumo` binary version (TraCI protocol drifts;
   a version mismatch fails on connect).
4. Smoke test: `traci.connect` a minimal `sumo --remote-port` session.

Binaries are flatpak-wrapped in `sim/vendor/bin/` (`sumo`, `sumo-gui`, `netconvert`, `duarouter`);
`run.py` resolves them automatically (override with `--sumo-bin` or `SUMO_BIN`).

### Watching a run (sumo-gui)

- **Traffic only, no control** — `sim/vendor/bin/sumo-gui -c sim/corridor/corridor.low.1.sumocfg`
  (or `smoke.sumocfg`). Plays autonomously; signals run normally and the preemption/shunt control
  is *not* active (it lives in `control.py`, driven over TraCI).
- **A real green-corridor run** — `uv run python sim/run.py preempt --density low --seed 1 --r 200
  --gui` (also works on `baseline`/`smoke`). Launches `sumo-gui`, drives it over TraCI, and slows
  stepping to `GUI_STEP_DELAY` seconds per step (default 0.1 s ≈ 10 steps/s; e.g.
  `export GUI_STEP_DELAY=0.9` gives a step roughly every second). Corridor runs load
  `sim/corridor/gui.settings.xml`, which starts on a ~650 m window around the EV's origin and
  then follows the ambulance (`traci.gui.trackVehicle`) so the narrow camera stays on it; zoom
  with `GUI_ZOOM` (default 500, 100 = whole net) and set `GUI_FOLLOW=0` for a static view. The
  ambulance is drawn red so it is easy to pick out. R=0 (`baseline`) shows the no-intervention
  case for comparison; `--r inf` shows the ceiling.

## Corridor (M1)

Chosen study corridor (issue #2). Candidates were auto-detected from the Bombay region net `mumbai.net.xml`; the corridor net was cropped to ~5 km² around the pick.

- **Corridor:** Lal Bahadur Shastri Marg
- **BBox (lon,lat):** 72.888, 19.08, 72.914, 19.097
- **Signalized intersections on the route:** 9
- **Length:** 2.34 km
- **Speed / lanes:** 120 km/h, 2 lanes (median)
- **Origin edge:** `257804750#0`
- **Destination edge:** `315614623#5`
- **Net:** `sim/corridor/corridor.net.xml`; regenerate with
  `uv run python sim/corridor.py crop 72.888,19.08,72.914,19.097`
- **Region net (candidate detection):** regenerate with
  `uv run python sim/corridor.py crop 72.82,18.95,72.94,19.12 --osm-out sim/scratch/mumbai.osm --net-out sim/scratch/mumbai.net.xml`

Alternatives (kept, auto-detected):

1. Swami Vivekanand Road: 51 signals, 12.59 km, 100 km/h, 2 lanes, O=1235213725 D=1235442610
2. Bandra Kurla Complex Road: 36 signals, 6.38 km, 100 km/h, 3 lanes, O=1253603264 D=27055240
3. Swatantrya Veer Savarkar Marg: 18 signals, 4.25 km, 100 km/h, 2 lanes, O=22849589#0 D=236102845#1
4. Sion Panvel Highway: 17 signals, 4.65 km, 100 km/h, 4 lanes, O=1268424017 D=1140776078#2
5. Juhu Tara Road: 15 signals, 5.73 km, 100 km/h, 1 lanes, O=1424718871 D=1142033831
6. Linking Road: 14 signals, 3.05 km, 100 km/h, 1 lanes, O=1226336530#0 D=1293641194#3
7. Sion Bandra Link Road: 14 signals, 2.82 km, 100 km/h, 2 lanes, O=1238603168#1 D=1237732054#7
8. Dr Babasaheb Ambedkar Marg (Vincent Road): 11 signals, 1.87 km, 100 km/h, 3 lanes, O=100841107#0-AddedOnRampEdge D=102161593#3
9. Guru Hargovindji Road: 10 signals, 2.30 km, 100 km/h, 2 lanes, O=1264143643 D=1251127693

## Milestones

0. **M0 — Tooling:** conda SUMO env + `uv add traci`, TraCI connect + version match.
1. **M1 — Corridor:** bbox-crop Bombay `.pbf` -> netconvert -> `.net.xml`; auto-pick 3 candidate
   corridors (arterial + >=4 signalized intersections), pick one with the user. **Done** — LBS Marg
   corridor, see [Corridor (M1)](#corridor-m1).
2. **M2 — Baseline runs:** background flows + 1 fixed EV, R=0; verify travel-time measurement. **Done** — see `sim/results/R0_baseline.csv`.
3. **M3 — Green corridor:** preemption trigger (R) + lane clearance; verify the EV never stops
   mid-corridor and TLCs hold green. **Done** — signal preemption (`control.py` `PreemptionController`)
   and lane clearance (`control.py` `LaneClearanceController`, see [Lane clearance](#lane-clearance-mechanic)).
   Risk item on the 2-lane corridor resolved: cars pull over onto the facing lane the EV is not using;
   no shunt-attributable teleports/collisions on repeat high-congestion runs.
4. **M4 — Batch run:** 150-run matrix with paired seeds.
   **Done** — `run.py batch` runs the full R ∈ {0,50,200,500,∞} × congestion {low,med,high}
   × seeds 1–10 matrix headless: auto-builds missing `(density, seed)` scenarios (seed-1
   files reused; randomTrips is seed-deterministic), runs cells in parallel
   (`--parallel 4` default), per-cell failure containment, `--resume` skips completed
   cells (crash-safe relaunch), and appends one uniform row per cell to a single
   `sim/results/results.csv`. Pairedness is structural: background traffic depends only on
   `(density, seed)` while the EV route/departure are fixed, so every R cell of a seed
   sees the identical traffic. R=0 uses the unified telemetry path (no preemption, no
   shunt), so baseline rows share the full schema including `teleports/collisions/n_shunt_*`;
   R=∞ rows carry `red_stop_violation`. Single-run `baseline`/`preempt` commands still write
   the legacy `R0_baseline.csv` / `R_preempt.csv`, which the batch leaves untouched.
   Verified on smoke subsets (schema, determinism via identical reruns, resume, med/high
   completion); full 150-row run reproducible with `uv run python sim/run.py batch`.
5. **M5 — Analysis:** travel time vs R per congestion, seconds saved, % of ceiling, plots.
   **Done** — `sim/analysis/analyze.py` aggregates `sim/results/results.csv` into medians + IQR
   per (R, congestion), paired seconds-saved vs R=0, and % of the R=∞ ceiling captured
   (`sim/analysis/aggregated.csv`; figures in `sim/analysis/figures/`). Written finding:
   `sim/analysis/report.md` — R ≥ 50 m buys ~55% (~140 s) of response time at all densities,
   R=200 is the robust completion optimum, and residual gridlocks are junction-box blockages,
   not shunt artifacts.

## Risks / unknowns

- pip `traci` <-> conda `sumo` version drift.
- Lane-clearance shuttling behavior on a congested arterial (M3).
- Whole-city `Bombay.net.xml` is 1 GB and slow: always operate on the cropped corridor net
  after M1.
- Detection latencies of the real `sireNN` model are not in the loop for A/C; they enter only in
  the later live-detector wiring phase.