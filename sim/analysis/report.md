# Green-corridor analysis (M5) — detection-range payoff vs congestion

Study data: 150 headless SUMO runs — R ∈ {0, 50, 200, 500, ∞} m × congestion
{low, med, high} × 10 paired seeds (identical per-seed traffic across R). EV
travels the 2.34 km LBS Marg corridor (LBS road), 2 lanes per direction,
9 signalized intersections; free-flow over the route is 82.4 s. Aggregates are
medians over seeds that completed (EV arrived before the 1500 s window);
completion rates are reported because 15/150 cells gridlocked. See
`aggregated.csv` (all numbers) and `figures/grid.png` (all four panels).

## Headline numbers

**Median EV travel time (s) and completion, by R and congestion**

| R (m) | low         | compl | med         | compl | high        | compl |
|-------|-------------|-------|-------------|-------|-------------|-------|
| 0     | 256.2 (256–256.5) | 10/10 | 256.5 (256.5–346.5) | 10/10 | 263.0 (256.5–346.5) | 9/10 |
| 50    | 118.0 (117.0–120.5) | 10/10 | 128.0 (122.5–129.0) | 9/10 | 130.5 (128.0–135.0) | 7/10 |
| 200   | 113.5 (113.0–119.5) | 10/10 | 122.0 (120.0–123.5) | 9/10 | 129.5 (124.0–129.5) | 9/10 |
| 500   | 115.0 (113.0–120.0) | 10/10 | 123.3 (121.5–128.0) | 8/10 | 129.5 (123.0–139.5) | 9/10 |
| ∞     | 115.0 (113.0–119.5) | 10/10 | 122.0 (121.5–124.5) | 8/10 | 127.5 (123.0–133.5) | 7/10 |

Parentheses = IQR; travel/free-flow column reads 3.11x at R0 → 1.38–1.58x for R ≥ 50.

**Seconds saved vs R=0 (per-seed paired median) and % of the R=∞ ceiling captured**

| R (m) | saved low | saved med | saved high | %ceil low | %ceil med | %ceil high |
|-------|-----------|-----------|------------|-----------|-----------|------------|
| 50  | 139.0 (10) | 134.0 (9) | 131.5 (6) | 98% | 96% | 98% |
| 200 | 142.5 (10) | 137.5 (9) | 137.2 (8) | 101% | 100% | 99% |
| 500 | 141.2 (10) | 133.5 (8) | 137.2 (8) | 100% | 99% | 99% |
| ∞   | 141.5 (10) | 135.0 (8) | 133.5 (7) | 100% | 100% | 100% |

( ) = number of seeds completed at both R=0 and R (pairs used). % ceil is computed
on density medians; raw values, so >100% just means the median at that R is
slightly below the R=∞ median (only low/R200: 113.5 vs 115.0).

## Findings

**1. The response-time payoff is won at a very short range.** Every R ≥ 50 m cuts
median travel time by ~55% (≈ 140 s saved, from ~256–263 s to ~115–130 s) at all
three congestion levels, and gets within ~2–4% of the full R=∞ ceiling. On low
density the corridor is essentially free-flowing above R=50; a 50 m look-ahead is
already the ceiling in practice.

**2. Congestion does not reduce the relative payoff — it moves the risk.** Median
travel times are similar across densities at R≥200 (~122 s med, ~129.5 s high),
but completion fidelity is what worsens: R∞ completes only 7/10 (high) and
8/10 (med) seeds, and R50 also drops to 7/10 (high). R=200 is the robust optimum —
9/10 at both med and high with the best median at med (122.0 s). R is therefore
**non-monotone in its reliability**: more look-ahead is not better.

**3. The failures are preemption-induced junction gridlocks, not shunt artifacts.**
In every incomplete intervention cell the shunt had moved ≤ 3 cars (med6: 91
requests → 0 deposits — no slots in the neighbour lane), so the lane-clearance is
exonerated; teleports/collisions were 0/0. The EV instead starves conflicting
flows by holding its approach greens, the conflicting stream fills the late
junction box, and the EV's green cannot expel vehicles already inside it — it
sits at v=0 in box lane `:11042291308_0_1` (95% of the route) for the rest of the
window. Representative seeds: **med6** completes at R0 (256.5 s) but gridlocks at
every R>0; **high10** completes at R200 (178.5 s) yet gridlocks at R50/R500/R∞;
**high6** completes only at R500 (122.5 s); while **high8** is *rescued* by the
intervention — gridlocked at R0, it completes at R50/200/500 (~128–135 s). Zero
R=∞ red-stop violations occurred (stalls are box-blockage, not stopped-at-red).

**4. Conservative medians confirm the headline numbers are not artifacts.** Re-counting
incomplete cells as a full 1500 s window shifts med/high medians by at most ~2 s
(e.g. high R50 130.5 → 132.8 s; med R50 128.0 → 128.3 s), because gridlock is a
seed-tail event rather than the median case — but completion must be reported
alongside any median.

## Implication for deployment

A fixed, short detection range (order 100–200 m) captures essentially the full
green-corridor benefit with the best completion reliability at congestion. The
"infinite look-ahead" ideal is misleading on a busy 2-lane arterial: it provokes
cross-flow starvation gridlock at mid-route junctions. The real structural
bottleneck is the final junction box — a 2-lane corridor cannot absorb cleared
cross-flow, so a south-of-corridor intersection treatment (or a wider target
corridor) would be needed to push below the ~1.5x-free-flow floor seen here.

Figures: `figures/tt_vs_r.png` (travel time vs R, completion tagged),
`figures/completion.png` (completion heatmap), `figures/saved_vs_r.png`
(paired savings), `figures/ceiling.png` (% ceiling captured).

Reproduce: `uv run python sim/run.py batch` (150-run matrix) →
`uv run python sim/analysis/analyze.py` (this table + figures). Free-flow
82.4 s as per `corridor.net.xml`.