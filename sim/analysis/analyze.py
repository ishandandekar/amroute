"""M5 analysis (issue #7): aggregate results.csv -> medians/IQR + headline plots.

Reads the batch output (default sim/results/results.csv, override with --results),
computes per (R, congestion) aggregates over COMPLETED cells plus completion
rates, paired seconds-saved vs R=0, and the % of the R=inf ceiling captured, then
writes sim/analysis/aggregated.csv and figures/:
  tt_vs_r.png      median travel time vs R per density (IQR bars, completion tags)
  saved_vs_r.png   median paired seconds saved vs R=0
  ceiling.png      % of the R=inf ceiling captured per density
  completion.png   completion-rate heatmap over the (R, density) matrix
  grid.png         2x2 combination for the report

Incomplete cells (EV gridlocked, no travel time) are excluded from TT medians but
their count is tracked via the completion fraction; a conservative median that
treats incomplete cells as the full sim window (1500 s) is reported alongside so
the selection bias is visible. Baseline numerics verified against the batch run:
free-flow over the EV route is a constant 82.37 s.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = REPO_ROOT / "sim" / "results" / "results.csv"
OUT_DIR = REPO_ROOT / "sim" / "analysis"
FIG_DIR = OUT_DIR / "figures"

R_ORDER = ["0", "50", "200", "500", "inf"]
DENSITIES = ["low", "med", "high"]
FREEFLOW_S = 82.36628835669747
WINDOW_S = 1500.0  # sim end (hard cap); upper bound for an incomplete cell


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def num(v) -> float:
    return float(v) if v != "" else float("nan")


def quantiles(values: list[float], q1_frac: int = 1, q3_frac: int = 3) -> tuple[float, float]:
    s = sorted(values)
    return s[len(s) // 4], s[(3 * len(s)) // 4]


def summarize_rows(rows: list[dict], r: str, density: str) -> dict:
    cells = [x for x in rows if x["r"] == r and x["density"] == density]
    done = [float(x["travel_time_s"]) for x in cells if x["completed"] == "True"]
    n_total = len(cells)
    n_done = len(done)
    row: dict = {
        "r": r,
        "density": density,
        "n_completed": n_done,
        "n_total": n_total,
        "completion": (n_done / n_total) if n_total else 0.0,
    }
    if done:
        q1, q3 = quantiles(done)
        row.update({
            "median_tt": statistics.median(done),
            "q1_tt": q1,
            "q3_tt": q3,
            "min_tt": min(done),
            "median_ratio": statistics.median(t / FREEFLOW_S for t in done),
            "conservative_median_tt": statistics.median(
                done + [WINDOW_S] * (n_total - n_done)),
        })
    else:
        row.update({
            "median_tt": "", "q1_tt": "", "q3_tt": "", "min_tt": "",
            "median_ratio": "",
            "conservative_median_tt": statistics.median([WINDOW_S] * n_total),
        })
    return row


def paired_savings(rows: list[dict], r: str, density: str) -> dict:
    r0 = {x["seed"]: float(x["travel_time_s"])
          for x in rows if x["r"] == "0" and x["density"] == density
          and x["completed"] == "True"}
    cells = {x["seed"]: float(x["travel_time_s"])
             for x in rows if x["r"] == r and x["density"] == density
             and x["completed"] == "True"}
    diffs = [r0[s] - cells[s] for s in cells if s in r0]
    if not diffs:
        return {"median_saved": "", "n_pairs": 0, "min_saved": ""}
    return {"median_saved": statistics.median(diffs),
            "n_pairs": len(diffs),
            "min_saved": min(diffs)}


def ceiling_capture(med_by_r: dict[str, dict], r: str, density: str) -> float | None:
    med0 = med_by_r["0"][density]["median_tt"]
    ceil = med_by_r["inf"][density]["median_tt"]
    med = med_by_r[r][density]["median_tt"]
    if med0 in (None, "") or med in (None, "") or ceil in (None, "") or med0 == ceil:
        return None
    return (med0 - med) / (med0 - ceil) * 100.0


def plot_tt(agg: list[dict], med_by_r: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = list(range(len(R_ORDER)))
    for density in DENSITIES:
        meds = [num(med_by_r[r][density]["median_tt"]) for r in R_ORDER]
        q1s = [num(med_by_r[r][density]["q1_tt"]) for r in R_ORDER]
        q3s = [num(med_by_r[r][density]["q3_tt"]) for r in R_ORDER]
        comp = [med_by_r[r][density]["completion"] for r in R_ORDER]
        lbl = f"{density}"
        ax.plot(xs, meds, marker="o", label=lbl, linewidth=1.8,
                linestyle="--" if density == "high" else "-")
        ax.errorbar(xs, meds, yerr=[[m - q for m, q in zip(meds, q1s)],
                                    [q - m for m, q in zip(meds, q3s)]],
                    fmt="none", ecolor="0.45", capsize=3)
        for x, m, c in zip(xs, meds, comp):
            if c < 1.0:
                txt = f"{m:.0f}s\n{c:.0%}" if m == m else f"-\n{c:.0%}"
                ax.annotate(txt,
                            (x, m if m == m else 0), textcoords="offset points",
                            xytext=(0, 8),
                            ha="center", fontsize=7, color="0.3")
    ax.axhline(FREEFLOW_S, color="0.7", linestyle=":", linewidth=1.2)
    ax.text(0.02, FREEFLOW_S + 3, "free-flow 82.4 s", fontsize=7, color="0.4")
    ax.set_xticks(xs)
    ax.set_xticklabels([("inf" if r == "inf" else r) + "  " for r in R_ORDER])
    ax.set_xlabel("detection range R (m)")
    ax.set_ylabel("EV travel time (s)")
    ax.set_title("Travel time vs R by congestion level\n(median of completed seeds, IQR bars; % = completed)")
    ax.legend(title="congestion")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_saved(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = list(range(1, len(R_ORDER)))
    for density in DENSITIES:
        sav = [paired_savings(rows, r, density) for r in R_ORDER[1:]]
        ys = [num(s["median_saved"]) for s in sav]
        ns = [s["n_pairs"] for s in sav]
        ax.plot(xs, ys, marker="s", label=density, linewidth=1.8)
        for x, y, n in zip(xs, ys, ns):
            if n < 10 and y == y:
                ax.annotate(f"{y:.0f}s\n(n={n})", (x, y),
                            textcoords="offset points", xytext=(0, -16),
                            ha="center", fontsize=7, color="0.3")
    ax.axhline(0, color="0.5", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([R_ORDER[i] for i in xs])
    ax.set_xlabel("detection range R (m)")
    ax.set_ylabel("seconds saved vs R=0 (median, paired)")
    ax.set_title("Seconds saved over R=0 by R and congestion\n(n = completed seed pairs used)")
    ax.legend(title="congestion")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ceiling(agg: list[dict], med_by_r: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = list(range(1, len(R_ORDER)))
    for density in DENSITIES:
        pcts = []
        for r in R_ORDER[1:]:
            p = ceiling_capture(med_by_r, r, density)
            pcts.append(p if p is not None else float("nan"))
        ax.plot(xs, pcts, marker="^", label=density, linewidth=1.8)
        for x, p in zip(xs, pcts):
            if p == p and (p > 100 or p < 0):
                ax.annotate(f"{p:.0f}%", (x, p), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7, color="0.4")
    ax.axhline(100, color="0.6", linestyle=":", linewidth=1.1)
    ax.text(0.02, 100.5, "100% = R=inf ceiling", fontsize=7, color="0.4")
    ax.set_xticks(xs)
    ax.set_xticklabels([R_ORDER[i] for i in xs])
    ax.set_xlabel("detection range R (m)")
    ax.set_ylabel("% of R=inf ceiling captured")
    ax.set_title("% of R=inf ceiling captured vs R (median-based)")
    ax.legend(title="congestion")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_completion(agg: list[dict], path: Path) -> None:
    data = [[med_by_r_block_completion(agg, r, d) for r in range(len(R_ORDER))]
            for d in range(len(DENSITIES))]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(R_ORDER)))
    ax.set_xticklabels([("inf" if R_ORDER[i] == "inf" else R_ORDER[i]) for i in range(len(R_ORDER))])
    ax.set_yticks(range(len(DENSITIES)))
    ax.set_yticklabels(DENSITIES)
    for d in range(len(DENSITIES)):
        for r in range(len(R_ORDER)):
            ax.text(r, d, f"{data[d][r]:.0%}", ha="center", va="center", fontsize=9,
                    color="0.15")
    ax.set_xlabel("detection range R (m)")
    ax.set_ylabel("congestion")
    ax.set_title("EV completion rate over 10 paired seeds")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def med_by_r_block_completion(agg: list[dict], r: int, d: int) -> float:
    for row in agg:
        if row["r"] == R_ORDER[r] and row["density"] == DENSITIES[d]:
            return row["completion"]
    return 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M5 aggregation of the batch results.csv")
    ap.add_argument("--results", type=Path, default=RESULTS_CSV,
                    help="batch output csv (default sim/results/results.csv)")
    args = ap.parse_args(argv)
    if not args.results.is_file():
        print(f"results not found: {args.results} (run sim/run.py batch first)", file=__import__("sys").stderr)
        return 1

    rows = load(args.results)
    agg = [summarize_rows(rows, r, d) for d in DENSITIES for r in R_ORDER]
    med_by_r: dict[str, dict[str, dict]] = {
        r: {d: next(x for x in agg if x["r"] == r and x["density"] == d)
            for d in DENSITIES}
        for r in R_ORDER
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    agg_path = OUT_DIR / "aggregated.csv"
    cols = ["r", "density", "n_completed", "n_total", "completion",
            "median_tt", "q1_tt", "q3_tt", "min_tt", "median_ratio",
            "conservative_median_tt"]
    with agg_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in agg:
            w.writerow({k: row.get(k, "") for k in cols})

    print_table(agg, rows, med_by_r)

    plot_tt(agg, med_by_r, FIG_DIR / "tt_vs_r.png")
    plot_saved(rows, FIG_DIR / "saved_vs_r.png")
    plot_ceiling(agg, med_by_r, FIG_DIR / "ceiling.png")
    plot_completion(agg, FIG_DIR / "completion.png")
    make_grid(FIG_DIR)
    print(f"[analyze] wrote {agg_path}")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"[analyze] wrote {f}")
    return 0


def make_grid(fig_dir: Path) -> None:
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    order = [("tt_vs_r.png", "Travel time vs R"),
             ("completion.png", "Completion over 10 seeds"),
             ("saved_vs_r.png", "Seconds saved vs R=0"),
             ("ceiling.png", "% of R=inf ceiling captured")]
    for ax, (name, _title) in zip(axes.flat, order):
        img = mpimg.imread(fig_dir / name)
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "grid.png", dpi=150)
    plt.close(fig)


def print_table(agg: list[dict], rows: list[dict], med_by_r: dict[str, dict[str, dict]]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    for density in DENSITIES:
        table = Table(title=f"{density.upper()} congestion — median over completed seeds (free-flow 82.4 s)")
        table.add_column("R (m)")
        table.add_column("completed")
        table.add_column("median TT (s)")
        table.add_column("IQR (s)")
        table.add_column("med travel/fcf")
        table.add_column("saved vs R0 (s)")
        table.add_column("n pairs")
        table.add_column("% ceil")
        table.add_column("conservative med (1500s)")
        for r in R_ORDER:
            a = med_by_r[r][density]
            sav = paired_savings(rows, r, density)
            pct = ceiling_capture(med_by_r, r, density)
            table.add_row(
                "inf" if r == "inf" else r,
                f"{a['n_completed']}/{a['n_total']}",
                f"{a['median_tt']:.1f}" if a["median_tt"] != "" else "-",
                f"({a['q1_tt']:.1f}-{a['q3_tt']:.1f})" if a["q1_tt"] != "" else "-",
                f"{a['median_ratio']:.2f}x" if a["median_ratio"] != "" else "-",
                f"{sav['median_saved']:.1f}" if sav["median_saved"] != "" else "-",
                str(sav["n_pairs"]),
                f"{pct:.0f}%" if pct is not None else "-",
                f"{a['conservative_median_tt']:.1f}" if isinstance(a["conservative_median_tt"], float) else "-",
            )
        console.print(table)


if __name__ == "__main__":
    raise SystemExit(main())