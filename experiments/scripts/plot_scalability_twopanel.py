#!/usr/bin/env python3
"""
Two-panel scalability figure (time | memory) from scalability_summary.csv.

    python plot_scalability_twopanel.py --summary scalability_summary.csv

Left:  runtime vs N (log-log), median + min-max band, LGTD highlighted,
       O(N) reference guide, LGTD large-N slope annotation.
Right: peak Python-side memory vs N (log-log).
Shared x-axis. OneShotSTL is marked (Java subprocess; see caption caveat).
Java methods are excluded from the memory panel (tracemalloc can't see JVM heap).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# LGTD bold red, on top. OneShotSTL gets a dagger in its label for the caveat.
STYLE = {
    "LGTD":          dict(color="#d62728", ls="-",  lw=3.4, z=10, marker="o", label="LGTD"),
    "OneShotSTL":    dict(color="#2ca02c", ls=":",  lw=2.2, z=4,  marker="s", label="OneShotSTL$^\\dagger$"),
    "STL":           dict(color="#1f77b4", ls=":",  lw=2.2, z=3,  marker="^", label="STL"),
    "OnlineSTL":     dict(color="#ff7f0e", ls="-.", lw=2.0, z=3,  marker="v", label="OnlineSTL"),
    "ASTD":          dict(color="#e377c2", ls="--", lw=2.0, z=3,  marker="D", label="ASTD"),
    "STR":           dict(color="#8c564b", ls="-",  lw=2.0, z=3,  marker="X", label="STR"),
    "FastRobustSTL": dict(color="#7f7f7f", ls="--", lw=2.0, z=3,  marker="P", label="FastRobustSTL"),
    "RobustSTL":     dict(color="#9467bd", ls="-",  lw=2.0, z=3,  marker="*", label="RobustSTL"),
}
DEFAULT = dict(color="#555", ls="-", lw=1.8, z=2, marker="o", label=None)
JAVA = {"OneShotSTL"}  # excluded from memory panel


def order_methods(present):
    ordered = [m for m in STYLE if m in present]
    ordered += [m for m in present if m not in ordered]
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--title", default="Empirical Scaling of STD Methods")
    args = ap.parse_args()

    spath = Path(args.summary)
    out_dir = Path(args.out_dir) if args.out_dir else spath.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(spath)
    df = df[df["status"] == "ok"].copy()
    for c in ["n", "median_s", "min_s", "max_s", "peak_mem_mb"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["n", "median_s"]).sort_values(["method", "n"])

    present = df["method"].unique().tolist()
    methods = order_methods(present)

    fig, (axT, axS) = plt.subplots(1, 2, figsize=(13.6, 5.6), sharex=True)

    # ---- (a) TIME ----
    lgtd_slope = None
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        s = STYLE.get(m, DEFAULT)
        axT.plot(sub["n"], sub["median_s"], s["ls"], color=s["color"], lw=s["lw"],
                 marker=s["marker"], ms=5, zorder=s["z"], label=s["label"] or m)
        if m == "LGTD" and len(sub) >= 4:
            up = sub.sort_values("n").iloc[len(sub) // 2:]
            lgtd_slope = float(np.polyfit(np.log10(up["n"]), np.log10(up["median_s"]), 1)[0])

    anchor = df[df["method"] == "LGTD"]
    anchor = anchor if not anchor.empty else df
    n0, t0 = anchor["n"].iloc[0], anchor["median_s"].iloc[0]
    nn = np.array([df["n"].min(), df["n"].max()], float)
    axT.plot(nn, t0 * (nn / n0), color="#999", ls=(0, (6, 4)), lw=1.4, zorder=1,
             label="O(N) reference")

    axT.set_yscale("log"); axT.set_xscale("log")
    axT.set_ylabel("Runtime per decomposition (s)", fontsize=12)
    axT.set_xlabel("Number of data points  $N$", fontsize=12)
    axT.set_title("(a) Runtime", fontsize=12.5, pad=8)
    axT.grid(True, which="major", ls="-", alpha=0.25)
    axT.grid(True, which="minor", ls=":", alpha=0.12)
    if lgtd_slope is not None:
        axT.text(0.97, 0.05,
                 f"LGTD slope $\\approx$ {lgtd_slope:.2f} (large $N$)\nslope 1 = linear",
                 transform=axT.transAxes, ha="right", va="bottom", fontsize=9,
                 style="italic", color="#333",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.95))

    # ---- (b) MEMORY ----
    mdf = df.dropna(subset=["peak_mem_mb"])
    mdf = mdf[~mdf["method"].isin(JAVA)]
    for m in order_methods(mdf["method"].unique().tolist()):
        sub = mdf[mdf["method"] == m]
        if sub.empty:
            continue
        s = STYLE.get(m, DEFAULT)
        axS.plot(sub["n"], sub["peak_mem_mb"], s["ls"], color=s["color"], lw=s["lw"],
                 marker=s["marker"], ms=5, zorder=s["z"], label=s["label"] or m)

    axS.set_yscale("log"); axS.set_xscale("log")
    axS.set_ylabel("Peak Python-side memory (MB)", fontsize=12)
    axS.set_xlabel("Number of data points  $N$", fontsize=12)
    axS.set_title("(b) Memory", fontsize=12.5, pad=8)
    axS.grid(True, which="major", ls="-", alpha=0.25)
    axS.grid(True, which="minor", ls=":", alpha=0.12)

    for ax in (axT, axS):
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.set_xlim(df["n"].min(), df["n"].max())

    # single shared legend (from the time panel, LGTD first)
    h, l = axT.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: (l[i] != "LGTD", l[i]))
    axT.legend([h[i] for i in order], [l[i] for i in order],
               fontsize=9, loc="upper left", framealpha=0.95)

    fig.suptitle(args.title, fontsize=13.5, y=1.00)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"scalability_twopanel.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote scalability_twopanel.png / .pdf"
          + (f"   (LGTD slope ~ {lgtd_slope:.2f})" if lgtd_slope else ""))
    print("Caption reminder: $\\dagger$ OneShotSTL runs as a Java subprocess; its "
          "runtime is dominated by per-call JVM startup at these N and is not "
          "directly comparable to the in-process methods. It is omitted from (b) "
          "because tracemalloc cannot observe the JVM heap.")


if __name__ == "__main__":
    main()
