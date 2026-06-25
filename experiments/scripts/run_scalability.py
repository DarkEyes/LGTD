#!/usr/bin/env python3
"""
Scalability benchmark for LGTD vs. baseline seasonal-trend decomposition methods.

Measures wall-clock runtime (and, optionally, peak Python-side memory) of each
method's `fit_transform` as a function of series length N, to empirically back
LGTD's O(K*N) linear-time claim.

Run from the repository root (after `pip install -e .` and activating the env):

    python experiments/scripts/run_scalability.py                       # default: all methods, N up to 100k, time+nothing
    python experiments/scripts/run_scalability.py --measure-memory       # also record peak memory
    python experiments/scripts/run_scalability.py --methods LGTD STL ASTD # subset
    python experiments/scripts/run_scalability.py --max-n 1000000         # push to 1M
    python experiments/scripts/run_scalability.py --reps 7 --period 120   # more reps, explicit period

Methodology
-----------
* Timing wraps ONLY `fit_transform` (not construction, imports, or data generation),
  using time.perf_counter().
* One warmup run per (method, N) is discarded (absorbs import/JIT/cache effects,
  e.g. torch / statsmodels / JVM start). Then `--reps` timed runs; we report the
  MEDIAN with min/max (median is robust to OS scheduling jitter).
* Period-based methods (STL, STR, RobustSTL, FastRobustSTL, OnlineSTL, OneShotSTL)
  receive a FIXED period (`--period`, default 120) independent of N. This is the
  fair, deployment-realistic choice and isolates N-scaling. LGTD and ASTD use no
  period. The fixed period is exactly the best case for the period-based methods.
* Auto-stop: once a method's median time at some N exceeds `--budget` seconds, it is
  not run at larger N (status="stopped"). Per-method hard caps in PER_METHOD_MAX_N
  guard against pathological hangs (RobustSTL is ~O(N^2)).
* Memory (optional): measured in a SEPARATE pass via tracemalloc so it does not
  distort timing. tracemalloc only sees Python allocations, so OneShotSTL (a Java
  subprocess) is reported as NaN for memory — its heap lives in the JVM.

CAVEAT — OneShotSTL: it runs as a Java subprocess. Its measured time includes a
roughly constant JVM-startup overhead per call, which inflates small-N points. Only
its asymptotic SLOPE is meaningful for the scaling comparison, not absolute values
at small N. It is included because the repo invokes it this way; interpret with care.

Outputs (under --out-dir, default experiments/results/scalability/):
    scalability_raw.csv      method, n, rep, time_s
    scalability_summary.csv  method, n, median_s, min_s, max_s, n_reps, peak_mem_mb, status
"""

import argparse
import csv
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

# --- path setup: repo root on sys.path so `experiments...` imports resolve;
#     `lgtd` is expected to be importable (pip install -e .). We also add src/
#     defensively in case the package was not installed.
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #

DEFAULT_GRID = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
                200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]

# Hard safety caps so pathological methods don't hang the sweep even if a single
# call is below the per-method budget. None = use global --max-n.
PER_METHOD_MAX_N = {
    "RobustSTL": 20_000,     # ~O(N^2): blows up fast
    "STR": 50_000,
    "FastRobustSTL": 100_000,
}

ALL_METHODS = [
    "LGTD",          # period-free  (the method under test)
    "STL",           # period-based (statsmodels, C-backed LOESS)
    "ASTD",          # period-free  (sliding DFT)
    "OnlineSTL",     # period-based
    "OneShotSTL",    # period-based, JAVA subprocess (see caveat)
    "STR",           # period-based
    "FastRobustSTL", # period-based (torch)
    "RobustSTL",     # period-based, ~O(N^2)
]

PERIOD_BASED = {"STL", "OnlineSTL", "OneShotSTL", "STR", "FastRobustSTL", "RobustSTL"}
JAVA_METHODS = {"OneShotSTL"}  # tracemalloc cannot see subprocess memory


# ----------------------------------------------------------------------------- #
# Data generation (self-contained; content does not affect timing materially)
# ----------------------------------------------------------------------------- #

def make_series(n: int, period: int, seed: int) -> np.ndarray:
    """Linear trend + fixed-period sinusoid + Gaussian noise, length n."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    trend = 0.02 * t
    seasonal = 50.0 * np.sin(2.0 * np.pi * t / period)
    noise = rng.normal(0.0, 1.0, size=n)
    return trend + seasonal + noise


# ----------------------------------------------------------------------------- #
# Model factory — mirrors experiments/runners/experiment_runner.py dispatch.
# Returns a zero-arg callable that runs fit_transform on `data`.
# Construction happens OUTSIDE this callable so only fit_transform is timed.
# ----------------------------------------------------------------------------- #

def build_runner(method: str, data: np.ndarray, period: int, lgtd_trend: str = "linear"):
    """Return a no-arg function that executes `method`'s decomposition on `data`.

    lgtd_trend fixes LGTD's global-trend model G for the scaling study. We default
    to 'linear' because LGTD's 'auto' mode picks linear-vs-LOWESS from the data,
    which makes runtime data-dependent (LOWESS is much slower) and would add a
    spurious discontinuity to the scaling curve. The AutoTrend-LLT local-trend
    stage that carries the O(K*N) claim is identical regardless of G; 'auto' only
    adds a one-time O(T) R^2 test. Use --lgtd-trend auto|lowess to override.
    """
    if method in ("LGTD", "LGTD_Linear", "LGTD_LOWESS"):
        from lgtd import lgtd
        trend = lgtd_trend
        if method == "LGTD_Linear":
            trend = "linear"
        elif method == "LGTD_LOWESS":
            trend = "lowess"
        model = lgtd(window_size=5, error_percentile=50, trend_selection=trend)
        return lambda: model.fit_transform(data)

    if method == "STL":
        from experiments.baselines.stl import STLDecomposer
        model = STLDecomposer(period=period)
        return lambda: model.fit_transform(data)

    if method == "ASTD":
        from experiments.baselines.astd import ASTDDecomposer
        model = ASTDDecomposer(seasonality_smoothing=0.7, mode="batch",
                               init_window_size=300)
        if not getattr(model, "_astd_available", True):
            raise ImportError("ASTD module not available in baselines directory")
        return lambda: model.fit_transform(data)

    if method == "OnlineSTL":
        from experiments.baselines.online_stl import OnlineSTLDecomposer
        model = OnlineSTLDecomposer(periods=[period])
        return lambda: model.fit_transform(data)

    if method == "OneShotSTL":
        from experiments.baselines.oneshot_stl import OneShotSTLDecomposer
        model = OneShotSTLDecomposer(period=period)  # validates JAR + java in __init__
        return lambda: model.fit_transform(data)

    if method == "STR":
        from experiments.baselines.str_decomposer import STRDecomposer
        model = STRDecomposer(seasonal_periods=[period])
        return lambda: model.fit_transform(data)

    if method == "FastRobustSTL":
        from experiments.baselines.fast_robust_stl import FastRobustSTLDecomposer
        model = FastRobustSTLDecomposer(period=period)
        return lambda: model.fit_transform(data)

    if method == "RobustSTL":
        from experiments.baselines.robust_stl import RobustSTLDecomposer
        model = RobustSTLDecomposer(period=period)
        return lambda: model.fit_transform(data)

    raise ValueError(f"Unknown method: {method}")


# ----------------------------------------------------------------------------- #
# Measurement
# ----------------------------------------------------------------------------- #

def time_once(run_fn) -> float:
    gc.collect()
    t0 = time.perf_counter()
    run_fn()
    return time.perf_counter() - t0


def peak_memory_mb(method: str, data: np.ndarray, period: int, lgtd_trend: str = "linear") -> float:
    """Peak Python-side allocation (MB) for one fit_transform. NaN for Java methods."""
    if method in JAVA_METHODS:
        return float("nan")
    try:
        run_fn = build_runner(method, data, period, lgtd_trend)
    except Exception:
        return float("nan")
    gc.collect()
    tracemalloc.start()
    try:
        run_fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024.0 * 1024.0)


# ----------------------------------------------------------------------------- #
# Main sweep
# ----------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS,
                    help=f"Subset of {ALL_METHODS}")
    ap.add_argument("--max-n", type=int, default=100_000,
                    help="Largest series length to attempt (default 100000)")
    ap.add_argument("--grid", type=int, nargs="+", default=None,
                    help="Explicit N grid (overrides --max-n filtering of DEFAULT_GRID)")
    ap.add_argument("--reps", type=int, default=5, help="Timed reps per (method,N)")
    ap.add_argument("--period", type=int, default=120,
                    help="Fixed season length for period-based methods (default 120)")
    ap.add_argument("--lgtd-trend", choices=["linear", "auto", "lowess"], default="linear",
                    help="LGTD global-trend model for the scaling study (default linear; see build_runner docstring)")
    ap.add_argument("--budget", type=float, default=90.0,
                    help="Per-method auto-stop budget in seconds (median > budget => stop scaling)")
    ap.add_argument("--measure-memory", action="store_true",
                    help="Also record peak Python-side memory (separate pass)")
    ap.add_argument("--seed", type=int, default=69, help="Base RNG seed")
    ap.add_argument("--out-dir", type=str,
                    default=str(REPO_ROOT / "experiments" / "results" / "scalability"))
    args = ap.parse_args()

    grid = args.grid if args.grid else [n for n in DEFAULT_GRID if n <= args.max_n]
    grid = sorted(set(grid))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "scalability_raw.csv"
    sum_path = out_dir / "scalability_summary.csv"

    # Open CSVs and flush incrementally so partial results survive Ctrl-C / crash.
    raw_f = open(raw_path, "w", newline="")
    sum_f = open(sum_path, "w", newline="")
    raw_w = csv.writer(raw_f)
    sum_w = csv.writer(sum_f)
    raw_w.writerow(["method", "n", "rep", "time_s"])
    sum_w.writerow(["method", "n", "median_s", "min_s", "max_s",
                    "n_reps", "peak_mem_mb", "status"])
    raw_f.flush(); sum_f.flush()

    print(f"Methods : {args.methods}")
    print(f"N grid  : {grid}")
    print(f"Reps    : {args.reps} (+1 warmup)   Period: {args.period}   "
          f"Budget: {args.budget}s   Memory: {args.measure_memory}")
    print(f"Output  : {out_dir}\n")

    for method in args.methods:
        cap = PER_METHOD_MAX_N.get(method)
        print(f"=== {method} ===")

        # One global warmup on a tiny series to trigger all lazy initialization
        # (imports, statsmodels/torch first-use caching, JVM spin-up) BEFORE the
        # first measured point, so N=grid[0] is not penalized by cold start.
        try:
            warm = make_series(max(4 * args.period, 600), args.period, seed=args.seed)
            build_runner(method, warm, args.period, args.lgtd_trend)()
        except Exception:
            pass  # availability is re-checked per-N below

        for n in grid:
            if cap is not None and n > cap:
                print(f"  N={n:>9}  skipped (per-method cap {cap})")
                sum_w.writerow([method, n, "", "", "", 0, "", "capped"]); sum_f.flush()
                continue

            data = make_series(n, args.period, seed=args.seed + n)

            # Build once to surface unavailable methods early (e.g. missing JAR / torch).
            try:
                run_fn = build_runner(method, data, args.period, args.lgtd_trend)
            except Exception as e:
                print(f"  N={n:>9}  UNAVAILABLE: {type(e).__name__}: {e}")
                sum_w.writerow([method, n, "", "", "", 0, "", f"unavailable:{type(e).__name__}"])
                sum_f.flush()
                break  # if it can't build at the smallest N, it won't at larger N

            # Warmup (discarded)
            try:
                run_fn()
            except Exception as e:
                print(f"  N={n:>9}  FAILED (warmup): {type(e).__name__}: {e}")
                sum_w.writerow([method, n, "", "", "", 0, "", f"failed:{type(e).__name__}"])
                sum_f.flush()
                break

            # Timed reps (fresh model each rep; time only fit_transform)
            times = []
            failed = False
            for r in range(args.reps):
                try:
                    run_fn = build_runner(method, data, args.period, args.lgtd_trend)
                    dt = time_once(run_fn)
                except Exception as e:
                    print(f"  N={n:>9}  FAILED (rep {r}): {type(e).__name__}: {e}")
                    failed = True
                    break
                times.append(dt)
                raw_w.writerow([method, n, r, f"{dt:.6f}"])
            raw_f.flush()

            if failed or not times:
                sum_w.writerow([method, n, "", "", "", 0, "", "failed"]); sum_f.flush()
                break

            med = float(np.median(times))
            lo, hi = float(np.min(times)), float(np.max(times))

            mem = ""
            if args.measure_memory:
                m = peak_memory_mb(method, data, args.period, args.lgtd_trend)
                mem = "" if (m != m) else f"{m:.3f}"  # NaN -> blank

            sum_w.writerow([method, n, f"{med:.6f}", f"{lo:.6f}", f"{hi:.6f}",
                            len(times), mem, "ok"])
            sum_f.flush()

            memstr = f"  mem={mem}MB" if mem else ""
            print(f"  N={n:>9}  median={med:8.4f}s  (min={lo:7.4f}, max={hi:7.4f}){memstr}")

            # Auto-stop scaling this method if it blew the budget.
            if med > args.budget:
                print(f"  -> median exceeded budget ({args.budget}s); not scaling {method} further.")
                if n < (grid[-1] if grid else n):
                    sum_w.writerow([method, "", "", "", "", 0, "", "stopped_after_budget"])
                    sum_f.flush()
                break

        print()

    raw_f.close(); sum_f.close()
    print(f"Done. Raw -> {raw_path}\n     Summary -> {sum_path}")
    print("Plot with:  python experiments/scripts/plot_scalability.py "
          f"--summary {sum_path}")


if __name__ == "__main__":
    main()
