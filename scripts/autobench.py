#!/usr/bin/env python3
"""autobench.py — Parametric benchmarking with power-law time fitting.

Usage:
    autobench.py /path/to/exe {L=1:5} --arg1 VAL --sweeps {sw=10:10:50}

Sweep syntax in any argument position:
    {name=start:end}        — integer steps of 1
    {name=start:step:end}   — explicit step (int or float)
"""

import sys
import re
import subprocess
import time
import math

SWEEP_RE = re.compile(r'\{([\w\^]+)=([^}]+)\}')


def parse_range(spec: str) -> list:
    parts = spec.split(':')
    if len(parts) == 2:
        start, end, step = float(parts[0]), float(parts[1]), 1.0
    elif len(parts) == 3:
        start, step, end = float(parts[0]), float(parts[1]), float(parts[2])
    else:
        raise ValueError(f"Bad sweep spec {spec!r}: expected start:end or start:step:end")

    values = []
    v = start
    eps = 1e-9 * max(abs(step), 1)
    while v <= end + eps:
        r = round(v, 10)
        values.append(int(r) if r == int(r) else r)
        v += step
    return values


def substitute(template: str, mapping: dict) -> str:
    return SWEEP_RE.sub(lambda m: str(mapping[m.group(1)]), template)


def fit_model_1d(name: str, xs: list, ts: list, exponent=None):
    """Fit T = A + B·x^p via scipy.curve_fit. Returns (A, B, p, r2) or None."""
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError:
        print("  (scipy/numpy not available — skipping fit)")
        return None

    xf = np.array([float(x) for x in xs])
    tf = np.array(ts, dtype=float)

    if len(xf) < 3:
        print(f"  (need ≥3 points for 3-parameter fit, got {len(xf)} — skipping)")
        return None

    def model(x, A, B, p):
        return A + B * x**p

    def model_set_p(x, A, B):
        return A + B * x**exponent

    # A is bounded above by min(T) so the power-law term stays non-negative.
    try:
        if exponent is None:
            popt, pcov = curve_fit(
                model, xf, tf,
                p0=[tf.min() * 0.1, tf.mean(), 1.0],
                bounds=([-np.inf, 0, -20], [float(tf.min()), np.inf, 20]),
                maxfev=20000,
            )

            A, B, p = popt

        else:
            popt, pcov = curve_fit(
                model_set_p, xf, tf,
                p0=[tf.min() * 0.1, tf.mean()],
                bounds=([-np.inf, 0], [float(tf.min()), np.inf]),
                maxfev=20000,
            )
            A, B = popt
            p = exponent
    except Exception as e:
        print(f"  Warning: curve_fit failed ({e})")
        return None


    perr = np.append(np.sqrt(np.diag(pcov)),[0])
    pred = model(xf, A, B, p)

    ss_res = np.sum((tf - pred) ** 2)
    ss_tot = np.sum((tf - tf.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print(f"  fit: T = {A:.4g} + {B:.4g}·{name}^{p:.3f}  "
          f"(R²={r2:.4f}, σ_A={perr[0]:.2g}, σ_p={perr[2]:.2g})")
    return (A, B, p, r2)


def stitch_fits(fits: dict, baselines: dict, var_names: list):
    """
    Each 1D sweep i was measured with other variables at baseline x_{j0}.
    Full separable model:  T = A + B_global · ∏_i x_i^p_i
    Per-sweep relation:    B_i = B_global · ∏_{j≠i} x_{j0}^p_j
    => B_global = B_i / ∏_{j≠i} x_{j0}^p_j  (one estimate per sweep)
    """
    ps = {n: fits[n][2] for n in var_names}

    A_global = sum(fits[n][0] for n in var_names) / len(var_names)

    B_estimates = []
    for name in var_names:
        other_prod = math.prod(
            float(baselines[n]) ** ps[n]
            for n in var_names if n != name
        )
        B_estimates.append(fits[name][1] / other_prod if other_prod != 0 else fits[name][1])
    B_global = sum(B_estimates) / len(B_estimates)

    power_str = " · ".join(f"{n}^{ps[n]:.3f}" for n in var_names)
    baseline_label = ", ".join(f"{n}={baselines[n]}" for n in var_names)
    baseline_scaling = math.prod(float(baselines[n]) ** ps[n] for n in var_names)
    baseline_pred = A_global + B_global * baseline_scaling

    print("=" * 60)
    print(f"Stitched model:  T = {A_global:.4g} + {B_global:.4g} · {power_str}")
    print(f"  Predict at baseline ({baseline_label}): {baseline_pred:.4g} s")
    print("=" * 60)


def main():
    raw_args = sys.argv[1:]
    if not raw_args:
        print(__doc__)
        sys.exit(1)

    # Collect sweep variables (preserve first-seen order)
    variables: dict[str, list] = {}
    for arg in raw_args:
        for m in SWEEP_RE.finditer(arg):
            name = m.group(1)
            if name not in variables:
                variables[name] = parse_range(m.group(2))

    var_names = list(variables.keys())

    if not var_names:
        t0 = time.perf_counter()
        proc = subprocess.run(raw_args)
        elapsed = time.perf_counter() - t0
        print(f"\nElapsed: {elapsed:.3f}s  (exit={proc.returncode})")
        return

    # Each variable swept independently; all others held at their minimum
    baseline = {name: variables[name][0] for name in var_names}
    total_runs = sum(len(variables[name]) for name in var_names)
    print(f"Sweep variables: {', '.join(f'{k}={variables[k]}' for k in var_names)}")
    print(f"Baseline (min) values: {', '.join(f'{k}={baseline[k]}' for k in var_names)}")
    print(f"Total runs: {total_runs} ({' + '.join(str(len(variables[k])) for k in var_names)})\n")

    # sweep_results[name] = list of (value, elapsed)
    sweep_results: dict[str, list[tuple]] = {name: [] for name in var_names}
    sweep_fits: dict = {}

    for sweep_var in var_names:
        print(f"--- Sweeping {sweep_var} ---")
        for val in variables[sweep_var]:
            mapping = {**baseline, sweep_var: val}
            label = f"{sweep_var}={val}"
            cmd = [substitute(a, mapping) for a in raw_args]

            print(f"[{label}]  {' '.join(cmd)}")
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
            elapsed = time.perf_counter() - t0

            status = "ok" if proc.returncode == 0 else f"exit={proc.returncode}"
            print(f"  -> {elapsed:.3f}s  ({status})")
            sweep_results[sweep_var].append((val, elapsed))

        xs = [v for v, _ in sweep_results[sweep_var]]
        ts = [t for _, t in sweep_results[sweep_var]]

        p = None
        print(sweep_var)
        if '^' in sweep_var:
            p = float(sweep_var.split('^')[1])
            print(f"Using fixed p={p}")
        result = fit_model_1d(sweep_var, xs, ts, exponent=p)

        if result is not None:
            sweep_fits[sweep_var] = result
        print()

    print("Summary:")
    for sweep_var in var_names:
        print(f"  {sweep_var}:")
        for val, elapsed in sweep_results[sweep_var]:
            print(f"    {sweep_var}={val:<10}  {elapsed:.3f}s")

    fitted_vars = [n for n in var_names if n in sweep_fits]
    if len(fitted_vars) >= 1:
        print()
        stitch_fits(sweep_fits, baseline, fitted_vars)


if __name__ == '__main__':
    main()
