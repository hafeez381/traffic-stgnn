"""Question 6(a): the comparison table and predicted-versus-actual plots, built from saved runs.

Every number comes from results/runs/<model>/metrics.json. The script refuses to build the table
if the four runs were not evaluated on the same test windows with the same settings.
It also computes a last-value baseline (repeat the latest reading for all four horizons) and
flags any trained model that does not beat it.
Writes results/comparison.{json,md,tex} and results/figures/q6a_forecasts.pdf.
Usage: python -m scripts.compare [--data DIR]
"""
import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts import style
from stgnn.config import CONFIG, GRAPHS
from stgnn.data import load, split_masks, window_ends

ROWS = [("temporal", "Temporal-only", "None"), ("fixed", "STGNN", "Fixed"),
        ("adaptive", "STGNN", "Adaptive"), ("hybrid", "STGNN", "Hybrid")]


def last_value(flow, mask):
    """MAE and RMSE of predicting x_t for t+1..t+4, over windows whose x_t and target both exist."""
    ends = window_ends(mask.sum())
    x = flow[mask]
    err_abs = err_sq = count = 0.0
    for h in range(1, CONFIG["out_len"] + 1):
        pred, target = x[ends], x[ends + h]
        ok = ~np.isnan(pred) & ~np.isnan(target)
        d = (pred - target)[ok].astype(np.float64)
        err_abs += np.abs(d).sum()
        err_sq += (d ** 2).sum()
        count += ok.sum()
    return float(err_abs / count), float(math.sqrt(err_sq / count))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/largest-sd-15min")
    args = ap.parse_args()
    runs = {m: json.load(open(f"results/runs/{m}/metrics.json")) for m in GRAPHS}

    shas = {r["test_ends_sha256"] for r in runs.values()}
    windows = {r["test_windows"] for r in runs.values()}
    if len(shas) != 1 or windows != {26389}:
        raise SystemExit(f"runs were not evaluated on the same test windows: {windows}, {len(shas)} hashes")
    configs = {r["config_sha256"] for r in runs.values()}
    if len(configs) != 1 or {r["mode"] for r in runs.values()} != {"full"}:
        raise SystemExit("runs used different settings or a smoke run is mixed in")

    flow, ts, ids = load(f"{args.data}/sd_2020_2021.npz")
    masks = split_masks(ts)
    floor = {s: last_value(flow, masks[s]) for s in ("val", "test")}
    # A trained model that loses to repeating the last reading is flagged in the output, not hidden.
    for m, r in runs.items():
        if r["val_mae"] >= floor["val"][0]:
            print(f"WARNING: {m} does not beat the last-value baseline on validation "
                  f"({r['val_mae']:.2f} vs {floor['val'][0]:.2f})")

    table = []
    for key, model, adjacency in ROWS:
        r = runs[key]
        table.append({"model": model, "adjacency": adjacency, "val_mae": r["val_mae"], "val_rmse": r["val_rmse"],
                      "test_mae": r["test_mae"], "test_rmse": r["test_rmse"], "fit_seconds": r["fit_seconds"],
                      "seconds_per_epoch": r["seconds_per_epoch"], "epochs_run": r["epochs_run"],
                      "best_epoch": r["best_epoch"], "parameters": r["parameters"], "alpha": r.get("alpha"),
                      "beats_last_value_val": bool(r["val_mae"] < floor["val"][0])})
    out = {"rows": table, "last_value": {"val_mae": floor["val"][0], "val_rmse": floor["val"][1],
                                         "test_mae": floor["test"][0], "test_rmse": floor["test"][1]},
           "device": runs["temporal"]["device"]}
    json.dump(out, open("results/comparison.json", "w"), indent=1)

    md = ["| Model | Adjacency | Test MAE | Test RMSE | Val MAE | Val RMSE | Training time (s) | Epochs (best) | Parameters |",
          "|---|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
           r"Model & Adjacency & MAE & RMSE & Val MAE & Val RMSE & Training time (s) & Epochs (best) & Params \\",
           r"\midrule"]
    for row in table:
        cells = [row["model"], row["adjacency"], f"{row['test_mae']:.2f}", f"{row['test_rmse']:.2f}",
                 f"{row['val_mae']:.2f}", f"{row['val_rmse']:.2f}", f"{row['fit_seconds']:.0f}",
                 f"{row['epochs_run']} ({row['best_epoch']})", f"{row['parameters']:,}"]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(cells).replace(",", "{,}") + r" \\")
    lv = out["last_value"]
    md.append(f"| Last value | None | {lv['test_mae']:.2f} | {lv['test_rmse']:.2f} | {lv['val_mae']:.2f} | {lv['val_rmse']:.2f} | – | – | 0 |")
    tex += [r"\midrule", f"Last value & None & {lv['test_mae']:.2f} & {lv['test_rmse']:.2f} & {lv['val_mae']:.2f} & "
            f"{lv['val_rmse']:.2f} & -- & -- & 0 \\\\", r"\bottomrule", r"\end{tabular}"]
    open("results/comparison.md", "w").write("\n".join(md) + "\n")
    open("results/comparison.tex", "w").write("\n".join(tex) + "\n")
    print("\n".join(md))

    forecast_plot(runs)


def forecast_plot(runs, start="2021-10-04", days=7):
    sel = json.load(open("results/selected_sensors.json"))
    preds = {m: np.load(f"results/runs/{m}/test_preds.npz") for m in GRAPHS}
    base = preds["temporal"]
    t_target = pd.to_datetime(base["end_timestamps"]) + pd.Timedelta(minutes=15)  # horizon-1 target time
    w = (t_target >= pd.Timestamp(start)) & (t_target < pd.Timestamp(start) + pd.Timedelta(days=days))
    style.apply()
    fig, axes = plt.subplots(3, 1, figsize=(style.WIDTH, 5.0), sharex=True)
    colours = {"temporal": "tab:gray", "fixed": "tab:blue", "adaptive": "tab:orange", "hybrid": "tab:green"}
    names = {"temporal": "temporal-only", "fixed": "fixed", "adaptive": "adaptive", "hybrid": "hybrid"}
    for k, ax in enumerate(axes):
        actual = np.where(base["mask"][w, 0, k], base["target"][w, 0, k], np.nan)
        ax.plot(t_target[w], actual, color="k", lw=1.0, label="actual")
        for m in GRAPHS:
            ax.plot(t_target[w], preds[m]["pred"][w, 0, k], color=colours[m], lw=0.6, alpha=0.9, label=names[m])
        style.tag(ax, "abc"[k])
        ax.set_ylabel("veh / 15 min")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=5, loc="upper left")
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d %b"))
    axes[-1].set_xlabel("Mon 4 to Sun 10 October 2021")
    fig.savefig("results/figures/q6a_forecasts.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
