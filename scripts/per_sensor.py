"""Where the fixed-graph model loses to the temporal-only model, sensor by sensor.

Re-evaluates both saved checkpoints on the test windows and records each sensor's test MAE.
Each sensor's "level gap" is how far its mean training flow is from the row-normalised road-graph
average of its neighbours' mean flows (self included, as the graph layer averages). If the graph
layer pulls a sensor toward its neighbours, the fixed model's extra error grows with this gap.
Writes results/per_sensor.json and results/figures/q4c_per_sensor.pdf.
Usage: python -m scripts.per_sensor [--data DIR]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts import style
from stgnn.config import CONFIG
from stgnn.data import build_splits, load, split_masks
from stgnn.graph import normalize_fixed
from stgnn.train import make_model


def per_sensor_mae(model, split, scaler, batch=256):
    model.eval()
    abs_sum = torch.zeros(split.x.shape[1], dtype=torch.float64)
    count = torch.zeros_like(abs_sum)
    with torch.no_grad():
        for i in range(0, len(split.ends), batch):
            x, y, m = split.windows(split.ends[i:i + batch])
            err = (scaler.inverse(model(x)) - y).abs() * m
            abs_sum += err.sum(dim=(0, 1)).cpu().double()
            count += m.sum(dim=(0, 1)).cpu().double()
    return (abs_sum / count.clamp(min=1)).numpy(), count.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/largest-sd-15min")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    flow, ts, ids = load(f"{args.data}/sd_2020_2021.npz")
    adj = np.load(f"{args.data}/sd_rn_adj.npy")
    splits, scaler = build_splits(flow, ts, device)
    mae = {}
    for kind in ("temporal", "fixed"):
        model, _ = make_model(kind, adj, CONFIG)
        model.load_state_dict(torch.load(f"results/runs/{kind}/best.pt", map_location="cpu"))
        mae[kind], count = per_sensor_mae(model.to(device), splits["test"], scaler)

    a_hat, _ = normalize_fixed(adj)
    level = np.nanmean(flow[split_masks(ts)["train"]], axis=0)
    ok = (count > 0) & ~np.isnan(level)
    lv = np.where(np.isnan(level), np.nanmean(level), level)
    gap = np.abs(lv - a_hat @ lv)
    extra = mae["fixed"] - mae["temporal"]
    rho = float(np.corrcoef(gap[ok], extra[ok])[0, 1])
    order = np.argsort(-extra[ok])
    share_top10pct = float(np.sort(extra[ok])[::-1][: max(1, ok.sum() // 10)].sum() / extra[ok].sum())

    out = {
        "sensors_scored": int(ok.sum()),
        "self_weight_median": float(np.median(np.diag(a_hat))),
        "fixed_worse_share": float((extra[ok] > 0).mean()),
        "extra_mae_median": float(np.median(extra[ok])),
        "extra_mae_share_from_worst_10pct_sensors": share_top10pct,
        "level_gap_median": float(np.median(gap[ok])),
        "corr_level_gap_vs_extra_mae": rho,
        "worst5": [{"id": str(ids[np.flatnonzero(ok)[k]]), "temporal_mae": float(mae["temporal"][np.flatnonzero(ok)[k]]),
                    "fixed_mae": float(mae["fixed"][np.flatnonzero(ok)[k]]), "level": float(lv[np.flatnonzero(ok)[k]]),
                    "level_gap": float(gap[np.flatnonzero(ok)[k]])} for k in order[:5]],
    }
    json.dump(out, open("results/per_sensor.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

    style.apply()
    fig, ax = plt.subplots(figsize=(style.WIDTH / 2, 2.2))
    ax.scatter(gap[ok], extra[ok], s=3, alpha=0.6)
    ax.set_xlabel("level gap to graph-averaged neighbours (veh / 15 min)")
    ax.set_ylabel("fixed minus temporal-only test MAE")
    ax.axhline(0, color="k", lw=0.5)
    fig.savefig("results/figures/q4c_per_sensor.pdf")


if __name__ == "__main__":
    main()
