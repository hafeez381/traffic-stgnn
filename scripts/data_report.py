"""Question 3: the extracted data and its road graph.

  (a) one week of flow at the three selected sensors; per-sensor missing shares
  (b) node and edge counts, symmetry, adjacency heatmap, neighbour-count distribution
  (c) one- and two-hop neighbourhoods of the median-degree sensor, and how repeated averaging
      over the graph flattens differences between sensors (the oversmoothing question)
Writes results/data_report.json and figures to results/figures/.
Usage: python -m scripts.data_report [--data DIR]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from stgnn.data import load, split_masks
from scripts import style
from stgnn.graph import normalize_fixed

FIG = "results/figures"


def week_plot(flow, ts, sel):
    start, end = pd.Timestamp("2020-02-03").value, pd.Timestamp("2020-02-10").value
    w = (ts >= start) & (ts < end)
    t = pd.to_datetime(ts[w])
    fig, ax = plt.subplots(figsize=(style.WIDTH, 2.2))
    for i, sensor_id, fwy, label in zip(sel["indices"], sel["ids"], sel["freeway"], ("quiet", "typical", "busy")):
        ax.plot(t, flow[w, i], lw=0.7, label=f"{sensor_id} ({fwy}, {label})")
    ax.set_ylabel("flow (vehicles / 15 min)")
    ax.set_xlabel("Mon 3 to Sun 9 February 2020")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d"))
    ax.legend(loc="upper left", ncol=3)
    ax.grid(alpha=0.3)
    fig.savefig(f"{FIG}/q3a_week.pdf", bbox_inches="tight")
    return int(np.isnan(flow[w][:, sel["indices"]]).sum())


def missing_plot(flow, masks):
    fig, ax = plt.subplots(figsize=(style.WIDTH / 2, 2.0))
    for name in ("train", "val", "test"):
        share = np.isnan(flow[masks[name]]).mean(0)
        ax.hist(share, bins=np.linspace(0, 1, 41), histtype="step", label=name)
    ax.set_yscale("log")
    ax.set_xlabel("share of 15-minute steps missing")
    ax.set_ylabel("sensors")
    ax.legend()
    fig.savefig(f"{FIG}/q3a_missing.pdf", bbox_inches="tight")


def graph_report(adj):
    off = adj.copy()
    np.fill_diagonal(off, 0)
    edges = off > 0
    out_deg, in_deg = edges.sum(1), edges.sum(0)
    asym = (off > 0) != (off.T > 0)
    rep = {
        "nodes": int(adj.shape[0]),
        "nonzero_entries_with_diagonal": int((adj > 0).sum()),
        "nonzero_edges_without_diagonal": int(edges.sum()),
        "diagonal_values": sorted(set(np.round(np.diag(adj), 6).tolist()))[:5],
        "symmetric_exact": bool(np.array_equal(adj, adj.T)),
        "symmetric_1e-6": bool(np.allclose(adj, adj.T, atol=1e-6)),
        "pattern_asymmetric_pairs": int(asym.sum() // 2),
        "weight_asymmetric_pairs_over_1e-6": int((np.abs(off - off.T) > 1e-6).sum() // 2),
        "max_abs_weight_difference": float(np.abs(off - off.T).max()),
        "weight_min_nonzero": float(off[edges].min()),
        "weight_max": float(off.max()),
        "out_degree": {"mean": float(out_deg.mean()), "median": float(np.median(out_deg)),
                       "min": int(out_deg.min()), "max": int(out_deg.max()), "isolated": int((out_deg == 0).sum())},
        "in_degree": {"mean": float(in_deg.mean()), "median": float(np.median(in_deg))},
    }

    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 2.6))
    im = axes[0].imshow(np.where(adj > 0, adj, np.nan), norm=LogNorm(vmin=0.01, vmax=1), cmap="viridis",
                        interpolation="nearest")
    style.tag(axes[0], "a")
    axes[0].set_xlabel("sensor $j$")
    axes[0].set_ylabel("sensor $i$")
    fig.colorbar(im, ax=axes, fraction=0.03, label="$A_{ij}$")
    axes[1].imshow(np.where(adj[:80, :80] > 0, adj[:80, :80], np.nan), norm=LogNorm(vmin=0.01, vmax=1),
                   cmap="viridis", interpolation="nearest")
    style.tag(axes[1], "b")
    axes[1].set_xlabel("sensor $j$")
    fig.savefig(f"{FIG}/q3b_adjacency.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(style.WIDTH / 2, 2.0))
    bins = np.arange(0, max(out_deg.max(), in_deg.max()) + 2) - 0.5
    ax.hist(out_deg, bins=bins, alpha=0.7, label="out-neighbours (row)")
    ax.hist(in_deg, bins=bins, histtype="step", color="k", label="in-neighbours (column)")
    ax.set_xlabel("number of neighbours (self excluded)")
    ax.set_ylabel("sensors")
    ax.legend()
    fig.savefig(f"{FIG}/q3b_degree.pdf", bbox_inches="tight")
    return rep, edges


def neighbourhood(edges, centre, meta, flow, ts_all, adj):
    one = set(np.flatnonzero(edges[centre]).tolist())
    reach2 = edges.astype(int) + edges.astype(int) @ edges.astype(int)
    two = set(np.flatnonzero(reach2[centre]).tolist()) - {centre}
    rep = {"sensor_index": centre, "sensor_id": int(meta.ID[centre]), "freeway": str(meta.Fwy[centre]),
           "one_hop": sorted(one), "one_hop_size": len(one), "two_hop_size": len(two),
           "two_hop_only_size": len(two - one), "one_hop_freeways": sorted(set(meta.Fwy[list(one)]))}

    fig, ax = plt.subplots(figsize=(style.WIDTH / 2, style.WIDTH / 2))
    ax.scatter(meta.Lng, meta.Lat, s=3, c="0.8", label="all SD sensors")
    ax.scatter(meta.Lng[list(two - one)], meta.Lat[list(two - one)], s=12, c="tab:orange", label="two-hop only")
    ax.scatter(meta.Lng[list(one)], meta.Lat[list(one)], s=12, c="tab:blue", label="one-hop")
    ax.scatter(meta.Lng[centre], meta.Lat[centre], s=80, marker="*", c="red", label=f"sensor {meta.ID[centre]}")
    lo, la = meta.Lng[list(two)], meta.Lat[list(two)]
    pad = 0.02
    ax.set_xlim(lo.min() - pad, lo.max() + pad)
    ax.set_ylim(la.min() - pad, la.max() + pad)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best")
    fig.savefig(f"{FIG}/q3c_neighbourhood.pdf", bbox_inches="tight")

    # Oversmoothing: apply the normalised graph k times to a real snapshot and watch the spread shrink.
    a_hat, _ = normalize_fixed(adj)
    snap = flow[np.flatnonzero(ts_all == pd.Timestamp("2021-04-06 17:00").value)[0]]  # Tuesday 17:00, test period
    x = np.where(np.isnan(snap), np.nanmean(snap), snap).astype(np.float64)
    spread = {}
    for k in range(0, 65):
        if k in (0, 1, 2, 4, 8, 16, 32, 64):
            spread[k] = float(x.std())
        x = a_hat @ x
    rep["oversmoothing_std_after_k_layers"] = spread
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/largest-sd-15min")
    args = ap.parse_args()
    flow, ts, ids = load(f"{args.data}/sd_2020_2021.npz")
    adj = np.load(f"{args.data}/sd_rn_adj.npy")
    meta = pd.read_csv(f"{args.data}/sd_meta.csv")
    assert list(meta.ID.astype(str)) == list(ids), "metadata and flow sensor order differ"
    sel = json.load(open("results/selected_sensors.json"))
    masks = split_masks(ts)

    style.apply()
    rep = {"flow_shape": list(flow.shape), "missing": json.load(open(f"{args.data}/missing_report.json"))}
    rep["week_plot_missing_points"] = week_plot(flow, ts, sel)
    missing_plot(flow, masks)
    rep["graph"], edges = graph_report(adj)
    rep["neighbourhood"] = neighbourhood(edges, sel["neighbourhood_sensor"]["index"], meta, flow, ts, adj)
    json.dump(rep, open("results/data_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "missing"}, indent=1))


if __name__ == "__main__":
    main()
