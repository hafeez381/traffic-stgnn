"""Questions 5(b) and 6(b): what the learned adjacency looks like and how it relates to the road graph.

For the adaptive run (and the learned part of the hybrid run), with each sensor's 10 strongest
learned neighbours, self excluded:
  overlap      share that are also road-graph neighbours, against the chance rate
  distance     great-circle distance of learned pairs vs road edges vs random pairs
  long range   share not in the road graph and farther than 4 km (LargeST's query radius)
  correlation  training-period flow correlation of learned pairs vs road edges vs random pairs
A row-shuffled copy of the learned matrix is a control: its overlap must sit near chance.
Writes results/graph_analysis.json and results/figures/q5b_*.pdf, q6b_*.pdf.
Usage: python -m scripts.graph_analysis [--data DIR]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from scripts import style
from stgnn.data import load, split_masks

K = 10
FIG = "results/figures"


def haversine_km(lat1, lng1, lat2, lng2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat, dlng = p2 - p1, np.radians(lng2 - lng1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlng / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def top_k(a, k=K):
    a = a.copy()
    np.fill_diagonal(a, -np.inf)
    idx = np.argsort(-a, axis=1)[:, :k]
    assert not np.any(idx == np.arange(len(a))[:, None]), "a self-edge entered the top-k"
    return idx


def flow_correlation(flow_train):
    x = np.where(np.isnan(flow_train), np.nanmean(flow_train, axis=0), flow_train)
    x = np.nan_to_num(x)  # sensors with no training data at all
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), np.inf)
    return (x.T @ x) / len(x)


def analyse(learned, road, dist, corr, rng):
    n = len(learned)
    idx = top_k(learned)
    rows = np.repeat(np.arange(n), K)
    cols = idx.ravel()
    in_road = road[rows, cols]
    chance = road.sum() / (n * (n - 1))

    shuffled = np.array([rng.permutation(r) for r in learned])
    sidx = top_k(shuffled)
    control = road[np.repeat(np.arange(n), K), sidx.ravel()].mean()

    road_r, road_c = np.nonzero(road)
    rand_r, rand_c = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    keep = rand_r != rand_c
    rand_r, rand_c = rand_r[keep], rand_c[keep]
    offdiag = learned[~np.eye(n, dtype=bool)]
    # strongest 1% of learned entries over the whole matrix
    thresh = np.quantile(offdiag, 0.99)
    sr, sc = np.nonzero((learned >= thresh) & ~np.eye(n, dtype=bool))
    long_range = (~in_road) & (dist[rows, cols] > 4.0)
    return {
        "row_sum_max_error": float(np.abs(learned.sum(1) - 1).max()),
        "max_row_entry_median": float(np.median(learned.max(1))),
        "uniform_level": 1.0 / n,
        "self_weight_median": float(np.median(np.diag(learned))),
        "top10_overlap_with_road": float(in_road.mean()),
        "top1_overlap_with_road": float(road[np.arange(n), idx[:, 0]].mean()),
        "strongest_1pct_overlap_with_road": float(road[sr, sc].mean()),
        "chance_overlap": float(chance),
        "shuffled_control_overlap": float(control),
        "top10_distance_km_median": float(np.median(dist[rows, cols])),
        "road_edge_distance_km_median": float(np.median(dist[road_r, road_c])),
        "random_pair_distance_km_median": float(np.median(dist[rand_r, rand_c])),
        "top10_share_within_4km": float((dist[rows, cols] <= 4.0).mean()),
        "top10_long_range_share": float(long_range.mean()),
        "top10_corr_median": float(np.median(corr[rows, cols])),
        "top10_long_range_corr_median": float(np.median(corr[rows[long_range], cols[long_range]])) if long_range.any() else None,
        "road_edge_corr_median": float(np.median(corr[road_r, road_c])),
        "random_pair_corr_median": float(np.median(corr[rand_r, rand_c])),
        "hub_in_degree_top5": np.bincount(cols, minlength=n)[np.argsort(-np.bincount(cols, minlength=n))[:5]].tolist(),
        "distinct_targets_in_top10": int(len(np.unique(cols))),
    }, (rows, cols, rand_r, rand_c, road_r, road_c)


def figures(name, learned, road, dist, meta, pairs):
    rows, cols, rand_r, rand_c, road_r, road_c = pairs
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH - 0.3, 2.6))
    im = axes[0].imshow(learned, norm=LogNorm(vmin=max(learned.min(), 1e-6), vmax=learned.max()),
                        cmap="viridis", interpolation="nearest")
    style.tag(axes[0], "a")
    axes[0].set_xlabel("sensor $j$")
    axes[0].set_ylabel("sensor $i$")
    fig.colorbar(im, ax=axes, fraction=0.03, label="learned weight")
    z = learned[:80, :80]
    axes[1].imshow(z, norm=LogNorm(vmin=max(z.min(), 1e-6), vmax=z.max()), cmap="viridis", interpolation="nearest")
    rr, cc = np.nonzero(road[:80, :80])
    axes[1].scatter(cc, rr, s=1, c="red", label="road-graph edge")
    style.tag(axes[1], "b")
    axes[1].legend(loc="lower left")
    fig.savefig(f"{FIG}/q5b_learned_adjacency_{name}.pdf", bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 2.4))
    bins = np.linspace(0, 80, 81)
    for (r, c, label, kw) in ((rand_r, rand_c, "random pairs", {"histtype": "step", "color": "0.4"}),
                              (road_r, road_c, "road-graph edges", {"alpha": 0.5}),
                              (rows, cols, "learned top-10", {"alpha": 0.5})):
        axes[0].hist(dist[r, c], bins=bins, density=True, label=label, **kw)
    axes[0].set_xlabel("distance between the two sensors (km)")
    axes[0].set_ylabel("density")
    axes[0].legend()
    style.tag(axes[0], "a")
    axes[1].scatter(meta.Lng, meta.Lat, s=2, c="0.75")
    strongest = np.argsort(-learned[rows, cols])[:300]
    for e in strongest:
        i, j = rows[e], cols[e]
        colour = "tab:blue" if road[i, j] else "tab:red"
        axes[1].plot([meta.Lng[i], meta.Lng[j]], [meta.Lat[i], meta.Lat[j]], color=colour, lw=0.6, alpha=0.7)
    axes[1].plot([], [], color="tab:blue", label="also a road edge")
    axes[1].plot([], [], color="tab:red", label="not in road graph")
    style.tag(axes[1], "b")
    axes[1].set_xlabel("longitude")
    axes[1].set_ylabel("latitude")
    axes[1].legend(loc="lower left")
    fig.savefig(f"{FIG}/q6b_learned_vs_road_{name}.pdf", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/largest-sd-15min")
    args = ap.parse_args()
    flow, ts, ids = load(f"{args.data}/sd_2020_2021.npz")
    meta = pd.read_csv(f"{args.data}/sd_meta.csv")
    assert list(meta.ID.astype(str)) == list(ids), "metadata and flow sensor order differ"
    adj = np.load(f"{args.data}/sd_rn_adj.npy")
    road = adj > 0
    np.fill_diagonal(road, False)
    lat, lng = meta.Lat.to_numpy(), meta.Lng.to_numpy()
    dist = haversine_km(lat[:, None], lng[:, None], lat[None, :], lng[None, :])
    corr = flow_correlation(flow[split_masks(ts)["train"]])  # training period only
    rng = np.random.default_rng(0)

    out = {}
    for name in ("adaptive", "hybrid"):
        learned = np.load(f"results/runs/{name}/adjacency.npy").astype(np.float64)
        out[name], pairs = analyse(learned, road, dist, corr, rng)
        figures(name, learned, road, dist, meta, pairs)
    alpha = json.load(open("results/runs/hybrid/metrics.json"))["alpha"]
    out["hybrid_alpha"] = alpha
    json.dump(out, open("results/graph_analysis.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    a = out["adaptive"]
    if a["max_row_entry_median"] < 2 * a["uniform_level"]:
        raise SystemExit("learned graph is close to uniform: the embeddings did not learn")
    if abs(a["shuffled_control_overlap"] - a["chance_overlap"]) > 3 * a["chance_overlap"]:
        raise SystemExit("shuffled control is far from chance: overlap code is wrong")


if __name__ == "__main__":
    main()
