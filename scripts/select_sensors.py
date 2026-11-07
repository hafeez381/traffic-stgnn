"""Pick the sensors that every plot uses, by a fixed rule rather than by hand.

Three flow sensors: rank sensors by mean 2020 flow and take the ones nearest the 10th, 50th and
90th percentiles (quiet, typical, busy), among sensors with at most 20% missing in both the
training and the test period. One neighbourhood sensor: the median out-degree in the road graph.
Writes results/selected_sensors.json.
Usage: python -m scripts.select_sensors [--data DIR]
"""
import argparse
import json

import numpy as np
import pandas as pd

from stgnn.data import load, split_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/largest-sd-15min")
    args = ap.parse_args()
    flow, ts, ids = load(f"{args.data}/sd_2020_2021.npz")
    adj = np.load(f"{args.data}/sd_rn_adj.npy")
    masks = split_masks(ts)
    train, test = flow[masks["train"]], flow[masks["test"]]
    ok = (np.isnan(train).mean(0) <= 0.2) & (np.isnan(test).mean(0) <= 0.2)
    mean = np.full(flow.shape[1], np.nan)
    mean[ok] = np.nanmean(train[:, ok], axis=0)
    eligible = np.flatnonzero(ok)
    picks = []
    for q in (0.1, 0.5, 0.9):
        target = np.quantile(mean[eligible], q)
        picks.append(int(eligible[np.argmin(np.abs(mean[eligible] - target))]))

    a = adj.copy()
    np.fill_diagonal(a, 0)
    deg = (a > 0).sum(axis=1)
    med = np.median(deg[ok])
    centre = int(eligible[np.argmin(np.abs(deg[eligible] - med))])

    meta = pd.read_csv(f"{args.data}/sd_meta.csv")
    out = {
        "indices": picks,
        "ids": [str(ids[i]) for i in picks],
        "labels": ["10th percentile flow", "median flow", "90th percentile flow"],
        "mean_2020_flow": [float(mean[i]) for i in picks],
        "freeway": [str(meta.Fwy[i]) for i in picks],
        "neighbourhood_sensor": {"index": centre, "id": str(ids[centre]), "out_degree": int(deg[centre]),
                                 "median_out_degree": float(med)},
        "rule": "nearest to 10/50/90th percentile of mean 2020 flow among sensors with <=20% missing in train and test",
    }
    json.dump(out, open("results/selected_sensors.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
