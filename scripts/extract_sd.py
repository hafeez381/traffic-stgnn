"""Extract LargeST San Diego (District 11) traffic flow for 2020-2021 at 15-minute resolution.

Reads the California-wide raw files from the LargeST Kaggle dataset and writes to OUT_DIR:
  sd_2020_2021.npz   flow (T, 716) float32 with NaN for missing, timestamps (T,) int64 ns, ids (716,)
  sd_rn_adj.npy      (716, 716) road-distance adjacency, rows/columns in the same sensor order
  sd_meta.csv        LargeST metadata for the 716 sensors, same order
  missing_report.json, sha256.txt

PeMS 5-minute flow is a vehicle count per interval, so three 5-minute counts are summed into one
15-minute count. A 15-minute value is missing if any of its three readings is missing: a partial
sum would undercount.

Usage: python -m scripts.extract_sd --mode full   (on Kaggle, LargeST attached)
       python -m scripts.extract_sd --mode smoke  (local, on a small synthetic file in the same layout)
"""
import argparse
import glob
import hashlib
import json
import os
import tempfile
import time

import h5py
import numpy as np
import pandas as pd

YEARS = (2020, 2021)
STEPS_PER_DAY_5MIN = 288
ROW_CHUNK = 8192


def find(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {name} under {root}, found {hits}")
    return hits[0]


def describe_h5(path):
    """Print the HDF5 layout so the log records which read path was used."""
    with h5py.File(path, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"    {name}: shape={obj.shape} dtype={obj.dtype} chunks={obj.chunks} compression={obj.compression}")
        f.visititems(show)


def read_columns(path, ids):
    """Read the given sensor columns from a pandas HDF file, returning (values (T, len(ids)), index ns).

    For the pandas 'fixed' layout (axis0 = column labels, axis1 = row index, block0_values = data),
    rows are read in chunks and only the wanted columns kept, so the full CA table (about 7 GB) is
    never in memory at once. Any other layout falls back to pandas.
    """
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        group = f[keys[0]]
        if len(keys) == 1 and {"axis0", "axis1", "block0_items", "block0_values"} <= set(group.keys()) \
                and "block1_values" not in group:
            labels = [x.decode() if isinstance(x, bytes) else str(x) for x in group["block0_items"][:]]
            pos = {label: i for i, label in enumerate(labels)}
            cols = np.array([pos[i] for i in ids])
            # The stored integers' unit is in the 'kind' attribute (older pandas: plain datetime64 = ns).
            kind = group["axis1"].attrs["kind"].decode()
            unit = kind[kind.index("[") + 1:-1] if "[" in kind else "ns"
            index = group["axis1"][:].astype(f"datetime64[{unit}]").astype("datetime64[ns]").astype("int64")
            values = group["block0_values"]
            out = np.empty((values.shape[0], len(ids)), dtype=np.float32)
            for start in range(0, values.shape[0], ROW_CHUNK):
                block = values[start:start + ROW_CHUNK]
                out[start:start + len(block)] = block[:, cols]
            print(f"    read via h5py fixed layout, {len(labels)} columns in file")
            return out, index
    print("    read via pandas fallback")
    df = pd.read_hdf(path)
    df.columns = [str(c) for c in df.columns]
    return df[ids].to_numpy(np.float32), df.index.as_unit("ns").asi8


def check_grid(index, year):
    days = 366 if year % 4 == 0 else 365
    expect = days * STEPS_PER_DAY_5MIN
    assert len(index) == expect, f"{year}: {len(index)} rows, expected {expect}"
    assert index[0] == pd.Timestamp(f"{year}-01-01").value, f"{year} starts at {pd.Timestamp(index[0])}"
    steps = np.diff(index)
    assert np.all(steps == 5 * 60 * 10**9), f"{year}: index is not a regular 5-minute grid"


def to_15min(flow5):
    t = flow5.shape[0] // 3
    return flow5[: 3 * t].reshape(t, 3, -1).sum(axis=1)  # plain sum: one NaN makes the bin NaN


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def missing_stats(flow, timestamps, label):
    miss = np.isnan(flow)
    per_sensor = miss.mean(axis=0)
    # longest run of consecutive missing steps at any sensor
    longest = 0
    for j in range(flow.shape[1]):
        m = miss[:, j].astype(np.int8)
        if m.any():
            edges = np.flatnonzero(np.diff(np.concatenate(([0], m, [0]))))
            longest = max(longest, int((edges[1::2] - edges[::2]).max()))
    return {
        "label": label,
        "steps": int(flow.shape[0]),
        "missing_share": float(miss.mean()),
        "sensors_any_missing": int((per_sensor > 0).sum()),
        "sensors_over_20pct": int((per_sensor > 0.2).sum()),
        "sensors_all_missing": int((per_sensor == 1).sum()),
        "per_sensor_share_quantiles": {q: float(np.quantile(per_sensor, q)) for q in (0.5, 0.9, 0.99, 1.0)},
        "longest_gap_steps": longest,
        "first": str(pd.Timestamp(timestamps[0])),
        "last": str(pd.Timestamp(timestamps[-1])),
    }


def extract(src, out):
    os.makedirs(out, exist_ok=True)
    meta = pd.read_csv(find(src, "ca_meta.csv"))
    sd = meta[meta.District == 11].reset_index(drop=True)
    assert len(sd) == 716 or os.environ.get("SMOKE"), f"{len(sd)} SD sensors, expected 716"
    ids = sd.ID.astype(str).tolist()
    id2 = sd.ID2.to_numpy()

    flows5, stamps5, flows15, stamps15 = [], [], [], []
    for year in YEARS:
        path = find(src, f"ca_his_raw_{year}.h5")
        print(f"{year}: {path}")
        describe_h5(path)
        t0 = time.time()
        flow5, index = read_columns(path, ids)
        print(f"    {flow5.shape} in {time.time() - t0:.0f}s")
        check_grid(index, year)
        flows5.append(flow5)
        stamps5.append(index)
        flows15.append(to_15min(flow5))
        stamps15.append(index[::3])

    flow = np.concatenate(flows15)
    timestamps = np.concatenate(stamps15)
    assert flow.shape == (sum(len(s) for s in stamps15), len(ids))
    npz = os.path.join(out, "sd_2020_2021.npz")
    np.savez_compressed(npz, flow=flow, timestamps=timestamps, ids=np.array(ids))

    # Adjacency rows/columns come from ID2 of the same metadata rows that ordered the flow columns.
    ca_adj = np.load(find(src, "ca_rn_adj.npy"), mmap_mode="r")
    adj = np.asarray(ca_adj[np.ix_(id2, id2)], dtype=np.float32)
    np.save(os.path.join(out, "sd_rn_adj.npy"), adj)
    sd.to_csv(os.path.join(out, "sd_meta.csv"), index=False)

    flow5 = np.concatenate(flows5)
    ts5 = np.concatenate(stamps5)
    report = {
        "5min_all": missing_stats(flow5, ts5, "5-minute, 2020-2021"),
        "15min_all": missing_stats(flow, timestamps, "15-minute, 2020-2021"),
    }
    splits = {"train": ("2020-01-01", "2021-01-01"), "val": ("2021-01-01", "2021-04-01"),
              "test": ("2021-04-01", "2022-01-01")}
    for name, (a, b) in splits.items():
        sel = (timestamps >= pd.Timestamp(a).value) & (timestamps < pd.Timestamp(b).value)
        report[f"15min_{name}"] = missing_stats(flow[sel], timestamps[sel], f"15-minute, {name}")
    # The raw count scale: weekday 17:00 median across sensors in February 2020, both resolutions
    feb = pd.DatetimeIndex(ts5)
    sel5 = (feb.month == 2) & (feb.year == 2020) & (feb.dayofweek < 5) & (feb.hour == 17) & (feb.minute == 0)
    report["scale_check"] = {"median_5min_count_weekday_17h_feb2020": float(np.nanmedian(flow5[sel5]))}
    feb15 = pd.DatetimeIndex(timestamps)
    sel15 = (feb15.month == 2) & (feb15.year == 2020) & (feb15.dayofweek < 5) & (feb15.hour == 17) & (feb15.minute == 0)
    report["scale_check"]["median_15min_sum_weekday_17h_feb2020"] = float(np.nanmedian(flow[sel15]))
    json.dump(report, open(os.path.join(out, "missing_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))

    with open(os.path.join(out, "sha256.txt"), "w") as f:
        for name in ("sd_2020_2021.npz", "sd_rn_adj.npy", "sd_meta.csv", "missing_report.json"):
            f.write(f"{sha256(os.path.join(out, name))}  {name}\n")
    print(open(os.path.join(out, "sha256.txt")).read())


def make_smoke_source(root, n_ca=40, n_sd=12):
    """A tiny stand-in for the LargeST files in the same layout (fixed-format HDF, npy, csv)."""
    rng = np.random.default_rng(0)
    ids = rng.choice(np.arange(100000, 999999), n_ca, replace=False)
    meta = pd.DataFrame({"ID": ids, "Lat": rng.uniform(32.5, 33.2, n_ca), "Lng": rng.uniform(-117.3, -116.9, n_ca),
                         "District": [11] * n_sd + [7] * (n_ca - n_sd), "ID2": np.arange(n_ca)})
    meta = meta.sample(frac=1, random_state=0)  # SD rows not first, ID2 not in row order
    meta.to_csv(os.path.join(root, "ca_meta.csv"), index=False)
    np.save(os.path.join(root, "ca_rn_adj.npy"), rng.random((n_ca, n_ca)) * (rng.random((n_ca, n_ca)) < 0.2))
    for year in YEARS:
        days = 366 if year % 4 == 0 else 365
        idx = pd.date_range(f"{year}-01-01", periods=days * STEPS_PER_DAY_5MIN, freq="5min")
        vals = rng.poisson(100, (len(idx), n_ca)).astype(float)
        vals[rng.random(vals.shape) < 0.02] = np.nan
        pd.DataFrame(vals, index=idx, columns=ids).to_hdf(os.path.join(root, f"ca_his_raw_{year}.h5"), key="t", mode="w")


def check_smoke(src, out):
    """Every extracted column must equal the pandas read of the same sensor, summed in threes."""
    data = np.load(os.path.join(out, "sd_2020_2021.npz"))
    adj = np.load(os.path.join(out, "sd_rn_adj.npy"))
    meta = pd.read_csv(os.path.join(src, "ca_meta.csv"))
    ca_adj = np.load(os.path.join(src, "ca_rn_adj.npy"))
    sd = meta[meta.District == 11].reset_index(drop=True)
    df = pd.concat([pd.read_hdf(os.path.join(src, f"ca_his_raw_{y}.h5")) for y in YEARS])
    df.columns = [str(c) for c in df.columns]
    for j, sensor in enumerate(data["ids"]):
        expect = df[sensor].to_numpy().reshape(-1, 3).sum(axis=1)
        np.testing.assert_allclose(data["flow"][:, j], expect, rtol=1e-6)
    np.testing.assert_allclose(adj, ca_adj[np.ix_(sd.ID2, sd.ID2)], rtol=1e-6)
    assert list(data["ids"]) == sd.ID.astype(str).tolist()
    print("smoke: columns, 15-minute sums and adjacency order match pandas: PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--src", default="/kaggle/input")
    args = ap.parse_args()
    out = os.environ.get("OUT_DIR", "/kaggle/working")
    if args.mode == "smoke":
        os.environ["SMOKE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
            os.makedirs(src)
            make_smoke_source(src)
            extract(src, out)
            check_smoke(src, out)
    else:
        extract(args.src, out)
