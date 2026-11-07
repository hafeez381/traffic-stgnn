"""Train one model (temporal-only, fixed, adaptive or hybrid graph) and write its results.

Writes to OUT_DIR (default results/runs/<model>):
  metrics.json      val/test MAE and RMSE in vehicles per 15 minutes, timings, epochs, parameters
  train_log.jsonl   one line per epoch
  best.pt           the best-validation checkpoint
  test_preds.npz    test forecasts and targets for the plotted sensors
  adjacency.npy     the learned matrix (adaptive and hybrid only)

Usage: python -m stgnn.train --model fixed --mode full --data /path/sd_2020_2021.npz --adj /path/sd_rn_adj.npy
       python -m stgnn.train --model hybrid --mode smoke    (synthetic data, a few batches, CPU)
"""
import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import torch

from stgnn.config import CONFIG, GRAPHS, SMOKE
from stgnn.data import build_splits, load, synthetic
from stgnn.graph import AdaptiveAdjacency, FixedAdjacency, HybridAdjacency, normalize_fixed
from stgnn.metrics import masked_mae, ErrorSums
from stgnn.model import STModel


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(kind, adj, cfg):
    torch.manual_seed(cfg["seed"])  # same initial encoder/head weights for every model
    n, h, e = cfg["num_nodes"], cfg["hidden"], cfg["embed_dim"]
    if kind == "temporal":
        graph, norm = None, None
    elif kind == "fixed":
        a_hat, norm = normalize_fixed(adj)
        graph = FixedAdjacency(a_hat)
    elif kind == "adaptive":
        graph, norm = AdaptiveAdjacency(n, e), None
    else:
        a_hat, norm = normalize_fixed(adj)
        graph = HybridAdjacency(a_hat, n, e)
    model = STModel(h, graph, cfg["out_len"])
    return model, norm


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def evaluate(model, split, scaler, device, batch, limit=None):
    """Error sums in original units over every window of the split, all horizons and sensors."""
    model.eval()
    sums = ErrorSums()
    ends = split.ends if limit is None else split.ends[:limit]
    with torch.no_grad():
        for i in range(0, len(ends), batch):
            x, y, m = split.windows(ends[i:i + batch])
            pred = scaler.inverse(model(x))
            sums.add(pred, y, m)
    return sums


def train_step(model, optimizer, split, t, scaler):
    # Only training windows may update weights, including the learned graph and alpha.
    if split.name != "train":
        raise RuntimeError(f"optimizer step on a {split.name} batch")
    model.train()
    x, y, m = split.windows(t)
    loss = masked_mae(model(x), scaler.transform(y), m)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=GRAPHS, required=True)
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--data")
    ap.add_argument("--adj")
    ap.add_argument("--sensors", default="results/selected_sensors.json")
    args = ap.parse_args()
    cfg = dict(CONFIG)
    smoke = args.mode == "smoke"
    if smoke:
        cfg.update(SMOKE)
    out = os.environ.get("OUT_DIR", f"results/runs/{args.model}")
    os.makedirs(out, exist_ok=True)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() and not smoke else "cpu")

    if smoke:
        flow, timestamps, ids, adj = synthetic(cfg["num_nodes"])
        sensors = [0, 1, 2]
    else:
        flow, timestamps, ids = load(args.data)
        adj = np.load(args.adj)
        sensors = json.load(open(args.sensors))["indices"]
    assert flow.shape[1] == cfg["num_nodes"] == adj.shape[0]
    splits, scaler = build_splits(flow, timestamps, device, cfg)
    model, norm = make_model(args.model, adj, cfg)
    model.to(device)
    fixed_before = model.graph.fixed().clone() if args.model == "hybrid" else \
        model.graph.a_hat.clone() if args.model == "fixed" else None
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params} parameters, normalisation {norm}, device {device}", flush=True)

    train = splits["train"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    best, best_epoch, waited, log = float("inf"), 0, 0, []
    max_eval = cfg.get("max_eval_windows")
    sync(device)
    fit_start = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        sync(device)
        t0 = time.perf_counter()
        order = train.ends[torch.randperm(len(train.ends), generator=gen).to(device)]
        n_batches = len(order) // cfg["batch_size"]
        if smoke:
            n_batches = min(n_batches, cfg["max_train_batches"])
        losses = [train_step(model, optimizer, train, order[i * cfg["batch_size"]:(i + 1) * cfg["batch_size"]], scaler)
                  for i in range(n_batches)]
        sync(device)
        train_s = time.perf_counter() - t0
        val = evaluate(model, splits["val"], scaler, device, cfg["eval_batch_size"], max_eval)
        sync(device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": val.mae, "val_rmse": val.rmse,
               "train_seconds": train_s, "epoch_seconds": time.perf_counter() - t0}
        if args.model == "hybrid":
            row["alpha"] = model.graph.alpha.item()
        log.append(row)
        print(json.dumps(row), flush=True)
        with open(os.path.join(out, "train_log.jsonl"), "a" if epoch > 1 else "w") as f:
            f.write(json.dumps(row) + "\n")
        if val.mae < best:
            best, best_epoch, waited = val.mae, epoch, 0
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break
    sync(device)
    fit_seconds = time.perf_counter() - fit_start

    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
    if fixed_before is not None:
        now = model.graph.fixed() if args.model == "hybrid" else model.graph.a_hat
        assert torch.equal(now, fixed_before), "the fixed road graph changed during training"
    val = evaluate(model, splits["val"], scaler, device, cfg["eval_batch_size"], max_eval)
    test = evaluate(model, splits["test"], scaler, device, cfg["eval_batch_size"], max_eval)
    test_split = splits["test"]

    # Forecasts for the plotted sensors over the whole test period (all horizons)
    preds, targets, masks = [], [], []
    model.eval()
    with torch.no_grad():
        ends = test_split.ends if max_eval is None else test_split.ends[:max_eval]
        for i in range(0, len(ends), cfg["eval_batch_size"]):
            x, y, m = test_split.windows(ends[i:i + cfg["eval_batch_size"]])
            preds.append(scaler.inverse(model(x))[:, :, sensors].cpu())
            targets.append(y[:, :, sensors].cpu())
            masks.append(m[:, :, sensors].cpu())
    np.savez_compressed(os.path.join(out, "test_preds.npz"), pred=torch.cat(preds).numpy(),
                        target=torch.cat(targets).numpy(), mask=torch.cat(masks).numpy(), sensors=np.array(sensors),
                        end_timestamps=test_split.timestamps[ends.cpu().numpy()])
    if args.model in ("adaptive", "hybrid"):
        g = model.graph if args.model == "adaptive" else model.graph.adaptive
        with torch.no_grad():
            np.save(os.path.join(out, "adjacency.npy"), g().cpu().numpy())

    metrics = {
        "model": args.model,
        "mode": args.mode,
        "normalisation": norm,
        "parameters": n_params,
        "epochs_run": len(log),
        "best_epoch": best_epoch,
        "fit_seconds": fit_seconds,
        "train_seconds": float(sum(r["train_seconds"] for r in log)),
        "seconds_per_epoch": float(np.mean([r["epoch_seconds"] for r in log])),
        "val_mae": val.mae, "val_rmse": val.rmse,
        "test_mae": test.mae, "test_rmse": test.rmse,
        "test_windows": int(len(test_split.ends)) if max_eval is None else int(min(max_eval, len(test_split.ends))),
        "test_ends_sha256": test_split.ends_sha256(),
        "config": cfg,
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "scaler": {"mean": scaler.mean, "std": scaler.std},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else device.type,
        "torch": torch.__version__,
    }
    if args.model == "hybrid":
        metrics["alpha"] = model.graph.alpha.item()
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
