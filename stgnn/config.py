"""Every setting shared by the four models. Only the graph differs between runs."""

CONFIG = {
    # data (LargeST San Diego, 15-minute flow counts)
    "num_nodes": 716,
    "train": ("2020-01-01", "2021-01-01"),  # [start, end)
    "val": ("2021-01-01", "2021-04-01"),
    "test": ("2021-04-01", "2022-01-01"),
    "split_steps": {"train": 35136, "val": 8640, "test": 26400},
    "in_len": 8,
    "out_len": 4,
    # model
    "hidden": 32,
    "embed_dim": 10,
    # training
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 5,
    "seed": 2026,
    "eval_batch_size": 256,
}

SMOKE = {"max_epochs": 2, "batch_size": 8, "max_train_batches": 10, "max_eval_windows": 256}
GRAPHS = ("temporal", "fixed", "adaptive", "hybrid")
