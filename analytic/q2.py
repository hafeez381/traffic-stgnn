"""Question 2: direction of a moving slowdown on a 40-sensor ring.

  (a)i  least-squares fit of the symmetric layer to one isolated slowdown
  (a)ii roll that layer forward 20 steps from a Gaussian bump vs the true shift (heatmaps)
  (b)   direction-aware weights, as a circular 1D convolution
  (c)   symmetric weights plus the previous snapshot, under backward and forward motion
Exits non-zero if a check fails. Writes results/figures/q2_rollout.pdf.
Usage: python -m analytic.q2
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts import style

N = 40
behind = lambda x: np.roll(x, 1)   # value at i is x[i-1]
ahead = lambda x: np.roll(x, -1)   # value at i is x[i+1]
true_step = ahead                  # backward motion: x_i(t+1) = x_{i+1}(t)
forward_step = behind              # forward motion: x_i(t+1) = x_{i-1}(t)

assert behind(np.arange(N))[5] == 4 and ahead(np.arange(N))[5] == 6


def symmetric_layer(x, a, b):
    return a * x + b * (behind(x) + ahead(x))


def part_a_fit():
    x = np.zeros(N)
    x[20] = 1.0
    y = true_step(x)
    # Least squares over all 40 sensors: columns are the features multiplying a and b.
    features = np.stack([x, behind(x) + ahead(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(features, y, rcond=None)
    sse = np.sum((symmetric_layer(x, a, b) - y) ** 2)
    print(f"(a)i  least squares: a = {a:.6f}, b = {b:.6f}, SSE = {sse:.6f}")
    for i in (19, 20, 21):
        print(f"      sensor {i}: predicted {symmetric_layer(x, a, b)[i]:.3f}, correct {y[i]:.0f}")
    return a, b


def part_a_rollout(a, b, out="results/figures/q2_rollout.pdf"):
    i = np.arange(N)
    x0 = np.exp(-((i - 20) ** 2) / 8)
    pred, true = [x0], [x0]
    for _ in range(20):
        pred.append(symmetric_layer(pred[-1], a, b))
        true.append(true_step(true[-1]))
    pred, true = np.array(pred), np.array(true)
    assert all(np.argmax(true[t]) == (20 - t) % N for t in range(21))

    print("(a)ii  t   true peak   pred peak   pred max   pred sum   true max")
    for t in (0, 1, 2, 5, 10, 20):
        print(f"      {t:3d}   {np.argmax(true[t]):6d}   {np.argmax(pred[t]):9d}   {pred[t].max():8.3f}"
              f"   {pred[t].sum():8.3f}   {true[t].max():8.3f}")

    vmin, vmax = min(pred.min(), true.min()), max(pred.max(), true.max())
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 2.3), sharey=True)
    for ax, data, letter in zip(axes, (pred, true), "ab"):
        im = ax.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis",
                       extent=(-0.5, N - 0.5, -0.5, 20.5))
        style.tag(ax, letter)
        ax.set_xlabel("sensor index $i$")
    axes[0].set_ylabel("time step $t$")
    fig.colorbar(im, ax=axes, label="slowdown $x_i(t)$")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"      saved {out}")


def part_b(rng):
    a, w_minus, w_plus = 0.0, 0.0, 1.0
    for _ in range(100):
        x = rng.normal(size=N)
        assert np.max(np.abs(a * x + w_minus * behind(x) + w_plus * ahead(x) - true_step(x))) <= 1e-12
    conv = torch.nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular", bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.tensor([[[w_minus, a, w_plus]]])  # multiplies [x_{i-1}, x_i, x_{i+1}]
    x = rng.normal(size=N)
    out = conv(torch.tensor(x).float().view(1, 1, N)).detach().numpy().ravel()
    assert np.allclose(out, true_step(x), atol=1e-6)
    print(f"(b)   a = {a}, w- = {w_minus}, w+ = {w_plus}: exact on 100 random snapshots;"
          f" Conv1d kernel {[w_minus, a, w_plus]} with circular padding matches: PASS")


def history_layer(x_now, x_prev, a, b, c):
    return a * x_now + b * (behind(x_now) + ahead(x_now)) + c * x_prev


def part_c(rng):
    a, b, c = 0.0, 1.0, -1.0
    for name, step in (("backward", true_step), ("forward", forward_step)):
        worst = 0.0
        for _ in range(100):
            x_prev = rng.normal(size=N)
            x_now = step(x_prev)
            worst = max(worst, np.max(np.abs(history_layer(x_now, x_prev, a, b, c) - step(x_now))))
        print(f"(c)   a = {a}, b = {b}, c = {c}, {name} motion: max error {worst:.1e}")
        assert worst <= 1e-12


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a, b = part_a_fit()
    part_a_rollout(a, b)
    part_b(rng)
    part_c(rng)
