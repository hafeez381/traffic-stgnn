"""Question 1: what a sum of neighbour messages can and cannot remember.

Runs three checks and exits non-zero if any fails:
  (a) affine messages: two neighbourhoods with equal size and total trend give equal sums
  (b) the place-value message recovers every (slowing, steady, speeding) count triple for K = 8
  (c) brute-force verdicts for the sum / mean / max table, with a witness pair for every "No"
Usage: python -m analytic.q1
"""
import sys
from fractions import Fraction

import numpy as np

K = 8
TRENDS = (-1, 0, 1)


def neighbourhoods(k=K):
    """Every allowed neighbourhood as a count triple (slowing, steady, speeding), 1 <= size <= k."""
    return [(s, z, p) for s in range(k + 1) for z in range(k + 1) for p in range(k + 1) if 1 <= s + z + p <= k]


def expand(counts):
    s, z, p = counts
    return [-1] * s + [0] * z + [1] * p


# ---------- (a) ----------
def check_affine():
    rng = np.random.default_rng(0)
    pair = ([-1, 1], [0, 0])  # n = 2 and S = 0 for both, but 0 and 2 steady neighbours
    for _ in range(3):
        w, c = rng.normal(size=2)
        sums = [sum(w * x + c for x in nb) for nb in pair]
        closed = [w * sum(nb) + c * len(nb) for nb in pair]
        assert np.isclose(sums[0], sums[1]) and np.allclose(sums, closed), (w, c, sums)
    print(f"(a) pair {pair}: equal aggregated message for 3 random (w, c): PASS")


# ---------- (b) ----------
def place_value_message(base):
    return {-1: 1, 0: base, 1: base**2}


def decode(total, base):
    rest, s = divmod(total, base)
    p, z = divmod(rest, base)
    return s, z, p


def check_place_value():
    triples = neighbourhoods()
    assert len(triples) == 164, len(triples)
    phi = place_value_message(K + 1)
    sums = [sum(phi[x] for x in expand(t)) for t in triples]
    assert len(set(sums)) == len(triples), "two neighbourhoods share a sum"
    assert all(decode(total, K + 1) == t for total, t in zip(sums, triples)), "decoding failed"
    print(f"(b) phi = {phi}: {len(triples)} triples, all sums distinct, all decoded: PASS")

    # One base smaller is not enough: K slowing neighbours carry into the steady digit.
    small = place_value_message(K)
    seen = {}
    for t in triples:
        total = sum(small[x] for x in expand(t))
        if total in seen:
            print(f"    base K = {K} collides: {seen[total]} and {t} both sum to {total}")
            break
        seen[total] = t
    else:
        sys.exit("expected a collision with base K")

    nonsteady = {x: x * x for x in TRENDS}
    # No size bound is needed; checked here up to 30 neighbours.
    for t in neighbourhoods(30):
        assert sum(nonsteady[x] for x in expand(t)) == t[0] + t[2]
    print(f"(b) non-steady counter phi(x) = x^2 = {nonsteady}: PASS")


# ---------- (c) ----------
ONE_HOT = {-1: (1, 0, 0), 0: (0, 1, 0), 1: (0, 0, 1)}


def aggregate(counts, method):
    s, z, p = counts
    n = s + z + p
    if method == "sum":
        return (s, z, p)
    if method == "mean":
        return (Fraction(s, n), Fraction(z, n), Fraction(p, n))
    return tuple(int(c > 0) for c in counts)  # max of one-hot vectors = presence


TASKS = {
    "(i) count steady": lambda t: t[1],
    "(ii) more than half slowing": lambda t: 2 * t[0] > sum(t),
    "(iii) any slowing": lambda t: t[0] > 0,
}


def possible(task, method):
    """True if the aggregated vector always determines the answer; else (False, witness pair)."""
    answer = TASKS[task]
    groups = {}
    for t in neighbourhoods():
        groups.setdefault(aggregate(t, method), {}).setdefault(answer(t), t)
    for vec, answers in groups.items():
        if len(answers) > 1:
            (a1, t1), (a2, t2) = list(answers.items())[:2]
            return False, (vec, t1, a1, t2, a2)
    return True, None


def check_table():
    for task in TASKS:
        for method in ("sum", "mean", "max"):
            ok, witness = possible(task, method)
            if ok:
                print(f"(c) {task:30s} {method:4s}  Yes")
            else:
                vec, t1, a1, t2, a2 = witness
                vec = tuple(str(v) for v in vec)
                print(f"(c) {task:30s} {method:4s}  No   e.g. {t1}->{a1} vs {t2}->{a2}, both give {vec}")


if __name__ == "__main__":
    check_affine()
    check_place_value()
    check_table()
