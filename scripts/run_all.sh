#!/usr/bin/env bash
# Train the four models one after another with the same data, seed and protocol.
# Each run writes results/runs/<model>/ and records the code commit it ran from.
# Optional: BACKUP=<dir> copies each finished run there as a second copy.
# Usage: scripts/run_all.sh [models...]   (default: temporal fixed adaptive hybrid)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="${PYTHON:-python}"
DATA="${DATA:-data/largest-sd-15min}"
models=("${@:-temporal fixed adaptive hybrid}")
[[ -n "$(git status --porcelain -- stgnn scripts)" ]] && { echo "commit code changes first" >&2; exit 1; }
for m in ${models[@]}; do
  out="results/runs/$m"
  mkdir -p "$out"
  git rev-parse HEAD > "$out/COMMIT"
  "$PY" -c "import platform, torch; print(platform.platform(), platform.processor(), 'torch', torch.__version__, 'mps', torch.backends.mps.is_available())" > "$out/machine.txt"
  sysctl -n machdep.cpu.brand_string hw.memsize >> "$out/machine.txt" 2>/dev/null || true
  OUT_DIR="$out" "$PY" -u -m stgnn.train --model "$m" --mode full \
    --data "$DATA/sd_2020_2021.npz" --adj "$DATA/sd_rn_adj.npy" > "$out/stdout.log" 2>&1
  (cd "$out" && shasum -a 256 metrics.json train_log.jsonl best.pt test_preds.npz $( [[ -f adjacency.npy ]] && echo adjacency.npy) > sha256.txt)
  if [[ -n "${BACKUP:-}" ]]; then
    mkdir -p "$BACKUP/$m" && cp "$out"/* "$BACKUP/$m/" && (cd "$BACKUP/$m" && shasum -a 256 -c sha256.txt)
  fi
  echo "done $m"
done
