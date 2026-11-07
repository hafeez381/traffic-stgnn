#!/usr/bin/env bash
# Package the committed code as a new version of the private Kaggle dataset abdul381/stgnn-code.
# Usage: kaggle/build_code.sh   (from anywhere in the repo, with a clean working tree)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
KAGGLE="${KAGGLE:-$HOME/.local/bin/kaggle}"

if [[ -n "$(git status --porcelain -- . ':!report')" ]]; then
  echo "working tree not clean: commit first, so results trace to a commit" >&2
  exit 1
fi

sha=$(git rev-parse HEAD)
git archive HEAD | gzip > kaggle/code/code.tgz
echo "$sha" > kaggle/code/COMMIT

if "$KAGGLE" datasets status abdul381/stgnn-code >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p kaggle/code -m "$sha" -q
else
  "$KAGGLE" datasets create -p kaggle/code -q
fi
# Kernels attach the latest version, so wait until this one is ready before pushing.
sleep 15
until [[ "$("$KAGGLE" datasets status abdul381/stgnn-code 2>/dev/null)" == *ready* ]]; do sleep 10; done
echo "uploaded $sha"
