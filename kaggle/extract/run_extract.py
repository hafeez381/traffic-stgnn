"""Kaggle entry point: unpack the code dataset, log the machine, then extract LargeST San Diego."""
import glob
import os
import shutil
import subprocess
import sys
import tarfile

OUT = "/kaggle/working"
CODE = "/tmp/code"

found = glob.glob("/kaggle/input/**/code.tgz", recursive=True)
if len(found) != 1:
    sys.exit(f"expected exactly one code.tgz under /kaggle/input, found {found}")
os.makedirs(CODE, exist_ok=True)
with tarfile.open(found[0]) as tar:
    tar.extractall(CODE, filter="data")
shutil.copy(os.path.join(os.path.dirname(found[0]), "COMMIT"), OUT)
print("commit", open(os.path.join(OUT, "COMMIT")).read().strip(), flush=True)

with open(os.path.join(OUT, "machine.txt"), "w") as f:
    for cmd in (["lscpu"], ["free", "-g"], ["df", "-h", "/kaggle/working"],
                [sys.executable, "-c", "import numpy, pandas, h5py; print(numpy.__version__, pandas.__version__, h5py.__version__)"]):
        f.write("$ " + " ".join(cmd) + "\n" + subprocess.run(cmd, capture_output=True, text=True).stdout + "\n")
print(open(os.path.join(OUT, "machine.txt")).read(), flush=True)
subprocess.run(["ls", "-laR", "/kaggle/input"])

env = dict(os.environ, OUT_DIR=OUT, PYTHONPATH=CODE)
proc = subprocess.run([sys.executable, "-u", "-m", "scripts.extract_sd", "--mode", "full"], cwd=CODE, env=env)
sys.exit(proc.returncode)
