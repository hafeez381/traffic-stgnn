# Graph Aggregation and Spatio-Temporal Traffic Forecasting on LargeST San Diego

Two parts. `analytic/` checks what a sum, mean or max of neighbour messages can compute, and how a graph layer can learn the direction of a moving slowdown on a ring road. `stgnn/` forecasts the next hour of 15-minute traffic flow at the 716 San Diego sensors of LargeST from the previous two hours, with a temporal convolution encoder, one graph convolution and a linear head, under four graphs: none, the road-distance graph, a learned graph and a learned mixture of the two.

| Model | Graph | Test MAE | Test RMSE | Val MAE | Training time (min) | Epochs (best) |
|---|---|---|---|---|---|---|
| Temporal-only | None | 65.7 | 104.3 | 58.1 | 20 | 13 (8) |
| STGNN | Fixed road graph | 141.0 | 218.0 | 124.9 | 76 | 50 (49) |
| STGNN | Adaptive | 82.7 | 130.7 | 65.5 | 81 | 50 (50) |
| STGNN | Hybrid (α = 0.22) | 77.3 | 122.2 | 61.4 | 78 | 50 (50) |
| Last value | None | 87.1 | 132.8 | 81.2 | – | – |

Errors are in vehicles per 15 minutes, averaged over the four horizons and all observed readings of the test period (April–December 2021). Missing readings are excluded through a mask.

## Setup

```bash
conda create -n dl python=3.12 && conda activate dl
pip install torch numpy pandas h5py matplotlib pytest
python -m pytest -q tests
```

The runs used Python 3.12, PyTorch 2.15.0, NumPy 2.5.4 and pandas 3.1.1 locally, and NumPy 2.1.3, pandas 2.3.3 and h5py 3.14.0 on Kaggle for the extraction. Every entry point takes `--mode smoke` (synthetic data of the same shape, a few batches, CPU) or `--mode full`. The seed is 2026.

## Compute

- **Extraction:** a Kaggle CPU kernel (4 cores, 31 GB) with the LargeST dataset attached, since each yearly California file is about 7 GB. Its output is the private Kaggle dataset `abdul381/largest-sd-15min`.
- **Training and evaluation:** an Apple M1 (8 GB) through the PyTorch MPS backend, about 95 s per epoch. Checkpoints, logs, metrics and test predictions are in `results/runs/<model>/`.

## Reproducing the results

### Analytic checks

```bash
python -m analytic.q1     # sum, mean and max over all 164 neighbourhoods
python -m analytic.q2     # ring-road fits, rollout heatmaps, direction checks
```

### Extract LargeST San Diego

```bash
python -m scripts.extract_sd --mode smoke            # local check on a synthetic file
kaggle/build_code.sh                                 # package the committed code as a Kaggle dataset
kaggle kernels push -p kaggle/extract
kaggle kernels output abdul381/stgnn-extract -p data/largest-sd-15min
```

The kernel writes `sd_2020_2021.npz` (15-minute flow, NaN for missing), `sd_rn_adj.npy`, `sd_meta.csv`, `missing_report.json` and `sha256.txt`.

### Data and graph statistics

```bash
python -m scripts.select_sensors
python -m scripts.data_report
```

### Train the four models

```bash
scripts/run_all.sh                         # temporal, fixed, adaptive, hybrid in sequence
python -m stgnn.train --model hybrid --mode full \
  --data data/largest-sd-15min/sd_2020_2021.npz --adj data/largest-sd-15min/sd_rn_adj.npy
```

### Tables and figures

```bash
python -m scripts.compare          # comparison table, forecast plots
python -m scripts.graph_analysis   # learned adjacency, overlap and distance analysis
python -m scripts.per_sensor       # per-sensor error of fixed vs temporal-only
```

## Attribution

- **Data:** LargeST (Liu et al., 2023), CC BY-NC 4.0, from Kaggle `liuxu77/largest`. The San Diego subset follows the repository's `generate_sd_dataset.ipynb` (District 11, adjacency indexed by `ID2`).
- **Libraries:** PyTorch, NumPy, pandas, h5py, matplotlib.
- **Method:** the adaptive adjacency Softmax(ReLU(E₁E₂ᵀ)) is from Graph WaveNet (Wu et al., 2019).

No code was copied from public implementations. Code was written with LLM assistance.

## References

- Liu et al. *LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting.* NeurIPS Datasets and Benchmarks, 2023.
- Wu et al. *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* IJCAI, 2019.
- Kipf and Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR, 2017.
- Xu et al. *How Powerful are Graph Neural Networks?* ICLR, 2019.
