| Model | Adjacency | Test MAE | Test RMSE | Val MAE | Val RMSE | Training time (s) | Epochs (best) | Parameters |
|---|---|---|---|---|---|---|---|---|
| Temporal-only | None | 65.68 | 104.33 | 58.12 | 95.58 | 1211 | 13 (8) | 6,500 |
| STGNN | Fixed | 140.96 | 218.04 | 124.93 | 199.64 | 4583 | 50 (49) | 6,500 |
| STGNN | Adaptive | 82.70 | 130.72 | 65.55 | 110.95 | 4883 | 50 (50) | 20,820 |
| STGNN | Hybrid | 77.25 | 122.24 | 61.42 | 104.62 | 4672 | 50 (50) | 20,821 |
| Last value | None | 87.08 | 132.76 | 81.16 | 124.91 | – | – | 0 |
