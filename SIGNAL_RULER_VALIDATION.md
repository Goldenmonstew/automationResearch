# Signal-ruler real-data validation — distributional oversmoothing reversal

Reproduced on real experiment artifacts (`fullrun/{model}`, 24 runs each,
`distributional_oversmoothing`). Final-epoch values across all `acc_*` series:

| model    | n  | ==1.0 (saturated) | real middle values | verdict          |
|----------|----|-------------------|--------------------|------------------|
| deepseek | 72 | 60 (83%)          | 12                 | degenerate (S=0) |
| gpt-4o   | 72 | 60 (83%)          | 9                  | degenerate (S=0) |
| gpt-5.5  | 72 | 8 (11%)           | 55 (~0.6-0.69)     | real signal (S>0)|

Distinct per-model fingerprints (a4f7c211 / 2e8c19d4 / 7d3e9a04) confirm the data
is genuinely different per model (not a display artifact).

The automated reviewer scored the saturated/degenerate deepseek run Overall 4
(highest) and the real-signal gpt-5.5 run Overall 3 — score and signal inverted.
The signal ruler flags the saturated runs as zero-signal, correcting the inversion.

Required fix (found via this real data, not visible on constructed samples):
`extract_numeric` must accept numpy-scalar series (`np.number`), otherwise real
`.npy` metric series stored as lists of `np.float32` are silently dropped
(commit: numpy-scalar metric extraction fix).
