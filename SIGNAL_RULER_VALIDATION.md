# Signal-ruler real-data validation — distributional oversmoothing reversal

Reproduced on real experiment artifacts (24 runs/model, distributional_oversmoothing).
Final-epoch values across all acc_* series:

| model    | n  | saturated (==1.0) | real middle values | verdict          |
|----------|----|-------------------|--------------------|------------------|
| deepseek | 72 | 60 (83%)          | 12                 | degenerate (S=0) |
| gpt-4o   | 72 | 60 (83%)          | 9                  | degenerate (S=0) |
| gpt-5.5  | 72 | 8 (11%)           | 55 (~0.6-0.69)     | real signal (S>0)|

Distinct per-model fingerprints confirm genuinely different data (not artifact).
The automated reviewer scored the saturated/degenerate run Overall 4 (highest) and
the real-signal run Overall 3 — the signal ruler inverts that, flagging saturated
runs as zero-signal. Fix required (real npy only): extract_numeric must accept
numpy-scalar series (np.number), else float32 metric series are silently dropped.
