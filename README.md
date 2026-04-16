# STRATA ML Baselines — Subprocess Interface

External ML baselines (TimeGAN, QuantGAN, Diffusion-TS, ...) plug into STRATA
through a subprocess + JSON-spec contract. The main pipeline (`50_v2.py`)
saves data to disk, calls a runner script, reads back a numpy file of
generated paths, and evaluates those paths through STRATA's existing metric
pipeline. **STRATA computes all metrics — runners only generate paths.**
This guarantees that all baselines (classical + ML) are scored identically.

## Folder layout

```
project/
├── 50_v2.py                       # main STRATA pipeline
├── baselines/
│   ├── README.md                  # this file
│   ├── timegan_runner.py          # PLACEHOLDER — replace with real impl
│   └── <name>_runner.py           # add new baselines here
└── strata_outputs/
    └── baseline_io/
        ├── inputs/                # spec JSONs, returns .npy, option CSVs
        └── outputs/               # generated paths .npy, result JSONs
```

## Spec JSON (input contract)

STRATA writes one of these per (ticker, seed, baseline). Path is passed as `--spec`.

```json
{
  "baseline_name": "TimeGAN",
  "ticker": "SPY",
  "seed": 42,
  "s0": 685.14,
  "risk_free": 0.0,
  "horizon_days": 30,
  "n_paths_eval": 1500,
  "train_returns_path":  "strata_outputs/baseline_io/inputs/SPY_seed42_train.npy",
  "val_returns_path":    "strata_outputs/baseline_io/inputs/SPY_seed42_val.npy",
  "test_returns_path":   "strata_outputs/baseline_io/inputs/SPY_seed42_test.npy",
  "option_train_csv":    "strata_outputs/baseline_io/inputs/SPY_seed42_option_train.csv",
  "option_holdout_csv":  "strata_outputs/baseline_io/inputs/SPY_seed42_option_holdout.csv",
  "paths_output_path":   "strata_outputs/baseline_io/outputs/SPY_seed42_timegan_paths.npy",
  "result_json_path":    "strata_outputs/baseline_io/outputs/SPY_seed42_timegan_result.json"
}
```

## Returns convention (CRITICAL)

- `train/val/test_returns_path` are **raw per-day log-returns**, 1-D float arrays.
  They are NOT normalized. Train your model on these directly.
- The output paths .npy must contain **raw per-day log-returns**, shape
  `(n_paths, horizon_days)`. **Not** cumulative, **not** prices, **not** normalized.

If your model naturally produces prices, convert: `r_t = log(P_t / P_{t-1})`.

## Output contract

Two files, both at paths given in the spec:

1. **`paths_output_path`** (.npy, REQUIRED): float array `(n_paths, horizon_days)`
   of raw per-day log-returns. STRATA validates the shape and skips the baseline
   with a warning if it doesn't match.

2. **`result_json_path`** (.json, REQUIRED): metadata only. STRATA does NOT read
   metrics from this file — it computes its own from the paths .npy.
   ```json
   {
     "baseline_name": "TimeGAN",
     "ticker": "SPY",
     "seed": 42,
     "status": "ok",
     "generated_returns_path": "...",
     "shape": [1500, 30]
   }
   ```

## Adding a new baseline

1. Create `baselines/<n>_runner.py` (lowercase). Use `timegan_runner.py` as a template.
2. Run STRATA with `--ml_baselines TimeGAN <YourName>`. STRATA will call
   `python baselines/<yourname>_runner.py --spec <path>` for each (ticker, seed).
3. Failures (non-zero exit, missing output file, wrong shape, timeout) are logged
   as warnings. The sweep continues; the baseline is omitted from results for
   that run only.

## Failure modes STRATA handles automatically

- **Runner script missing**: warning, baseline skipped for the run.
- **Subprocess crashes** (non-zero return code): warning, skipped.
- **Timeout** (default 7200s = 2h per run): warning, skipped.
- **Missing or wrong-shape paths file**: warning, skipped.

## What runners must NOT do

- Don't compute or report your own KS, MMD, or RMSE metrics. STRATA does this.
- Don't normalize the output paths. Return raw log-returns.
- Don't write to any path other than the two given in the spec.
- Don't assume STRATA's Python environment. Activate your own venv if needed
  inside the runner (e.g. `subprocess` to a different python interpreter).

## Replacing the TimeGAN placeholder

`baselines/timegan_runner.py` ships as a placeholder that emits noise-matched
random walks (essentially GBM). It satisfies the contract but is **not**
TimeGAN. Reference implementations:

- Original (TF1): https://github.com/jsyoon0823/TimeGAN
- PyTorch port:   https://github.com/birdx0810/timegan-pytorch

Replace the `# GENERATE block` in `timegan_runner.py` with actual training
and sampling code. Numbers from the placeholder must NEVER appear in any
manuscript — STRATA logs `status: ok_placeholder` so it's grep-able.

