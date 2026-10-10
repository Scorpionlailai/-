# E22/W02 six-dataset transfer protocol

This protocol adds six datasets without changing the H/V/T implementation or
the two-stage E22/W02 training logic. Every formal run uses `seq96 -> pred96`,
seed 2021, `variable_missing`, and one of four locked missing scenarios.

## Dataset registry

| CLI dataset | Config directory | Variables | Expected rows | Train / validation / test boundaries | Input file |
|---|---|---:|---:|---|---|
| `ETTh2` | `configs/datasets/etth2/` | 7 | 17,420 | 0 / 8,640 / 11,520 / 14,400 | `data/raw/ETTh2.csv` |
| `ETTm1` | `configs/datasets/ettm1/` | 7 | 69,680 | 0 / 34,560 / 46,080 / 57,600 | `data/raw/ETTm1.csv` |
| `ETTm2` | `configs/datasets/ettm2/` | 7 | 69,680 | 0 / 34,560 / 46,080 / 57,600 | `data/raw/ETTm2.csv` |
| `Exchange` | `configs/datasets/exchange/` | 8 | 7,588 | 0 / 5,311 / 6,071 / 7,588 | `data/processed/exchange_rate.csv` |
| `Appliances` | `configs/datasets/appliances/` | 26 | 19,735 | 0 / 13,814 / 15,788 / 19,735 | `data/processed/appliances_energy.csv` |
| `BeijingAirQuality` | `configs/datasets/beijing_air_quality/` | 12 | 35,064 | 0 / 24,544 / 28,052 / 35,064 | `data/processed/beijing_pm25_12stations.csv` |

All boundaries are chronological. ETT follows the same fixed 12/4/4-month
family protocol used by the existing ETTh1 experiment. The other three use a
chronological 70/10/20 split.

## Locked missing scenarios

Each dataset directory contains exactly these four formal configurations:

| Config | `start_rate` | `block_len` | Role |
|---|---:|---:|---|
| `sr010_bl5.yaml` | 0.10 | 5 | low missing pressure |
| `sr010_bl8.yaml` | 0.10 | 8 | main E22/W02 pressure |
| `sr015_bl8.yaml` | 0.15 | 8 | high missing pressure |
| `sr020_bl8.yaml` | 0.20 | 8 | extreme missing pressure |

Do not create `sr010_bl3` in advance. Add it only for a dataset whose completed
`sr010_bl5` Clean--Missing gap is still judged large enough to require a lower
pressure point. The decision is made from the recorded test MSE and MAE gaps;
it is not an automatic mask-generation rule. ETTm1 triggered this condition on
2026-09-24: at `sr010_bl5`, Missing degraded from Clean by 14.81% MSE and 9.20%
MAE, so `configs/datasets/ettm1/sr010_bl3.yaml` was added as a dataset-specific
supplement. Exchange also triggered it on 2026-09-28: its `sr010_bl5` gap was
18.34% MSE and 9.88% MAE, so `configs/datasets/exchange/sr010_bl3.yaml` was
added. The other four datasets retain only the four core scenarios.

The small scenario files inherit their full E22 architecture from
`configs/datasets/_e22_common.yaml`. Loading a config resolves inheritance
before hashing, so manifests still contain the complete effective config hash.

## Data preparation

The official ETT CSV files already have the required `date` plus seven numeric
columns and can be placed directly under `data/raw/`.

### Local source provenance (2026-09-24)

The six files have been downloaded from their official/canonical repositories
and validated locally. Their immutable source hashes are:

| Dataset | Local source | SHA-256 |
|---|---|---|
| ETTh2 | `data/raw/ETTh2.csv` | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` |
| ETTm1 | `data/raw/ETTm1.csv` | `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e` |
| ETTm2 | `data/raw/ETTm2.csv` | `db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd` |
| Exchange | `data/downloads/exchange_rate.txt.gz` | `6a4fbb561dcdee43f071a83badbeec9223a335552278348656738b556f1c2e9d` |
| Appliances | `data/downloads/appliances_energy_prediction.zip` | `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a` |
| BeijingAirQuality | `data/downloads/beijing_multi_site_air_quality.zip` | `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4` |

The generated files have the following hashes:

| Dataset | Prepared file | SHA-256 |
|---|---|---|
| Exchange | `data/processed/exchange_rate.csv` | `dee1339f1423b046a69e0643f5c8c7605e4f34dfc1afbddd7d5e41eda1359973` |
| Appliances | `data/processed/appliances_energy.csv` | `3d0689d7312dcf97c2c45669d2f92e38b57699da08a14a1b0f0fb14795f9a6ff` |
| BeijingAirQuality | `data/processed/beijing_pm25_12stations.csv` | `1f342ca365d228aafe9c00f832b8ef4d7e617c88dc309ad4f1de08c08c267aee` |
| Beijing native mask | `data/processed/beijing_pm25_12stations.natural_missing.npz` | `7d3ffbb9fca649df6d982029bd0d00a1954f51f5d54e53b36398bf6a7cf89cf8` |

The other sources are converted deterministically:

```bash
PYTHONPATH=.:src conda run --no-capture-out -n malr \
  python -m malr.data.benchmark_prep Exchange \
  --source data/downloads/exchange_rate.txt.gz \
  --output data/processed/exchange_rate.csv

PYTHONPATH=.:src conda run --no-capture-out -n malr \
  python -m malr.data.benchmark_prep Appliances \
  --source data/downloads/appliances_energy_prediction.zip \
  --output data/processed/appliances_energy.csv

PYTHONPATH=.:src conda run --no-capture-out -n malr \
  python -m malr.data.benchmark_prep BeijingAirQuality \
  --source data/downloads/beijing_multi_site_air_quality.zip \
  --output data/processed/beijing_pm25_12stations.csv
```

Appliances removes only the published random controls `rv1` and `rv2`, leaving
26 meaningful channels. Beijing uses synchronized PM2.5 from 12 stations. For
the four artificial-missing experiments, native Beijing gaps are completed by
causal forward fill and then a training-only station median for leading gaps.
The original native observation mask is retained in
`beijing_pm25_12stations.natural_missing.npz`. Therefore these four runs must be
reported as artificial-missing experiments on a causal reference completion,
not yet as a real-missing recovery benchmark.

## Validation and artifact layout

Validate every prepared file before generating masks. For example:

```bash
PYTHONPATH=.:src conda run --no-capture-out -n malr python -m malr.cli validate-data \
  --config configs/datasets/ettm1/sr010_bl8.yaml \
  --dataset ETTm1
```

Use the existing artifact convention:

```text
artifacts/<Dataset>/seq96_pred96/sr<rate>_bl<length>/experiments/e22_transfer_v1/
```

For example, the ETTm1 main experiment belongs at:

```text
artifacts/ETTm1/seq96_pred96/sr010_bl8/experiments/e22_transfer_v1/
```

Clean is invariant across the four missing scenarios and should be trained and
cached once per dataset. Missing must be trained for every scenario. Ours is
trained from scratch on the main `sr010_bl8` protocol first; the other three
protocols can then record both zero-shot stress evaluation and, where required,
same-protocol retraining, following the existing E22/W02 procedure.

## Initial batch profiles

The initial batch sizes are 128 for ETTh2/ETTm1/ETTm2, 64 for Exchange, 16 for
Appliances, and 32 for BeijingAirQuality. These preserve the chosen PatchTST
profile while accounting for the approximately quadratic variable-candidate
cost in V. If a memory probe fails, lower only the runtime repair batch first;
do not silently alter H/V/T or the missing protocol.
