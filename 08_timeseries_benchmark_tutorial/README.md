# Time series benchmarking tutorial (Hydrology CAMELS US with Chronos 2)

<h3 align="center"><strong>DS5110 Final Project Example (Hydrology)</strong></h3>
<p align="center"><strong>Junyang He</strong></p>

---

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Methodology](#methodology)
- [Experiment Process](#experiment-process)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Findings and Discussion](#findings-and-discussion)
- [Data Quality and Possible Anomalies](#data-quality-and-possible-anomalies)
- [How to Set the Project Environment and Replicate Results](#how-to-set-the-project-environment-and-replicate-results)
- [Dataset and Model Links](#dataset-and-model-links)
- [Conclusion and Future Improvements](#conclusion-and-future-improvements)

---

## Introduction

This project benchmarks zero-shot timeseries forecasting performance on CAMELS-US hydrology data using Chronos-2 foundation model from Amazon. The objective is to build a reproducible, machine-readable benchmark pipeline that runs in Google Colab to verify the CAMELS-US dataset and quantify the capabilities of the Chronos-2 model.

The project evaluates how well a timeseries foundation model (Chronos-2) can forecast next-day streamflow across 671 US river basins. This study only explores univariate forecasting, which forecasts streamflow `QObs(mm/d)` only with prior available streamflow data within lookback window. We do not leverage any additional exogenous streams or static features in input.

Task definition:
- Forecast type: single-step forecasting
- Forecast horizon: 1 day
- Target variable: `QObs(mm/d)`
- Input mode: univariate (past target only)
- Lookback window: 30 days
- Evaluation metric: RMSE
- Reporting: one overall RMSE on held-out test locations

## Data

Dataset source:
- CAMELS: Catchment Attributes and MEteorology for Large-sample Studies
- Dataset page: `https://zenodo.org/records/15529996`

Input file used in this repository:
- `data/raw/BasicInputTimeSeries_us.csv`

Expected key columns:
- `Year_Mnth_Day` (timestamp)
- `basin_id` (location/time-series identifier)
- `QObs(mm/d)` (streamflow target)
- unnamed first column (dropped in preprocessing)

For streamflow, we selected the interval of 7,031 days, spanning from October 2, 1989, to December 31, 2008. This largely aligns with the start date of a water year (October 1st), as defined by the U.S. Geological Survey. In preprocessing, we parse timestamps from `Year_Mnth_Day`, sort by `basin_id` and timestamp, drop the unnamed index-like first column, remove rows missing required time/target fields, and forward-fill missing values within each basin. The benchmark uses strictly univariate input (`QObs(mm/d)` history) and a location holdout split by `basin_id` (80% train locations, 20% test locations, random seed 42). An analysis of NaNs (missing values) in data revealed no NaN values in time series data, yet some exists in the static exogenous features. Since this benchmark does not involve static exogenous features, we do not perform data interpolation or categorical encoding on static features.

Note: train test split is not required since we are doing inference only (no training), we evaluate on 20% test data split by location for the purpose of demonstration.

## Methodology

Model:
- Chronos-2 (`amazon/chronos-2`) for zero-shot forecasting

Inference protocol:
- rolling one-step prediction with past-only context
- context window: 30 steps
- for held-out locations with no prior context, the evaluation loop optionally seeds with early test history (`allow_test_cold_start=True`) before scoring subsequent points

Metric:
- RMSE:
  - `RMSE = sqrt(mean((y_true - y_pred)^2))`

## Experiment Process

1. Install dependencies in Colab.
2. Load raw CSV and validate schema.
3. Preprocess and save processed artifacts to `data/processed/`.
4. Create location-based train/test split by `basin_id`.
5. Run Chronos-2 rolling one-step inference.
6. Compute and report RMSE on test evaluation points.
7. Save outputs to standardized JSON/CSV and figures.

## Repository Structure

```text
timeseries_benchmark_tutorial/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── metadata/
│   ├── experiment_config.json
│   └── ts_characteristics.json
├── results/
│   ├── benchmark_results.json
│   ├── predictions_test.csv
│   └── figures/
├── notebooks/
│   ├── main.ipynb
│   └── final_benchmark_colab.ipynb
└── src/
    ├── data_utils.py
    ├── eval_utils.py
    ├── plotting_utils.py
    └── metrics.py
```

## Results

| Quantity | Value |
|---|---|
| Dataset | CAMELS-US |
| Model | chronos-2 |
| Target variable | `QObs(mm/d)` |
| Lookback window | 30 |
| Forecast horizon | 1 |
| RMSE | **2.451361768118934** |
### Actual vs Predicted Visualization

![Actual vs Predicted Visualization](results/figures/actual_vs_predicted_visualization.png)

Note:
- Before execution, `benchmark_results.json` contains template values (`rmse: null`).
- After execution, notebook cells overwrite with run outputs.

## Findings and Discussion

The benchmark achieved an overall **RMSE of 2.4514** on the location-holdout test set (`20%` unseen basins), indicating that Chronos-2 captures the dominant streamflow dynamics reasonably well in a zero-shot setting. From the Actual vs Predicted visualization, the model captures recurring seasonal patterns and baseline fluctuations closely, with predicted values generally following the same temporal structure as observations. The main error appears during sharp peak-flow events, where predictions are smoother and tend to under-estimate extreme spikes. This is consistent with the observed spread around high-flow periods and likely contributes most of the RMSE. Overall, the results suggest strong trend-level generalization across unseen locations, with remaining performance gaps concentrated in high-variance or extreme-event regimes.

## Data Quality and Possible Anomalies
CAMELS-US Hydrology data exhibits clear seasonal patterns, with extremely long sequence lengths available (20 years of daily data), across 671 locations. These characteristics makes it ideal for time series experiments. Furthermore, the timeseries streams have no missing values. The minimal missing values in the static features can be trivially interpolated with techniques like forward-filling without significantly breaking patterns.

Even with strong overall quality, several anomaly modes remain important for interpretation. Streamflow is typically right-skewed, so rare high-flow peaks can dominate RMSE and make performance look worse than visual fit during normal-flow periods. Cross-basin heterogeneity is also high, good fit on the current set of 20% locations does not guarantee good fit on all locations.


## How to Set the Project Environment and Replicate Results

### Google Colab (recommended)

1. Open `notebooks/main.ipynb` in Colab.
2. Run setup cells (`git clone`, `%cd`, `pip install`, `git pull`).
3. Ensure raw file path resolves to `data/raw/BasicInputTimeSeries_us.csv` (or set `INPUT_CSV_PATH` env var).
4. Run all cells top-to-bottom.
5. Verify outputs under `results/` and `metadata/`.


## Dataset and Model Links

- Dataset: `https://zenodo.org/records/15529996`
- Model: `https://huggingface.co/amazon/chronos-2`

## Conclusion and Future Improvements

Current pipeline provides a reproducible baseline for zero-shot single-step streamflow forecasting on CAMELS-US with standardized outputs.

Possible improvements:
- try multivariate input (include all available timeseries streams and static features)
- add explicit naive baseline comparison for context
- add per-basin RMSE diagnostics
- evaluate alternative lookback sizes and model parameters