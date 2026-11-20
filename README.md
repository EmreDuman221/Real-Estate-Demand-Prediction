# China Real Estate Demand Prediction — Kaggle Solution

**Time-series forecasting | Polars data engineering | CatBoost | Custom evaluation metric | Heuristic ensembling**

This repository documents my solution to the [Kaggle Real Estate Demand Prediction competition](https://www.kaggle.com/competitions/china-real-estate-demand-prediction). The task is to forecast **monthly new-house transaction amounts** for residential sectors in China, combining transaction history with other sector and market data. The competition target is denominated in **10,000 Chinese yuan**.

The notebook builds a sector-by-month analytical dataset, creates lagged and seasonal features, trains a **CatBoost regression model with a custom objective**, and combines its forecast with historical-value heuristics for the twelve-month submission horizon.

> **Scope:** This README describes the specific uploaded notebook. It does **not** include LightGBM, XGBoost, Optuna, or a multi-model AutoML search; those algorithms are not present in this version.

## Results from the saved notebook execution

| Item | Recorded value |
| --- | ---: |
| Feature matrix | **6,432 rows × 1,011 columns** |
| Chronological training observations | **4,416** |
| One-month validation observations | **96** |
| Best **custom validation metric** | **0.579873** |
| Best CatBoost iteration | **13,046** |
| Final submission predictions | **1,152** |
| Nonzero submission predictions | **936** |

The numbers above come from the **saved execution output**, preserved in [`notebooks/original_executed_notebook.ipynb`](notebooks/original_executed_notebook.ipynb). They have **not been independently rerun** for this repository. **0.579873 is an internal holdout score, not a verified Kaggle public or private leaderboard score.** A leaderboard placement is not asserted here.

## 1. Problem and data

The competition asks participants to estimate each sector's new-house transaction amount for future months. The output format is:

```text
id,new_house_transaction_amount
2024 Aug_sector 1,<predicted amount>
...
```

The solution reads eight training tables and one test table from the Kaggle-provided directory:

| Input | Role in model |
| --- | --- |
| `train/new_house_transactions.csv` | Main transaction history and target |
| `train/new_house_transactions_nearby_sectors.csv` | Nearby-sector new-house activity |
| `train/pre_owned_house_transactions.csv` | Existing-home transactions |
| `train/pre_owned_house_transactions_nearby_sectors.csv` | Nearby-sector existing-home activity |
| `train/land_transactions.csv` | Land transaction features |
| `train/land_transactions_nearby_sectors.csv` | Nearby-sector land activity |
| `train/sector_POI.csv` | Sector point-of-interest features |
| `train/city_indexes.csv` | City-level indicators |
| `test.csv` | Prediction identifiers / required row order |

**No competition data files are redistributed in this repository.** Obtain them directly from Kaggle in accordance with the competition rules.

## 2. Solution overview

```text
Eight source tables + test identifiers
              |
              v
Create complete sector × month panel
   + month, sector, and time indexes
              |
              v
Join transactions, POI, and city indicators
   + missing-value and dtype handling
              |
              v
Build lagged, rolling, EWM, and seasonal features
              |
              v
Next-month target + chronological holdout
              |
              v
CatBoost with competition-style metric/objective
              |
              v
One-step CatBoost forecasts
              |
              v
Blend with last value + weighted geometric mean
   + trend, zero-activity, and December adjustments
              |
              v
12-month submission, 1,152 sector-month rows
```

### Data integration

The code uses **Polars** to create a regular sector–month panel, which helps represent sector-month combinations not explicitly present in a source table. Every source's non-key fields are prefixed (`nht_`, `pht_`, `lt_`, etc.) before joining to prevent collisions.

After the primary new-house transaction join, missing values are filled with **0**. Subsequent auxiliary table joins generally use **-1** as a missing-value sentinel. Numeric columns are downcast where possible, and the resulting joined panel is passed to the time-series feature builder.

### Time-series and market features

| Feature group | Implementation |
| --- | --- |
| Monthly lags | **1, 2, 12 months** |
| Rolling statistics | Mean and standard deviation over **3, 6, 12 months** |
| Exponentially weighted means | Alpha **0.3, 0.5, 0.7** |
| Momentum | Changes over **1 and 3 months** |
| Year-over-year proxy | Ratio against values **12 months earlier** |
| Supply proxy | Available-for-sale inventory / transaction count |
| Seasonality | Cyclic sine and cosine features with **12-, 6-, and 3-month** periods |
| Other engineered ratios | Including an area-per-unit ratio named `price_per_unit_trend` in the code |

The `price_per_unit_trend` variable name is retained from the submitted notebook, but **its formula uses area per unit, not observed sale prices**. It should be understood as an area-related ratio, not a verified pricing feature.

## 3. Training and validation

The supervised label is the **following month's new-house transaction amount** within each sector, implemented using a `shift(-1)`.

The notebook trains on earlier monthly observations (**4,416 rows**) and validates on **one later month (96 rows)** rather than a shuffled train/test split. The code fits a `CatBoostRegressor` with a custom loss derivative and uses a custom evaluation metric for early stopping, with **27,000 maximum iterations** and **1,500 rounds** of early stopping patience. The saved run stopped early; the best validation iteration was **13,046**.

The evaluation metric follows the competition's two-stage percentage-error logic:

1. Compute absolute percentage error (APE). If **more than 30%** of samples have APE above **100%**, assign **0**.
2. Otherwise, average the APE for the retained samples (APE ≤ 100%), divide by the retained-sample fraction, and calculate `max(0, 1 − scaled_MAPE)`.

The notebook additionally defines a **hand-crafted, sign-based derivative** (`CustomObjective`) for CatBoost optimization. This differs from optimizing a conventional squared-error or absolute-error loss.

## 4. Twelve-month forecasting and ensembling

A CatBoost forecast is first generated from the final observed month. The notebook then creates twelve monthly submission values with a rule-based blending scheme:

- **Recent level:** latest observed amount in the sector.
- **Historical level:** exponentially weighted geometric mean of recent positive observations (**14 months**, alpha **0.55**).
- **CatBoost signal:** one-step model prediction.
- **Adaptive weights:** CatBoost receives **50% weight** when the sector's trailing coefficient of variation exceeds 0.5, otherwise **40%**; the remainder is split across the two historical signals.
- **Small trend adjustment:** multiply by **1.05** or **0.95** for sufficiently positive/negative recent three-month trend.
- **Zero-activity rule:** force forecasts to zero for sectors meeting the specified recent inactivity conditions.
- **December effect:** multiply December predictions by a sector-specific historical December ratio, clipped to **0.82–1.45**.

The output is built against the Kaggle test identifiers, preserving their row order, with columns `id` and `new_house_transaction_amount`, then saved as `/kaggle/working/submission.csv`.

### Important modeling limitations

This notebook is an **experimental competition solution**, not a validated production forecasting system:

- The **96-row holdout measures one-month-ahead model behavior**, not the full twelve-month blended forecast. The reported validation score should not be used as evidence of twelve-month accuracy.
- The final horizon **reuses one CatBoost forecast for all twelve future months**, with heuristic trend/December modifications; it does **not** train separate horizon-specific models or recursively reestimate feature values.
- Rolling/weighted features are constructed from the supplied observed panel. A production rolling forecast would need an explicit as-of-date framework to ensure all features are available at each forecast origin.
- The zero-filling of missing primary transactions is a modeling assumption, not proof that all unrecorded transactions equal zero.
- The validation uses a single chronological period, not repeated rolling-origin time-series cross-validation, and the CatBoost custom objective is not a standard probabilistic or differentiable forecasting loss.
- The results are only grounded in the saved notebook output; neither Kaggle leaderboard performance nor reproducibility in a fresh runtime is independently verified here.

## 5. Reproduce

**Recommended:** Run in a Kaggle notebook with the competition dataset attached.

1. Open [`real-estate-demand-prediction.ipynb`](real-estate-demand-prediction.ipynb) and import it into Kaggle.
2. Add the [competition dataset](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/data) to the notebook so the files are available under `/kaggle/input/china-real-estate-demand-prediction/`.
3. Install dependencies if your runtime does not already have them (see `requirements.txt`).
4. Run the notebook in order. Training may be computationally intensive because the model is configured for up to 27,000 iterations.
5. Download `/kaggle/working/submission.csv` after the final cell completes.

```bash
pip install -r requirements.txt
```

The path conventions and output location assume **Kaggle's runtime**. They are not directly portable to an arbitrary local computer without modifying the data and output paths.

## 6. Repository contents

```text
China-Real-Estate-Demand-Prediction/
├── README.md
├── real-estate-demand-prediction.ipynb   # annotated, organized walkthrough
├── notebooks/
│   └── original_executed_notebook.ipynb # original notebook + saved output
├── requirements.txt
└── .gitignore
```

The organized notebook preserves the original code verbatim across cells, with new Markdown explanations. Its outputs were cleared because it has not been executed after reorganization; the original executed notebook provides the provenance for the reported results.

## 7. Skills demonstrated

**Data integration with Polars and pandas · relational joins · time-series feature engineering · chronological validation · customized model objectives and metrics · CatBoost regression · domain-informed forecast blending · competition submission engineering.**

## Reference

- [Kaggle — Real Estate Demand Prediction competition overview](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview)
- [Kaggle — Evaluation and submission requirements](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview)
