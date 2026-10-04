# Final notebook: `predict_catboost.ipynb`

The deliverable. It loads the trained model (`model.pkl`), scores a test file and writes one fraud score per row.
It also walks through the data audit, the EDA story and the feature engineering, so it reads as the full
solution. **It does no training.** Scoring 12,000 rows takes about a second.

| | |
|---|---|
| **Model** | 3 CatBoost classifiers (seeds 0, 1, 2), depth 4, lr 0.03 × 800, L2 10, averaged |
| **Features** | 24 per-row features built from the 10 raw columns (no labels, no cross-row statistics) |
| **Out-of-fold PR-AUC** | 0.238 (95% CI 0.194–0.292); random ranker 0.018 |
| **Public leaderboard PR-AUC** | **0.19586**, the team's best of 15 uploads |
| **Output** | `prediction` ∈ [0, 1]; the top 1.65% (the CV-optimal F1 share) sit at ≥ 0.5 |

## Run it

From the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # full environment; or: pip install -r final_notebook/requirements.txt
cd final_notebook
jupyter notebook predict_catboost.ipynb    # Run All
```

Or headless, with outputs saved into the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace final_notebook/predict_catboost.ipynb
```

## Inputs and outputs

| What | Where it's read from / written to |
|---|---|
| Model | `model.pkl` (next to the notebook) |
| Test file | `$DATATHON_INPUT_PATH`, else `test.csv`, else `../data/Track 2 Testing Dataset.csv` |
| Training file (EDA only) | `train.csv`, else `../data/Track 2 Training Dataset.csv`. If neither exists, the EDA cells are skipped and prediction still runs. |
| Predictions | `$DATATHON_OUTPUT_PATH`, else `predictions.csv`. Columns: `id, prediction` if the test file has ids, else just `prediction`, in the test file's row order. |

To score a **new, unseen file** (the private leaderboard):

```bash
DATATHON_INPUT_PATH=/path/to/new_test.csv DATATHON_OUTPUT_PATH=/path/to/out.csv \
  jupyter nbconvert --to notebook --execute final_notebook/predict_catboost.ipynb --output /tmp/run.ipynb
```

The cleaning step (`sanitise`) makes new files safe to score. Columns are matched by name, so missing columns
become blank and extra ones are ignored. Impossible values (negative amounts, hour 25, `new_device` = 2), text
and infinities become blank. Category spelling is normalised (`" sg"` → `"SG"`), and unseen categories map to
`Unknown`. None of this changes anything on the competition files.

## Where `model.pkl` comes from

`model.pkl` is a `joblib` dict holding the three fitted `CatBoostClassifier`s, the flag share (0.0165) and the
constants the scoring code depends on (feature order, category lists, cleaning rules). The notebook's first model
cell asserts that those constants equal its own, so a mismatched model fails loudly instead of scoring wrongly.

To rebuild it from scratch (about a minute) and check that it is identical:

```bash
python -m fraudlab train      # writes results/final/model.pkl
python -m fraudlab verify     # rebuilt test scores == shipped model's; submission MD5 == the 0.19586 upload
```

`notebooks/01_final_catboost.ipynb` trains the same model step by step and saves the same bundle to
`results/catboost/model.pkl`.

> **Reproducibility note.** CatBoost's fitted trees depend on the thread count. The model is trained with
> `thread_count=3` and CatBoost 1.2.10, which reproduce the uploaded file byte for byte (MD5
> `d666afc9c1454c2933b2aca8e21933fb`). Prediction does not depend on threads.

## Notebook outline

1. **Data processing:** load, data-quality audit, `sanitise`.
2. **EDA:** fraud is rare → single signals jump → amounts overlap → risky categories → signals stack up → the test set is shifted.
3. **Feature engineering:** the 24 features and why; `n_flags` check; silent fraud.
4. **Model:** load `model.pkl`, settings table, feature importance.
5. **Predictions:** average of 3 models → order-preserving map with the cut-off at 0.5 → CSV.
6. **What else was tried:** public-leaderboard comparison, and the model's assumptions.
