# Results

| Path | Written by | Contents |
|---|---|---|
| [`leaderboard.csv`](leaderboard.csv) | hand-recorded | **All 15 public-leaderboard uploads**: score, where each came from, archived file |
| [`leaderboard_uploads/`](leaderboard_uploads/) | archived, read-only | The uploaded CSVs that survive, named by leaderboard rank. `01_catboost_3seeds_FINAL.csv` is the final submission (MD5 `d666afc9…`) |
| [`model_selection/`](model_selection/) | `fraudlab select` / notebook 02 | [`REPORT.md`](model_selection/REPORT.md) (the model-zoo comparison), `leaderboard.csv` (every metric), `oof_scores.csv` (out-of-fold scores of every candidate), `selected_model.json` |
| `final/` | `fraudlab train` / `predict` | `submission.csv`, byte-identical to the final upload. `model.pkl` is git-ignored; rebuild it with `make train` |
| `eda/` | `fraudlab eda` | `data_quality.csv` (blank share per column), `train_test_shift.csv` |
| `catboost/` | notebook 01 | CV metrics, `submission_best_catboost.csv` (= the final upload), `model.pkl` (git-ignored) |
| `shift_reanchoring/` | notebook 03 | re-anchored submission + label-free diagnostics |

Everything except `leaderboard_uploads/` can be regenerated (`make pipeline`, `make notebooks`). That folder is the
historical record and must not be overwritten.
