# Notebooks

Three notebooks, each answering one question. They are executed and saved with outputs, so GitHub shows every
table and figure. The deliverable itself is [`../final_notebook/`](../final_notebook/README.md).

| # | Notebook | Question | Answer |
|---|---|---|---|
| 01 | [01_final_catboost](01_final_catboost.ipynb) | How is the final model built? Every step with its "why": cleaning → 24 features → CatBoost → CV → alert share → final fit | CV PR-AUC **0.238**, F1 0.305 at the top 1.65%. Reproduces the 0.19586 upload **byte for byte** (MD5 check) and saves the `model.pkl` the final notebook loads |
| 02 | [02_model_comparison](02_model_comparison.ipynb) | Which model should we submit? Logistic, random forest, LightGBM, XGBoost, in-context kNN, CatBoost and 4 blends, all on identical folds | CV can't separate CatBoost from its blends (the CatBoost + logistic blend even edges ahead, 0.239 vs 0.238). On unseen data the blend lost (public 0.183 vs 0.196), so the pre-stated rule selects **plain CatBoost** |
| 03 | [03_shift_reanchoring](03_shift_reanchoring.ipynb) | Is the train → test shift just inputs drifting? If so, mapping non-Singapore amounts and counts back to their training scale should help | **No: public 0.1844 vs 0.19586.** The extra bursts in the test set carry real fraud, so the feature–fraud *relationship* changed. That is why the final model is left uncorrected |

**Run:** `make notebooks` (~10 min on a laptop CPU), or open one in Jupyter and *Run All*. They read `../data/`; in
Google Colab, 01 and 03 open an upload button instead. CatBoost's thread count is fixed at 3 because its fitted
trees depend on it.

> **CV** = 5-fold stratified out-of-fold PR-AUC on the training set. **Public** = leaderboard PR-AUC on the
> 12,000-row test set; the same model moves ±0.004 between seeds, and paired differences have SD ≈ 0.012.
> A random ranker scores 0.018.

## Experiments not kept as notebooks

To keep the repo small, the other experiments are recorded as results rather than code. Every public score is in
[`../results/leaderboard.csv`](../results/leaderboard.csv), and every experiment, with its numbers, is in
[`../docs/TECHNICAL_RECORD.md` §7](../docs/TECHNICAL_RECORD.md).

| Experiment | Result | Lesson |
|---|---|---|
| Logistic-regression baseline + 12 ablations from fraud papers | CV 0.233; class weights, WoE, splines, interactions all lost | A strong, explainable floor; recall comes from the threshold, not from weights |
| TabPFN + LightGBM + logistic blend | public 0.1935 | More models didn't beat one well-regularised CatBoost |
| CatBoost configuration search (one-hot, rsm, Depthwise, deeper/slower) | the baseline config was the best of 8 | The final settings are well tuned |
| Leaderboard probes (no `new_device`, defaults, raw features, class weights) | every one lost on CV | Supports the 24-feature, unweighted design |
| Monotone constraints | CV and stress tests favoured it; **public ~0.15** | Synthetic stress tests didn't predict the real shift |
| Context-relative features, flag-excess prior, test rows as negatives | public 0.1893 / 0.1878; CV −0.03 for context-only | Every shift correction lost on unseen data |
| Bagged MLPs | CV 0.183 | Neural nets don't help with 353 frauds |

Notebooks committed earlier are still in git history (`git log --all -- src/`).
