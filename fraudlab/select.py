"""Stage 2 - model selection: cross-validate every model family and blend on identical folds, then pick one.

Selection rule (fixed before looking at the results):
  1. Rank candidates by out-of-fold PR-AUC.
  2. Shortlist every candidate *statistically tied* with the CV leader: the paired bootstrap 95% interval of
     (candidate - leader) PR-AUC contains 0. CV cannot separate these.
  3. Among the shortlist, prefer evidence from unseen data: the highest public-leaderboard PR-AUC
     (results/leaderboard.csv). The public set is used only as this external tie-breaker, never for tuning.
  4. If no shortlisted candidate was uploaded, take the simplest (fewest members), then the best CV score.
"""
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from .data import load_train
from .evaluate import bootstrap_ap, folds, metrics, pick_share
from .features import make_features
from .models import BASE_MODELS, CANDIDATES, combine, fit_score
from .plots import BLUE, INK, MUTED, ORANGE, plt, save


def out_of_fold(X, y, names):
    """Out-of-fold fraud scores for each base learner, all on the same 5 folds."""
    oof = {}
    for name in names:
        t0, s = time.time(), np.zeros(len(y))
        for tr, te in folds(y):
            s[te] = fit_score(name, X.iloc[tr], y[tr], X.iloc[te], C.CV_SEEDS)
        oof[name] = s
        print(f"  {name:14s} OOF PR-AUC {metrics(y, s)['PR-AUC']:.4f}   ({time.time() - t0:5.1f}s)", flush=True)
    return oof


def choose(table):
    shortlist = table[table["tied with CV leader"]]
    uploaded = shortlist.dropna(subset=["public PR-AUC"])
    if len(uploaded):
        return uploaded["public PR-AUC"].idxmax(), "highest public PR-AUC among the candidates tied on CV"
    simplest = shortlist.sort_values(["members", "PR-AUC"], ascending=[True, False])
    return simplest.index[0], "simplest candidate tied on CV (no public evidence)"


def figure(table, chosen):
    t = table.sort_values("PR-AUC")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hlines(y, t["CI low"], t["CI high"], color=BLUE, lw=2, alpha=0.35)
    ax.plot(t["PR-AUC"], y, "o", color=BLUE, ms=8, mec="#fcfcfb", mew=2, label="cross-validation (95% bootstrap CI)")
    ax.plot(t["public PR-AUC"], y, "D", color=ORANGE, ms=8, mec="#fcfcfb", mew=2, label="public leaderboard")
    ax.axvline(t["base rate"].iloc[0], color=MUTED, lw=1)
    ax.set_yticks(y, [f"{n}  ← selected" if n == chosen else n for n in t.index])
    for lbl, name in zip(ax.get_yticklabels(), t.index):
        if name == chosen:
            lbl.set_fontweight("bold"), lbl.set_color(INK)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="PR-AUC (random ranker = base rate, grey line)",
           title="Model zoo: CV can't separate the top candidates; unseen data can")
    ax.legend(loc="upper left")
    save(fig, "model_selection.png")


def alert_dial(y, score, name):
    """Precision and recall as the alert share grows: the operating-point trade-off."""
    shares = np.linspace(0.002, 0.10, 99)
    order = np.argsort(-score, kind="stable")
    tp = np.cumsum(y[order])
    k = np.maximum((shares * len(y)).astype(int), 1)
    prec, rec = tp[k - 1] / k, tp[k - 1] / y.sum()
    best = pick_share(y, score)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(100 * shares, prec, color=BLUE, lw=2, label="precision")
    ax.plot(100 * shares, rec, color=ORANGE, lw=2, label="recall")
    ax.axvline(100 * best["share"], color=MUTED, lw=1)
    ax.text(100 * best["share"] + 0.15, 0.62, f"chosen: top {best['share']:.2%}\nF1 {best['F1']:.3f}", fontsize=8)
    ax.set(xlabel="alert rate (% of transactions flagged)", ylabel="out-of-fold value", ylim=(0, 0.7),
           title=f"{name}: one dial trades missed fraud against blocked customers")
    ax.legend(loc="upper right")
    save(fig, "alert_dial.png")


def run(skip=()):
    t0 = time.time()
    df = load_train()
    X, y = make_features(df, extra=True), df[C.TARGET].values
    candidates = [c for c in CANDIDATES if not set(c.members) & set(skip)]
    names = [n for n in BASE_MODELS if any(n in c.members for c in candidates)]
    print(f"  {len(candidates)} candidates from {len(names)} base learners, {C.N_SPLITS}-fold CV")

    oof = out_of_fold(X, y, names)
    scores = {c.name: combine(c, oof) for c in candidates}
    boot = bootstrap_ap(y, scores)
    public = pd.read_csv(C.LEADERBOARD_CSV).set_index("submission")["public_pr_auc"]

    rows = []
    for c in candidates:
        m = metrics(y, scores[c.name], probabilities=not c.is_blend)
        lo, hi = np.percentile(boot[c.name], [2.5, 97.5])
        rows.append({"candidate": c.name, **m, "CI low": lo, "CI high": hi, "members": len(c.members),
                     "family": " + ".join(BASE_MODELS[n][1] for n in c.members),
                     "public PR-AUC": public.get(c.public_name, np.nan), "base rate": y.mean()})
    table = pd.DataFrame(rows).set_index("candidate").sort_values("PR-AUC", ascending=False)
    leader = table.index[0]
    for name in table.index:
        d = boot[name] - boot[leader]
        table.loc[name, "delta vs leader"] = table.loc[name, "PR-AUC"] - table.loc[leader, "PR-AUC"]
        table.loc[name, "delta 95% low"], table.loc[name, "delta 95% high"] = np.percentile(d, [2.5, 97.5])
    table["tied with CV leader"] = (table["delta 95% low"] <= 0) & (table["delta 95% high"] >= 0)
    chosen, reason = choose(table)

    C.SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(C.SELECTION_DIR / "leaderboard.csv")
    pd.DataFrame(scores).assign(fraud=y).to_csv(C.SELECTION_DIR / "oof_scores.csv", index=False)
    selection = {"selected": chosen, "reason": reason, "cv_leader": leader,
                 "shortlist": list(table.index[table["tied with CV leader"]]),
                 "flag_share": float(table.loc[chosen, "flag share"]),
                 "cv_pr_auc": float(table.loc[chosen, "PR-AUC"]),
                 "public_pr_auc": None if np.isnan(table.loc[chosen, "public PR-AUC"]) else float(table.loc[chosen, "public PR-AUC"])}
    (C.SELECTION_DIR / "selected_model.json").write_text(json.dumps(selection, indent=2))
    write_report(table, selection, time.time() - t0)
    figure(table, chosen)
    alert_dial(y, scores[chosen], chosen)

    show = ["PR-AUC", "CI low", "CI high", "top-100 precision", "F1", "recall", "public PR-AUC", "tied with CV leader"]
    print(table[show].round(4).to_string())
    print(f"\n  SELECTED: {chosen}  ({reason})")
    return selection


def write_report(table, sel, seconds):
    cols = ["family", "PR-AUC", "CI low", "CI high", "delta vs leader", "tied with CV leader",
            "top-100 precision", "F1", "recall", "flag share", "public PR-AUC"]
    t = table[cols].copy()
    t["tied with CV leader"] = t["tied with CV leader"].map({True: "yes", False: "no"})
    t["flag share"] = (100 * t["flag share"]).round(2).astype(str) + "%"
    md = [
        "# Model selection report",
        "",
        f"Generated by `python -m fraudlab select` ({seconds / 60:.1f} min). Every candidate is scored on the same "
        f"{C.N_SPLITS} stratified folds; blends are rank averages of the members' out-of-fold scores.",
        "",
        f"**Selected: `{sel['selected']}`.** Reason: {sel['reason']}. CV leader: `{sel['cv_leader']}`. "
        f"Shortlist (tied on CV): {', '.join(f'`{s}`' for s in sel['shortlist'])}.",
        "",
        "![model selection](../../docs/figures/model_selection.png)",
        "",
        t.round(4).to_markdown(),
        "",
        "**How to read this.** `tied with CV leader` = the paired bootstrap 95% interval of the PR-AUC difference "
        "contains 0. `public PR-AUC` is shown only for candidates that were uploaded (see `results/leaderboard.csv`). "
        "F1 / recall / precision are at the flag share that maximises smoothed out-of-fold F1.",
        "",
        "Selection rule: see the docstring of `fraudlab/select.py`.",
    ]
    (C.SELECTION_DIR / "REPORT.md").write_text("\n".join(md) + "\n")
