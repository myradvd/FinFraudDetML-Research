"""Stage 1 - audit + EDA: data-quality table, train-vs-test shift table and the README figures."""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config as C
from .data import load_test, load_train, sanitise
from .features import engineer
from .plots import BLUE, MUTED, ORANGE, plt, save


def _rate(t, by):
    g = t.groupby(by, observed=True)["fraud"].agg(rows="size", frauds="sum")
    return g.assign(rate_pct=100 * g.frauds / g.rows)


def _bars(ax, table, title, base, horizontal=False, label=True):
    bars = (ax.barh if horizontal else ax.bar)([str(i) for i in table.index], table.rate_pct, 0.6, color=BLUE)
    (ax.axvline if horizontal else ax.axhline)(100 * base, color=MUTED, lw=1)
    if label:
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    if horizontal:
        ax.invert_yaxis()
    ax.grid(axis="y" if horizontal else "x", visible=False)
    ax.set_title(title)


def audit(train, test):
    """Blank share per column, duplicates, and the headline shift statistics."""
    tr, te = sanitise(train), sanitise(test)
    quality = pd.DataFrame({"train % blank": 100 * train.isna().mean(), "test % blank": 100 * test.isna().mean()}).round(2)
    checks = {
        "rows with any blank (%)": lambda d: 100 * d[C.RAW_COLS].isna().any(axis=1).mean(),
        "new device (%)": lambda d: 100 * d.new_device.mean(),
        ">= 4 transactions in 1h (%)": lambda d: 100 * (d.transactions_last_1h >= 4).mean(),
        "amount P90": lambda d: d.transaction_amount.quantile(0.9),
        "amount P95": lambda d: d.transaction_amount.quantile(0.95),
        "account age median (days)": lambda d: d.account_age.median(),
        "Singapore (%)": lambda d: 100 * (d.country == "SG").mean(),
    }
    shift = pd.DataFrame({"train": {k: f(tr) for k, f in checks.items()},
                          "test": {k: f(te) for k, f in checks.items()}}).round(2)
    dup = {"duplicate ids": int(train.id.duplicated().sum()),
           "duplicate rows": int(train.drop(columns="id").duplicated().sum()),
           "train/test overlapping rows": int(pd.merge(train.drop(columns=["id", "fraud"]).dropna(),
                                                      test.drop(columns="id").dropna()).shape[0])}
    return quality, shift, dup


def figures(train, test):
    t = sanitise(train).assign(fraud=train.fraud)
    te = sanitise(test)
    base = t.fraud.mean()

    # 1. Single signals: flat, then a jump
    panels = [
        (pd.cut(t.transaction_amount, [0, 50, 100, 500, 1000, np.inf], right=False,
                labels=["<50", "50-100", "100-500", "500-1k", "1k+"]), "Amount: jump above 500"),
        (pd.cut(t.transactions_last_1h, [0, 1, 2, 4, 6, np.inf], right=False,
                labels=["0", "1", "2-3", "4-5", "6+"]), "Transactions in last hour: jump at 4+"),
        (pd.cut(t.transactions_last_24h, [0, 6, 11, 16, np.inf], right=False,
                labels=["1-5", "6-10", "11-15", "16+"]), "Transactions in last 24h: jump at 11+"),
        (t.new_device.map({0: "known", 1: "new"}), "Device: new is 6x riskier"),
        (pd.cut(t.account_age, [0, 31, 91, 366, 1096, np.inf], right=False,
                labels=["<=30", "31-90", "91-365", "366-1095", "1096+"]), "Account age (days): youngest riskiest"),
        (t.transaction_hour.astype("Int64"), "Hour of day: riskier at night (23-04)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (groups, title) in zip(axes.flat, panels):
        _bars(ax, _rate(t, groups.rename(None)), title, base, label=not title.startswith("Hour"))
    axes[1, 2].set_xticks(axes[1, 2].get_xticks()[::3])
    for ax in axes[:, 0]:
        ax.set_ylabel("fraud rate (%)")
    fig.suptitle(f"Fraud rate by single signal (grey line: overall {base:.1%})", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "fraud_rate_by_signal.png")

    # 2. Signals stack up: device x 1h bursts
    burst = pd.cut(t.transactions_last_1h, [0, 1, 2, 4, 6, np.inf], right=False, labels=["0", "1", "2-3", "4-5", "6+"])
    device = t.new_device.map({0: "known device", 1: "new device"})
    cells = t.groupby([device, burst], observed=True)["fraud"].agg(["mean", "size"])
    rate, rows = 100 * cells["mean"].unstack(), cells["size"].unstack()
    fig, ax = plt.subplots(figsize=(8, 2.6))
    im = ax.imshow(rate.values, cmap=LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#0d366b"]), aspect="auto")
    for (i, j), v in np.ndenumerate(rate.values):
        ax.text(j, i, f"{v:.1f}%\nn={rows.values[i, j]:,}", ha="center", va="center", fontsize=8,
                color="white" if v > np.nanmax(rate.values) / 2 else "#0b0b0b")
    ax.set_xticks(range(rate.shape[1]), rate.columns)
    ax.set_yticks(range(rate.shape[0]), rate.index)
    ax.grid(False)
    ax.set(xlabel="transactions in the last hour", title="Fraud rate: device x transactions in the last hour")
    fig.colorbar(im, ax=ax, label="fraud rate (%)")
    save(fig, "device_x_burst.png")

    # 3. n_flags: fraud concentrates where flags stack
    n_flags = engineer(train).n_flags.clip(upper=5).astype(int)
    table = _rate(t.assign(k=n_flags.map(lambda k: "5+" if k == 5 else str(k))), "k")
    fig, ax = plt.subplots(figsize=(7, 3))
    _bars(ax, table, "Fraud rate by number of risk flags fired (n_flags)", base)
    ax.set_xticks(range(len(table)), [f"{i}\nn={r:,}" for i, r in zip(table.index, table.rows)])
    ax.set_ylabel("fraud rate (%)")
    save(fig, "n_flags.png")

    # 4. The shift
    checks = {"rows with a blank": lambda d: d[C.RAW_COLS].isna().any(axis=1).mean(),
              "new device": lambda d: d.new_device.mean(),
              "4+ transactions in 1h": lambda d: (d.transactions_last_1h >= 4).mean(),
              "amount > 1,000": lambda d: (d.transaction_amount > 1000).mean(),
              "Singapore": lambda d: (d.country == "SG").mean()}
    shares = pd.DataFrame({n: {k: 100 * f(d) for k, f in checks.items()} for n, d in [("train", t), ("test", te)]})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.2]})
    y = np.arange(len(shares))
    for k, (name, color) in enumerate([("train", BLUE), ("test", ORANGE)]):
        bars = ax1.barh(y + (k - 0.5) * 0.4, shares[name], 0.36, color=color, label=name)
        ax1.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    ax1.set_yticks(y, shares.index)
    ax1.invert_yaxis()
    ax1.grid(axis="y", visible=False)
    ax1.set(xlabel="% of rows", title="Share of rows: train vs test")
    ax1.legend()
    bins = np.logspace(np.log10(2), np.log10(10_000), 40)
    for name, d, color in [("train", t, BLUE), ("test", te, ORANGE)]:
        x = d.transaction_amount.dropna()
        ax2.hist(x, bins, weights=np.full(len(x), 1 / len(x)), histtype="step", color=color, lw=2, label=name)
    ax2.set_xscale("log")
    ax2.set(xlabel="transaction amount (log scale)", ylabel="share of rows", title="Amounts: test has a heavier large-amount tail")
    ax2.legend()
    fig.tight_layout()
    save(fig, "train_test_shift.png")


def leaderboard_figure():
    lb = pd.read_csv(C.LEADERBOARD_CSV).sort_values("public_pr_auc")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [ORANGE if r == 1 else BLUE for r in lb["rank"]]
    bars = ax.barh(lb.submission, lb.public_pr_auc, 0.6, color=colors)
    approx = lb.approximate.eq("yes")
    ax.bar_label(bars, [f"~{v:.2f}" if a else f"{v:.4f}" for v, a in zip(lb.public_pr_auc, approx)], padding=2, fontsize=8)
    best = lb.public_pr_auc.max()
    ax.axvspan(best - 0.012, best, color=MUTED, alpha=0.12, lw=0)
    ax.text(best - 0.0115, -0.9, "within paired noise\n(SD 0.012) of the best", fontsize=7, color=MUTED, va="bottom")
    ax.set_xlim(0.14, 0.205)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="public leaderboard PR-AUC (12,000 test rows)",
           title="Every public upload: the plain final CatBoost (orange) was never beaten")
    save(fig, "public_leaderboard.png")


def run():
    train, test = load_train(), load_test()
    quality, shift, dup = audit(train, test)
    out = C.RESULTS_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    quality.to_csv(out / "data_quality.csv")
    shift.to_csv(out / "train_test_shift.csv")
    print(f"  {train.fraud.sum()} frauds in {len(train):,} rows ({train.fraud.mean():.2%}); test {len(test):,} rows")
    print(f"  integrity: {dup}")
    print(shift.to_string())
    figures(train, test)
    leaderboard_figure()
