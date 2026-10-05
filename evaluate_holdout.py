"""Out-of-sample evaluation of the Prophet forecasts against simple baselines.

The notebook fits Prophet and reports MAPE on the same days the model was
trained on. This script measures accuracy on days the model has NOT seen:
the last 6 weeks of the data are held out, every model is trained only on the
earlier data, and all models are scored on the same held-out open days.

Models compared, per store:
  * naive_last_week   - each held-out day gets the sales of the same weekday in
                        the last training week (a standard seasonal-naive baseline)
  * weekday_avg_8w    - average sales for that weekday over the last 8 training weeks
  * prophet_holidays  - Prophet with the store's school and state holidays (as in the notebook)
  * prophet_promo     - the same, plus the Promo flag as a regressor (promos are planned in advance)

Usage (download train.csv from https://www.kaggle.com/c/rossmann-store-sales/data first):
    python evaluate_holdout.py --train data/train.csv --stores 50
    python evaluate_holdout.py --train data/train.csv --all        # all 1,115 stores (slow)

Outputs:
    results/holdout_by_store.csv   one row per store with each model's MAPE and RMSPE
    results/holdout_summary.md     summary table to paste into the README
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd

HOLDOUT_DAYS = 42  # 6 weeks, the same horizon as the Rossmann Kaggle competition


def mape(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def rmspe(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean(((actual - predicted) / actual) ** 2)) * 100)


def to_markdown(df):
    """Small markdown table writer (avoids an extra dependency)."""
    cols = [df.index.name or "Model"] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f"{v:.2f}" for v in row]) + " |")
    return "\n".join(lines)


def load(train_path):
    df = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    return df.sort_values(["Store", "Date"])


def store_holidays(store_df):
    """School and state holidays for one store, in Prophet's format."""
    state = store_df.loc[store_df["StateHoliday"].isin(["a", "b", "c"]), "Date"]
    school = store_df.loc[store_df["SchoolHoliday"] == 1, "Date"]
    return pd.concat([
        pd.DataFrame({"ds": state, "holiday": "state_holiday"}),
        pd.DataFrame({"ds": school, "holiday": "school_holiday"}),
    ], ignore_index=True)


def evaluate_store(store_df, cutoff):
    from prophet import Prophet

    open_days = store_df[(store_df["Open"] == 1) & (store_df["Sales"] > 0)]
    train = open_days[open_days["Date"] < cutoff]
    test = open_days[open_days["Date"] >= cutoff]
    if len(test) == 0 or len(train) < 365:
        return None

    actual = test["Sales"].values
    preds = {}

    # Baseline 1: same weekday in the last training week
    last_week = train[train["Date"] >= cutoff - pd.Timedelta(days=7)]
    by_dow_last = last_week.groupby("DayOfWeek")["Sales"].mean()
    # Baseline 2: weekday average over the last 8 training weeks
    last_8w = train[train["Date"] >= cutoff - pd.Timedelta(weeks=8)]
    by_dow_8w = last_8w.groupby("DayOfWeek")["Sales"].mean()
    preds["naive_last_week"] = test["DayOfWeek"].map(by_dow_last).fillna(test["DayOfWeek"].map(by_dow_8w)).values
    preds["weekday_avg_8w"] = test["DayOfWeek"].map(by_dow_8w).values

    holidays = store_holidays(store_df)
    fit_df = train[["Date", "Sales", "Promo"]].rename(columns={"Date": "ds", "Sales": "y"})
    future = test[["Date", "Promo"]].rename(columns={"Date": "ds"})

    m = Prophet(holidays=holidays)
    m.fit(fit_df[["ds", "y"]])
    preds["prophet_holidays"] = m.predict(future[["ds"]])["yhat"].values

    m = Prophet(holidays=holidays)
    m.add_regressor("Promo")
    m.fit(fit_df)
    preds["prophet_promo"] = m.predict(future)["yhat"].values

    row = {"Store": int(store_df["Store"].iloc[0]), "test_days": len(test)}
    for name, p in preds.items():
        row[f"{name}_mape"] = mape(actual, p)
        row[f"{name}_rmspe"] = rmspe(actual, p)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--stores", type=int, default=50, help="number of randomly chosen stores")
    ap.add_argument("--all", action="store_true", help="evaluate every store")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    df = load(args.train)
    cutoff = df["Date"].max() - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    stores = sorted(df["Store"].unique())
    if not args.all:
        rng = np.random.default_rng(args.seed)
        stores = sorted(rng.choice(stores, size=min(args.stores, len(stores)), replace=False))

    rows = []
    for i, s in enumerate(stores, 1):
        r = evaluate_store(df[df["Store"] == s], cutoff)
        if r:
            rows.append(r)
        print(f"[{i}/{len(stores)}] store {s} done")

    res = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, "holdout_by_store.csv"), index=False)

    models = ["naive_last_week", "weekday_avg_8w", "prophet_holidays", "prophet_promo"]
    summary = pd.DataFrame({
        "Mean MAPE (%)": [res[f"{m}_mape"].mean() for m in models],
        "Median MAPE (%)": [res[f"{m}_mape"].median() for m in models],
        "Mean RMSPE (%)": [res[f"{m}_rmspe"].mean() for m in models],
    }, index=models).round(2)

    best_base = summary.loc[["naive_last_week", "weekday_avg_8w"], "Mean MAPE (%)"].idxmin()
    best_prophet = summary.loc[["prophet_holidays", "prophet_promo"], "Mean MAPE (%)"].idxmin()
    b, p = summary.at[best_base, "Mean MAPE (%)"], summary.at[best_prophet, "Mean MAPE (%)"]
    improvement = (b - p) / b * 100
    wins = (res[f"{best_prophet}_mape"] < res[f"{best_base}_mape"]).mean() * 100

    text = (
        f"# Holdout evaluation\n\n"
        f"Stores evaluated: {len(res)} | Holdout: last {HOLDOUT_DAYS} days "
        f"({cutoff.date()} to {df['Date'].max().date()}), open days with sales > 0 | "
        f"Models trained only on data before the holdout.\n\n"
        f"{to_markdown(summary)}\n\n"
        f"Best Prophet model ({best_prophet}) vs best baseline ({best_base}): "
        f"mean MAPE {p:.2f}% vs {b:.2f}%, a {improvement:.1f}% relative reduction in error; "
        f"Prophet had the lower MAPE in {wins:.0f}% of stores.\n"
    )
    with open(os.path.join(args.out, "holdout_summary.md"), "w") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
