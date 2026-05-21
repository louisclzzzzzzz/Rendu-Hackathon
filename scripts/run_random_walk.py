"""
Compute RandomWalk baseline metrics on all existing combo combinations,
then merge into hackathon_scores.csv.
"""
import sys, json, logging
import numpy as np
import pandas as pd
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_xlsx_file, resample_weekly, filter_period,
    discover_files, LOOK_BACK, PERIODS,
)
from src.indicators import compute_all_indicators, INDICATOR_TARGETS
from src.data_loader import DataPipeline
from src.models import RandomWalkModel
from src.evaluate import evaluate_predictions, ResultsStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data" / "raw"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
HORIZON     = 15

# ── Discover which combos we already have results for ────────────────────────
with open(METRICS_DIR / "results.json") as f:
    existing = json.load(f)

combos = set(
    (r["ticker"], r["timeframe"], r["period"], r["indicator"])
    for r in existing
)
tickers = sorted({r["ticker"] for r in existing})

logger.info(f"Found {len(combos)} unique combos across {tickers}")

# ── Load raw data ─────────────────────────────────────────────────────────────
file_map = discover_files(str(DATA_DIR), tickers)

store = ResultsStore(str(METRICS_DIR))

for ticker in tickers:
    fp = file_map.get(ticker)
    if fp is None:
        logger.warning(f"No file for {ticker}, skipping")
        continue

    df_daily = load_xlsx_file(fp)

    for timeframe in ("daily", "weekly"):
        df_tf = resample_weekly(df_daily) if timeframe == "weekly" else df_daily
        look_back = LOOK_BACK[timeframe]

        for period in PERIODS:
            for indicator in ("macd", "stochastic"):
                if (ticker, timeframe, period, indicator) not in combos:
                    continue  # skip combos not in original run

                df_p = filter_period(df_tf, period)
                indicator_col = INDICATOR_TARGETS[indicator]
                df_ind = compute_all_indicators(df_p).dropna(subset=[indicator_col])

                min_rows = look_back + HORIZON + 10
                if len(df_ind) < min_rows:
                    logger.warning(f"  {ticker}/{timeframe}/{period}/{indicator}: too few rows ({len(df_ind)}), skipping")
                    continue

                series = df_ind[indicator_col].values.astype(np.float32)
                pipe = DataPipeline(look_back=look_back, horizon=HORIZON)
                (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = pipe.prepare(series)

                if len(X_test) == 0:
                    logger.warning(f"  {ticker}/{timeframe}/{period}/{indicator}: empty test set, skipping")
                    continue

                model = RandomWalkModel()
                model.fit(X_train, y_train, X_val, y_val)
                raw_preds = model.predict(X_test, horizon=HORIZON)

                y_test_inv  = pipe.inverse_transform(y_test)
                preds_inv   = pipe.inverse_transform(raw_preds)

                metrics = evaluate_predictions(y_test_inv, preds_inv)
                store.add(ticker, indicator, timeframe, period, "RandomWalk", metrics)
                logger.info(
                    f"  {ticker}/{timeframe}/{period}/{indicator} -> "
                    f"trend={metrics['trend_accuracy']:.3f} pearson={metrics['pearson_correlation']:.3f}"
                )

store.save()
logger.info("results.json updated with RandomWalk entries")

# ── Recompute hackathon_scores.csv ────────────────────────────────────────────
with open(METRICS_DIR / "results.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

pivot_df = df.pivot(
    index=["ticker", "timeframe", "period", "model"],
    columns="indicator",
    values=["trend_accuracy", "pearson_correlation"],
).reset_index()

pivot_df.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0]
    for col in pivot_df.columns
]

T_MACD   = pivot_df["trend_accuracy_macd"]
rho_MACD = pivot_df["pearson_correlation_macd"]
T_Stoch  = pivot_df["trend_accuracy_stochastic"]
rho_Stoch= pivot_df["pearson_correlation_stochastic"]

pivot_df["hackathon_score"] = (
    T_MACD.fillna(0)
    + np.maximum(0, rho_MACD.fillna(0))
    + T_Stoch.fillna(0)
    + np.maximum(0, rho_Stoch.fillna(0))
) / 4

out_path = METRICS_DIR / "hackathon_scores.csv"
pivot_df.to_csv(out_path, index=False)
logger.info(f"hackathon_scores.csv saved to {out_path}")

# ── Quick summary ─────────────────────────────────────────────────────────────
rw = pivot_df[pivot_df["model"] == "RandomWalk"].sort_values("hackathon_score", ascending=False)
print("\n=== RandomWalk scores ===")
print(rw[["ticker", "timeframe", "period", "model", "hackathon_score"]].to_string(index=False))

all_means = pivot_df.groupby("model")["hackathon_score"].mean().sort_values(ascending=False)
print("\n=== Mean hackathon score per model ===")
print(all_means.to_string())
