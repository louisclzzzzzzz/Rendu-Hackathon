"""Main training and evaluation pipeline — with period dimension."""

import logging
import numpy as np
from pathlib import Path

from .data_loader import DataPipeline, LOOK_BACK
from .indicators import compute_all_indicators, INDICATOR_TARGETS
from .models import get_all_models, model_save_path
from .evaluate import evaluate_predictions, ResultsStore

logger = logging.getLogger(__name__)

STAT_MODELS = {"ARIMA", "SARIMA", "HoltWinters"}


def run_combination(ticker: str, indicator: str, timeframe: str, period: str,
                    df, models_dir: str, horizon: int,
                    store: ResultsStore, predictions_cache: dict):
    """Train and evaluate all models for one (ticker × indicator × timeframe × period)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {ticker} | {indicator} | {timeframe} | {period}")

    indicator_col = INDICATOR_TARGETS[indicator]
    look_back = LOOK_BACK.get(timeframe, 60)
    min_rows = look_back + horizon + 10

    df_ind = compute_all_indicators(df).dropna(subset=[indicator_col])

    if len(df_ind) < min_rows:
        logger.warning(f"  Not enough data ({len(df_ind)} rows, need {min_rows}) — skipping")
        return

    series = df_ind[indicator_col].values.astype(np.float32)
    pipe = DataPipeline(look_back=look_back, horizon=horizon)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = pipe.prepare(series)

    if len(X_test) == 0:
        logger.warning(f"  Empty test set — skipping")
        return

    models = get_all_models(horizon=horizon)

    for model in models:
        model_name = model.name
        save_path = model_save_path(models_dir, model_name, ticker, indicator, timeframe, period)

        try:
            logger.info(f"  Training {model_name}...")
            model.fit(X_train, y_train, X_val, y_val)
            raw_preds = model.predict(X_test, horizon=horizon)

            y_test_inv = pipe.inverse_transform(y_test)
            preds_inv = pipe.inverse_transform(raw_preds)

            metrics = evaluate_predictions(y_test_inv, preds_inv)
            store.add(ticker, indicator, timeframe, period, model_name, metrics)

            key = (ticker, indicator, timeframe, period, model_name)
            predictions_cache[key] = {
                "y_true": y_test_inv[-1],
                "y_pred": preds_inv[-1],
                "metrics": metrics,
            }
            model.save(save_path)

        except Exception as e:
            logger.error(f"  FAILED {model_name}: {e}", exc_info=True)
            store.add(ticker, indicator, timeframe, period, model_name, {
                "trend_accuracy": float("nan"),
                "pearson_correlation": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
            })


def run_pipeline(data: dict, tickers: list, timeframes: list, periods: list,
                 indicators: list, horizon: int,
                 models_dir: str, metrics_dir: str) -> tuple:
    """
    Run all 60 combinations.
    data: {ticker: {timeframe: {period: pd.DataFrame}}}
    Returns (ResultsStore, predictions_cache).
    """
    store = ResultsStore(metrics_dir)
    predictions_cache = {}
    total = len(tickers) * len(timeframes) * len(periods) * len(indicators)
    done = 0

    for ticker in tickers:
        for timeframe in timeframes:
            for period in periods:
                df = data.get(ticker, {}).get(timeframe, {}).get(period)
                if df is None:
                    logger.warning(f"No data for {ticker}/{timeframe}/{period}, skipping")
                    continue
                for indicator in indicators:
                    done += 1
                    logger.info(f"\n[{done}/{total}] {ticker}/{timeframe}/{period}/{indicator}")
                    run_combination(
                        ticker=ticker, indicator=indicator,
                        timeframe=timeframe, period=period,
                        df=df, models_dir=models_dir, horizon=horizon,
                        store=store, predictions_cache=predictions_cache,
                    )

    store.save()
    return store, predictions_cache
