"""Evaluation metrics and result storage — with period dimension."""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def trend_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% of steps where predicted direction matches actual direction."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) < 2:
        return np.nan
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    return float(np.mean(np.sign(true_diff) == np.sign(pred_diff)))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import pearsonr
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    corr, _ = pearsonr(y_true, y_pred)
    return float(corr)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all metrics for a batch of (n_samples, horizon) predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    trend_accs, pearsons, rmses, maes = [], [], [], []
    for i in range(len(y_true)):
        trend_accs.append(trend_accuracy(y_true[i], y_pred[i]))
        pearsons.append(pearson_correlation(y_true[i], y_pred[i]))
        rmses.append(rmse(y_true[i], y_pred[i]))
        maes.append(mae(y_true[i], y_pred[i]))

    def safe_mean(lst):
        vals = [v for v in lst if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    return {
        "trend_accuracy": safe_mean(trend_accs),
        "pearson_correlation": safe_mean(pearsons),
        "rmse": safe_mean(rmses),
        "mae": safe_mean(maes),
    }


class ResultsStore:
    """Accumulate and persist evaluation results across runs."""

    KEY_COLS = ("ticker", "indicator", "timeframe", "period", "model")

    def __init__(self, metrics_dir: str):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = self._load_existing()

    def _load_existing(self) -> list:
        json_path = self.metrics_dir / "results.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    records = json.load(f)
                logger.info(f"Loaded {len(records)} existing results from {json_path}")
                return records
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
        return []

    def _record_key(self, r: dict) -> tuple:
        return tuple(r.get(c, "") for c in self.KEY_COLS)

    def add(self, ticker: str, indicator: str, timeframe: str, period: str,
            model_name: str, metrics: dict):
        record = {
            "ticker": ticker,
            "indicator": indicator,
            "timeframe": timeframe,
            "period": period,
            "model": model_name,
            **metrics,
        }
        key = (ticker, indicator, timeframe, period, model_name)
        self.records = [r for r in self.records if self._record_key(r) != key]
        self.records.append(record)

        ta = metrics.get("trend_accuracy", float("nan"))
        pc = metrics.get("pearson_correlation", float("nan"))
        logger.info(
            f"  [{ticker}/{indicator}/{timeframe}/{period}] {model_name}: "
            f"trend_acc={ta:.3f}, pearson={pc:.3f}"
        )

    def save(self):
        json_path = self.metrics_dir / "results.json"
        csv_path = self.metrics_dir / "results.csv"

        def _clean(x):
            if isinstance(x, float) and np.isnan(x):
                return None
            return x

        clean = [{k: _clean(v) for k, v in r.items()} for r in self.records]
        with open(json_path, "w") as f:
            json.dump(clean, f, indent=2)
        df = pd.DataFrame(self.records)
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved: {len(self.records)} records → {json_path}")
        return df

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def best_model_per_combination(self) -> dict:
        """Return {(ticker, indicator, timeframe, period): best_model_name}."""
        df = self.to_dataframe()
        if df.empty:
            return {}
        result = {}
        group_cols = [c for c in self.KEY_COLS if c != "model"]
        for keys, grp in df.groupby(group_cols):
            valid = grp.dropna(subset=["trend_accuracy"])
            if not valid.empty:
                best = valid.loc[valid["trend_accuracy"].idxmax(), "model"]
                result[keys] = best
        return result
