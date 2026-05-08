"""Visualization: indicator plots, prediction charts, heatmaps, rankings."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

THEME = {
    "bg": "#ffffff", "fg": "#333333",
    "accent": "#007bff", "accent2": "#dc3545",
    "accent3": "#ffc107", "grid": "#eeeeee",
}

PERIOD_COLORS = {
    "bull_pre_covid":  "#00d4ff",
    "covid_crash":     "#ff6b6b",
    "bear_inflation":  "#ffd93d",
    "ai_rebound":      "#7bed9f",
    "full_10y":        "#a29bfe",
}


def _apply_dark_theme(fig, axes):
    fig.patch.set_facecolor(THEME["bg"])
    ax_list = axes.flatten() if hasattr(axes, "flatten") else ([axes] if not hasattr(axes, "__iter__") else list(axes))
    for ax in ax_list:
        ax.set_facecolor(THEME["bg"])
        ax.tick_params(colors=THEME["fg"])
        for spine in ax.spines.values():
            spine.set_color(THEME["grid"])
        ax.xaxis.label.set_color(THEME["fg"])
        ax.yaxis.label.set_color(THEME["fg"])
        ax.title.set_color(THEME["fg"])
        ax.yaxis.set_tick_params(labelcolor=THEME["fg"])
        ax.xaxis.set_tick_params(labelcolor=THEME["fg"])
    return fig


def plot_indicators(data: dict, plots_dir: str):
    """Plot MACD and Stochastic for each ticker — one plot per ticker showing full_10y daily."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .indicators import compute_all_indicators

    plots_dir = Path(plots_dir)

    for ticker, tf_data in data.items():
        # Prefer full_10y daily; fallback to any available slice
        df_raw = None
        for tf in ["daily", "weekly"]:
            tf_slices = tf_data.get(tf, {})
            for period in ["full_10y"] + list(tf_slices.keys()):
                if period in tf_slices:
                    df_raw = tf_slices[period]
                    break
            if df_raw is not None:
                break
        if df_raw is None:
            continue

        try:
            df_ind = compute_all_indicators(df_raw).dropna()
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            _apply_dark_theme(fig, axes)
            idx = df_ind.index

            axes[0].plot(idx, df_ind["close"], color=THEME["accent"], lw=0.8)
            axes[0].set_title(f"{ticker} — Close Price", fontsize=12)
            axes[0].set_ylabel("Price", color=THEME["fg"])

            axes[1].plot(idx, df_ind["macd"], color=THEME["accent"], lw=0.8, label="MACD")
            axes[1].plot(idx, df_ind["macd_signal"], color=THEME["accent2"], lw=0.8, label="Signal")
            axes[1].bar(idx, df_ind["macd_hist"], color=THEME["accent3"], alpha=0.4, width=0.8)
            axes[1].set_title("MACD(12,26,9)", fontsize=11)
            axes[1].set_ylabel("MACD", color=THEME["fg"])
            axes[1].legend(facecolor=THEME["bg"], labelcolor=THEME["fg"], fontsize=8)

            axes[2].plot(idx, df_ind["stoch_k"], color=THEME["accent"], lw=0.8, label="%K")
            axes[2].plot(idx, df_ind["stoch_d"], color=THEME["accent2"], lw=0.8, label="%D")
            axes[2].axhline(80, color=THEME["grid"], ls="--", lw=0.6)
            axes[2].axhline(20, color=THEME["grid"], ls="--", lw=0.6)
            axes[2].set_title("Stochastic(14,3,5)", fontsize=11)
            axes[2].set_ylabel("%K/%D", color=THEME["fg"])
            axes[2].set_ylim(0, 100)
            axes[2].legend(facecolor=THEME["bg"], labelcolor=THEME["fg"], fontsize=8)

            plt.tight_layout()
            out = plots_dir / f"indicators_{ticker}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=THEME["bg"])
            plt.close(fig)
            logger.info(f"Saved {out}")
        except Exception as e:
            logger.error(f"Indicator plot failed for {ticker}: {e}")


def plot_predictions(predictions_cache: dict, best_models: dict, plots_dir: str):
    """Best model prediction vs actual — one plot per (ticker, indicator, timeframe, period)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)

    for keys, best_model_name in best_models.items():
        if len(keys) == 4:
            ticker, indicator, timeframe, period = keys
        else:
            continue

        cache_key = (ticker, indicator, timeframe, period, best_model_name)
        if cache_key not in predictions_cache:
            continue

        entry = predictions_cache[cache_key]
        y_true, y_pred = entry["y_true"], entry["y_pred"]
        metrics = entry["metrics"]

        try:
            steps = np.arange(1, len(y_true) + 1)
            fig, ax = plt.subplots(figsize=(12, 5))
            _apply_dark_theme(fig, ax)

            ax.plot(steps, y_true, color=THEME["accent"], lw=2, label="Actual", marker="o", ms=3)
            ax.plot(steps, y_pred, color=THEME["accent2"], lw=2, ls="--",
                    label=f"Predicted ({best_model_name})", marker="s", ms=3)
            spread = np.abs(y_pred) * 0.05 + 1e-8
            ax.fill_between(steps, y_pred - spread, y_pred + spread, alpha=0.15, color=THEME["accent2"])

            ta = metrics.get("trend_accuracy", float("nan"))
            pc = metrics.get("pearson_correlation", float("nan"))
            ax.set_title(
                f"{ticker} | {indicator} | {timeframe} | {period}\n"
                f"{best_model_name} — Trend Acc: {ta:.3f}  Pearson: {pc:.3f}",
                color=THEME["fg"], fontsize=10
            )
            ax.set_xlabel("Forecast Step", color=THEME["fg"])
            ax.set_ylabel(indicator.upper(), color=THEME["fg"])
            ax.legend(facecolor=THEME["bg"], labelcolor=THEME["fg"])

            out = plots_dir / f"prediction_{ticker}_{indicator}_{timeframe}_{period}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=THEME["bg"])
            plt.close(fig)
            logger.info(f"Saved {out}")
        except Exception as e:
            logger.error(f"Prediction plot failed {keys}: {e}")


def plot_heatmap(results_df: pd.DataFrame, plots_dir: str):
    """Heatmap: model × ticker for each indicator, averaged across periods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = Path(plots_dir)

    for metric in ["trend_accuracy", "pearson_correlation"]:
        for indicator in results_df["indicator"].unique() if "indicator" in results_df.columns else []:
            sub = results_df[results_df["indicator"] == indicator]
            try:
                pivot = sub.pivot_table(index="model", columns="ticker", values=metric, aggfunc="mean")
                if pivot.empty:
                    continue
                fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.5), max(5, pivot.shape[0] * 0.6)))
                _apply_dark_theme(fig, ax)
                sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                            linewidths=0.5, linecolor=THEME["grid"], cbar_kws={"shrink": 0.8})
                ax.set_title(f"{metric.replace('_',' ').title()} — {indicator.upper()} (avg over periods)",
                             fontsize=12, color=THEME["fg"])
                plt.xticks(rotation=0, color=THEME["fg"])
                plt.yticks(rotation=0, color=THEME["fg"])
                out = plots_dir / f"heatmap_{metric}_{indicator}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=THEME["bg"])
                plt.close(fig)
                logger.info(f"Saved {out}")
            except Exception as e:
                logger.error(f"Heatmap failed {metric}/{indicator}: {e}")


def plot_ranking(results_df: pd.DataFrame, plots_dir: str):
    """Ranked bar chart of models by average score across all combinations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    try:
        avg = results_df.groupby("model")[["trend_accuracy", "pearson_correlation"]].mean().dropna()
        avg["score"] = avg.mean(axis=1)
        avg = avg.sort_values("score", ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        _apply_dark_theme(fig, axes)

        for ax, col, color in zip(axes, ["trend_accuracy", "pearson_correlation"],
                                   [THEME["accent"], THEME["accent2"]]):
            vals = avg[col]
            bars = ax.barh(avg.index, vals, color=color, alpha=0.85)
            ax.set_xlabel(col.replace("_", " ").title(), color=THEME["fg"])
            ax.set_title(col.replace("_", " ").title(), color=THEME["fg"])
            ax.axvline(vals.mean(), color=THEME["accent3"], ls="--", lw=1, label="Mean")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", color=THEME["fg"], fontsize=8)
            ax.legend(facecolor=THEME["bg"], labelcolor=THEME["fg"])

        plt.suptitle("Model Ranking — Average Across All 60 Combinations", color=THEME["fg"], fontsize=13)
        plt.tight_layout()
        out = plots_dir / "model_ranking.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=THEME["bg"])
        plt.close(fig)
        logger.info(f"Saved {out}")
    except Exception as e:
        logger.error(f"Ranking plot failed: {e}")


def plot_period_comparison(results_df: pd.DataFrame, plots_dir: str):
    """Line chart: trend accuracy per market period, per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    if "period" not in results_df.columns:
        return

    PERIOD_ORDER = ["bull_pre_covid", "covid_crash", "bear_inflation", "ai_rebound", "full_10y"]

    try:
        for metric in ["trend_accuracy", "pearson_correlation"]:
            pivot = results_df.groupby(["model", "period"])[metric].mean().unstack("period")
            pivot = pivot.reindex(columns=[p for p in PERIOD_ORDER if p in pivot.columns])
            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(13, 6))
            _apply_dark_theme(fig, ax)

            for model in pivot.index:
                ax.plot(pivot.columns, pivot.loc[model], marker="o", lw=1.5, ms=5, label=model)

            ax.set_title(f"{metric.replace('_',' ').title()} per Market Period", color=THEME["fg"], fontsize=12)
            ax.set_xlabel("Period", color=THEME["fg"])
            ax.set_ylabel(metric.replace("_", " ").title(), color=THEME["fg"])
            ax.legend(facecolor=THEME["bg"], labelcolor=THEME["fg"], fontsize=7, ncol=2)
            plt.xticks(rotation=15, color=THEME["fg"])

            out = plots_dir / f"period_comparison_{metric}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=THEME["bg"])
            plt.close(fig)
            logger.info(f"Saved {out}")
    except Exception as e:
        logger.error(f"Period comparison plot failed: {e}")


def plot_daily_vs_weekly(results_df: pd.DataFrame, plots_dir: str):
    """Side-by-side bar: daily vs weekly performance per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    try:
        for metric in ["trend_accuracy", "pearson_correlation"]:
            pivot = results_df.groupby(["model", "timeframe"])[metric].mean().unstack("timeframe")
            if pivot.empty or pivot.shape[1] < 2:
                continue
            pivot = pivot.sort_values(pivot.columns[0], ascending=False)
            x = np.arange(len(pivot))
            width = 0.35

            fig, ax = plt.subplots(figsize=(14, 6))
            _apply_dark_theme(fig, ax)

            for i, (col, color) in enumerate(zip(pivot.columns, [THEME["accent"], THEME["accent2"]])):
                ax.bar(x + i * width, pivot[col], width, label=col, color=color, alpha=0.85)

            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(pivot.index, rotation=30, ha="right", color=THEME["fg"])
            ax.set_ylabel(metric.replace("_", " ").title(), color=THEME["fg"])
            ax.set_title(f"Daily vs Weekly — {metric.replace('_', ' ').title()}", color=THEME["fg"], fontsize=12)
            ax.legend(facecolor=THEME["bg"], labelcolor=THEME["fg"])

            out = plots_dir / f"daily_vs_weekly_{metric}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=THEME["bg"])
            plt.close(fig)
            logger.info(f"Saved {out}")
    except Exception as e:
        logger.error(f"Daily vs weekly plot failed: {e}")


def generate_all_plots(data: dict, results_df: pd.DataFrame,
                       predictions_cache: dict, best_models: dict, plots_dir: str):
    plot_indicators(data, plots_dir)
    plot_predictions(predictions_cache, best_models, plots_dir)
    plot_heatmap(results_df, plots_dir)
    plot_ranking(results_df, plots_dir)
    plot_period_comparison(results_df, plots_dir)
    plot_daily_vs_weekly(results_df, plots_dir)
