"""
Financial Indicator Forecasting Pipeline — Hackathon Entry Point

Usage:
    python main.py --data_dir data/raw --tickers AAPL MSFT NVDA \
                   --timeframes daily weekly \
                   --periods bull_pre_covid covid_crash bear_inflation ai_rebound full_10y \
                   --indicators macd stochastic \
                   --horizon 15

Matrix: 3 tickers × 2 timeframes × 5 periods × 2 indicators = 60 experiments
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "results" / "training.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")

ALL_PERIODS = ["bull_pre_covid", "covid_crash", "bear_inflation", "ai_rebound", "full_10y"]


def parse_args():
    parser = argparse.ArgumentParser(description="Financial Indicator Forecasting Pipeline")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--timeframes", nargs="+", default=["daily", "weekly"])
    parser.add_argument("--periods", nargs="+", default=ALL_PERIODS)
    parser.add_argument("--indicators", nargs="+", default=["macd", "stochastic"])
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def check_dependencies():
    required = ["numpy", "pandas", "sklearn", "statsmodels", "scipy", "matplotlib", "seaborn", "openpyxl"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(f"Missing required packages: {missing}")
        sys.exit(1)
    for pkg, desc in {"torch": "LSTM/GRU/Transformer", "xgboost": "XGBoost"}.items():
        try:
            __import__(pkg)
            logger.info(f"  [OK] {pkg} — {desc}")
        except ImportError:
            logger.warning(f"  [SKIP] {pkg} — {desc} will be skipped")


def print_summary(store, elapsed: float):
    df = store.to_dataframe()
    if df.empty:
        print("\nNo results.")
        return

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — average across all combinations")
    print("=" * 70)

    avg = df.groupby("model")[["trend_accuracy", "pearson_correlation"]].mean()
    avg["score"] = avg.mean(axis=1)
    avg = avg.sort_values("score", ascending=False)

    print(f"\n{'Model':<20} {'Trend Acc':>12} {'Pearson':>12} {'Score':>10}")
    print("-" * 58)
    for model, row in avg.iterrows():
        def fmt(v): return f"{v:.4f}" if v == v else "N/A"
        print(f"{model:<20} {fmt(row['trend_accuracy']):>12} {fmt(row['pearson_correlation']):>12} {fmt(row['score']):>10}")

    winner = avg.dropna().head(1)
    if not winner.empty:
        print(f"\nBest overall model: {winner.index[0]} (score={winner['score'].iloc[0]:.4f})")

    # Per-period breakdown
    if "period" in df.columns:
        print(f"\n{'Period':<20} {'Trend Acc':>12} {'Pearson':>12}")
        print("-" * 46)
        for period, grp in df.groupby("period"):
            ta = grp["trend_accuracy"].mean()
            pc = grp["pearson_correlation"].mean()
            print(f"{period:<20} {ta:>12.4f} {pc:>12.4f}")

    print(f"\nTotal records: {len(df)} / {len(df.drop_duplicates(['ticker','indicator','timeframe','period','model']))}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Results: results/metrics/  |  Plots: results/plots/  |  Report: report.md")
    print("=" * 70)


def main():
    args = parse_args()

    data_dir    = PROJECT_ROOT / args.data_dir
    models_dir  = PROJECT_ROOT / args.models_dir
    results_dir = PROJECT_ROOT / args.results_dir
    plots_dir   = results_dir / "plots"
    metrics_dir = results_dir / "metrics"

    for d in [data_dir, models_dir, plots_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    total = len(args.tickers) * len(args.timeframes) * len(args.periods) * len(args.indicators)
    logger.info("=" * 60)
    logger.info("Financial Indicator Forecasting Pipeline")
    logger.info(f"Tickers:    {args.tickers}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"Periods:    {args.periods}")
    logger.info(f"Indicators: {args.indicators}")
    logger.info(f"Horizon:    {args.horizon} steps")
    logger.info(f"Total combos: {total} experiments × N models")
    logger.info("=" * 60)

    check_dependencies()

    if args.dry_run:
        from src.data_loader import discover_files, PERIODS
        print("\n[DRY RUN] Checking structure...")
        fm = discover_files(str(data_dir), args.tickers)
        for t in args.tickers:
            print(f"  {t}: {'FOUND → ' + fm[t] if t in fm else 'MISSING'}")
        print(f"\nPeriods available: {list(PERIODS.keys())}")
        print(f"\nTotal experiments planned: {total}")
        return

    # ── Load data ──────────────────────────────────────────────────────────────
    from src.data_loader import load_all_data
    logger.info("Loading and slicing data...")
    data = load_all_data(str(data_dir), args.tickers, args.timeframes, args.periods)

    total_slices = sum(1 for t in data for tf in data[t] for p in data[t][tf])
    if total_slices == 0:
        logger.error("No data slices loaded. Check that XLSX files are in data/raw/")
        sys.exit(1)
    logger.info(f"Loaded {total_slices} data slices")

    # ── Run pipeline ───────────────────────────────────────────────────────────
    from src.pipeline import run_pipeline
    t0 = time.time()
    store, predictions_cache = run_pipeline(
        data=data,
        tickers=args.tickers,
        timeframes=args.timeframes,
        periods=args.periods,
        indicators=args.indicators,
        horizon=args.horizon,
        models_dir=str(models_dir),
        metrics_dir=str(metrics_dir),
    )
    elapsed = time.time() - t0

    # ── Visualizations ─────────────────────────────────────────────────────────
    try:
        from src.visualize import generate_all_plots
        results_df = store.to_dataframe()
        best_models = store.best_model_per_combination()
        logger.info("Generating plots...")
        generate_all_plots(data, results_df, predictions_cache, best_models, str(plots_dir))
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)

    # ── Report ─────────────────────────────────────────────────────────────────
    try:
        from src.report_gen import generate_report
        results_df = store.to_dataframe()
        report_path = PROJECT_ROOT / "report.md"
        generate_report(results_df, data, str(plots_dir), str(report_path))
        logger.info(f"Report saved to {report_path}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)

    print_summary(store, elapsed)


if __name__ == "__main__":
    main()
