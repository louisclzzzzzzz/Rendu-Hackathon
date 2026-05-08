import sys
import os
import json
import pandas as pd
from pathlib import Path
import logging
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import ResultsStore
from src.data_loader import load_all_data, LOOK_BACK, DataPipeline
from src.indicators import compute_all_indicators, INDICATOR_TARGETS
from src.models import get_all_models, model_save_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("replot")

def replot():
    results_dir = PROJECT_ROOT / "results"
    metrics_dir = results_dir / "metrics"
    plots_dir = results_dir / "plots"
    models_dir = PROJECT_ROOT / "models"
    data_dir = PROJECT_ROOT / "data" / "raw"
    
    store = ResultsStore(str(metrics_dir))
    results_df = store.to_dataframe()
    
    if results_df.empty:
        print("No results found to replot.")
        return

    best_models = store.best_model_per_combination()
    
    print("Regenerating summary plots (Rankings, Comparisons, Global Heatmaps)...")
    from src.visualize import plot_heatmap, plot_ranking, plot_period_comparison, plot_daily_vs_weekly, plot_indicators
    
    plot_heatmap(results_df, str(plots_dir))
    plot_ranking(results_df, str(plots_dir))
    plot_period_comparison(results_df, str(plots_dir))
    plot_daily_vs_weekly(results_df, str(plots_dir))
    
    # Regenerate metric heatmaps folder
    print("Regenerating metric heatmaps folder...")
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run(["python3", "project/src/generate_heatmaps.py", "--metric", "pearson_correlation"], env=env)
    subprocess.run(["python3", "project/src/generate_heatmaps.py", "--metric", "trend_accuracy"], env=env)
    
    for f in (plots_dir / "heatmap").glob("*_trend_heatmap.png"):
        target = plots_dir / "heatmap" / f.name.replace("_trend_heatmap.png", "_heatmap.png")
        import shutil
        shutil.copy(str(f), str(target))

    # Regenerate prediction plots
    print("Regenerating prediction plots (requires loading models and data)...")
    tickers = results_df['ticker'].unique()
    timeframes = results_df['timeframe'].unique()
    periods = results_df['period'].unique()
    indicators = results_df['indicator'].unique()
    
    data = load_all_data(str(data_dir), tickers, timeframes, periods)
    plot_indicators(data, str(plots_dir)) # Full history plots
    
    from src.visualize import plot_predictions
    predictions_cache = {}
    
    for combo, best_model_name in best_models.items():
        ticker, indicator, timeframe, period = combo
        df = data.get(ticker, {}).get(timeframe, {}).get(period)
        if df is None: continue
        
        indicator_col = INDICATOR_TARGETS[indicator]
        look_back = LOOK_BACK.get(timeframe, 60)
        df_ind = compute_all_indicators(df).dropna(subset=[indicator_col])
        series = df_ind[indicator_col].values.astype(np.float32)
        
        # We need the horizon used during training (default 15)
        horizon = 15 
        pipe = DataPipeline(look_back=look_back, horizon=horizon)
        _, _, (X_test, y_test), _ = pipe.prepare(series)
        
        if len(X_test) == 0: continue
        
        # Load model
        save_path = model_save_path(str(models_dir), best_model_name, ticker, indicator, timeframe, period)
        if not Path(save_path).exists():
            # For ARIMA/SARIMA/HW, the path might be .pkl
            save_path = save_path.replace(".pt", ".pkl")
            if not Path(save_path).exists():
                logger.warning(f"Model file not found for {combo}: {save_path}")
                continue
        
        try:
            # We need to instantiate the model to call .load()
            from src.models import get_all_models
            models = get_all_models(horizon=horizon)
            model = next(m for m in models if m.name == best_model_name)
            model.load(save_path)
            
            raw_preds = model.predict(X_test, horizon=horizon)
            y_test_inv = pipe.inverse_transform(y_test)
            preds_inv = pipe.inverse_transform(raw_preds)
            
            # Re-evaluate to get metrics
            from src.evaluate import evaluate_predictions
            metrics = evaluate_predictions(y_test_inv, preds_inv)
            
            cache_key = (ticker, indicator, timeframe, period, best_model_name)
            predictions_cache[cache_key] = {
                "y_true": y_test_inv[-1],
                "y_pred": preds_inv[-1],
                "metrics": metrics,
            }
        except Exception as e:
            logger.error(f"Error regenerating prediction for {combo}: {e}")

    if predictions_cache:
        plot_predictions(predictions_cache, best_models, str(plots_dir))

if __name__ == "__main__":
    replot()
