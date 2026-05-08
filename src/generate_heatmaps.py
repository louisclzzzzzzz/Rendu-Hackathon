import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

def generate_heatmaps(csv_path, output_dir, metric='pearson_correlation'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Filter for relevant tickers, timeframes, indicators
    tickers = df['ticker'].unique()
    timeframes = df['timeframe'].unique()
    indicators = df['indicator'].unique()
    
    # Style configuration
    plt.style.use('default')
    bg_color = 'white'
    
    for ticker in tickers:
        for timeframe in timeframes:
            for indicator in indicators:
                subset = df[(df['ticker'] == ticker) & 
                            (df['timeframe'] == timeframe) & 
                            (df['indicator'] == indicator)]
                
                if subset.empty:
                    continue
                
                # Pivot for heatmap: Period x Model
                pivot_df = subset.pivot(index='period', columns='model', values=metric)
                
                # Sort periods for consistency
                period_order = ['bull_pre_covid', 'covid_crash', 'bear_inflation', 'ai_rebound', 'full_10y']
                existing_periods = [p for p in period_order if p in pivot_df.index]
                other_periods = [p for p in pivot_df.index if p not in period_order]
                pivot_df = pivot_df.reindex(existing_periods + other_periods)
                
                # Sort models by average performance in this subset to make heatmap cleaner
                avg_model_perf = pivot_df.mean(axis=0).sort_values(ascending=False)
                pivot_df = pivot_df[avg_model_perf.index]

                from matplotlib.colors import TwoSlopeNorm
                
                fig, ax = plt.subplots(figsize=(14, 9))
                fig.patch.set_facecolor(bg_color)
                ax.set_facecolor(bg_color)
                
                if metric == 'trend_accuracy':
                    # Force 0.4 to be in the red zone and 0.6 to be the start of green
                    # We use a TwoSlopeNorm to define the center point clearly
                    vmin, vcenter, vmax = 0.35, 0.55, 1.0
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                    cmap = 'RdYlGn'
                    cbar_label = 'Trend Accuracy'
                else: # pearson_correlation
                    vmin, vcenter, vmax = -1.0, 0.0, 1.0
                    # Check if all values are positive to adjust scale
                    if pivot_df.min().min() >= 0:
                        vmin, vcenter, vmax = 0.0, 0.5, 1.0
                        cmap = 'YlGnBu'
                    else:
                        cmap = 'seismic'
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                    cbar_label = 'Pearson Correlation'
                
                sns.heatmap(pivot_df, annot=True, cmap=cmap, fmt='.3f', 
                            norm=norm,
                            cbar_kws={'label': cbar_label},
                            annot_kws={'weight': 'bold', 'size': 11},
                            linewidths=1.0, linecolor='#f0f0f0', ax=ax)
                
                plt.title(f'{metric.replace("_", " ").title()} Heatmap: {ticker} | {timeframe} | {indicator}', 
                          fontsize=14, pad=20, color='black')
                plt.xlabel('Model', fontsize=12, color='black')
                plt.ylabel('Market Period', fontsize=12, color='black')
                
                plt.xticks(rotation=45, ha='right', color='black')
                plt.yticks(color='black')
                plt.tight_layout()
                
                suffix = 'pearson' if metric == 'pearson_correlation' else 'trend'
                filename = f"{ticker}_{timeframe}_{indicator}_{suffix}_heatmap.png"
                save_path = output_dir / filename
                plt.savefig(save_path, dpi=150, facecolor=bg_color)
                plt.close()
                print(f"Generated heatmap: {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default='project/results/metrics/results.csv')
    parser.add_argument("--out", default='project/results/plots/heatmap')
    parser.add_argument("--metric", default='pearson_correlation')
    args = parser.parse_args()
    
    generate_heatmaps(args.csv, args.out, args.metric)
