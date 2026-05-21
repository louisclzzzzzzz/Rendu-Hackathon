# Financial Indicator Forecasting Pipeline

> University hackathon project — benchmarks 9 forecasting models on MACD and Stochastic indicators across labeled market regimes for AAPL, MSFT, and NVDA.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-optional-EE4C2C?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Project structure

```
.
├── main.py                  # Pipeline entry point
├── requirements.txt
├── src/
│   ├── data_loader.py       # XLSX loading, period slicing, weekly resampling
│   ├── indicators.py        # MACD and Stochastic computation
│   ├── models.py            # All 9 model implementations
│   ├── pipeline.py          # Training and evaluation loop
│   ├── evaluate.py          # Metrics (trend accuracy, Pearson, RMSE, MAE)
│   ├── visualize.py         # Plot generation
│   ├── generate_heatmaps.py # Heatmap utilities
│   ├── report_gen.py        # Markdown report generator
│   └── replot.py            # Regenerate plots from saved metrics
├── scripts/
│   ├── compute_scores.py        # Compute hackathon scoring formula
│   ├── run_random_walk.py       # Run random walk baseline on existing combos
│   ├── make_daily_weekly_bar.py # Daily vs weekly bar chart
│   └── make_regime_heatmaps.py  # Model × market regime heatmaps
├── data/
│   └── raw/                 # AAPL.xlsx, MSFT.xlsx, NVDA.xlsx
└── docs/
    ├── presentation.pdf
    └── rapport_hackathon.pdf
```

---

## Key features

- **60 experiments** — 3 tickers × 2 timeframes (daily/weekly) × 5 market regimes × 2 indicators
- **9 models** — Random Walk baseline, ARIMA, SARIMA, Holt-Winters, Random Forest, SVR, XGBoost (optional), LSTM, GRU, Transformer
- **Labeled market regimes** — Bull pre-COVID, COVID crash, Bear/inflation 2022, AI rebound 2023–24, full 10-year window
- **Automatic outputs** — per-combination metrics saved to JSON/CSV, forecast plots, heatmaps, and a Markdown report
- **Graceful degradation** — PyTorch and XGBoost are optional; the pipeline skips them cleanly if not installed

---

## Installation

```bash
git clone https://github.com/louisclzzzzzzz/Rendu-Hackathon.git
cd Rendu-Hackathon
pip install -r requirements.txt
```

Deep learning models (LSTM, GRU, Transformer) require PyTorch:
```bash
pip install torch
```

---

## Usage

### Run the full pipeline

```bash
python main.py
```

This trains all models, evaluates them, saves results under `results/`, and writes a `report.md`.

### Dry run (check data files without training)

```bash
python main.py --dry_run
```

### Custom run

```bash
python main.py --tickers AAPL NVDA \
               --timeframes daily \
               --periods bull_pre_covid ai_rebound \
               --indicators macd \
               --horizon 15
```

### Regenerate plots from saved results

```bash
python src/replot.py
```

### Compute hackathon scores

```bash
python scripts/compute_scores.py
```

---

## Results snapshot

Sample output from `results/plots/`:

| Chart | Description |
|---|---|
| `model_ranking.png` | Average score per model across all 60 combos |
| `heatmap_trend_accuracy.png` | Trend accuracy heatmap: models × tickers |
| `regime_heatmap_all.png` | Score by model × market regime |
| `forecast_AAPL_macd_weekly_ai_rebound.png` | Example forecast overlay |

*Full results (CSV/JSON) are gitignored — run the pipeline to generate them.*

---

## License

MIT