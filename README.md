# Financial Indicator Forecasting Pipeline

Ce dépôt contient l'intégralité de la pipeline de prévision d'indicateurs financiers (MACD et Stochastic) développée pour le Hackathon.

## Structure du projet
- `main.py` : Point d'entrée de la pipeline.
- `src/` : Code source (Data Loading, Indicators, Models, Evaluation, Visualization, Report Gen).
- `data/raw/` : Données historiques (XLSX).

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/louisclzzzzzzz/Rendu-Hackathon.git
   cd Rendu-Hackathon
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### 1. Lancer la pipeline complète
Cette commande charge les données, entraîne les modèles, évalue les performances et génère les graphiques ainsi que le rapport final.
```bash
python main.py
```

### 2. Régénérer uniquement les graphiques
Si les résultats existent déjà dans `results/metrics/`, vous pouvez mettre à jour les visuels sans ré-entraîner les modèles :
```bash
python src/replot.py
```

## Fonctionnalités
- Support multi-ticker (AAPL, MSFT, NVDA).
- Analyse par périodes de marché (Bull market, Covid crash, AI Rebound, etc.).
- Comparaison Daily vs Weekly.
- 9 modèles inclus (ARIMA, SARIMA, Holt-Winters, RandomForest, SVR, LSTM, GRU, Transformer).
- Génération automatique de rapports Markdown et Heatmaps.
