"""Compute the hackathon scoring formula from results.json and export to hackathon_scores.csv."""
import json
import pandas as pd
import numpy as np

with open('results/metrics/results.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Pivoter la table pour avoir MACD et Stochastic sur la meme ligne
pivot_df = df.pivot(
    index=['ticker', 'timeframe', 'period', 'model'],
    columns='indicator',
    values=['trend_accuracy', 'pearson_correlation']
).reset_index()

# Aplatir le MultiIndex des colonnes
pivot_df.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in pivot_df.columns]

# Extraire les metriques
T_MACD = pivot_df['trend_accuracy_macd']
rho_MACD = pivot_df['pearson_correlation_macd']
T_Stoch = pivot_df['trend_accuracy_stochastic']
rho_Stoch = pivot_df['pearson_correlation_stochastic']

# Calculer le score (rho_MACD et rho_Stoch peuvent avoir des NaN s'il manque des donnees)
pivot_df['hackathon_score'] = (
    T_MACD.fillna(0) + 
    np.maximum(0, rho_MACD.fillna(0)) + 
    T_Stoch.fillna(0) + 
    np.maximum(0, rho_Stoch.fillna(0))
) / 4

# Afficher les meilleurs scores
result_valide = pivot_df[pivot_df['hackathon_score'] >= 0.55].sort_values('hackathon_score', ascending=False)
print("Configurations ayant un score >= 0.55 :")
print(result_valide[['ticker', 'period', 'model', 'hackathon_score']].head(20).to_string(index=False))

# Sauvegarder dans un CSV
pivot_df.to_csv('results/metrics/hackathon_scores.csv', index=False)
print("\nLes scores ont ete sauvegardes dans results/metrics/hackathon_scores.csv")
