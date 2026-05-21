# Rapport Final — Prévision d'Indicateurs Techniques sur le Marché Actions

---

## Table des Matières

1. [Contexte et Problématique](#1-contexte-et-problématique)
2. [Données et Méthodologie](#2-données-et-méthodologie)
3. [Analyse Exploratoire : Impact du Régime de Marché](#3-analyse-exploratoire--impact-du-régime-de-marché)
4. [Justification du Choix de la Période Bull Market Pré-COVID](#4-justification-du-choix-de-la-période-bull-market-pré-covid)
5. [Analyse Détaillée — Période Bull Market (Pré-COVID)](#5-analyse-détaillée--période-bull-market-pré-covid)
6. [Score Hackathon](#6-score-hackathon)
7. [Conclusions et Recommandations](#7-conclusions-et-recommandations)

---

## 1. Contexte et Problématique

### 1.1 Objectif

L'objectif de ce projet est de **prévoir les valeurs futures d'indicateurs techniques** (MACD et Stochastique) à partir de données de prix historiques d'actions cotées en bourse. Plus précisément, pour un horizon de **15 pas de temps** (jours ou semaines), nous cherchons à déterminer quel modèle de prévision produit les projections les plus fiables en termes de :
- **Direction** (la tendance prédite est-elle la bonne ?) → métrique `trend_accuracy`
- **Amplitude** (les valeurs prédites sont-elles proches des valeurs réelles ?) → métrique `pearson_correlation`

### 1.2 Enjeu

La prévision d'indicateurs techniques est un pilier de la finance quantitative. Un modèle capable de projeter correctement le MACD ou le Stochastique pourrait alimenter des stratégies de trading systématique. Cependant, les marchés financiers traversent des **régimes** très différents (hausse, krach, inflation, rebond spéculatif), et un modèle performant dans un contexte peut s'effondrer dans un autre. C'est cette question fondamentale que nous avons voulu adresser.

---

## 2. Données et Méthodologie

### 2.1 Univers de données

| Paramètre | Valeurs |
| :--- | :--- |
| **Tickers (actifs)** | AAPL (Apple), NVDA (Nvidia), MSFT (Microsoft) |
| **Granularités temporelles** | Daily (journalier), Weekly (hebdomadaire, par ré-échantillonnage) |
| **Indicateurs techniques** | MACD, Stochastique |
| **Horizon de prévision** | 15 pas de temps |

### 2.2 Périodes de marché (stratification)

Plutôt que de tester nos modèles sur une unique fenêtre temporelle arbitraire, nous avons **stratifié les données en 5 régimes de marché distincts**, chacun correspondant à une dynamique financière fondamentalement différente :

| Période | Dates | Caractéristique |
| :--- | :--- | :--- |
| `bull_pre_covid` | Janv. 2017 — Fév. 2020 | Marché haussier soutenu, faible volatilité |
| `covid_crash` | Janv. 2020 — Déc. 2021 | Choc brutal suivi d'un rebond en V rapide |
| `bear_inflation` | Janv. 2022 — Déc. 2022 | Marché baissier piloté par l'inflation et les taux |
| `ai_rebound` | Janv. 2023 — Déc. 2024 | Rebond spéculatif autour de l'IA, forte volatilité sectorielle |
| `full_10y` | Janv. 2015 — Déc. 2024 | Ensemble complet couvrant tous les régimes |

### 2.3 Modèles testés

Nous avons comparé **8 modèles** couvrant trois grandes familles algorithmiques :

| Famille | Modèles |
| :--- | :--- |
| **Statistiques temporelles** | ARIMA, SARIMA, Holt-Winters |
| **Machine Learning classique** | Random Forest, SVR (Support Vector Regression) |
| **Deep Learning** | LSTM, GRU, Transformer |

### 2.4 Métriques d'évaluation

Pour chaque combinaison (ticker × timeframe × période × indicateur × modèle), nous avons calculé :
- **`trend_accuracy`** : proportion de pas de temps où le modèle prédit correctement la direction (hausse ou baisse) de l'indicateur.
- **`pearson_correlation`** : corrélation de Pearson entre les valeurs prédites et les valeurs réelles, mesurant la fidélité de la forme du signal.

Chaque métrique est analysée séparément pour le **MACD** et le **Stochastique**, donnant 4 axes d'évaluation :
`trend_accuracy_macd`, `trend_accuracy_stochastic`, `pearson_correlation_macd`, `pearson_correlation_stochastic`.

### 2.5 Matrice expérimentale

Au total, le pipeline a généré **3 tickers × 2 timeframes × 5 périodes × 2 indicateurs × 8 modèles = 480 expériences** (432 lignes exploitables après exclusion des cas sans données suffisantes).

---

## 3. Analyse Exploratoire : Impact du Régime de Marché

### 3.1 Tous les régimes ne se valent pas

Les performances moyennes (tous modèles, tous tickers, tous timeframes confondus) révèlent une hiérarchie claire selon le régime de marché :

**Précision de la Tendance (Trend Accuracy)** :

| Période | MACD | Stochastique |
| :--- | :---: | :---: |
| `full_10y` | **0.5401** | 0.4801 |
| `bull_pre_covid` | 0.5388 | **0.5030** |
| `covid_crash` | 0.5305 | 0.5011 |
| `bear_inflation` | 0.5209 | 0.4548 |
| `ai_rebound` | 0.5166 | 0.4751 |

**Corrélation de Pearson** :

| Période | MACD | Stochastique |
| :--- | :---: | :---: |
| `covid_crash` | **0.1766** | 0.1029 |
| `full_10y` | 0.1414 | **0.1426** |
| `bear_inflation` | 0.1389 | 0.0165 |
| `bull_pre_covid` | 0.0803 | 0.0079 |
| `ai_rebound` | 0.0737 | 0.0039 |

### 3.2 Variabilité critique des performances

Un même modèle peut voir ses performances varier de manière dramatique selon le régime. Voici des exemples concrets :

| Ticker | Modèle | Période la plus forte | Période la plus faible | Écart |
| :--- | :--- | :--- | :--- | :--- |
| **NVDA** | ARIMA | Bull Market (0.914) | AI Rebound (0.511) | **40.3%** |
| **AAPL** | SVR | Bull Market (0.847) | Covid Crash (0.345) | **50.2%** |
| **MSFT** | RandomForest | Bear Inflation (0.693) | Covid Crash (0.623) | **7.0%** |

> **Constat clé** : Certains modèles comme ARIMA ou SVR présentent une amplitude de variation de 40–50% entre leur meilleure et leur pire période, tandis que RandomForest est remarquablement stable (~7% de variation).

### 3.3 Pourquoi ne pas choisir une période au hasard ?

Tester un modèle sur une seule période (ou uniquement sur `full_10y`) expose à deux risques majeurs :
- **Biais de survie** : si on teste par hasard sur le Bull Market, on surestime massivement la robustesse de la stratégie.
- **Risque de ruine** : en ignorant la performance en Bear Inflation, on pourrait déployer un modèle qui perdrait tout son capital lors du prochain cycle inflationniste.

**La stratification par période est donc la seule méthode rigoureuse** pour garantir qu'un modèle est prêt à affronter l'imprévisibilité du marché réel.

---

## 4. Justification du Choix de la Période Bull Market Pré-COVID

Après cette analyse exploratoire multi-régimes, nous avons choisi de concentrer notre étude approfondie sur la période **`bull_pre_covid`** (Janvier 2017 — Février 2020). Voici pourquoi :

1. **Meilleure Trend Accuracy sur le Stochastique** (0.5030) : C'est la seule période où les modèles arrivent en moyenne à dépasser le seuil de 50% de prédiction correcte de la direction sur cet indicateur difficile.
2. **Forte Trend Accuracy sur le MACD** (0.5388) : Quasiment à égalité avec le `full_10y` mais sur une fenêtre temporelle homogène (un seul régime).
3. **Concentration des configurations gagnantes** : Sur les 6 configurations ayant atteint le seuil d'excellence $S \geq 0.55$ du Score Hackathon, **4 proviennent du Bull Market pré-COVID**.
4. **Régime le plus représentatif d'un marché "normal"** : Contrairement au crash COVID ou au rebond IA, le Bull Market pré-COVID offre une période prolongée de croissance organique, reflétant le comportement le plus fréquent des marchés sur le long terme (~70% du temps historiquement).
5. **Pertinence opérationnelle** : Un trader souhaitant déployer un modèle cherche d'abord à performer dans les conditions de marché les plus courantes avant de se protéger contre les événements extrêmes.

---

## 5. Analyse Détaillée — Période Bull Market (Pré-COVID)

### 5.1 Performances par Ticker

Pour chaque actif, nous avons identifié les meilleurs modèles selon les 4 métriques clés.

#### Apple (AAPL)
- **Meilleur pour `trend_accuracy_macd`** : SVR (Weekly) avec **0.8476**
- **Meilleur pour `trend_accuracy_stochastic`** : SVR (Weekly) avec **0.5549**
- **Meilleur pour `pearson_correlation_macd`** : SVR (Weekly) avec **0.9374**
- **Meilleur pour `pearson_correlation_stochastic`** : LSTM (Weekly) avec **0.1754**

*Conclusion AAPL :* Le modèle SVR domine la quasi-totalité des métriques sur une base hebdomadaire, avec une corrélation quasi-parfaite sur le MACD. Le Stochastique reste globalement difficile à modéliser, le LSTM s'en sortant le mieux en corrélation.

#### Nvidia (NVDA)
- **Meilleur pour `trend_accuracy_macd`** : ARIMA (Weekly) avec **0.9143**
- **Meilleur pour `trend_accuracy_stochastic`** : ARIMA (Weekly) avec **0.6374**
- **Meilleur pour `pearson_correlation_macd`** : Holt-Winters (Weekly) avec **0.8546**
- **Meilleur pour `pearson_correlation_stochastic`** : SARIMA (Weekly) avec **0.4109**

*Conclusion NVDA :* Les approches statistiques classiques (ARIMA, SARIMA, Holt-Winters) dominent largement sur NVDA. La croissance explosive de ce titre durant cette période favorise les modèles de suivi de tendance simples.

#### Microsoft (MSFT)
- **Meilleur pour `trend_accuracy_macd`** : SVR (Daily) avec **0.5849**
- **Meilleur pour `trend_accuracy_stochastic`** : Random Forest (Weekly) avec **0.6264**
- **Meilleur pour `pearson_correlation_macd`** : SVR (Daily) avec **0.2715**
- **Meilleur pour `pearson_correlation_stochastic`** : SVR (Weekly) avec **0.6430**

*Conclusion MSFT :* Le SVR et le Random Forest dominent. Contrairement aux autres titres, le MACD est mieux capté en Daily, tandis que le Stochastique fonctionne mieux en Weekly.

### 5.2 Comportement Global des Actifs (Difficulté de prédiction)

En observant les métriques indépendamment, les actifs ne se comportent pas de la même manière face aux indicateurs :

- **Tendance (MACD)** : **NVDA** est le plus facile à lire (0.610 de précision en moyenne), très loin devant AAPL (0.521) et MSFT (0.484).
- **Tendance (Stochastique)** : **NVDA** (0.520) et **MSFT** (0.515) sont au coude-à-coude, AAPL étant plus difficile à prévoir (0.472).
- **Corrélation de Pearson (MACD)** : Seul **NVDA** offre une corrélation moyenne significative (0.262). AAPL et MSFT sont quasiment à zéro.
- **Corrélation de Pearson (Stochastique)** : C'est l'inverse — **MSFT** (0.061) tire légèrement son épingle du jeu, tandis qu'AAPL est en moyenne négatif (-0.038).

> **Enseignement** : La « lisibilité » d'un actif dépend fortement de l'indicateur technique utilisé. NVDA est idéal pour le MACD, MSFT pour le Stochastique. Il n'existe pas de modèle unique adapté à tous les cas.

---

### 5.3 Comparaison Générale des Modèles

#### Palmarès des meilleures prédictions globales
- **Meilleur pour `trend_accuracy_macd`** : **ARIMA** sur **NVDA** (Weekly) → **0.9143**
- **Meilleur pour `trend_accuracy_stochastic`** : **ARIMA** sur **NVDA** (Weekly) → **0.6374**
- **Meilleur pour `pearson_correlation_macd`** : **SVR** sur **AAPL** (Weekly) → **0.9374**
- **Meilleur pour `pearson_correlation_stochastic`** : **SVR** sur **MSFT** (Weekly) → **0.6430**

#### Top 3 des Modèles en Moyenne (Tous Tickers confondus)

**`trend_accuracy_macd`** :
1. **SVR** (0.619)
2. **HoltWinters** (0.605)
3. **SARIMA** (0.589)

**`trend_accuracy_stochastic`** :
1. **RandomForest** (0.543)
2. **SVR** (0.524)
3. **SARIMA** (0.518)

**`pearson_correlation_macd`** :
1. **SVR** (0.292)
2. **HoltWinters** (0.190)
3. **SARIMA** (0.187)

**`pearson_correlation_stochastic`** :
1. **SVR** (0.130)
2. **SARIMA** (0.072)
3. **RandomForest** (0.070)

> **Le SVR se détache nettement** comme le modèle le plus polyvalent, apparaissant dans le Top 2 de chacune des 4 métriques.

---

### 5.4 Comparaison par Famille de Modèles

Si l'on regroupe les algorithmes par famille :

- **Précision de Tendance (MACD)** : Les modèles **Statistiques** dominent légèrement (0.573 vs 0.559 pour le ML). En revanche, le **Machine Learning** prend la tête sur la tendance **Stochastique** (0.534).
- **Corrélation de Pearson** : Le **Machine Learning** écrase la concurrence avec une corrélation MACD de 0.185 (contre 0.132 pour les stats) et une corrélation Stochastique de 0.100 (contre des valeurs négatives pour le reste).
- **Deep Learning (LSTM, GRU, Transformer)** : Systématiquement la famille la moins performante, affichant la pire tendance MACD (0.490) et des corrélations moyennes négatives (-0.041 et -0.029).

---

### 5.5 Le « Flop 3 » : Les Pires Prédictions

Il est aussi instructif d'observer ce qui échoue :

- **Pires modèles pour la tendance** : Le **GRU** et le **Transformer** ont la pire précision MACD (≈ 0.485). Pour le Stochastique, c'est **ARIMA** (0.443) qui échoue le plus lourdement.
- **Le pire contre-sens (Corrélation MACD)** : Le modèle **ARIMA sur AAPL (Weekly)** a produit une corrélation MACD de **-0.9564** ! Il a parié presque parfaitement à l'envers de la vraie dynamique du marché.
- **Le pire contre-sens (Corrélation Stochastique)** : Le modèle **GRU sur AAPL (Weekly)** s'effondre avec une corrélation de **-0.5018**.

> **Enseignement** : Les modèles statistiques (ARIMA) et le Deep Learning (GRU) peuvent produire des prédictions **diamétralement opposées** à la réalité quand le signal ne correspond pas à leurs hypothèses structurelles. C'est un risque opérationnel majeur.

---

### 5.6 Comparaison des Timeframes (Daily vs Weekly)

L'échelle de temps a un impact majeur sur la capacité prédictive :

| Métrique | Weekly | Daily | Gagnant |
| :--- | :---: | :---: | :---: |
| `trend_accuracy_macd` | **0.5688** | 0.5088 | Weekly |
| `trend_accuracy_stochastic` | **0.5165** | 0.4896 | Weekly |
| `pearson_correlation_macd` | **0.1524** | 0.0081 | Weekly |
| `pearson_correlation_stochastic` | -0.0298 | **0.0456** | Daily |

> **L'échelle hebdomadaire (Weekly) domine très nettement** pour détecter la direction et la corrélation lors d'un marché haussier, surtout sur le MACD. La seule exception concerne la corrélation sur le Stochastique, où le Daily s'en tire marginalement mieux (mais reste proche de zéro dans les deux cas).

---

## 6. Score Hackathon

Le Score Hackathon $S$ agrège les 4 métriques en un indicateur unique :

$$S = \frac{T_{\text{MACD}} + \max(0,\rho_{\text{MACD}}) + T_{\text{Stoch}} + \max(0,\rho_{\text{Stoch}})}{4}$$

L'objectif est d'atteindre $S \geq 0.55$.

### 6.1 Configurations atteignant l'objectif

Sur l'ensemble des 217 combinaisons testées, **seules 6 configurations** ont dépassé le seuil. Toutes proviennent de données **hebdomadaires (Weekly)** :

| Actif | Période | Modèle | Tendance MACD | Corr. MACD | Tendance Stoch | Corr. Stoch | **Score** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NVDA** | `bull_pre_covid` | **SARIMA** | 82.8% | 0.654 | 60.4% | 0.411 | **0.6246** |
| **AAPL** | `bull_pre_covid` | **SVR** | 84.7% | 0.937 | 55.4% | 0.097 | **0.6093** |
| **NVDA** | `bull_pre_covid` | **ARIMA** | 91.4% | 0.809 | 63.7% | 0.000 | **0.5902** |
| **NVDA** | `bull_pre_covid` | **HoltWinters** | 91.4% | 0.854 | 50.5% | 0.012 | **0.5716** |
| **NVDA** | `covid_crash` | **GRU** | 63.1% | 0.807 | 57.1% | 0.228 | **0.5596** |
| **NVDA** | `covid_crash` | **SARIMA** | 58.3% | 0.386 | 66.0% | 0.578 | **0.5521** |

### 6.2 Moyenne du Score Hackathon par Modèle

Si l'on ne devait retenir qu'un modèle « gagnant » pour un marché haussier :
1. **SVR** (0.404)
2. **SARIMA** (0.366)
3. **HoltWinters** (0.343)
4. **RandomForest** (0.326)

---

## 7. Conclusions et Recommandations

### 7.1 Résultats clés

1. **La période compte autant que le modèle.** Un même algorithme peut varier de 50% de performance selon le régime de marché. La stratification temporelle n'est pas un luxe, c'est une nécessité méthodologique.
2. **Le Bull Market pré-COVID est le régime le plus propice** à la prévision d'indicateurs techniques, car la tendance est forte, stable et directionnelle. C'est là que les modèles atteignent leurs meilleures performances (4 des 6 meilleures configurations globales).
3. **L'échelle Weekly surpasse le Daily** sur 3 des 4 métriques. Le bruit intra-journalier détruit la capacité des modèles à capter un signal significatif.
4. **Le SVR est le modèle le plus polyvalent**, apparaissant systématiquement dans le Top 2 de chaque métrique et dominant le classement du Score Hackathon.
5. **Le Deep Learning déçoit sur cette tâche.** Les LSTM, GRU et Transformer sous-performent systématiquement par rapport au Machine Learning classique et aux modèles statistiques. Cela s'explique probablement par la faible quantité de données disponibles sur chaque sous-période et par la nature non-stationnaire des séries financières.
6. **Le MACD est plus prédictible que le Stochastique.** Les corrélations de Pearson sur le MACD atteignent 0.93 dans les meilleures configurations, tandis que le Stochastique dépasse rarement 0.40-0.50.

### 7.2 Recommandations opérationnelles

- **Pour un déploiement en production** : Utiliser un modèle **SVR** ou **SARIMA** sur des données agrégées en **Weekly**, en ciblant prioritairement le signal MACD.
- **Adapter le modèle au régime** : Ne pas entraîner sur `full_10y` aveuglément. Utiliser une détection de régime de marché et basculer sur le modèle adapté (ARIMA/SARIMA pour les marchés calmes, RandomForest pour absorber la variance en période de turbulence).
- **Approche hybride (Ensemble)** : Combiner un ARIMA/SARIMA pour capter les signaux forts en Bull Market avec un RandomForest comme « filet de sécurité » robuste lorsque la volatilité augmente.
- **Ne pas négliger la dimension « ticker »** : NVDA est beaucoup plus lisible que MSFT avec le MACD, mais l'inverse est vrai avec le Stochastique. Le choix du modèle doit être contextualisé par actif.
