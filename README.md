# ML-HousePrices
Prédiction de prix immobiliers avec un Random Forest Regressor — Projet ML (Kaggle)

## House Prices - Prédiction de prix immobiliers

Deuxième projet d'initiation au Machine Learning, réalisé sur le dataset Kaggle "House Prices - Advanced Regression Techniques".

## Objectif
Prédire le prix de vente de maisons à partir de 76 variables (superficie, qualité, année de construction, type de quartier...).

## Démarche

### 1. Exploration des données (EDA)
- Dataset de 1460 maisons, 81 variables au départ
- Analyse de la distribution des prix (asymétrique vers la droite, médiane à 163 000$)
- Identification de 19 colonnes avec des valeurs manquantes

### 2. Nettoyage des données
- Suppression des colonnes avec trop de manquants (>80%) : PoolQC, Alley, Fence, MiscFeature
- Remplacement des NaN par "None" ou 0 quand ils signifient une absence (garage, sous-sol, cheminée...)
- Remplacement par la médiane pour les vraies valeurs manquantes (LotFrontage)
- Encodage des variables texte en nombres via `.cat.codes`

### 3. Modélisation
- Algorithme : Random Forest Regressor (sklearn)
- Séparation train/validation : 80% / 20%
- 100 arbres de décision (`n_estimators=100`)

### 4. Évaluation
- Métrique principale : MAE (Mean Absolute Error) — erreur moyenne en dollars
- Métrique secondaire : RMSE — pénalise davantage les grosses erreurs

## Résultats
MAE (validation) | environ 17 465 $ 
RMSE (validation) | environ 28 342 $ 

Erreur moyenne d'environ 10% par rapport au prix médian (163 000$).


## Technologies
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Fichiers
- `notebook-houseprices.ipynb` : code complet et commenté du projet
- `submission.csv` : prédictions finales sur le set de test Kaggle

