# Machine Learning Algorithms Repository

Ce dépôt contient une collection d'algorithmes de machine learning implémentés en Python, couvrant des approches supervisées et non supervisées. Chaque répertoire contient des fichiers Python (`.py`) et leurs versions correspondantes sous forme de notebooks Jupyter (`.ipynb`).

## Structure du Dépôt

Le dépôt est organisé par type d'algorithme :

### 1. [Analyse en Composantes Principales (PCA)](./Analyse_en_Composantes_Principales/)
- **Objectif** : Réduction de dimensionnalité et visualisation.
- **Données** : Iris dataset.
- **Techniques** : Normalisation des données, projection 2D/3D.

### 2. [Arbres de Décision](./Arbres_de_Décision/)
- **Objectif** : Classification binaire.
- **Données** : Titanic survivors dataset.
- **Modèle** : `DecisionTreeClassifier`.

### 3. [Clustering - K-Means](./Clustering_-_KMEANS/)
- **Objectif** : Segmentation de données (apprentissage non supervisé).
- **Données** : Mall Customers (segmentation de clientèle).

### 4. [Gradient Boosting](./Gradient_Boosting/)
- **Objectif** : Régression pour la prédiction de prix.
- **Données** : Immobilier au Maroc (Location Data).
- **Modèle** : `GradientBoostingRegressor`.

### 5. [Naive Bayes](./Naive_Bayes/)
- **Objectif** : Classification probabiliste.
- **Données** : Titanic dataset.
- **Variantes** : Gaussian, Multinomial, Categorical Naive Bayes.

### 6. [Régression Linéaire](./Regression_Lineaire/)
- **Objectif** : Prédiction de valeurs continues.
- **Données** : Coûts et revenus de films.

### 7. [Régression Logistique](./Regression_logistique/)
- **Objectif** : Classification binaire (médicale).
- **Données** : Breast Cancer Diagnostic dataset.

### 8. [Support Vector Machine (SVM)](./Support_Vector_Machine/)
- **Objectif** : Classification et prédiction de ventes.
- **Données** : Ventes commerciales et données utilisateurs.

---

## Comment utiliser ce dépôt

Chaque répertoire contient son propre fichier `README.md` avec des détails spécifiques sur les implémentations et les jeux de données utilisés. Les scripts Python peuvent être exécutés directement, tandis que les notebooks offrent une exploration plus interactive.

### Prérequis
- Python 3.x
- Bibliothèques : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
