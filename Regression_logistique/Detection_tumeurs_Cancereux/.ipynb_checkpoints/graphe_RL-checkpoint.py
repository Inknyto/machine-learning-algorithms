import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# Charger le jeu de données depuis 'breast-cancer.csv'
data = pd.read_csv('breast-cancer.csv')

# Remplacer les valeurs dans la colonne 'diagnosis'
data['diagnosis'] = data['diagnosis'].replace({'M': 0, 'B': 1})

# Séparer les variables de caractéristiques (X) et la variable cible (y)
X = data.drop(columns=['id', 'diagnosis'])  
y = data['diagnosis']

# Divisez les données en ensembles d'apprentissage (70%) et de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer un modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle sur l'ensemble d'apprentissage
model.fit(X_train, y_train)

# Prédire les probabilités à l'aide du modèle
probabilities = model.predict_proba(X_test)[:, 1]

# Tracer la fonction sigmoïde
X_range = np.linspace(-10, 10, 1000)
sigmoid = 1 / (1 + np.exp(-X_range))

# Tracer les points de données en fonction des probabilités
plt.figure(figsize=(10, 6))
plt.plot(X_range, sigmoid, label="Fonction Sigmoïde", color='blue')
plt.scatter(X_test[probabilities < 0.5].iloc[:, 0], [0.5] * np.sum(probabilities < 0.5), marker='o', color='red', label="Tumeurs Malignes (Prob < 0.5)")
plt.scatter(X_test[probabilities >= 0.5].iloc[:, 0], [0.5] * np.sum(probabilities >= 0.5), marker='o', color='green', label="Tumeurs Bénignes (Prob >= 0.5)")
plt.xlabel("Valeurs de X")
plt.ylabel("Seuil de Probabilité (0.5)")
plt.legend()
plt.title("Fonction Sigmoïde et Séparation des Tumeurs")
plt.grid(True)
plt.show()
