import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from scipy.special import expit

# Charger le jeu de données depuis 'breast-cancer.csv'
data = pd.read_csv('breast-cancer.csv')

# Remplacer les valeurs dans la colonne 'diagnosis'
data['diagnosis'] = data['diagnosis'].replace({'M': 0, 'B': 1})

# Séparer les variables de caractéristiques (X) et la variable cible (y)
X = data[['symmetry_worst']]  # Sélectionner uniquement la colonne 'symmetry_worst'
y = data['diagnosis']

# Divisez les données en ensembles d'apprentissage (70%) et de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer un modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle sur l'ensemble d'apprentissage
model.fit(X_train, y_train)

# Prédire les étiquettes sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer la performance du modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Précision du modèle : {:.2f}%".format(accuracy * 100))
print("Matrice de confusion :")
print(conf_matrix)
print("Rapport de classification :")
print(class_report)

# Tracer le graphe de la classification
plt.figure(figsize=(10, 6))
plt.scatter(X_test[y_pred == 0], [0] * np.sum(y_pred == 0), marker='o', color='red', label="Tumeurs Malignes")
plt.scatter(X_test[y_pred == 1], [0] * np.sum(y_pred == 1), marker='o', color='green', label="Tumeurs Bénignes")
plt.xlabel("Symmetry Worst")
plt.ylabel("Diagnosis (0: Maligne, 1: Bénigne)")
plt.legend()
plt.title("Classification des Tumeurs en fonction de Symmetry Worst")
plt.grid(True)

# Tracer la fonction sigmoïde
sigmoid_x = np.linspace(X['symmetry_worst'].min(), X['symmetry_worst'].max(), 100)
sigmoid_y = expit(sigmoid_x * model.coef_ + model.intercept_).ravel()
plt.plot(sigmoid_x, sigmoid_y, color='orange', label='Fonction Sigmoïde')

plt.legend()

plt.show()
