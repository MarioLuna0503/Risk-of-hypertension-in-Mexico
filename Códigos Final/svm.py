# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:07:39 2025
@author: valer
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
import numpy as np

# Cargar y balancear datos
df = pd.read_csv("r4.csv")
df_0 = df[df['riesgo_hipertension'] == 0].sample(n=1200, random_state=42)
df_1 = df[df['riesgo_hipertension'] == 1].sample(n=1200, random_state=42)
df_balanced = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)

# Separar X y y
X = df_balanced.drop(columns=["riesgo_hipertension", "FOLIO_I"])
y = df_balanced['riesgo_hipertension']

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Entrenar modelo SVM lineal
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predicciones
y_pred = svm_model.predict(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No riesgo", "Riesgo"])
disp.plot(cmap='Blues')
plt.title(f"Matriz de Confusión\nAccuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1-score: {f1:.3f}")
plt.show()

# PCA para visualización
pca = PCA(n_components=2, random_state=42)
X_test_2D = pca.fit_transform(X_test)

# Entrenar SVM sobre datos 2D para visualización (sólo visual)
svm_vis = SVC(kernel='linear')
svm_vis.fit(X_test_2D, y_test)

# Crear malla para graficar frontera de decisión
x_min, x_max = X_test_2D[:, 0].min() - 1, X_test_2D[:, 0].max() + 1
y_min, y_max = X_test_2D[:, 1].min() - 1, X_test_2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Gráfica de clasificación
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
sns.scatterplot(x=X_test_2D[:, 0], y=X_test_2D[:, 1], hue=y_test, palette='coolwarm', edgecolor='k')

# Calcular hiperplano y márgenes
w = svm_vis.coef_[0]
b = svm_vis.intercept_[0]
x_vals = np.linspace(x_min, x_max, 300)
y_vals = -(w[0] / w[1]) * x_vals - b / w[1]
margin = 1 / np.sqrt(np.sum(w ** 2))
y_margin_up = y_vals + margin
y_margin_down = y_vals - margin

# Dibujar hiperplano y márgenes
plt.plot(x_vals, y_vals, 'k--', label='Hiperplano')
plt.plot(x_vals, y_margin_up, 'k:', label='Margen superior')
plt.plot(x_vals, y_margin_down, 'k:', label='Margen inferior')

# Vectores de soporte
support_vectors = svm_vis.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k', label='Vectores de soporte')

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("SVM lineal: Visualización del hiperplano en 2D")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Guardar modelo y escalador
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

with open("svm_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
