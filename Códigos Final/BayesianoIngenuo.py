# -*- coding: utf-8 -*- 
"""
Created on Wed May  7 00:14:13 2025
@author: mayit
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
datos = pd.read_csv("r4.csv")

# Se quita la primer columna (Folios)
datos = datos.iloc[:, 1:]

# Separar características y etiquetas
X = datos.iloc[:, :-1].values
y = datos.iloc[:, -1].values

# Dividir datos: 70% entrenamiento, 30% prueba (ANTES de normalizar)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Dentro del conjunto de entrenamiento: 70% entrenamiento, 30% validación
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full)

# Normalizar con estadísticas del conjunto de entrenamiento
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)


  

# Normalizar conjuntos
X_train = (X_train - X_min) / (X_max - X_min + 1e-6)
X_val   = (X_val   - X_min) / (X_max - X_min + 1e-6)
X_test  = (X_test  - X_min) / (X_max - X_min + 1e-6)


# funcion calcular parametros recibe caracteristicas y vector de clases
def calcular_parametros(X, y):
    #se obtienen las clases de la matriz
    clases = np.unique(y)
    #diccionario
    parametros = {}
    
    #ciclo for 
    for c in clases:
        #Filtro para las muestras pertenecientes a cada clase
        X_c = X[y == c]
        #Se obtiene la media de cada columna para cada caracteristica para la clase
        media = X_c.mean(axis=0)
        #Se obtiene la desviación estandar para cada caracteristica de la clase
        std = X_c.std(axis=0, ddof=1) + 1e-6  # para evitar división por cero
        #Se calcula la frecuencia relativa número de muestras de clase c dividido entre el total de muestras
        prior = X_c.shape[0] / X.shape[0]
        #Se almacenan los datos en el diccionario
        parametros[c] = {'media': media, 'std': std, 'prior': prior}
    return parametros




def calcular_log_prob(x, media, std):
    exponent = - ((x - media) ** 2) / (2 * std ** 2)
    probabilidad = exponent - np.log(std) - 0.5 * np.log(2 * np.pi)
    return np.sum(probabilidad)


# Predecir clase para cada muestra
def predecir(X, parametros):
    predicciones = []
    for x in X:
        log_probs = {}
        for c in parametros:
            media = parametros[c]['media']
            std = parametros[c]['std']
            prior = np.log(parametros[c]['prior'])
            log_likelihood = calcular_log_prob(x, media, std)
            log_probs[c] = prior + log_likelihood
        predicciones.append(max(log_probs, key=log_probs.get))
    return np.array(predicciones)


# Predecir probabilidades para cada clase
def predecir_probabilidades(X, parametros):
    probabilidades = []
    for x in X:
        log_probs = {}
        for c in parametros:
            media = parametros[c]['media']
            std = parametros[c]['std']
            prior = np.log(parametros[c]['prior'])
            log_likelihood = calcular_log_prob(x, media, std)
            log_probs[c] = prior + log_likelihood
        
        log_prob_vals = np.array(list(log_probs.values()))
        max_log = np.max(log_prob_vals)
        probs_exp = np.exp(log_prob_vals - max_log)  # Estabilidad numérica
        probs = probs_exp / np.sum(probs_exp)
        probabilidades.append(probs)
    return np.array(probabilidades)

# Entrenamiento
parametros = calcular_parametros(X_train, y_train)

# Validación para seleccionar los mejores parámetros
best_f1 = 0
best_parametros = parametros

for epoch in range(1, 21):
    pred_val = predecir(X_val, parametros)
    f1 = f1_score(y_val, pred_val)
    print(f"Epoch {epoch} - F1 Score: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_parametros = parametros
    else:
        print("Early stopping triggered.")
        break

# Evaluación final en test
y_pred = predecir(X_test, best_parametros)

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Métricas en el conjunto de prueba ---")
print(f"Exactitud (Accuracy):      {accuracy:.4f}")
print(f"Precisión (Precision):     {precision:.4f}")
print(f"Estabilidad (Recall):      {recall:.4f}")
print(f"Desempeño (F1-Score):      {f1:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()

# Obtener probabilidades de predicción
y_pred_prob = predecir_probabilidades(X_test, best_parametros)


# Mostrar primeras 10 filas
clases_ordenadas = sorted(np.unique(y))
columnas = [f'Prob de clase {c}' for c in clases_ordenadas]
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=columnas)
print("\nPrimeras 10 filas de probabilidades predichas:")
print(y_pred_prob_df.head(10))

# Histograma de probabilidades para la clase positiva
y_pred1 = y_pred_prob[:, clases_ordenadas.index(1)]

plt.rcParams['font.size'] = 12
plt.hist(y_pred1, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma de probabilidades predichas para clase 1')
plt.xlabel('Probabilidades predichas')
plt.ylabel('Frecuencia')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()


def entrenar_y_preparar():
    global best_parametros, X_min, X_max
    return best_parametros, X_min, X_max

def predecir_una_muestra(muestra, parametros, X_min, X_max):
    muestra_norm = (np.array(muestra) - X_min) / (X_max - X_min + 1e-6)
    proba = predecir_probabilidades([muestra_norm], parametros)
    pred = predecir([muestra_norm], parametros)
    return int(pred[0]), proba[0]
