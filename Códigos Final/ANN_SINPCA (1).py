
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:31:03 2025

@author: valer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Cargar datodata
data= pd.read_csv("r4.csv")
seed = 143
#f_0 = df[df['riesgo_hipertension'] == 0].sample(n=1200, random_state=42)
#f_1 = df[df['riesgo_hipertension'] == 1].sample(n=1200, random_state=42)
#ata = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)

# Preparar x e y
x = data.drop(["riesgo_hipertension", "FOLIO_I"], axis=1).to_numpy()
y = data['riesgo_hipertension'].to_numpy()

# División inicial: 70% entrenamiento + validación, 30% prueba final
x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=seed)

# División secundaria: de x_temp, 70% a entrenamiento y 30% a validación
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.3, shuffle=True, random_state=seed)

# Normalización
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Guardar scaler
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# One-hot encoding
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Modelo
shape = x_train_scaled.shape[1]
input_layer = Input(shape=(shape,))
hidden_layer1 = Dense(100, activation='sigmoid')(input_layer)
hidden_layer1 = tf.keras.layers.Dropout(0.2)(hidden_layer1)
hidden_layer2 = Dense(30, activation='sigmoid')(hidden_layer1)
h2 = tf.keras.layers.Dropout(0.1)(hidden_layer2)
h2 = Dense(30, activation='sigmoid')(h2)
output_layer = Dense(2, activation='softmax')(h2)

model = Model(inputs=input_layer, outputs=output_layer)

# Balanceo de clases
classes = np.unique(y_train.argmax(axis=1))
weights = compute_class_weight('balanced', classes=classes, y=y_train.argmax(axis=1))
d = dict(enumerate(weights, 0))

# Compilación
cce = tf.keras.losses.CategoricalFocalCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cce, metrics=['accuracy'])

model.summary()

# Checkpoint
checkpoint = ModelCheckpoint('Mejor_modelo_101_3.keras', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

# Entrenamiento
history = model.fit(x_train_scaled, y_train, epochs=20, class_weight=d,
                    validation_data=(x_val_scaled, y_val),
                    callbacks=[checkpoint], verbose=1)

# Evaluación
test_loss, test_acc = model.evaluate(x_test_scaled, y_test)
print(f"Test accuracy: {test_acc}, test loss {test_loss}")

# Curvas de entrenamiento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Precisión de entrenamiento')
plt.plot(epochs, val_acc, 'b', label='Precisión de validación')
plt.title('Precisión de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Pérdida de entrenamiento')
plt.plot(epochs, val_loss, 'b', label='Pérdida de validación')
plt.title('Pérdida de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Cargar modelo y scaler
loaded_model = tf.keras.models.load_model("Mejor_modelo_101_3.keras")
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Función de predicción
def predict_output(input_data):
    input_data_scaled = loaded_scaler.transform(input_data)
    return input_data_scaled

# Predicción y evaluación
x_a_probar = x_test
targets = y_test
prediction = loaded_model(predict_output(x_a_probar))
prediction_label = np.argmax(prediction, axis=1)
targets = np.argmax(targets, axis=1)

# Matriz de confusión
def confusion_matrix_custom(predictions, targets, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(predictions, targets):
        matrix[t, p] += 1
    return matrix

num_classes = 2
matrix = confusion_matrix_custom(prediction_label, targets, num_classes)
accuracy = accuracy_score(targets, prediction_label) * 100
precision = precision_score(targets, prediction_label, average='weighted')
recall = recall_score(targets, prediction_label, average='weighted')
f1 = f1_score(targets, prediction_label, average='weighted')

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f"Matriz de Confusión\nAccuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f}")
plt.show()

if f1>0.92:
    model.save('red90.keras')#G#arda en formato nativo de Keras
