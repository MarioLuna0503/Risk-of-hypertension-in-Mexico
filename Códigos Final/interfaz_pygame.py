import pygame
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
from BayesianoIngenuo import entrenar_y_preparar, predecir_una_muestra


# Inicializar Pygame
pygame.init()
pygame.font.init()

# Cargar datos y modelo
data = pd.read_csv("r4.csv")
columnas = data.columns[1:-1]
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = tf.keras.models.load_model("red90.keras")
svm_model = pickle.load(open("svm_model.pkl", "rb"))
svm_scaler = pickle.load(open("svm_scaler.pkl", "rb"))


#bayesiano
parametros, X_min, X_max = entrenar_y_preparar()



# Configuración
WIDTH, HEIGHT = 1000, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predicción de Riesgo de Hipertensión")

# Colores
OLIVE = (107, 142, 35)
PASTEL_GREEN = (200, 255, 200)
WHITE = (255, 255, 255)
DARK_GREEN = (34, 85, 34)
BLACK = (0, 0, 0)

# Fuentes
title_font = pygame.font.SysFont("georgia", 42, bold=True)
input_font = pygame.font.SysFont("arial", 20)
button_font = pygame.font.SysFont("arial", 24, bold=True)

# Campos de entrada
input_boxes = []
box_width, box_height = 200, 30
padding_x, padding_y = 100, 25
start_x = 80
start_y = 100
cols = 3

for i, col in enumerate(columnas):
    col_index = i % cols
    row_index = i // cols
    x = start_x + col_index * (box_width + padding_x)
    y = start_y + row_index * (box_height + padding_y)
    rect = pygame.Rect(x, y, box_width, box_height)
    input_boxes.append({"rect": rect, "text": "", "label": col})

# Botones
button_rect = pygame.Rect(WIDTH//2 - 220, HEIGHT - 80, 200, 50)  # Verificar
clear_button_rect = pygame.Rect(WIDTH//2 + 20, HEIGHT - 80, 200, 50)  # Borrar Todo

def predecir():
    try:
        valores = []
        for field in input_boxes:
            texto = field["text"]
            if texto.strip() == "":
                return "Llena todos los campos"
            valores.append(float(texto))
        X = np.array(valores).reshape(1, -1)

        # Red neuronal
        X_scaled_nn = scaler.transform(X)
        pred_nn = model.predict(X_scaled_nn)
        resultado_nn = np.argmax(pred_nn)
        texto_nn = "ALTO" if resultado_nn == 1 else "BAJO"

        # Bayesiano
        resultado_bayes = predecir_una_muestra(X[0], parametros, X_min, X_max)
        texto_bayes = "ALTO" if resultado_bayes == 1 else "BAJO"

        # SVM
        X_scaled_svm = svm_scaler.transform(X)
        resultado_svm = svm_model.predict(X_scaled_svm)[0]
        texto_svm = "ALTO" if resultado_svm == 1 else "BAJO"

        return f"Red Neuronal: {texto_nn}  |  Bayesiano: {texto_bayes}  |  SVM: {texto_svm}"

    except ValueError:
        return "Solo se permiten números válidos"

    
def limpiar_campos():
    for box in input_boxes:
        box["text"] = ""

# Loop principal
running = True
active_box = None
resultado_prediccion = ""
clock = pygame.time.Clock()

while running:
    screen.fill(PASTEL_GREEN)

    # Título
    title_surface = title_font.render("Predicción de Riesgo de Hipertensión", True, DARK_GREEN)
    screen.blit(title_surface, (WIDTH//2 - title_surface.get_width()//2, 20))

    # Dibujar campos
    for box in input_boxes:
        pygame.draw.rect(screen, WHITE, box["rect"], border_radius=5)
        pygame.draw.rect(screen, OLIVE, box["rect"], 2, border_radius=5)
        text_surface = input_font.render(box["text"], True, BLACK)
        screen.blit(text_surface, (box["rect"].x + 5, box["rect"].y + 5))
        label_surface = input_font.render(box["label"], True, DARK_GREEN)
        screen.blit(label_surface, (box["rect"].x, box["rect"].y - 20))

    # Botón Verificar
    pygame.draw.rect(screen, OLIVE, button_rect, border_radius=8)
    button_text = button_font.render("Verificar", True, WHITE)
    screen.blit(button_text, (button_rect.centerx - button_text.get_width() // 2,
                              button_rect.centery - button_text.get_height() // 2))

    # Botón Borrar Todo
    pygame.draw.rect(screen, DARK_GREEN, clear_button_rect, border_radius=8)
    clear_text = button_font.render("Borrar Todo", True, WHITE)
    screen.blit(clear_text, (clear_button_rect.centerx - clear_text.get_width() // 2,
                             clear_button_rect.centery - clear_text.get_height() // 2))

    # Resultado
    if resultado_prediccion:
        resultado_surface = button_font.render(resultado_prediccion, True, DARK_GREEN)
        screen.blit(resultado_surface, (WIDTH//2 - resultado_surface.get_width()//2, HEIGHT - 130))

    # Eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                resultado_prediccion = predecir()
            elif clear_button_rect.collidepoint(event.pos):
                limpiar_campos()
                resultado_prediccion = ""
            else:
                for box in input_boxes:
                    if box["rect"].collidepoint(event.pos):
                        active_box = box
                        break
                else:
                    active_box = None

        elif event.type == pygame.KEYDOWN:
            if active_box is not None:
                if event.key == pygame.K_BACKSPACE:
                    active_box["text"] = active_box["text"][:-1]
                elif event.key == pygame.K_RETURN:
                    pass
                else:
                    char = event.unicode
                    # Permitir solo números, punto decimal y signo negativo inicial
                    if (char.isdigit() or 
                        (char == "." and "." not in active_box["text"]) or 
                        (char == "-" and active_box["text"] == "")):
                        active_box["text"] += char

    pygame.display.flip()
    clock.tick(30)

pygame.quit()