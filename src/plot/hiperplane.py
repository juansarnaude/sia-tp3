import numpy as np
import plotly.graph_objs as go

# Definir la función de la línea
def calcular_x2(x1, w1, w2, b):
    return - (w1 * x1 + b) / w2

def graph_hiperplane(w1, w2, b):
    # Generar puntos en el eje x1
    x1 = np.linspace(-10, 10, 100)

    # Calcular los valores correspondientes de x2
    x2 = calcular_x2(x1, w1, w2, b)

    # Crear los puntos de ejemplo para las dos clases
    X = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
    y = np.array([1, 0, 1, 1])  # Etiquetas de clase (0 o 1)

    # Crear el trazo para la línea de decisión
    linea_decision = go.Scatter(
        x=x1,
        y=x2,
        mode='lines',
        name='Línea de decisión',
        line=dict(color='black')
    )

    # Crear el trazo para los puntos de la clase 0 (rojo)
    puntos_clase_0 = go.Scatter(
        x=X[y == 0][:, 0],
        y=X[y == 0][:, 1],
        mode='markers',
        name='Clase 0',
        marker=dict(color='red', size=10)
    )

    # Crear el trazo para los puntos de la clase 1 (azul)
    puntos_clase_1 = go.Scatter(
        x=X[y == 1][:, 0],
        y=X[y == 1][:, 1],
        mode='markers',
        name='Clase 1',
        marker=dict(color='blue', size=10)
    )

    # Crear la figura con los trazos
    layout = go.Layout(
        title="Línea de decisión de un Perceptrón",
        xaxis=dict(title="x1", range=[-1.5, 1.5]),  # Fijar rango de x1
        yaxis=dict(title="x2", range=[-1.5, 1.5]),  # Fijar rango de x2
        showlegend=True
    )

    fig = go.Figure(data=[linea_decision, puntos_clase_0, puntos_clase_1], layout=layout)

    # Mostrar la gráfica
    fig.show()

    
# # Pesos y sesgo después del entrenamiento del perceptrón
# w1, w2 = 2, -3  # Pesos
# b = 1  # Sesgo (bias)

# graph_hiperplane(w1,w2,b)
