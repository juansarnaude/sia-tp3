import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.utils.functions import gaussian_noise

# Cargar el DataFrame
df = pd.read_csv("./input/ej3/ej3.txt", delimiter=' ', header=None)

# Eliminar la última columna
df = df.iloc[:, :-1]

# Crear listas de matrices de 7 filas
matrix_list = [df.iloc[i:i + 7, :] for i in range(0, len(df), 7)]

# Seleccionar la matriz en la posición 3
matrix3 = matrix_list[3]

# Graficar la matriz
fig = go.Figure(data=go.Heatmap(
    z=matrix3,  # Matriz con los valores a graficar
    text=np.round(matrix3, 2),  # Texto que aparecerá en cada celda (redondeado a 2 decimales)
    texttemplate="%{text}",  # Muestra el valor dentro de cada celda
    textfont={"size":24},  # Tamaño del texto
    colorscale='Blues',  # Puedes cambiar el esquema de colores (Viridis, Plasma, etc.)
    colorbar=dict(
        titleside='right',
        titlefont={'size':20},  # Tamaño de la fuente del título de la barra de color
        tickfont={'size':20}  # Tamaño de la fuente de los números de la barra de color
    )
))

# Añadir título y ajustar diseño
fig.update_layout(
    title='Heatmap con valores numéricos',
    xaxis_showgrid=False,  # Oculta la cuadrícula del eje x
    yaxis_showgrid=False,  # Oculta la cuadrícula del eje y
    xaxis_showticklabels=False,  # Oculta las etiquetas del eje x
    yaxis_showticklabels=False   # Oculta las etiquetas del eje y
)

# Mostrar el gráfico
fig.show()

# Convertimos la matriz a float para que el ruido pueda sumarse correctamente
matriz_float = matrix3.astype(float)

# Generamos ruido gaussiano con la misma forma que la matriz
ruido = np.random.normal(loc=0, scale=0.2, size=matrix3.shape)

# Sumamos el ruido a la matriz original
matrix3_noisy = matriz_float + ruido

# # Podemos forzar los valores a estar entre 0 y 1 si fuera necesario (opcional)
# matrix3_noisy = np.clip(matrix3_noisy, 0, 1)

# Definir una escala de colores centrada en 0 (rojo-blanco-rojo)
red_0_center = [
    [0, 'red'],      # Valor mínimo
    [0.5, 'white'],  # Centro en 0
    [1, 'red']     # Valor máximo
]

# Graficar la matriz
fig2 = go.Figure(data=go.Heatmap(
    z=ruido,  # Matriz con los valores a graficar
    text=np.round(ruido, 2),  # Texto que aparecerá en cada celda (redondeado a 2 decimales)
    texttemplate="%{text}",  # Muestra el valor dentro de cada celda
    textfont={"size":24},  # Tamaño del texto
    colorscale=red_0_center,  # Puedes cambiar el esquema de colores (Viridis, Plasma, etc.)
    colorbar=dict(
        titleside='right',
        titlefont={'size':20},  # Tamaño de la fuente del título de la barra de color
        tickfont={'size':20}  # Tamaño de la fuente de los números de la barra de color
    )
))

# Añadir título y ajustar diseño
fig2.update_layout(
    title='Heatmap con valores numéricos',
    xaxis_showgrid=False,  # Oculta la cuadrícula del eje x
    yaxis_showgrid=False,  # Oculta la cuadrícula del eje y
    xaxis_showticklabels=False,  # Oculta las etiquetas del eje x
    yaxis_showticklabels=False   # Oculta las etiquetas del eje y
)

# Mostrar el gráfico
fig2.show()

# Graficar la matriz
fig3 = go.Figure(data=go.Heatmap(
    z=matrix3_noisy,  # Matriz con los valores a graficar
    text=np.round(matrix3_noisy, 2),  # Texto que aparecerá en cada celda (redondeado a 2 decimales)
    texttemplate="%{text}",  # Muestra el valor dentro de cada celda
    textfont={"size":24},  # Tamaño del texto
    colorscale='Blues',  # Puedes cambiar el esquema de colores (Viridis, Plasma, etc.)
    colorbar=dict(
        titleside='right',
        titlefont={'size':20},  # Tamaño de la fuente del título de la barra de color
        tickfont={'size':20}  # Tamaño de la fuente de los números de la barra de color
    )
))

# Añadir título y ajustar diseño
fig3.update_layout(
    title='Heatmap con valores numéricos',
    xaxis_showgrid=False,  # Oculta la cuadrícula del eje x
    yaxis_showgrid=False,  # Oculta la cuadrícula del eje y
    xaxis_showticklabels=False,  # Oculta las etiquetas del eje x
    yaxis_showticklabels=False   # Oculta las etiquetas del eje y
)

# Mostrar el gráfico
fig3.show()

