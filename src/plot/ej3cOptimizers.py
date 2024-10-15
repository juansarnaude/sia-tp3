import pandas as pd
import plotly.graph_objects as go

# Nombres de los archivos CSV (modifica estos nombres según tus archivos)
archivos = [
    "./output/ej3/output_gradient.csv",
    "./output/ej3/momentum.csv",
    "./output/ej3/adam.csv",
]

# Lista para almacenar los DataFrames y nombres de las arquitecturas
dataframes = []
nombres_modelos = ["Gradient Descent", "Momentum", "Adam"]

# Lista de colores a usar para cada modelo
colores = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# Leer cada archivo CSV y guardarlo con su nombre correspondiente
for archivo, nombre in zip(archivos, nombres_modelos):
    df = pd.read_csv(archivo)
    df["Modelo"] = nombre  # Añadir la columna de nombre del modelo
    dataframes.append(df)

# Concatenar todos los DataFrames en uno solo para facilitar el plotting
df_total = pd.concat(dataframes, ignore_index=True)

# Crear la figura usando plotly
fig = go.Figure()

# Agregar una línea para cada modelo
for nombre, color in zip(nombres_modelos, colores):
    df_modelo = df_total[df_total["Modelo"] == nombre]

    # Agregar la línea para el modelo con color específico
    fig.add_trace(go.Scatter(
        x=df_modelo["epoch"],
        y=df_modelo["f1_score"],
        mode='lines',
        line=dict(color=color),
        name=nombre
    ))

    # Marcar el último valor con un punto del mismo color que la línea, mostrando la época
    fig.add_trace(go.Scatter(
        x=[df_modelo["epoch"].iloc[-1]],  # Última época
        y=[df_modelo["f1_score"].iloc[-1]],  # Último f1_score
        mode='markers+text',
        marker=dict(size=10, color=color, symbol='circle'),  # Usar el mismo color
        text=[df_modelo["epoch"].iloc[-1]],  # Mostrar el valor de la época
        textposition="top center",  # Posicionar el texto encima del punto
        showlegend=False
    ))

# Configurar el layout del gráfico
fig.update_layout(
    title={
        'text': "Comparación de F1-Score por optimizador",
        'x': 0.5,  # Centrar el título
        'xanchor': 'center',
        'font': {'size': 28}  # Tamaño del título
    },
    xaxis={
        'title': {'text': "Época", 'font': {'size': 24}},  # Tamaño del título del eje X
        'tickfont': {'size': 20}  # Tamaño de los números del eje X
    },
    yaxis={
        'title': {'text': "F1-Score", 'font': {'size': 24}},  # Tamaño del título del eje Y
        'tickfont': {'size': 20}  # Tamaño de los números del eje Y
    },
    legend={
        'title': {'text': "Modelos", 'font': {'size': 24}},  # Tamaño del título de la leyenda
        'font': {'size': 20}  # Tamaño del texto de la leyenda
    },
    template="plotly"
)

# Mostrar la figura
fig.show()
