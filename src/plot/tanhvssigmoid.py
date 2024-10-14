import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Nombres de los archivos CSV (modifica estos nombres según tus archivos)
archivos = [
    "./output/ej3c/tanh.csv", 
    "./output/ej3c/sigmoid.csv",
]

# Lista para almacenar los DataFrames y nombres de las arquitecturas
dataframes = []
nombres_modelos = ["tanh", "sigmoid"]

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
for nombre in nombres_modelos:
    df_modelo = df_total[df_total["Modelo"] == nombre]
    fig.add_trace(go.Scatter(
        x=df_modelo["epoch"],
        y=df_modelo["f1_score"],
        mode='lines',
        name=nombre
    ))

# Configurar el layout del gráfico con tamaños más grandes
fig.update_layout(
    title={
        'text': "Comparación de F1-Score por función de activación",
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