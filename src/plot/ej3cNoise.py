import pandas as pd
import plotly.express as px

# Cargar los datos desde el CSV
df = pd.read_csv("./output/ej3c/noise.csv")

# Crear el gráfico de línea
fig = px.line(
    df, 
    x="standard_deviation", 
    y="f1_score", 
    title="F1 Score vs. Standard Deviation",
    labels={
        "standard_deviation": "Standard Deviation", 
        "f1_score": "F1 Score"
    },
    markers=True  # Agregar marcadores a la línea
)

# Personalizar tamaño de letra y ejes
fig.update_layout(
    font=dict(size=28),  # Tamaño general del texto
    xaxis_title_font=dict(size=24),  # Tamaño del título del eje X
    yaxis_title_font=dict(size=24),  # Tamaño del título del eje Y
    xaxis=dict(
        tickmode='linear',  # Modo de tick lineal
        dtick=0.5,  # Intervalo de 0.5 entre ticks
        tickfont=dict(size=24)  # Tamaño de fuente para los ticks del eje X
    ),
    yaxis=dict(tickfont=dict(size=24)),  # Tamaño de los números en el eje Y
)

# Mostrar el gráfico
fig.show()
