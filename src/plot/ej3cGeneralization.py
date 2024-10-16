import pandas as pd
import plotly.express as px

# Leer el CSV con los valores de recall por clase
data = pd.read_csv('./output/ej3c/generalization.csv')

# Preparar los datos para el gráfico
clases = data.columns
valores = data.iloc[0]

# Crear el gráfico con Plotly
fig = px.bar(
    x=clases, 
    y=valores, 
    labels={'x': 'Clase', 'y': 'Recall'}, 
    title='Recall por Clase'
)

# Personalizar el estilo del gráfico
fig.update_layout(
    title={'font': {'size': 24}},  # Tamaño del título
    xaxis={'title': {'font': {'size': 18}}, 'tickfont': {'size': 16}},  # Eje X
    yaxis={'title': {'font': {'size': 18}}, 'tickfont': {'size': 16}},  # Eje Y
)

# Mostrar el gráfico
fig.show()
