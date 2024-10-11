import plotly.graph_objects as go
import pandas as pd

# Cargar el CSV
data = pd.read_csv('./output/ej4/ej4.csv')

# Crear la figura
fig = go.Figure()

# Agregar los datos de las épocas y predicciones correctas
fig.add_trace(go.Scatter(x=data['Epoch'], y=data['CorrectPredictions'],
                         mode='lines+markers', name='Predicciones Correctas'))

# Personalizar el gráfico
fig.update_layout(title='Evolución de las Predicciones Correctas según las épocas a partir de los 10000 inputs de testing',
                  xaxis_title='Épocas',
                  yaxis_title='Predicciones Correctas',
                  template='plotly_white',
                  font=dict(size=22),  # Cambiar el tamaño de la fuente general
                    title_font=dict(size=26))

# Mostrar el gráfico
fig.show()