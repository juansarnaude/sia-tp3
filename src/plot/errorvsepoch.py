import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

# Leer los datos del archivo CSV
data = pd.read_csv('./output/ej3b/error.csv')

# Crear una figura con Plotly
fig = go.Figure()

# Añadir la línea de error de entrenamiento
fig.add_trace(go.Scatter(
    x=data['Epoch'],
    y=data['Error'],
    mode='lines',
    name='Error (Entrenamiento)',
    marker=dict(size=8),
    line=dict(width=2)
))

# Añadir la línea de error de prueba
fig.add_trace(go.Scatter(
    x=data['Epoch'],
    y=data['Test Error'],
    mode='lines',
    name='Error (Prueba)',
    marker=dict(size=8),
    line=dict(width=2, dash='dash')
))

# Configurar el diseño de la figura
fig.update_layout(
    title='Evolución del Error a lo Largo de las Épocas',
    xaxis_title='Época',
    yaxis_title='Error',
    template='plotly_white'
)

# Guardar la figura como un archivo HTML
pio.write_html(fig, 'error_evolution.html')

# Mostrar la figura
fig.show()
