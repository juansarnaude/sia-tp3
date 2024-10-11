import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from src.metrics.Evaluator import Evaluator
from src.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from src.optimizer.GradientDescent import GradientDescent
from src.optimizer.Momentum import Momentum
from src.optimizer.Adam import Adam
from src.utils.functions import sigmoid, tanh, gaussian_noise, index_of_max_value

import json

with open("./configs/ej3c.json") as file:
    config = json.load(file)

    df = pd.read_csv(config["input_file"], delimiter=' ', header=None)

    df = df.iloc[:, :-1]

    matrix_list = [df.iloc[i:i + 7, :] for i in range(0, len(df), 7)]
    
    # Configurar el diseño de la figura
    fig.update_layout(
        title='Evolución del error a lo Largo de las épocas con ruido gaussiano 0.5',
        xaxis_title='Época',
        yaxis_title='Error',
        template='plotly_white',
        font=dict(size=22),  # Cambiar el tamaño de la fuente general
        title_font=dict(size=26)  # Cambiar el tamaño de la fuente del título
    )

    flattened_matrixes = [matrix.values.flatten() for matrix in matrix_list]

    learning_rate=0.01

    # layers per output
    layer_sizes=config["layer_sizes"]

    activation_functions = ["tanh","sigmoid"]

    # Optimizer configuration
    optimizers = ["gradient_descent","adam","momentum"]

    # cutoffs
    epochs = 1000
    epsilon = 0.01

    # list of inputs
    inputs = flattened_matrixes


    standard_deviation = 0.2
    noisy_input = []
    for matrix in matrix_list:
        noisy_input.append(gaussian_noise(matrix=matrix, standard_deviation=standard_deviation).values.flatten())

    training_input = []
    for matrix in matrix_list:
        training_input.append(gaussian_noise(matrix=matrix, standard_deviation=0).values.flatten())

    for activation_function_str in activation_functions:
        if activation_function_str == "tanh":
            activation_function = tanh
        elif activation_function_str == "sigmoid":
            activation_function = sigmoid
        for optimizer_str in optimizers:
            optimizer = None
            if optimizer_str == "gradient_descent":
                optimizer = GradientDescent(learning_rate)
            elif optimizer_str == "momentum":
                optimizer = Momentum(learning_rate, 0.9)
            elif optimizer_str == "adam":
                optimizer = Adam(learning_rate, 0.9, 0.999)

            mlp = MultiLayerPerceptron(
                layer_sizes=layer_sizes,
                activation_function=activation_function,
                optimizer=optimizer,
                output_path=f"./output/ej3/mlp_{optimizer_str}_{activation_function_str}_05",
            )

            training_expected_values = [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            ]

            testing_expected_values = [0,1,2,3,4,5,6,7,8,9]

            evaluator = Evaluator(10, 10, 0.0001, index_of_max_value)

            accuracy, f1, precision, recall = evaluator.evaluate(mlp, training_input, training_expected_values, noisy_input, testing_expected_values, 1000)

            # Leer los datos del archivo CSV
            data = pd.read_csv(f"./output/ej3/mlp_{optimizer_str}_{activation_function_str}_05")

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
                title=f'Evolución del Error a lo Largo de las Épocas - {optimizer_str} - {activation_function_str} - Ruido Gaussiano {standard_deviation}',
                xaxis_title='Época',
                yaxis_title='Error',
                template='plotly_white'
            )

            # Guardar la figura como un archivo HTML
            pio.write_html(fig, 'error_evolution.html')

            # Mostrar la figura
            fig.show()
