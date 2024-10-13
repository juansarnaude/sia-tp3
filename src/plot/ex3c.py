import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

with open("./configs/ej3c.json") as file:
    config = json.load(file)

    output_path = config["output_file"]

    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv(output_path)

    # Step 2: Define the metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Step 3: Generate a plot for each metric comparing train vs test
    for metric in metrics:
        fig = go.Figure()

        # Add training metric trace
        fig.add_trace(go.Scatter(
            x=df['epoch'],
            y=df[metric],
            mode='lines',
            name=f'Train {metric.capitalize()}'
        ))

        # Add test metric trace
        fig.add_trace(go.Scatter(
            x=df['epoch'],
            y=df[f'test_{metric}'],
            mode='lines',
            name=f'Test {metric.capitalize()}'
        ))

        # Step 4: Customize layout
        fig.update_layout(
            title=f'{metric.capitalize()} over Epochs',
            xaxis_title='Epoch',
            yaxis_title=metric.capitalize(),
            legend=dict(title='Dataset', x=0.85, y=1),
            template='plotly_white'
        )

        # Step 5: Show the plot
        fig.show()
