import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend; required when running inside a web server
import matplotlib.pyplot as plt
import pandas as pd

def plot_sentiment_distribution(data, output_path='static/sentiment_distribution.png'):
    """Render a pie chart of sentiment counts and save it to `output_path`."""
    sentiment_counts = data['sentiment'].value_counts()
    labels = sentiment_counts.index
    values = sentiment_counts.values

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Sentiment Distribution')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path
