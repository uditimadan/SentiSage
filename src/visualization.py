import matplotlib.pyplot as plt
import pandas as pd

def plot_sentiment_distribution(data):
    sentiment_counts = data['sentiment'].value_counts()
    labels = sentiment_counts.index
    values = sentiment_counts.values

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Sentiment Distribution')
    plt.show()
