from flask import Flask, render_template, request
import pandas as pd
from src.preprocess import preprocess_data
from src.model import SentimentModel
from src.visualization import plot_sentiment_distribution

app = Flask(__name__)

# Load the pre-trained model
model = SentimentModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    preprocessed_data = preprocess_data(text)
    prediction = model.predict(preprocessed_data['tokens'].values[0])
    
    sentiment = ["Negative", "Neutral", "Positive"][prediction]
    return render_template('index.html', sentiment=sentiment, text=text)

@app.route('/visualize')
def visualize():
    data = pd.read_csv('data/dataset.csv')
    plot_sentiment_distribution(data)
    return "Sentiment distribution plotted!"

if __name__ == '__main__':
    app.run(debug=True)
