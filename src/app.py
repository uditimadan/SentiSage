import os

from flask import Flask, render_template, request
import pandas as pd
from src.model import SentimentModel
from src.visualization import plot_sentiment_distribution

# Resolve templates/static relative to the project root so the app works
# whether it's launched as `python -m src.app` or from another working dir.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
)

# Load the pre-trained model
model = SentimentModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = model.predict(text)

    sentiment = ["Negative", "Neutral", "Positive"][prediction]
    return render_template('index.html', sentiment=sentiment, text=text)

@app.route('/visualize')
def visualize():
    data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'dataset.csv'))
    plot_sentiment_distribution(data, output_path=os.path.join(BASE_DIR, 'static', 'sentiment_distribution.png'))
    return (
        '<div class="container text-center mt-5">'
        '<h2>Sentiment Distribution</h2>'
        '<img src="/static/sentiment_distribution.png" alt="Sentiment distribution chart" class="img-fluid">'
        '<p><a href="/">Back</a></p>'
        '</div>'
    )

if __name__ == '__main__':
    app.run(debug=True)
