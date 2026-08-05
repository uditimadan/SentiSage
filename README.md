# SentiSage
A simple but unique project for text classification and sentiment analysis using BERT with an emphasis on custom datasets and more advanced visualizations.

<img width="1178" height="1252" alt="SentiSagePredition" src="https://github.com/user-attachments/assets/8dd04581-0f4c-468d-93fe-a238eefebaa2" />


## Features
- **Sentiment Classification** (Positive, Negative, Neutral)
  Train the model on a custom dataset, such as movie reviews, product reviews, or social media posts.
- **Multi-Classification (Emotion Detection)**
  Detect emotions like Happy, Sad, Angry, etc., using a labeled dataset.
- **Data Preprocessing**
  Clean and preprocess textual data before feeding it into the BERT model.
- **Interactive Frontend with Graphs**
  Visualize the results of sentiment analysis using charts (like pie charts and bar graphs).
- **User Input**
  Allow users to input their own text for real-time sentiment analysis.
- **Custom Visualization Dashboard**
  Display results with clear visuals using libraries like Matplotlib or Plotly.

## Requirements
- Python 3.9+ (tested with 3.12)
- pip

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/uditimadan/sentiSage.git
   cd sentiSage
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python -m src.app
   ```
   Run it as a module (`-m src.app`) from the project root, not `python src/app.py` — the app imports its own
   `src` package (e.g. `from src.model import SentimentModel`), which only resolves correctly when Python is
   started from the project root with `-m`.

5. Open **http://127.0.0.1:5000** in your browser.
   - Enter text on the home page and submit to see a sentiment prediction.
   - Click "View Sentiment Distribution" (`/visualize`) to see a pie chart generated from `data/dataset.csv`.

On first run, the app downloads the `nlptown/bert-base-multilingual-uncased-sentiment` weights and
tokenizer from Hugging Face (~650MB), so the first startup takes a minute or two and requires an internet
connection and enough free disk space. Subsequent runs use the local Hugging Face cache
(`~/.cache/huggingface`) and start much faster.

### Running with Docker
```bash
docker build -t sentisage .
docker run -p 5000:5000 sentisage
```

## How sentiment prediction works
`SentimentModel` (`src/model.py`) uses [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment),
a BERT checkpoint already fine-tuned to rate text 1-5 stars, and buckets that rating down to the app's
Negative / Neutral / Positive labels (1-2 stars → Negative, 3 → Neutral, 4-5 → Positive). This gives real,
input-dependent predictions out of the box, without requiring you to train anything first.

Plain `bert-base-uncased` (the base encoder, no sentiment head) was tried first, but a freshly initialized
classification head just outputs noise — every prediction collapsed to the same label regardless of input.
`models/bert_sentiment_model.pth` is an empty placeholder and isn't loaded anywhere; if you fine-tune your
own checkpoint (e.g. starting from `notebooks/sentiment_analysis.ipynb`) and want to use it instead, point
`SentimentModel(checkpoint=...)` at it.

## Known limitations
- Sentiment models like this are calibrated on opinionated text (reviews, comments). Purely factual,
  opinion-free sentences ("the meeting is at 3pm") can land in any of the three buckets somewhat
  unpredictably, since the model has no real opinion to report — this is expected, not a bug.
- `data/dataset.csv` is a tiny 3-row sample, only enough to exercise the `/visualize` chart.
