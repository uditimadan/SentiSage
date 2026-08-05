import pandas as pd
from transformers import BertTokenizer

_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(file_path):
    """Load a CSV dataset and tokenize its `text` column (used for training/visualization)."""
    df = pd.read_csv(file_path)
    df['tokens'] = df['text'].apply(lambda x: _tokenizer.encode(x, add_special_tokens=True))
    return df

def preprocess_text(text):
    """Tokenize a single string of text (used for real-time prediction)."""
    return _tokenizer.encode(text, add_special_tokens=True)
