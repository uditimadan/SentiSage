import pandas as pd
from transformers import BertTokenizer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokens = df['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    df['tokens'] = tokens
    return df
