import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer

# Plain `bert-base-uncased` has no sentiment head at all — asking for
# num_labels=3 just bolts on a randomly-initialized classifier, so every
# prediction is noise (and, since it's the same random weights for every
# request in a run, it tends to collapse to one constant class). This
# checkpoint is BERT already fine-tuned on real review data to predict a
# 1-5 star rating, which we bucket down into Negative/Neutral/Positive so
# /predict returns genuine, input-dependent results without requiring a
# custom training run first.
DEFAULT_CHECKPOINT = 'nlptown/bert-base-multilingual-uncased-sentiment'


class SentimentModel:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
        self.model = BertForSequenceClassification.from_pretrained(checkpoint)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

    def train(self, inputs, labels):
        self.model.train()
        outputs = self.model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def predict(self, text):
        """Predict sentiment for a piece of text.

        Returns 0 (Negative), 1 (Neutral), or 2 (Positive), matching the
        label order the rest of the app expects.
        """
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        star_rating = torch.argmax(outputs.logits, dim=1).item()  # 0-4 => 1-5 stars

        if star_rating <= 1:   # 1-2 stars
            return 0  # Negative
        if star_rating == 2:   # 3 stars
            return 1  # Neutral
        return 2                # 4-5 stars

