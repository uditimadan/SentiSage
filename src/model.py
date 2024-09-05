import torch
from transformers import BertForSequenceClassification, AdamW

class SentimentModel:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
    
    def train(self, inputs, labels):
        self.model.train()
        outputs = self.model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        return torch.argmax(outputs.logits, dim=1).item()
