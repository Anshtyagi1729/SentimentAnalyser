import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset Preparation
DATA_PATH = "training.1600000.processed.noemoticon.csv"
logger.info(f"Loading dataset from {DATA_PATH}")
try:
    data = pd.read_csv(DATA_PATH, encoding="ISO-8859-1", header=None)
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Filter out neutral tweets if any
data = data[data['target'].isin([0, 4])]
data['target'] = data['target'].map({0: 0, 4: 1})  # Map to binary sentiment
data = data.sample(n=150000, random_state=42)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
logger.info(f"Dataset split: Train size = {len(train_data)}, Test size = {len(test_data)}")

# Custom Tokenizer
def simple_tokenizer(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text.split()

# Vocabulary Builder
class Vocabulary:
    def __init__(self, max_size=10000):
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.max_size = max_size
        self.word_counts = {}

    def build_vocabulary(self, sentences):
        # Count word frequencies
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

        # Sort words by frequency and create vocabulary
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, _ in sorted_words[:self.max_size-2]:
            if word not in self.word_to_index:
                index = len(self.word_to_index)
                self.word_to_index[word] = index
                self.index_to_word[index] = word
        
        logger.info(f"Vocabulary built. Total words: {len(self.word_to_index)}")

    def encode(self, words, max_length=100):
        # Convert words to indices
        encoded = [self.word_to_index.get(word, 1) for word in words[:max_length]]
        # Pad or truncate
        encoded = encoded + [0] * (max_length - len(encoded))
        return encoded

# Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, dataframe, vocab, max_length=100):
        self.texts = dataframe['text'].apply(simple_tokenizer).tolist()
        self.labels = dataframe['target'].values
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded_text = self.vocab.encode(text)
        
        return (torch.tensor(encoded_text, dtype=torch.long), 
                torch.tensor(label, dtype=torch.float32))


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # RNN
        _, (hidden, _) = self.rnn(embedded)
        
        # Fully connected layer
        output = self.fc(hidden.squeeze(0))
        return output

# Training Function
def train_model(vocab):
    train_dataset = SentimentDataset(train_data, vocab)
    test_dataset = SentimentDataset(test_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    vocab_size = len(vocab.word_to_index)
    embedding_dim = 128
    hidden_dim = 256

    logger.info(f"Model parameters: Vocab Size = {vocab_size}, Embedding Dim = {embedding_dim}, Hidden Dim = {hidden_dim}")

    # Initialize model
    model = SentimentRNN(vocab_size, embedding_dim, hidden_dim)
    
   
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    criterion = criterion.to(device)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        logger.info(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

    torch.save(model.state_dict(), 'sentiment_model.pth')
    logger.info("Model training completed and saved.")
    return model, vocab

vocab = Vocabulary()
vocab.build_vocabulary(train_data['text'].apply(simple_tokenizer))

def test_prediction(text, model, vocab):
    logger.info(f"Testing prediction for text: {text}")
    
    tokenized = simple_tokenizer(text)
    encoded = vocab.encode(tokenized)
    
    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = tensor.to(device)
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor))
    
    sentiment = 'Positive' if prediction.item() > 0.5 else 'Negative'
    logger.info(f"Prediction: {sentiment} (Confidence: {prediction.item():.4f})")
    return sentiment

app = Flask(__name__)
model, vocab = train_model(vocab)

@app.route('/test_prediction', methods=['GET'])
def prediction_test():
    test_texts = [
        "I love this product, it's amazing!",
        "This is the worst experience ever.",
        "Not sure how I feel about this."
    ]
    
    results = []
    for text in test_texts:
        sentiment = test_prediction(text, model, vocab)
        results.append({'text': text, 'sentiment': sentiment})
    
    return jsonify(results)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data.get('text')
    
    tokenized = simple_tokenizer(text)
    encoded = vocab.encode(tokenized)
    
    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = tensor.to(device)
    model.to(device)

    model.eval()
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor))
    
    sentiment = 'Positive' if prediction.item() > 0.5 else 'Negative'
    return jsonify({'sentiment': sentiment, 'confidence': float(prediction.item())})

if __name__ == '__main__':
   
    test_texts = [
        "I love this product, it's amazing!",
        "This is the worst experience ever.",
        "Not sure how I feel about this."
    ]
    
    for text in test_texts:
        test_prediction(text, model, vocab)
    
    app.run(debug=False)